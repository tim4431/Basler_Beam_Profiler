"""pypylon driver for Basler cameras.

Nothing here is specific to one model: pixel formats, trigger mode, I/O lines,
exposure limits and pixel size all come from `cameraconfig`, and anything the
camera turns out not to support is logged and skipped rather than raised.  So a
new Basler works as soon as its `pixel_size` is in `camera_config.yaml`, and it
still opens (with uncalibrated microns) before that.

Unplugging is detected with `IsCameraDeviceRemoved`, which pylon keeps current,
so a timeout (long exposure, missed trigger) is never mistaken for a lost
camera.  Set PYLON_CAMEMU=2 to get two emulated cameras for testing.
"""

from __future__ import annotations

import logging

import numpy as np
from pypylon import pylon

from .. import cameraconfig
from .base import Backend, Camera, CameraDisconnected, DeviceInfo


class BaslerBackend(Backend):
    name = "basler"
    title = "Basler (pylon)"
    install_hint = "install the pylon SDK and `pip install pypylon`"

    def enumerate(self, full: bool = False) -> list[DeviceInfo]:
        return [
            DeviceInfo(
                backend=self.name,
                model=d.GetModelName(),
                serial=d.GetSerialNumber(),
                vendor=d.GetVendorName() or "Basler",
                address=d.GetFullName(),
            )
            for d in pylon.TlFactory.GetInstance().EnumerateDevices()
        ]

    def open(self, info: DeviceInfo, mode: str) -> Camera:
        cam = BaslerCamera(mode=mode, serial=info.serial)
        cam.device = info
        return cam

    def probe(self, info: DeviceInfo) -> dict:
        for d in pylon.TlFactory.GetInstance().EnumerateDevices():
            if d.GetSerialNumber() == info.serial:
                return probe_device(d)
        return {**super().probe(info), "error": "no longer connected"}


class BaslerCamera(Camera):
    backend = "basler"

    def __init__(self, mode: str = "16Bit", device_idx: int = 0, serial: str | None = None):
        """Open the camera with this serial number, or the `device_idx`-th one
        enumerated when no serial is given."""
        factory = pylon.TlFactory.GetInstance()
        if serial:
            wanted = pylon.DeviceInfo()
            wanted.SetSerialNumber(serial)
            device = factory.CreateFirstDevice(wanted)
        else:
            devices = factory.EnumerateDevices()
            if device_idx >= len(devices):
                raise IndexError(f"Only {len(devices)} cameras found, idx={device_idx}")
            device = factory.CreateDevice(devices[device_idx])
        self.camera = pylon.InstantCamera(device)
        self.device_info = self.camera.GetDeviceInfo()
        self.model_name = self.device_info.GetModelName()
        self.serial = self.device_info.GetSerialNumber()
        logging.info(f"Connecting to camera: {self.model_name} ({self.serial})")

        settings = cameraconfig.resolve(self.model_name, self.serial)
        super().__init__(mode, settings)
        logging.info(cameraconfig.describe(settings, self.model_name))

        try:
            self._configure(settings)
        except Exception:
            self.close()  # never leave a half-opened device locked
            raise

    def _configure(self, settings):
        self.camera.Open()
        self.W = int(self.camera.Width.Max)
        self.H = int(self.camera.Height.Max)
        self._clamp_exposure_limits()
        self._roi = self._read_roi()

        self.converter = pylon.ImageFormatConverter()
        self._select_pixel_format()
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self._setup_trigger()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.Gain = settings.default_gain
        self.ExposureTime = settings.default_exposure

        # Mirrors the exposure window for scope / DAQ sync (Line2, active-low,
        # on most Baslers).
        self.exposure_line_available = self._setup_exposure_line()

        # A user-driven output, pulsed low around each acquisition (Line3).
        self.userdefined_line_available = self.set_userdefined_line(False)
        if settings.user_output_line and not self.userdefined_line_available:
            logging.warning(
                f"{settings.user_output_line} user output unavailable on {self.model_name}"
            )

    # ------------------------------ setup -------------------------------- #
    def apply_settings(self, settings):
        """Re-narrow to the hardware whenever new settings are adopted, so a
        reload cannot widen the window past what pylon accepts."""
        super().apply_settings(settings)
        if getattr(self, "camera", None) is not None and self.camera.IsOpen():
            self._clamp_exposure_limits()

    def _clamp_exposure_limits(self):
        """Keep the configured window inside what the camera actually accepts."""
        try:
            lo, hi = self.camera.ExposureTime.Min, self.camera.ExposureTime.Max
        except Exception:  # node absent on some models
            return
        narrowed = (
            max(self.settings.min_exposure, float(lo)),
            min(self.settings.max_exposure, float(hi)),
        )
        if narrowed[0] >= narrowed[1]:
            logging.warning(
                f"configured exposure {self.settings.min_exposure:g}-"
                f"{self.settings.max_exposure:g} us is outside the camera's "
                f"{lo:.0f}-{hi:.0f} us; using the camera's range"
            )
            narrowed = (float(lo), float(hi))
        self.set_exposure_limits(*narrowed)

    def available_pixel_formats(self) -> list[str]:
        try:
            return list(self.camera.PixelFormat.Symbolics)
        except Exception:
            return []

    def _select_pixel_format(self):
        """Set the first candidate for this mode that the camera accepts, and
        point the converter at the matching output depth."""
        candidates = (
            self.settings.pixel_format_8bit
            if self.mode == "8Bit"
            else self.settings.pixel_format_16bit
        )
        available = self.available_pixel_formats()
        for fmt in candidates:
            if available and fmt not in available:
                continue
            try:
                self.camera.PixelFormat.Value = fmt
            except Exception as e:
                logging.debug(f"PixelFormat {fmt} rejected: {e}")
                continue
            self.pixel_format = fmt
            logging.info(f"Camera operating in {fmt} ({self.mode} mode)")
            break
        else:
            self.pixel_format = self._current_pixel_format()
            logging.error(
                f"none of the configured {self.mode} formats {list(candidates)} are "
                f"available on {self.model_name} (it offers {available or 'unknown'}); "
                f"staying on {self.pixel_format}.  Set pixel_format_{self.mode.lower()} "
                "in camera_config.yaml."
            )
        self.converter.OutputPixelFormat = (
            pylon.PixelType_Mono8 if self.mode == "8Bit" else pylon.PixelType_Mono16
        )

    def _current_pixel_format(self) -> str:
        try:
            return str(self.camera.PixelFormat.Value)
        except Exception:
            return "unknown"

    def _setup_trigger(self):
        """Software triggering where the model has it.  `trigger_mode: off`
        free-runs; `auto` falls back to free-running silently enough."""
        self.software_trigger = False
        self._can_wait_trigger = False
        mode = self.settings.trigger_mode
        if mode == "off":
            try:
                self.camera.TriggerMode.Value = "Off"
            except Exception as e:
                logging.debug(f"could not disable the trigger: {e}")
            logging.info("Free-running (trigger_mode: off)")
            return
        try:
            self.camera.TriggerSelector.Value = "FrameStart"
            self.camera.TriggerMode.Value = "On"
            self.camera.TriggerSource.Value = "Software"
            self.software_trigger = True
            try:  # needs AcquisitionStatus or SoftwareTrigger.IsDone on the model
                self._can_wait_trigger = bool(self.camera.CanWaitForFrameTriggerReady())
            except Exception:
                self._can_wait_trigger = False
        except Exception as e:
            log = logging.error if mode == "software" else logging.warning
            log(f"Software trigger unavailable, free-running: {e}")

    def _setup_exposure_line(self) -> bool:
        line = self.settings.exposure_line
        if not line:
            return False
        try:
            self.camera.LineSelector.Value = line
            self.camera.LineMode.Value = "Output"
            self.camera.LineInverter.Value = self.settings.exposure_line_invert
            self.camera.LineSource.Value = "ExposureActive"
            return True
        except Exception as e:
            logging.warning(f"{line} exposure output unavailable: {e}")
            return False

    def set_userdefined_line(self, enable: bool) -> bool:
        line = self.settings.user_output_line
        if not line:
            self.userdefined_line_enabled = False
            return False
        try:
            self.camera.LineSelector.Value = line
            if enable:
                self.camera.LineMode.Value = "Output"
                self.camera.LineInverter.Value = False
                self.camera.LineSource.Value = self.settings.user_output_selector
                self.camera.UserOutputSelector.Value = self.settings.user_output_selector
                self.camera.UserOutputValue.Value = True
            else:
                self.camera.LineMode.Value = "Input"
        except Exception as e:
            logging.error(f"Failed to set user-defined line {line}: {e}")
            self.userdefined_line_enabled = False
            return False
        self.userdefined_line_enabled = enable
        return True

    def _set_user_output(self, high: bool):
        self.camera.UserOutputSelector.Value = self.settings.user_output_selector
        self.camera.UserOutputValue.Value = high

    # ------------------------------ properties --------------------------- #
    # Getters answer from the value last written (as read back from the
    # camera), so the per-frame HUD costs no USB round trips and they keep
    # working after the camera is unplugged.
    @property
    def Gain(self):
        return self._gain

    @Gain.setter
    def Gain(self, value):
        self.camera.Gain.Value = value
        self._gain = float(self.camera.Gain.Value)

    @property
    def ExposureTime(self):
        return self._exposure

    @ExposureTime.setter
    def ExposureTime(self, value):
        self.camera.ExposureTime.Value = float(
            np.clip(value, self.min_exposure, self.max_exposure)
        )
        self._exposure = float(self.camera.ExposureTime.Value)

    @property
    def ROI(self):
        return self._roi

    @ROI.setter
    def ROI(self, value):
        if not self.camera.IsOpen():
            logging.error("Camera is not open, cannot set ROI.")
            return
        try:
            was_grabbing = self.camera.IsGrabbing()
            if was_grabbing:
                self.camera.StopGrabbing()
            self._apply_roi(*value)
            if was_grabbing:
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            logging.error(f"Failed to set ROI: {e}")
        self._roi = self._read_roi()

    def _read_roi(self):
        try:
            return (
                int(self.camera.Width.Value),
                int(self.camera.Height.Value),
                int(self.camera.OffsetX.Value),
                int(self.camera.OffsetY.Value),
            )
        except Exception:  # unplugged mid-change: keep what we had
            return getattr(self, "_roi", (self.W, self.H, 0, 0))

    def _apply_roi(self, width, height, offset_x, offset_y):
        width = max(10, min(width, self.W))
        height = max(10, min(height, self.H))
        offset_x = min(offset_x, self.W - width)
        offset_y = min(offset_y, self.H - height)
        self.camera.Width.Value = int(self._align(width, self.camera.Width))
        self.camera.Height.Value = int(self._align(height, self.camera.Height))
        self.camera.OffsetX.Value = int(self._align(offset_x, self.camera.OffsetX))
        self.camera.OffsetY.Value = int(self._align(offset_y, self.camera.OffsetY))

    @staticmethod
    def _align(value, node):
        """Floor `value` to the node's nearest valid increment."""
        return value - (value - node.Min) % node.Inc

    # ------------------------------ acquisition -------------------------- #
    def is_connected(self) -> bool:
        try:
            return not self.camera.IsCameraDeviceRemoved()
        except Exception:
            return False

    def _grab(self) -> np.ndarray | None:
        if self.camera.IsCameraDeviceRemoved():
            raise CameraDisconnected(f"{self.label} ({self.serial}) was removed")
        if self.userdefined_line_enabled:
            self._set_user_output(False)  # pulse low around the acquisition
        if self.software_trigger:
            # A trigger sent before the camera is armed is silently dropped -
            # right after StartGrabbing, i.e. after every ROI change - and the
            # grab then times out.
            if self._can_wait_trigger:
                self.camera.WaitForFrameTriggerReady(
                    self.settings.grab_timeout_ms, pylon.TimeoutHandling_Return
                )
            self.camera.TriggerSoftware.Execute()
        result = self.camera.RetrieveResult(
            self.settings.grab_timeout_ms, pylon.TimeoutHandling_ThrowException
        )
        try:
            if self.userdefined_line_enabled:
                self._set_user_output(True)
            if not result.GrabSucceeded():
                return None
            return self.converter.Convert(result).GetArray().copy()
        finally:
            result.Release()

    def close(self):
        """Tear down in the order pylon wants; every step may fail on a camera
        that has been unplugged, and none of them may stop the others."""
        camera = getattr(self, "camera", None)
        if camera is None:
            return
        for step in (camera.StopGrabbing, camera.Close, camera.DestroyDevice):
            try:
                step()
            except Exception as e:
                logging.debug(f"{step.__name__} on close: {e}")


# --------------------------------------------------------------------------- #
#  discovery, for --list-cameras
# --------------------------------------------------------------------------- #
def probe_device(device_info) -> dict:
    """Open one camera briefly and report what `cameraconfig.yaml_stub` needs:
    sensor size, Mono pixel formats, exposure range, I/O lines, trigger
    sources.  Every read is optional - a model that lacks a node is simply
    absent from the result."""
    out: dict = {
        "model": device_info.GetModelName(),
        "serial": device_info.GetSerialNumber(),
        "vendor": device_info.GetVendorName(),
        "backend": BaslerBackend.name,
    }
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
    try:
        camera.Open()
    except Exception as e:
        out["error"] = str(e)
        return out
    try:
        out["sensor_size"] = (int(camera.Width.Max), int(camera.Height.Max))
        for key, read in (
            ("pixel_formats", lambda: list(camera.PixelFormat.Symbolics)),
            ("exposure_range", lambda: (
                float(camera.ExposureTime.Min), float(camera.ExposureTime.Max))),
            ("lines", lambda: list(camera.LineSelector.Symbolics)),
            ("trigger_sources", lambda: list(camera.TriggerSource.Symbolics)),
            # Pixel pitch is deliberately not probed: no Basler exposes it as a
            # GenICam node, and a guessed scale is worse than a blank to fill in.
        ):
            try:
                out[key] = read()
            except Exception:
                pass
    finally:
        camera.Close()
    return out


def list_devices() -> list[dict]:
    """`probe_device` for every connected Basler, in enumeration order."""
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    return [probe_device(d) for d in devices]
