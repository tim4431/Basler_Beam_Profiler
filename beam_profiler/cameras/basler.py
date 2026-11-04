"""pypylon driver for Basler cameras.

Nothing here is specific to one model: pixel formats, trigger mode, I/O lines,
exposure limits and pixel size all come from `cameraconfig`, and anything the
camera turns out not to support is logged and skipped rather than raised.  So a
new Basler works as soon as its `pixel_size` is in `camera_config.yaml`, and it
still opens (with uncalibrated microns) before that.
"""

from __future__ import annotations

import logging

import numpy as np
from pypylon import pylon

from .. import cameraconfig
from .base import Camera


class BaslerCamera(Camera):
    def __init__(self, mode: str = "16Bit", device_idx: int = 0):
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if device_idx >= len(devices):
            raise IndexError(f"Only {len(devices)} cameras found, idx={device_idx}")
        self.device_info = devices[device_idx]
        self.camera = pylon.InstantCamera(factory.CreateDevice(self.device_info))
        self.model_name = self.device_info.GetModelName()
        self.serial = self.device_info.GetSerialNumber()
        logging.info(f"Connected to camera: {self.model_name} ({self.serial})")

        settings = cameraconfig.resolve(self.model_name, self.serial)
        super().__init__(mode, settings)
        logging.info(cameraconfig.describe(settings, self.model_name))

        self.camera.Open()
        self.W = int(self.camera.Width.Max)
        self.H = int(self.camera.Height.Max)
        self._clamp_exposure_limits()

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
    @property
    def Gain(self):
        return self.camera.Gain.Value

    @Gain.setter
    def Gain(self, value):
        self.camera.Gain.Value = value

    @property
    def ExposureTime(self):
        return self.camera.ExposureTime.Value

    @ExposureTime.setter
    def ExposureTime(self, value):
        self.camera.ExposureTime.Value = float(
            np.clip(value, self.min_exposure, self.max_exposure)
        )

    @property
    def ROI(self):
        return (
            self.camera.Width.Value,
            self.camera.Height.Value,
            self.camera.OffsetX.Value,
            self.camera.OffsetY.Value,
        )

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
    def _grab(self) -> np.ndarray | None:
        if self.userdefined_line_enabled:
            self._set_user_output(False)  # pulse low around the acquisition
        if self.software_trigger:
            self.camera.TriggerSoftware.Execute()
        result = self.camera.RetrieveResult(
            self.settings.grab_timeout_ms, pylon.TimeoutHandling_ThrowException
        )
        if self.userdefined_line_enabled:
            self._set_user_output(True)
        if not result.GrabSucceeded():
            result.Release()
            return None
        img = self.converter.Convert(result).GetArray().copy()
        result.Release()
        return img

    def close(self):
        self.camera.StopGrabbing()
        self.camera.Close()


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
