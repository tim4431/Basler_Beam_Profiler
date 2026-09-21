"""FLIR USB3/GigE cameras via the vendor's Spinnaker Python bindings.

PySpin is deliberately imported only when hardware discovery is requested.
Every returned frame owns its memory; no SDK buffer escapes acquisition.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

from ..config import load_camera_config
from .base import Camera


class _Session:
    """Keep the SDK singleton alive until the last camera has been released."""
    def __init__(self, sdk):
        self.system = sdk.System.GetInstance()
        self.users = 0

    def release(self):
        self.users -= 1
        if self.users == 0:
            self.close()

    def close(self):
        if self.system is not None:
            self.system.ReleaseInstance()
            self.system = None


def open_flir_cameras(mode="16Bit"):
    if sys.platform == "darwin":
        # A frozen app can load the separately installed vendor Python binding.
        # Its Python ABI and architecture must match this app (cp312/arm64).
        bindings = os.environ.get("BEAM_PROFILER_PYSPIN_PATH", os.path.expanduser(
            "~/Library/Application Support/BeamProfiler/pyspin"))
        if os.path.isdir(bindings) and bindings not in sys.path:
            sys.path.append(bindings)
        if "SPINNAKER_GENTL64_CTI" not in os.environ:
            for path in ("/usr/local/lib/spinnaker-gentl/Spinnaker_GenTL.cti",
                         "/Applications/Spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti"):
                if Path(path).is_file():
                    os.environ["SPINNAKER_GENTL64_CTI"] = path
                    break
    try:
        import PySpin as sdk
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "FLIR requires the Spinnaker SDK and its matching spinnaker_python "
            "wheel (Python 3.12, macOS ARM64 on Apple Silicon). See docs/macos.md. "
            "Do not install the unrelated 'pyspin' package from PyPI."
        ) from exc
    if not hasattr(sdk, "System"):
        raise RuntimeError("This PySpin module is not FLIR Spinnaker; see docs/macos.md")
    session = _Session(sdk)
    cameras = []
    devices = None
    raw = None
    failure = None
    try:
        devices = session.system.GetCameras()
        for index in range(devices.GetSize()):
            raw = devices.GetByIndex(index)
            cameras.append(FlirCamera(raw, sdk, mode, session))
            raw = None
    except Exception as exc:
        # Keep the message, not the traceback: constructor frames can retain
        # CameraPtr/nodemap references and prevent the SDK from shutting down.
        failure = str(exc)
    raw = None
    if devices is not None:
        devices.Clear()
        devices = None
    if failure is not None:
        for camera in cameras:
            camera.close()
    if session.users == 0:
        session.close()
    if failure is not None:
        raise RuntimeError(failure)
    return cameras


class FlirCamera(Camera):
    def __init__(self, camera, sdk, mode="16Bit", session=None):
        super().__init__(mode)
        self.camera, self.sdk, self._session = camera, sdk, session
        self._running = False
        self._initialized = False
        self._nodes = None
        self.software_trigger = False
        self.exposure_line_available = False
        self._sync = {}
        try:
            camera.Init()
            self._initialized = True
            self._nodes = camera.GetNodeMap()
            # A previous application crash may leave device acquisition active
            # even though this new host-side CameraPtr is not streaming yet.
            if not sdk.IsWritable(self._node("Width", "Integer")):
                stop = sdk.CCommandPtr(self._nodes.GetNode("AcquisitionStop"))
                if sdk.IsWritable(stop):
                    stop.Execute()
                locked = sdk.CIntegerPtr(self._nodes.GetNode("TLParamsLocked"))
                if sdk.IsWritable(locked):
                    locked.SetValue(0)
                del stop, locked
            self.model_name = self._node("DeviceModelName", "String").GetValue()
            self.serial = self._node("DeviceSerialNumber", "String").GetValue()
            settings = load_camera_config().get(self.model_name)
            if settings is None:
                raise ValueError(
                    f"Add {self.model_name!r} and its pixel_size (meters) to camera_config.yaml "
                    "before using calibrated measurements."
                )
            self.pixel_size = float(settings["pixel_size"])
            if not math.isfinite(self.pixel_size) or self.pixel_size <= 0:
                raise ValueError("pixel_size must be a positive finite value in meters")
            self.default_roi = tuple(settings["default_roi"])
            self._sync = settings.get("sync", {})
            # Remove offsets first so Width/Height maxima describe the whole sensor.
            self._integer("OffsetX", 0)
            self._integer("OffsetY", 0)
            self.W = int(self._node("Width", "Integer").GetMax())
            self.H = int(self._node("Height", "Integer").GetMax())
            self.ROI = (self.W, self.H, 0, 0)
            self._enum("AcquisitionMode", "Continuous")
            self._enum("ExposureAuto", "Off")
            self._optional(lambda: self._enum("ExposureMode", "Timed"), "timed exposure")
            self._optional(lambda: self._enum("GainAuto", "Off"), "manual gain")
            self._optional(lambda: self._boolean("GammaEnable", False), "linear gamma")
            # This Blackfly exposes native Mono16 (12-bit ADC, MSB aligned).
            # Avoid converting packed/color data with ambiguous intensity scaling.
            self._enum("PixelFormat", "Mono16" if mode == "16Bit" else "Mono8")
            exposure = self._node("ExposureTime", "Float")
            self.min_exposure = float(exposure.GetMin())
            self.max_exposure = float(exposure.GetMax())
            self.Gain = 0
            self.ExposureTime = 200
            self._enum("TriggerMode", "Off")
            try:
                self._enum("TriggerSelector", "FrameStart")
                self._enum("TriggerSource", "Software")
                self._enum("TriggerMode", "On")
                self.software_trigger = True
            except Exception as exc:
                # Never leave a partially configured trigger armed.
                self._enum("TriggerMode", "Off")
                logging.warning("FLIR software trigger unavailable; free-running: %s", exc)
            stream = camera.GetTLStreamNodeMap()
            self._optional(
                lambda: self._enum("StreamBufferHandlingMode", "NewestOnly", stream),
                "latest-frame buffering",
            )
            self._configure_sync()
            camera.BeginAcquisition()
            self._running = True
            if session is not None:
                session.users += 1
            logging.info("Connected FLIR %s (%s), %s", self.model_name, self.serial, mode)
        except Exception:
            if self._initialized:
                camera.DeInit()
            self._nodes = None
            self.camera = None
            self._initialized = False
            raise

    def _node(self, name, kind, nodemap=None):
        mapping = self._nodes if nodemap is None else nodemap
        node = getattr(self.sdk, f"C{kind}Ptr")(mapping.GetNode(name))
        if not self.sdk.IsReadable(node):
            raise RuntimeError(f"Camera feature {name} is unavailable or unreadable")
        return node

    def _enum(self, name, value, nodemap=None):
        node = self._node(name, "Enumeration", nodemap)
        entry = node.GetEntryByName(value)
        if not self.sdk.IsReadable(entry):
            raise RuntimeError(f"Camera does not support {name}={value}")
        if node.GetIntValue() == entry.GetValue():
            return  # fixed-output lines can already be Output but read-only
        if not self.sdk.IsWritable(node):
            raise RuntimeError(f"Camera feature {name} is not writable")
        node.SetIntValue(entry.GetValue())

    def _boolean(self, name, value):
        node = self._node(name, "Boolean")
        if node.GetValue() == value:
            return
        if not self.sdk.IsWritable(node):
            raise RuntimeError(f"Camera feature {name} is not writable")
        node.SetValue(bool(value))

    def _integer(self, name, value):
        node = self._node(name, "Integer")
        lo, hi, inc = node.GetMin(), node.GetMax(), node.GetInc()
        value = max(lo, min(int(value), hi))
        value = lo + (value - lo) // inc * inc
        node.SetValue(value)

    def _float(self, name, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        node = self._node(name, "Float")
        node.SetValue(max(node.GetMin(), min(value, node.GetMax())))

    @staticmethod
    def _optional(action, feature):
        try:
            action()
            return True
        except Exception as exc:
            logging.warning("FLIR %s unavailable: %s", feature, exc)
            return False

    @property
    def ExposureTime(self):
        return self._node("ExposureTime", "Float").GetValue()

    @ExposureTime.setter
    def ExposureTime(self, value):
        self._float("ExposureTime", value)

    @property
    def Gain(self):
        return self._node("Gain", "Float").GetValue()

    @Gain.setter
    def Gain(self, value):
        self._float("Gain", value)

    @property
    def ROI(self):
        return tuple(int(self._node(n, "Integer").GetValue())
                     for n in ("Width", "Height", "OffsetX", "OffsetY"))

    @ROI.setter
    def ROI(self, value):
        if len(value) != 4 or not all(math.isfinite(v) for v in value):
            raise ValueError("ROI must contain four finite values: width, height, x, y")
        running = self._running
        if running:
            self.camera.EndAcquisition()
            self._running = False
        old = self.ROI
        try:
            self._apply_roi(value)
        except Exception:
            self._apply_roi(old)
            raise
        finally:
            if running:
                self.camera.BeginAcquisition()
                self._running = True

    def _apply_roi(self, value):
        width, height, x, y = value
        self._integer("OffsetX", 0)
        self._integer("OffsetY", 0)
        self._integer("Width", width)
        self._integer("Height", height)
        self._integer("OffsetX", x)
        self._integer("OffsetY", y)

    def _configure_sync(self):
        # Pin selection is per-model configuration; never assume Basler wiring.
        line = self._sync.get("exposure_line")
        if line:
            def configure():
                self._enum("LineSelector", line)
                self._enum("LineMode", "Output")
                self._boolean("LineInverter", True)
                self._enum("LineSource", "ExposureActive")
            self.exposure_line_available = self._optional(configure, "exposure sync")
        self.userdefined_line_available = bool(self._sync.get("user_line"))

    def _set_user_output(self, high):
        self._enum("UserOutputSelector", self._sync.get("user_output", "UserOutput0"))
        self._boolean("UserOutputValue", high)

    def set_userdefined_line(self, enable):
        if not self.userdefined_line_available:
            logging.warning("No FLIR user-output line configured in camera_config.yaml")
            return False
        def configure():
            self._enum("LineSelector", self._sync["user_line"])
            if enable:
                self._enum("LineMode", "Output")
                self._boolean("LineInverter", False)
                self._enum("LineSource", self._sync.get("user_output", "UserOutput0"))
                self._set_user_output(True)
            else:
                self._set_user_output(False)
                self._enum("LineSource", "Off")
        success = self._optional(configure, "user output")
        if success:
            self.userdefined_line_enabled = bool(enable)
        return success

    def _grab(self):
        image = None
        pulsed = self.userdefined_line_enabled
        try:
            if pulsed:
                self._set_user_output(False)
            if self.software_trigger:
                trigger = self.sdk.CCommandPtr(self._nodes.GetNode("TriggerSoftware"))
                if not self.sdk.IsWritable(trigger):
                    raise RuntimeError("Software trigger is not writable")
                trigger.Execute()
            timeout = max(1000, int(self.ExposureTime / 1000) + 1000)
            image = self.camera.GetNextImage(timeout)
            if image.IsIncomplete():
                logging.warning("Incomplete FLIR image: %s", image.GetImageStatus())
                return None
            array = image.GetNDArray().copy()
            expected = np.uint16 if self.mode == "16Bit" else np.uint8
            if array.ndim != 2 or array.dtype != expected:
                raise RuntimeError(f"Unexpected FLIR frame: {array.shape}, {array.dtype}")
            return array
        finally:
            try:
                if image is not None:
                    image.Release()
            finally:
                if pulsed:
                    self._set_user_output(True)

    def close(self):
        if self.camera is None:
            return
        try:
            if self.userdefined_line_enabled:
                self.set_userdefined_line(False)
            if self._running:
                self.camera.EndAcquisition()
                self._running = False
        finally:
            try:
                if self._initialized:
                    self.camera.DeInit()
            finally:
                self._initialized = False
                self._nodes = None
                self.camera = None
                if self._session is not None:
                    session, self._session = self._session, None
                    session.release()
