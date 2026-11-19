"""Camera interface shared by every backend (Basler, GenICam, UVC, simulator).

A *backend* knows how to find devices and open them; a *camera* is one opened
device.  Adding support for another vendor means one module with a `Backend`
subclass and a `Camera` subclass - see `cameras/__init__.py` for the registry.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .. import cameraconfig
from ..cameraconfig import CameraSettings

# A camera that fails this many grabs in a row is treated as gone, for backends
# that cannot ask the hardware directly (see `Camera.is_connected`).
MAX_CONSECUTIVE_FAILURES = 5


class CameraDisconnected(RuntimeError):
    """The device went away (unplugged, powered off, cable fault).  The viewer
    catches this, shows a placeholder and reopens the camera when it returns."""


@dataclass(frozen=True)
class DeviceInfo:
    """One enumerated device, before it is opened.

    `uid` identifies the physical unit across unplug/replug (backend + serial),
    so the app can recognise a camera that comes back and restore its state.
    `address` is whatever the backend needs to open it again (a pylon full
    name, a GenTL device id, /dev/video0)."""

    backend: str
    model: str
    serial: str = ""
    vendor: str = ""
    address: str = ""

    @property
    def uid(self) -> str:
        return f"{self.backend}:{self.serial or self.address or self.model}"


class Backend:
    """A family of cameras reachable through one SDK.

    Subclasses set the class attributes and implement `enumerate` and `open`.
    `enumerate` runs on a background thread every couple of seconds, so it must
    be cheap and must not disturb a camera that is already open.  `open` runs
    on a worker thread; the returned camera is then used from the main thread.
    """

    name = ""  # short id, used in uids and --backends
    title = ""  # what the UI shows
    install_hint = ""  # how to get it working when it is unavailable
    # Whether a device of this kind may be opened without the user asking - a
    # laptop webcam should never replace the profiler camera on its own.
    auto_open = True

    def available(self) -> tuple[bool, str]:
        """(usable, reason).  Called once at start-up; a backend whose SDK is
        missing is listed in the Cameras window with the reason, not hidden."""
        return True, ""

    def enumerate(self, full: bool = False) -> list[DeviceInfo]:
        """Devices present right now.  `full` is set on start-up and when the
        user presses Rescan, for backends that can only search expensively."""
        raise NotImplementedError

    def open(self, info: DeviceInfo, mode: str) -> Camera:
        raise NotImplementedError

    def probe(self, info: DeviceInfo) -> dict:
        """What --list-cameras prints: sensor size, pixel formats, exposure
        range, I/O lines, trigger sources.  Every key is optional."""
        return {"model": info.model, "serial": info.serial, "vendor": info.vendor}

    def close(self):
        """Release SDK-wide resources at exit."""


class AutoExposure:
    """Keeps the frame maximum near saturation: coarse 10% steps far from the
    target, PID fine-tuning inside the tolerance band."""

    def __init__(self, kp=0.05, ki=0.0, kd=0.3, target=0.8, tolerance=0.19):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target_frac, self.tol_frac = target, tolerance
        self._buffer: deque = deque(maxlen=5)
        self._integral = 0.0
        self._last_error = 0.0

    def step(self, max_val: float, exposure: float, saturation: int) -> float:
        """Return the next exposure time given the current frame maximum."""
        target = self.target_frac * saturation
        tol = self.tol_frac * saturation
        self._buffer.append(max_val)
        if abs(max_val - target) > tol:
            self._integral = self._last_error = 0.0
            return exposure * (1.1 if max_val < target else 0.9)
        error = (target - np.mean(self._buffer)) / saturation
        self._integral += error
        derivative = error - self._last_error
        self._last_error = error
        control = self.kp * error + self.ki * self._integral + self.kd * derivative
        return exposure * (1 + control)


class Camera:
    """Base class: subclasses implement _grab() and the ROI / ExposureTime /
    Gain properties, and set model_name, serial, W, H in __init__ before
    calling `apply_settings`.

    Everything that is per-model rather than per-backend - pixel size, exposure
    limits, pixel formats, I/O lines - comes from a `CameraSettings`, so adding
    a camera is a `camera_config.yaml` edit.  `pixel_size` is still a plain
    float attribute, falling back to `UNKNOWN_PIXEL_SIZE` when unconfigured;
    read `pixel_size_known` before presenting a micron figure as calibrated.

    The ROI / ExposureTime / Gain getters should answer from a cached value
    rather than the hardware: the HUD reads them every frame, and they must
    keep answering after the device is unplugged so the viewer can save the
    state it will restore on reconnect."""

    mode: str
    model_name: str
    serial: str
    W: int
    H: int
    backend = ""  # Backend.name, set by the backend that opened it
    device: DeviceInfo | None = None

    def __init__(self, mode: str = "16Bit", settings: CameraSettings | None = None):
        if mode not in ("8Bit", "16Bit"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.auto_exposure = False
        self._ae = AutoExposure()
        self.userdefined_line_available = False
        self.userdefined_line_enabled = False
        self._failures = 0
        self._last_failure_log = 0.0
        self.apply_settings(settings or CameraSettings())

    def apply_settings(self, settings: CameraSettings):
        """Adopt a resolved settings block.  Called by __init__; safe to call
        again to pick up an edited camera_config.yaml."""
        self.settings = settings
        self.pixel_size = settings.effective_pixel_size
        self.pixel_size_known = settings.pixel_size_known
        self.set_exposure_limits(settings.min_exposure, settings.max_exposure)

    def reload_settings(self):
        """Re-resolve this camera against the (possibly edited) config file."""
        self.apply_settings(cameraconfig.resolve(self.model_name, self.serial, warn=False))

    def set_exposure_limits(self, lo: float, hi: float):
        """Define the window this camera will honour: the configured one, which
        a driver may narrow further to what the hardware reports.  This is what
        auto-exposure is restored to, so the narrowing is never lost."""
        self._limits = (float(lo), float(hi))
        self.min_exposure, self.max_exposure = self._limits

    @property
    def uid(self) -> str:
        if self.device is not None:
            return self.device.uid
        return f"{self.backend}:{self.serial}"

    @property
    def label(self) -> str:
        """What to show the user: the configured label, else the model name."""
        return self.settings.label or self.model_name

    @property
    def default_roi(self) -> tuple[int, int, int, int]:
        """Configured start-up ROI, clamped to the sensor; full frame if unset."""
        roi = self.settings.default_roi
        if roi is None:
            return (self.W, self.H, 0, 0)
        w = max(1, min(int(roi[0]), self.W))
        h = max(1, min(int(roi[1]), self.H))
        return (w, h, min(int(roi[2]), self.W - w), min(int(roi[3]), self.H - h))

    @property
    def saturation(self) -> int:
        return 255 if self.mode == "8Bit" else 65535

    def AutoExposureOn(self, range: tuple[float, float] | None = None):
        """Enable auto-exposure.  `range` narrows the configured limits; it
        cannot widen them past what the model is configured to allow."""
        self.auto_exposure = True
        if range is not None:
            self.min_exposure = max(self._limits[0], range[0])
            self.max_exposure = min(self._limits[1], range[1])
            if self.min_exposure >= self.max_exposure:  # non-overlapping request
                logging.warning(
                    f"auto-exposure range {range} does not overlap this camera's "
                    f"{self._limits[0]:g}-{self._limits[1]:g} us; keeping the latter"
                )
                self.min_exposure, self.max_exposure = self._limits

    def AutoExposureOff(self):
        self.auto_exposure = False
        self.min_exposure, self.max_exposure = self._limits  # undo any narrowing

    def grab_image(self) -> np.ndarray | None:
        """The next frame, or None if none arrived in time.

        Raises `CameraDisconnected` once the device is gone.  Any other driver
        error is a missed frame - logged (at most every few seconds) and
        counted, so a backend without a hardware presence check still notices
        a camera that has stopped answering altogether."""
        try:
            img = self._grab()
        except CameraDisconnected:
            raise
        except Exception as e:
            self._failures += 1
            if not self.is_connected():
                raise CameraDisconnected(f"{self.label}: {e}") from e
            now = time.monotonic()
            if now - self._last_failure_log > 5:
                self._last_failure_log = now
                logging.warning(f"{self.label}: grab failed ({e})")
            return None
        if img is None:
            return None
        self._failures = 0
        if self.auto_exposure:
            new_exp = self._ae.step(float(img.max()), self.ExposureTime, self.saturation)
            self.ExposureTime = float(np.clip(new_exp, self.min_exposure, self.max_exposure))
        return img

    def is_connected(self) -> bool:
        """Whether the device is still there.  Backends that can ask the
        hardware override this; the default gives up after a run of failed
        grabs."""
        return self._failures < MAX_CONSECUTIVE_FAILURES

    def set_userdefined_line(self, enable: bool) -> bool:
        """Hardware sync output; only real cameras support this."""
        return False

    # --- to implement in subclasses -------------------------------------- #
    def _grab(self) -> np.ndarray | None:
        raise NotImplementedError

    ROI: tuple  # (w, h, ox, oy) property
    ExposureTime: float  # µs property
    Gain: float  # property

    def close(self):
        """Release the device.  Must not raise, even when it is already gone."""
