"""Camera interface shared by the Basler driver and the simulator."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from ..cameraconfig import CameraSettings


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
    read `pixel_size_known` before presenting a micron figure as calibrated."""

    mode: str
    model_name: str
    serial: str
    W: int
    H: int

    def __init__(self, mode: str = "16Bit", settings: CameraSettings | None = None):
        if mode not in ("8Bit", "16Bit"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.auto_exposure = False
        self._ae = AutoExposure()
        self.userdefined_line_available = False
        self.userdefined_line_enabled = False
        self.apply_settings(settings or CameraSettings())

    def apply_settings(self, settings: CameraSettings):
        """Adopt a resolved settings block.  Called by __init__; safe to call
        again to pick up an edited camera_config.yaml."""
        self.settings = settings
        self.pixel_size = settings.effective_pixel_size
        self.pixel_size_known = settings.pixel_size_known
        self.set_exposure_limits(settings.min_exposure, settings.max_exposure)

    def set_exposure_limits(self, lo: float, hi: float):
        """Define the window this camera will honour: the configured one, which
        a driver may narrow further to what the hardware reports.  This is what
        auto-exposure is restored to, so the narrowing is never lost."""
        self._limits = (float(lo), float(hi))
        self.min_exposure, self.max_exposure = self._limits

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
        img = self._grab()
        if img is not None and self.auto_exposure:
            new_exp = self._ae.step(float(img.max()), self.ExposureTime, self.saturation)
            self.ExposureTime = float(np.clip(new_exp, self.min_exposure, self.max_exposure))
        return img

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
        pass
