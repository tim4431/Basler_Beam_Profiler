"""ROI geometry: zoom/aspect model and sensor<->display coordinate mapping."""

from __future__ import annotations

import numpy as np


class ROIModel:
    """Maintain ROI as (scale, aspect, centre) -> (w, h, ox, oy) tuple."""

    MIN_SCALE = 0.02

    def __init__(self, sensor_w: int, sensor_h: int, roi: tuple | None = None):
        self.full_w = sensor_w
        self.full_h = sensor_h
        self.scale = 1.0  # fraction of sensor *width*
        self.aspect = sensor_w / sensor_h  # w / h
        self.cx = sensor_w / 2
        self.cy = sensor_h / 2
        if roi is not None:
            self.set_roi(roi)

    def set_roi(self, roi: tuple):
        """Adopt (w, h, offset_x, offset_y) sensor pixels as the current view -
        used for a camera's configured `default_roi`."""
        w, h, ox, oy = roi
        w = max(1, min(int(w), self.full_w))
        h = max(1, min(int(h), self.full_h))
        self.aspect = w / h
        self.scale = float(np.clip(w / self.full_w, self.MIN_SCALE, 1.0))
        self.cx = ox + w / 2
        self.cy = oy + h / 2

    @property
    def size(self) -> tuple[int, int]:
        w = max(1, int(self.full_w * self.scale))
        h = max(1, int(w / self.aspect))
        if h > self.full_h:  # shrink if we spill vertically
            h = self.full_h
            w = int(h * self.aspect)
        return w, h

    @property
    def tuple(self) -> tuple[int, int, int, int]:
        """(w, h, offset_x, offset_y) as expected by the camera."""
        w, h = self.size
        ox = int(np.clip(self.cx - w / 2, 0, self.full_w - w))
        oy = int(np.clip(self.cy - h / 2, 0, self.full_h - h))
        self.cx = ox + w / 2  # honour clamping
        self.cy = oy + h / 2
        return w, h, ox, oy

    def keep_point_fixed(
        self, sensor_x: float, sensor_y: float, new_scale=None, new_aspect=None
    ):
        """Update scale/aspect while keeping (sensor_x, sensor_y) at the same
        absolute sensor position.  Aspect changes keep the longest edge length
        constant so the zoom level feels intuitive."""
        w_old, h_old, ox_old, oy_old = self.tuple
        long_old = max(w_old, h_old)
        r_x = (sensor_x - ox_old) / w_old if w_old else 0.5
        r_y = (sensor_y - oy_old) / h_old if h_old else 0.5

        if new_aspect is not None:
            new_aspect = float(np.clip(new_aspect, 0.1, 20.0))
            w_tmp = self.full_w * self.scale
            long_tmp = max(w_tmp, w_tmp / new_aspect)
            if long_tmp > 0:
                self.scale *= long_old / long_tmp
            self.aspect = new_aspect

        if new_scale is not None:
            self.scale = float(np.clip(new_scale, self.MIN_SCALE, 1.0))

        w_new, h_new = self.size
        ox_new = sensor_x - r_x * w_new
        oy_new = sensor_y - r_y * h_new
        self.cx = np.clip(ox_new + w_new / 2, w_new / 2, self.full_w - w_new / 2)
        self.cy = np.clip(oy_new + h_new / 2, h_new / 2, self.full_h - h_new / 2)


class ViewTransform:
    """Maps the current ROI into a square window (letterboxed with padding)."""

    def __init__(self, roi: tuple[int, int, int, int], window: int):
        self.w, self.h, self.ox, self.oy = roi
        self.window = window
        self.scale = window / max(self.w, self.h)
        self.view_w = int(self.w * self.scale)
        self.view_h = int(self.h * self.scale)
        self.pad_l = (window - self.view_w) // 2
        self.pad_t = (window - self.view_h) // 2

    def contains(self, disp_x: int, disp_y: int) -> bool:
        return (self.pad_l <= disp_x < self.pad_l + self.view_w) and (
            self.pad_t <= disp_y < self.pad_t + self.view_h
        )

    def to_sensor(self, disp_x: float, disp_y: float) -> tuple[float, float]:
        return (
            self.ox + (disp_x - self.pad_l) / self.scale,
            self.oy + (disp_y - self.pad_t) / self.scale,
        )

    def to_display(self, sensor_x: float, sensor_y: float) -> tuple[int, int]:
        return (
            int((sensor_x - self.ox) * self.scale) + self.pad_l,
            int((sensor_y - self.oy) * self.scale) + self.pad_t,
        )

    def roi_to_display(self, x: float, y: float) -> tuple[int, int]:
        """ROI-relative pixel coords (e.g. spot fits) -> display coords."""
        return self.to_display(self.ox + x, self.oy + y)

    def roi_to_display_many(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised `roi_to_display` for whole spot columns."""
        return (
            (np.asarray(x) * self.scale).astype(np.int32) + self.pad_l,
            (np.asarray(y) * self.scale).astype(np.int32) + self.pad_t,
        )
