"""Interactive viewer: main loop, keyboard/mouse handling and overlay layout.

Keyboard shortcuts are listed in the README."""

from __future__ import annotations

import logging
import os
import sys
import time
from threading import Thread

import cv2
import numpy as np

from ..config import (
    DATA_DIR,
    FPS_LIMIT,
    MAX_EXPOSURE_US,
    MIN_EXPOSURE_US,
    PAD_COLOR,
    WEB_SERVER_PORT,
    WINDOW_SIZE,
)
from .. import fitconfig
from ..processing import SpotArray, classify_grid, detect_spots, gpu, region_stats
from ..roi import ROIModel, ViewTransform
from ..status import write_status
from . import overlays, settings

WINDOW = "Beam Profiler - Basler / FLIR"

# waitKeyEx arrow codes: Windows, Cocoa(macOS), and Linux(GTK) report different values
KEY_FACTORS = {
    2555904: 1.1, 65363: 1.1, 63235: 1.1,  # right: +10%
    2424832: 0.9, 65361: 0.9, 63234: 0.9,  # left:  -10%
    2490368: 10, 65362: 10, 63232: 10,  # up:    x10
    2621440: 0.1, 65364: 0.1, 63233: 0.1,  # down:  /10
}


class Viewer:
    def __init__(self, cameras: list):
        self._quit_requested = False
        self._native_close_handler = None
        self.cameras = cameras
        self.curr_idx = 0
        self.camera = cameras[0]
        self.roi_model = ROIModel(self.camera.W, self.camera.H)
        self.camera.ROI = self.roi_model.tuple
        self._cached_state: dict[str, dict] = {}  # per-camera, for seamless switching

        self._auto_exp = False
        self.do_fitting = False
        self.show_stats = False
        self.row_col_fitting = False
        self.exposure_us = 200
        self.last_mouse = (0, 0)  # sensor coords
        self.rect_disp = None  # white rectangle while dragging (display coords)
        self.rect_sensor = None
        self.fit_rect_disp = None  # green fitting rectangle while dragging
        self.fit_rect_sensor = None

        self.avg_dt = 1 / FPS_LIMIT
        self.ema = 0.8  # smoothing for the displayed std figures
        self.std_dx_ema = self.std_dy_ema = self.sigma_std_ema = 0.0
        self._reset_spot_stats()

        # shared with the HTTP server
        self.current_frame = None
        self.latest_spots = SpotArray.empty()
        self.latest_stats: dict = {}
        self.current_rect_stats = None
        self.web_server_enabled = False
        self.show_rect_stats = False
        self._web_thread = None

        # WINDOW_GUI_NORMAL drops the Qt toolbar and the pixel-hover overlay,
        # which render as black/garbled boxes (OpenCV's bundled Qt has no fonts)
        gpu.warm_up()  # pay the CuPy import off the critical path
        settings.prepare_gui()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW, WINDOW_SIZE, WINDOW_SIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        if sys.platform == "darwin":
            from .macos import attach_close_button
            self._native_close_handler = attach_close_button(WINDOW, self.request_close)

    def request_close(self):
        self._quit_requested = True


    # ------------------------------ properties --------------------------- #
    @property
    def auto_exp(self):
        return self._auto_exp

    @auto_exp.setter
    def auto_exp(self, value):
        self._auto_exp = value
        if value:
            self.camera.AutoExposureOn(range=(MIN_EXPOSURE_US, MAX_EXPOSURE_US))
        else:
            self.camera.AutoExposureOff()
            self.exposure_us = int(np.clip(self.exposure_us, MIN_EXPOSURE_US, MAX_EXPOSURE_US))

    @property
    def view(self) -> ViewTransform:
        return ViewTransform(self.camera.ROI, WINDOW_SIZE)

    # ------------------------------ main loop ----------------------------- #
    def run(self, max_frames=None):
        try:
            self._run_loop(max_frames=max_frames)
        finally:
            cv2.destroyAllWindows()
            settings.destroy_settings()

    def _run_loop(self, max_frames=None):
        frame_count = 0
        os.makedirs(DATA_DIR, exist_ok=True)
        while True:
            settings.pump_events()
            if self._quit_requested or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            t0 = time.time()
            frame = self.camera.grab_image()
            if frame is None:
                if not self._handle_key(cv2.waitKeyEx(10)):
                    break
                continue
            self.current_frame = frame
            frame_disp = (
                cv2.convertScaleAbs(frame, alpha=1 / 256.0)
                if self.camera.mode == "16Bit"
                else frame
            )
            self.frame_disp = frame_disp

            self.current_rect_stats = (
                region_stats(frame, self.camera.ROI, self.rect_sensor, self.camera.pixel_size)
                if self.show_rect_stats
                else None
            )

            spots = self._detect(frame) if self.do_fitting else SpotArray.empty()
            self.latest_spots = spots  # columnar; server.py serialises on request
            self._update_spot_stats(spots)
            if self.web_server_enabled:
                self._update_web_stats()

            disp = self._compose(frame_disp, spots)
            cv2.imshow(WINDOW, disp)
            if not self._handle_key(cv2.waitKeyEx(1)):
                break
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
            self._sleep_for_fps(time.time() - t0)

    def _detect(self, frame) -> SpotArray:
        spots = detect_spots(frame)
        if self.fit_rect_sensor is None or not len(spots):
            return spots
        _, _, ox, oy = self.camera.ROI
        x1, y1, x2, y2 = self.fit_rect_sensor
        sx, sy = ox + spots.x, oy + spots.y
        return spots[(sx >= x1) & (sx <= x2) & (sy >= y1) & (sy <= y2)]

    def _compose(self, frame_disp, spots) -> np.ndarray:
        """Resize + pad the frame to the window and draw all overlays.

        Everything is drawn in display space after the resize, so text and
        line widths do not scale with the ROI zoom."""
        view = self.view
        resized = cv2.resize(frame_disp, (view.view_w, view.view_h))
        padded = cv2.copyMakeBorder(
            resized,
            view.pad_t,
            WINDOW_SIZE - view.view_h - view.pad_t,
            view.pad_l,
            WINDOW_SIZE - view.view_w - view.pad_l,
            cv2.BORDER_CONSTANT,
            value=PAD_COLOR[0],
        )
        disp = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        if len(spots) and not self.row_col_fitting:
            overlays.draw_spots(disp, spots, view, self.camera.pixel_size)

        y = overlays.draw_hud(disp, self._hud_lines())
        if self.show_rect_stats:
            y = overlays.draw_rect_stats(disp, self.current_rect_stats, y)
        if self.show_stats:
            bar_y = overlays.draw_spot_stats(
                disp,
                y,
                self.std_dx_ema,
                self.std_dy_ema,
                float(np.mean(self.stats_dx)) * self.camera.pixel_size * 1e6,
                float(np.mean(self.stats_dy)) * self.camera.pixel_size * 1e6,
                self._curvature(self.stats_dx),
                self._curvature(self.stats_dy),
                self.sigma_std_ema,
                show_curvature=not self.row_col_fitting,
            )
            if self.row_col_fitting:
                overlays.draw_grid_stats(
                    disp, bar_y, self.rows, self.columns, self.grid_stats,
                    self.camera.pixel_size * 1e6,
                )
        self._draw_rectangles(disp, view)
        if self.row_col_fitting and self.do_fitting:
            overlays.draw_row_col(disp, view, self.rows, self.columns)
        return disp

    def _hud_lines(self) -> list[str]:
        exposure = self.camera.ExposureTime
        exposure_txt = f"Auto = {exposure:.0f}" if self.auto_exp else f"{exposure}"
        um = self.camera.pixel_size * 1e6
        rect_w = (self.rect_sensor[2] - self.rect_sensor[0]) * um if self.rect_sensor else 0
        rect_h = (self.rect_sensor[3] - self.rect_sensor[1]) * um if self.rect_sensor else 0
        return [
            f"Camera: {self.camera.model_name}, FPS: {1 / self.avg_dt:4.1f}",
            f"ROI: {self.camera.ROI}  Mouse: {self.last_mouse}",
            f"Exposure: {exposure_txt} us",
            f"Rect: {self.rect_sensor}, W = {rect_w:.1f} um, H = {rect_h:.1f} um",
            f"Fitting: {self.do_fitting} ({gpu.backend_name()}), Stats: {self.show_stats}, "
            f"Row-Col: {self.row_col_fitting}",
            f"Fit: {fitconfig.describe()} (Press 'o' for settings)",
            f"Web Server: {'ON' if self.web_server_enabled else 'OFF'}, "
            f"Rect Stats: {'ON' if self.show_rect_stats else 'OFF'} (Press 'w' to toggle)",
            f"User output: {self.camera.userdefined_line_enabled} (Press 'y' to toggle)",
        ]

    def _draw_rectangles(self, disp, view: ViewTransform):
        for rect, color in ((self.rect_disp, (255, 255, 255)), (self.fit_rect_disp, (0, 255, 0))):
            if rect is not None:
                cv2.rectangle(disp, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
        for rect, color in (
            (self.rect_sensor, (255, 255, 255)),
            (self.fit_rect_sensor, (0, 255, 0)),
        ):
            if rect is not None:
                p1 = view.to_display(rect[0], rect[1])
                p2 = view.to_display(rect[2], rect[3])
                cv2.rectangle(disp, p1, p2, color, 2)

    # ------------------------------ statistics ---------------------------- #
    def _reset_spot_stats(self):
        self.stats_dx = self.stats_dy = self.stats_sigma = np.array([0.0])
        self.rows, self.columns, self.grid_stats = [], [], {}

    def _update_spot_stats(self, spots: SpotArray):
        if len(spots) > 1:
            order = np.argsort(spots.x)
            self.stats_dx = np.diff(spots.x[order])
            self.stats_dy = np.diff(spots.y[order])
            self.stats_sigma = spots.sigma
            self.rows, self.columns, self.grid_stats = classify_grid(spots, eps=20)
        else:
            self._reset_spot_stats()
        self.std_dx_ema = self.ema * self.std_dx_ema + (1 - self.ema) * float(np.std(self.stats_dx))
        self.std_dy_ema = self.ema * self.std_dy_ema + (1 - self.ema) * float(np.std(self.stats_dy))
        self.sigma_std_ema = (
            self.ema * self.sigma_std_ema + (1 - self.ema) * float(np.std(self.stats_sigma))
        )

    @staticmethod
    def _curvature(deltas) -> float:
        """Mean second difference of the spot spacings (sign = bow direction)."""
        return float(np.mean(np.diff(deltas))) if len(deltas) > 1 else 0.0

    def _update_web_stats(self):
        um = self.camera.pixel_size * 1e6
        stats = {
            "basic_stats": {
                "dx_std": float(np.std(self.stats_dx)),
                "dy_std": float(np.std(self.stats_dy)),
                "dx_mean_um": float(np.mean(self.stats_dx)) * um,
                "dy_mean_um": float(np.mean(self.stats_dy)) * um,
                "sigma_std": self.sigma_std_ema,
            }
        }
        if self.grid_stats:
            grid = {
                axis: {
                    k: v for k, v in axis_stats.items() if isinstance(v, (int, float))
                }
                for axis, axis_stats in self.grid_stats.items()
            }
            for axis in grid:
                for key in list(grid[axis]):
                    if key.startswith("avg_"):
                        grid[axis][f"{key}_um"] = grid[axis][key] * um
            stats["grid_stats"] = grid
            stats["row_col_counts"] = {
                "num_rows": len(self.rows),
                "num_columns": len(self.columns),
            }
        self.latest_stats = stats

    # ------------------------------ mouse --------------------------------- #
    def on_mouse(self, event, x, y, flags, _param):
        view = self.view
        if view.contains(x, y):
            sx, sy = view.to_sensor(x, y)
            self.last_mouse = (int(sx), int(sy))
        ctrl = flags & (cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_SHIFTKEY)

        if event == cv2.EVENT_MOUSEWHEEL:
            if ctrl:  # change aspect ratio, keeping the long edge fixed
                if abs(self.roi_model.aspect) < 0.01:
                    return
                factor = 0.95 if flags > 0 else 1.05
                self.roi_model.keep_point_fixed(
                    *self.last_mouse, new_aspect=self.roi_model.aspect * factor
                )
            else:  # zoom
                delta = 0.05 if abs(self.roi_model.scale) > 0.1 else 0.01
                delta = -delta if flags > 0 else delta
                self.roi_model.keep_point_fixed(
                    *self.last_mouse, new_scale=self.roi_model.scale + delta
                )
            self.camera.ROI = self.roi_model.tuple

        elif event == cv2.EVENT_LBUTTONDOWN and view.contains(x, y):
            if ctrl:
                self.fit_rect_disp = [x, y, x, y]
            else:
                self.rect_disp = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.fit_rect_disp is not None:
                self.fit_rect_disp[2:] = [x, y]
            elif self.rect_disp is not None:
                self.rect_disp[2:] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.fit_rect_disp is not None:
                self.fit_rect_sensor = self._finish_rect(self.fit_rect_disp, view)
                self.fit_rect_disp = None
                write_status(self)
            elif self.rect_disp is not None:
                self.rect_sensor = self._finish_rect(self.rect_disp, view)
                self.rect_disp = None
                write_status(self)

    @staticmethod
    def _finish_rect(rect_disp, view: ViewTransform) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect_disp
        s1 = view.to_sensor(min(x1, x2), min(y1, y2))
        s2 = view.to_sensor(max(x1, x2), max(y1, y2))
        return (int(s1[0]), int(s1[1]), int(s2[0]), int(s2[1]))

    # ------------------------------ keys ---------------------------------- #
    def _handle_key(self, key: int) -> bool:
        if key == 27:  # ESC
            return False
        if key in KEY_FACTORS:
            self._adjust_exposure(KEY_FACTORS[key])
        elif key == ord("a"):
            self.auto_exp = not self.auto_exp
        elif key == ord("f"):
            self.do_fitting = not self.do_fitting
        elif key == ord("g"):
            self.show_stats = not self.show_stats
        elif key == ord("h"):
            self.row_col_fitting = not self.row_col_fitting
        elif key == ord("c"):
            self.rect_sensor = None
        elif key == ord("v"):
            self.fit_rect_sensor = None
        elif key == ord("s"):
            self._quick_save()
        elif key == ord("d"):
            self._dialog_save()
        elif key == ord("t"):
            self._switch_camera()
        elif key == ord("w"):
            self._toggle_web_server()
        elif key == ord("o"):
            settings.open_settings()
        elif key == ord("y"):
            self.camera.set_userdefined_line(not self.camera.userdefined_line_enabled)
            logging.info(
                f"userdefined line sync mode set to {self.camera.userdefined_line_enabled}"
            )
        if key != -1:
            write_status(self)
        return True

    def _adjust_exposure(self, factor: float):
        if self.auto_exp:
            return
        self.exposure_us = int(
            np.clip(self.camera.ExposureTime * factor, MIN_EXPOSURE_US, MAX_EXPOSURE_US)
        )
        self.camera.ExposureTime = self.exposure_us
        self.exposure_us = self.camera.ExposureTime

    # ------------------------------ actions ------------------------------- #
    def _save_frame(self, jpg_path: str):
        """JPEG of the display frame plus the raw frame as .npy."""
        cv2.imwrite(jpg_path, self.frame_disp)
        np.save(os.path.splitext(jpg_path)[0] + ".npy", self.current_frame)
        logging.info("Saved %s (+ .npy)", jpg_path)

    def _quick_save(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._save_frame(os.path.join(DATA_DIR, f"vipa_{ts}.jpg"))

    def _dialog_save(self):
        fp = settings.save_dialog(
            initialdir=DATA_DIR,
            initialfile=f"vipa_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
            title="Save Image",
            filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")),
            defaultextension=".jpg",
        )
        if fp:
            self._save_frame(fp)

    def _switch_camera(self):
        if len(self.cameras) < 2:
            return
        self._cached_state[self.camera.serial] = dict(
            roi_model=self.roi_model,
            exp=self.camera.ExposureTime,
            gain=self.camera.Gain,
            auto=self.auto_exp,
            rect=self.rect_sensor,
            fit_rect=self.fit_rect_sensor,
            do_fitting=self.do_fitting,
            show_stats=self.show_stats,
        )
        self.curr_idx = (self.curr_idx + 1) % len(self.cameras)
        self.camera = self.cameras[self.curr_idx]
        st = self._cached_state.get(self.camera.serial)
        if st:
            self.roi_model = st["roi_model"]
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = st["exp"]
            self.camera.Gain = st["gain"]
            self.auto_exp = st["auto"]
            self.rect_sensor = st["rect"]
            self.fit_rect_sensor = st["fit_rect"]
            self.do_fitting = st["do_fitting"]
            self.show_stats = st["show_stats"]
        else:
            self.roi_model = ROIModel(self.camera.W, self.camera.H)
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = self.exposure_us
            self.camera.Gain = 0
            self.auto_exp = False
            self.rect_sensor = self.fit_rect_sensor = None
            self.do_fitting = self.show_stats = False
        logging.info("Switched to %s", self.camera.model_name)

    def _toggle_web_server(self):
        if not self.web_server_enabled:
            if self._web_thread is None:
                from ..server import create_app

                app = create_app(self)
                self._web_thread = Thread(
                    target=lambda: app.run(
                        host="0.0.0.0", port=WEB_SERVER_PORT, debug=False, use_reloader=False
                    ),
                    daemon=True,
                )
                self._web_thread.start()
            self.web_server_enabled = True
            self.show_rect_stats = True
            logging.info(f"Web server started on http://localhost:{WEB_SERVER_PORT}")
        else:
            self.web_server_enabled = False
            self.show_rect_stats = False
            logging.info("Web server stopped and rect stats display disabled")

    def _sleep_for_fps(self, frame_dt: float):
        target = 1 / FPS_LIMIT
        if frame_dt < target:
            time.sleep(target - frame_dt)
            frame_dt = target
        self.avg_dt = 0.9 * self.avg_dt + 0.1 * frame_dt
