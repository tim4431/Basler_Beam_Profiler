"""Interactive viewer: main loop, keyboard/mouse handling and overlay layout.

Keyboard shortcuts are listed in the README.

Which camera is on screen - and what happens when it is unplugged, comes back,
or another one is plugged in - is `cameras.session.CameraSession`'s job; the
viewer only supplies its per-camera state (ROI, exposure, rectangles) to be
saved and restored.  Nothing a camera does may end the loop: a disconnect is
an offline screen, any other per-frame error is logged and the loop goes on.
Only Esc or closing the window quits."""

from __future__ import annotations

import logging
import os
import time
import traceback
from collections import deque
from threading import Thread

import cv2
import numpy as np

from ..cameras import CameraManager
from ..cameras.session import CameraSession
from ..config import (
    DATA_DIR,
    FPS_LIMIT,
    PAD_COLOR,
    WEB_SERVER_PORT,
    WINDOW_SIZE,
)
from .. import fitconfig
from ..processing import SpotArray, classify_grid, detect_spots, gpu, region_stats
from ..roi import ROIModel, ViewTransform
from ..status import write_status
from . import overlays

try:  # tkinter is a separate system package on Linux (python3-tk)
    from . import cameras_window, settings
except ImportError as _tk_err:
    _TK_HINT = f"{_tk_err}; install it with `sudo apt install python3-tk`"

    class settings:  # noqa: N801  (stands in for the module)
        @staticmethod
        def open_settings():
            logging.warning("Settings window unavailable: %s", _TK_HINT)

        @staticmethod
        def close_settings():
            pass

        @staticmethod
        def is_open() -> bool:
            return False

    class cameras_window:  # noqa: N801
        @staticmethod
        def open_cameras(_manager):
            logging.warning("Cameras window unavailable (%s); 't' still cycles cameras",
                            _TK_HINT)

        @staticmethod
        def close_cameras():
            pass

        @staticmethod
        def is_open() -> bool:
            return False

WINDOW = "Beam Profiler"

# waitKeyEx arrow codes: Windows and Linux(GTK) report different values
KEY_FACTORS = {
    2555904: 1.1, 65363: 1.1,  # right: +10%
    2424832: 0.9, 65361: 0.9,  # left:  -10%
    2490368: 10, 65362: 10,  # up:    x10
    2621440: 0.1, 65364: 0.1,  # down:  /10
}
TOAST_LIFETIME = 6.0
_BLANK = np.full((2, 2), 40, np.uint8)  # stands in for a frame before the first one


class Viewer:
    def __init__(self, manager: CameraManager, prefer: str | None = None, simulate: bool = False):
        self.manager = manager
        self._auto_exp = False
        self.do_fitting = False
        self.show_stats = False
        self.row_col_fitting = False
        self.exposure_us = 0
        self.roi_model: ROIModel | None = None
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
        self.frame_disp = None
        self.latest_spots = SpotArray.empty()
        self.latest_stats: dict = {}
        self.current_rect_stats = None
        self.web_server_enabled = False
        self.show_rect_stats = False
        self._web_thread = None

        self.toasts: deque = deque(maxlen=4)
        self._camera_state = None
        self._last_disp = None
        self._error_log: dict[str, float] = {}

        self.cams = CameraSession(
            manager, self._capture_state, self._restore_state, self._notify
        )
        self.cams.start(prefer=prefer, simulate=simulate)

        self.toolbar = overlays.Toolbar(
            [
                ("fit", "Fit (f)"),
                ("stats", "Stats (g)"),
                ("grid", "Grid (h)"),
                ("cameras", "Cameras (k)"),
                ("web", "Web (w)"),
                ("save", "Save (s)"),
                ("settings", "Settings (o)"),
            ],
            WINDOW_SIZE,
        )
        self._hover_button = None

        # WINDOW_GUI_NORMAL drops the Qt toolbar and the pixel-hover overlay,
        # which render as black/garbled boxes (OpenCV's bundled Qt has no fonts)
        gpu.warm_up()  # pay the CuPy import off the critical path
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW, WINDOW_SIZE, WINDOW_SIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

    # ------------------------------ properties --------------------------- #
    @property
    def camera(self):
        """The camera on screen.  While it is disconnected this is still the
        lost one: its getters answer from cache, its hardware calls are skipped."""
        return self.cams.camera

    @property
    def offline(self) -> bool:
        return self.cams.offline is not None

    @property
    def auto_exp(self):
        return self._auto_exp

    @auto_exp.setter
    def auto_exp(self, value):
        self._auto_exp = value
        if value:
            self.camera.AutoExposureOn()
        else:
            self.camera.AutoExposureOff()
            self.exposure_us = int(np.clip(self.exposure_us, *self.exposure_limits))

    @property
    def exposure_limits(self) -> tuple[float, float]:
        """Per-camera, from camera_config.yaml, narrowed to what the hardware allows."""
        return self.camera.min_exposure, self.camera.max_exposure

    @property
    def view(self) -> ViewTransform:
        return ViewTransform(self.camera.ROI, WINDOW_SIZE)

    # ------------------------- per-camera state -------------------------- #
    def _capture_state(self) -> dict:
        cam = self.camera
        return dict(
            roi_model=self.roi_model,
            exposure_us=self.exposure_us,
            exposure=cam.ExposureTime,
            gain=cam.Gain,
            auto=self.auto_exp,
            rect=self.rect_sensor,
            fit_rect=self.fit_rect_sensor,
            do_fitting=self.do_fitting,
            show_stats=self.show_stats,
            row_col=self.row_col_fitting,
            user_line=cam.userdefined_line_enabled,
        )

    def _restore_state(self, cam, state: dict | None):
        """Apply a camera's saved state, or its configured defaults the first
        time it is shown."""
        self._reset_spot_stats()
        self.current_frame = self.frame_disp = None
        self.latest_spots = SpotArray.empty()
        self.rect_disp = self.fit_rect_disp = None
        if state:
            self.roi_model = state["roi_model"]
            cam.ROI = self.roi_model.tuple
            cam.Gain = state["gain"]
            cam.ExposureTime = state["exposure"]
            self.exposure_us = state["exposure_us"]
            self.auto_exp = state["auto"]
            self.rect_sensor, self.fit_rect_sensor = state["rect"], state["fit_rect"]
            self.do_fitting, self.show_stats = state["do_fitting"], state["show_stats"]
            self.row_col_fitting = state["row_col"]
            if state["user_line"]:
                cam.set_userdefined_line(True)
        else:
            self.roi_model = ROIModel(cam.W, cam.H, cam.default_roi)
            cam.ROI = self.roi_model.tuple
            self.exposure_us = int(np.clip(cam.settings.default_exposure, *self.exposure_limits))
            cam.ExposureTime = self.exposure_us
            cam.Gain = cam.settings.default_gain
            self.auto_exp = False
            self.rect_sensor = self.fit_rect_sensor = None
            self.do_fitting = self.show_stats = False
        write_status(self)

    def _notify(self, message: str, level: str = "info"):
        """Show a message for a few seconds.  Safe from any thread."""
        self.toasts.append((time.monotonic(), message, level))

    # ------------------------------ main loop ----------------------------- #
    def run(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            while True:
                t0 = time.time()
                try:
                    disp = self.step()
                except Exception as e:  # a bug must not take the profiler down
                    self._report_error(e)
                    disp = self._last_disp
                if disp is None:
                    disp = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8)
                self._last_disp = disp
                cv2.imshow(WINDOW, disp)
                if not self._handle_key(cv2.waitKeyEx(1)):
                    break
                if not self._window_open():
                    break
                self._sleep_for_fps(time.time() - t0)
        finally:
            self.cams.close()
            cv2.destroyAllWindows()

    def step(self) -> np.ndarray:
        """One frame: follow camera events, grab, process, draw.  Returns the
        image to show; touches no window, so it can be driven headlessly."""
        self.cams.service()
        frame = self.cams.grab()
        state = (self.cams.uid, self.offline)
        if state != self._camera_state:  # mirror connects/disconnects to pylon_camera.json
            self._camera_state = state
            write_status(self)
        if frame is not None:
            self.current_frame = frame
            self.frame_disp = (
                cv2.convertScaleAbs(frame, alpha=1 / 256.0)
                if self.camera.mode == "16Bit"
                else frame
            )
            self._process(frame)
        elif self.offline:
            self.current_frame = None  # never serve a dead camera's last frame as current
        return self._compose(self.frame_disp, SpotArray.empty() if self.offline else self.latest_spots)

    def _process(self, frame):
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

    def _report_error(self, e: Exception):
        """Log a per-frame error with its traceback - once per 10 s per kind,
        not 60 times a second - and show it."""
        key = f"{type(e).__name__}: {e}"
        now = time.monotonic()
        if now - self._error_log.get(key, -1e9) > 10:
            self._error_log[key] = now
            logging.error("error in the main loop (continuing):\n%s", traceback.format_exc())
            self._notify(f"Error: {key[:90]} (see log)", "error")

    @staticmethod
    def _window_open() -> bool:
        """False once the user closes the window with the title-bar X.
        waitKeyEx pumps the GUI events, so the check right after it sees the
        close before the next imshow would silently recreate the window."""
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

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
        resized = cv2.resize(_BLANK if frame_disp is None else frame_disp,
                             (view.view_w, view.view_h))
        if self.offline:
            resized //= 3  # the last picture, dimmed, under the banner
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
        if self.show_stats and not self.offline:
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
        if self.row_col_fitting and self.do_fitting and not self.offline:
            overlays.draw_row_col(disp, view, self.rows, self.columns)
        self._draw_camera_status(disp)
        self.toolbar.draw(
            disp,
            {
                "fit": self.do_fitting,
                "stats": self.show_stats,
                "grid": self.row_col_fitting,
                "cameras": cameras_window.is_open(),
                "web": self.web_server_enabled,
                "settings": settings.is_open(),
            },
            hover=self._hover_button,
        )
        return disp

    def _draw_camera_status(self, disp):
        if self.offline:
            label = self.camera.label
            overlays.draw_banner(
                disp,
                "Camera disconnected",
                [
                    f"{label} stopped responding.",
                    "Plug it back in - it reconnects by itself and keeps its settings.",
                    "Or switch camera:  k = Cameras window,  t = next camera.",
                ],
                "error",
            )
        connecting = self.cams.connecting
        if connecting:
            if self.offline:  # under the banner
                overlays.draw_pill(disp, f"Connecting to {connecting}...", "warn",
                                   y=WINDOW_SIZE // 2 + 120)
            else:  # the old camera is still live: stay out of the picture and the HUD
                overlays.draw_pill(disp, f"Connecting to {connecting}...", "warn", right=True)
        overlays.draw_toasts(  # a copy: the Tk thread may append meanwhile
            disp, list(self.toasts), time.monotonic(),
            bottom=WINDOW_SIZE - overlays.Toolbar.HEIGHT - overlays.Toolbar.MARGIN - 12,
            lifetime=TOAST_LIFETIME,
        )

    def _hud_lines(self) -> list[str]:
        cam = self.camera
        exposure = cam.ExposureTime
        exposure_txt = f"Auto = {exposure:.0f}" if self.auto_exp else f"{exposure:.0f}"
        um = cam.pixel_size * 1e6
        # Without a configured pixel_size every "um" below is really a pixel
        # count - say so rather than let it read as calibrated.
        scale_warning = "" if cam.pixel_size_known else " [px size unset: um = px]"
        rect_w = (self.rect_sensor[2] - self.rect_sensor[0]) * um if self.rect_sensor else 0
        rect_h = (self.rect_sensor[3] - self.rect_sensor[1]) * um if self.rect_sensor else 0
        entry = self.manager.entry(self.cams.uid)
        source = entry.backend.title if entry is not None else cam.backend
        state = "DISCONNECTED" if self.offline else f"FPS: {1 / self.avg_dt:4.1f}"
        others = sum(
            1 for d in self.manager.snapshot()
            if d["present"] and not d["active"] and d["backend"] != "sim"
        )
        more = f"  (+{others} more, k)" if others else ""
        return [
            f"Camera: {cam.label} [{source}]{scale_warning}, {state}{more}",
            f"ROI: {cam.ROI}  Mouse: {self.last_mouse}",
            f"Exposure: {exposure_txt} us",
            f"Rect: {self.rect_sensor}, W = {rect_w:.1f} um, H = {rect_h:.1f} um",
            f"Fitting: {self.do_fitting} ({gpu.backend_name()}), Stats: {self.show_stats}, "
            f"Row-Col: {self.row_col_fitting}",
            f"Fit: {fitconfig.describe()} (Press 'o' for settings)",
            f"Web Server: {'ON' if self.web_server_enabled else 'OFF'}, "
            f"Rect Stats: {'ON' if self.show_rect_stats else 'OFF'} (Press 'w' to toggle)",
            f"exposure_sync: {cam.userdefined_line_enabled} (Press 'y' to toggle)",
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
        button = self.toolbar.hit(x, y)
        dragging = self.rect_disp is not None or self.fit_rect_disp is not None
        self._hover_button = None if dragging else button
        if button is not None and not dragging:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._toolbar_click(button)
            return
        view = self.view
        if view.contains(x, y):
            sx, sy = view.to_sensor(x, y)
            self.last_mouse = (int(sx), int(sy))
        ctrl = flags & cv2.EVENT_FLAG_CTRLKEY

        if event == cv2.EVENT_MOUSEWHEEL:
            if self.offline:  # nothing to zoom into; the ROI is restored on reconnect
                return
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
            self.cams.guard(setattr, self.camera, "ROI", self.roi_model.tuple)

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

    def _toolbar_click(self, name: str):
        if name == "fit":
            self.do_fitting = not self.do_fitting
        elif name == "stats":
            self.show_stats = not self.show_stats
        elif name == "grid":
            self.row_col_fitting = not self.row_col_fitting
        elif name == "cameras":
            self._toggle_cameras_window()
        elif name == "web":
            self._toggle_web_server()
        elif name == "save":
            self._quick_save()
        elif name == "settings":
            if settings.is_open():
                settings.close_settings()
            else:
                settings.open_settings()
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
            self.cams.next()
        elif key == ord("k"):
            self._toggle_cameras_window()
        elif key == ord("w"):
            self._toggle_web_server()
        elif key == ord("o"):
            settings.open_settings()
        elif key == ord("y"):
            cam = self.camera
            self.cams.guard(cam.set_userdefined_line, not cam.userdefined_line_enabled)
            logging.info(f"userdefined line sync mode set to {cam.userdefined_line_enabled}")
        if key != -1:
            write_status(self)
        return True

    def _adjust_exposure(self, factor: float):
        if self.auto_exp or self.offline:
            return
        self.exposure_us = int(np.clip(self.exposure_us * factor, *self.exposure_limits))
        self.cams.guard(setattr, self.camera, "ExposureTime", self.exposure_us)

    def _toggle_cameras_window(self):
        if cameras_window.is_open():
            cameras_window.close_cameras()
        else:
            cameras_window.open_cameras(self.manager)

    # ------------------------------ actions ------------------------------- #
    def _save_frame(self, jpg_path: str, frame=None, frame_disp=None):
        """JPEG of the display frame plus the raw frame as .npy."""
        frame = self.current_frame if frame is None else frame
        frame_disp = self.frame_disp if frame_disp is None else frame_disp
        cv2.imwrite(jpg_path, frame_disp)
        np.save(os.path.splitext(jpg_path)[0] + ".npy", frame)
        logging.info("Saved %s (+ .npy)", jpg_path)
        self._notify(f"Saved {os.path.basename(jpg_path)}")

    def _can_save(self) -> bool:
        if self.current_frame is None:
            self._notify("No live frame to save", "warn")
            return False
        return True

    def _quick_save(self):
        if not self._can_save():
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._save_frame(os.path.join(DATA_DIR, f"vipa_{ts}.jpg"))

    def _dialog_save(self):
        """Ask for a file name on the Tk thread and save the frame as it was
        when the key was pressed; the live view keeps running meanwhile."""
        if not self._can_save():
            return
        try:
            from . import tkhost
        except ImportError as e:
            logging.warning("Save dialog unavailable (%s); quick-saving instead", e)
            self._quick_save()
            return
        frame, frame_disp = self.current_frame, self.frame_disp

        def ask(root):
            from tkinter import filedialog

            fp = filedialog.asksaveasfilename(
                parent=root,
                initialdir=DATA_DIR,
                initialfile=f"vipa_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
                title="Save Image",
                filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")),
            )
            if fp:
                self._save_frame(fp, frame, frame_disp)

        tkhost.call(ask)

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
