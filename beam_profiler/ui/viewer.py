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
from .. import fitconfig, uiconfig
from ..processing import SpotArray, classify_grid, detect_spots, gpu, region_stats
from ..roi import ROIModel, ViewTransform
from ..status import write_status
from . import desktop, overlays

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
# key -> action (see Viewer._act); F1 differs between Windows and Linux too
KEY_ACTIONS = {
    ord("a"): "auto", ord("f"): "fit", ord("g"): "stats", ord("h"): "grid",
    ord("c"): "clear_rect", ord("v"): "clear_fit_rect", ord("s"): "save",
    ord("d"): "save_as", ord("t"): "next", ord("k"): "cameras", ord("w"): "web",
    ord("o"): "open_settings", ord("y"): "sync", ord("i"): "info", ord("?"): "help",
    7340032: "help", 65470: "help",  # F1
    ord("+"): "zoom_in", ord("="): "zoom_in", 65451: "zoom_in",  # keypad + (GTK)
    ord("-"): "zoom_out", 65453: "zoom_out", ord("0"): "zoom_reset",
}
TOAST_LIFETIME = 6.0
TOOLTIP_DELAY = 0.3  # s of hovering before a button explains itself
TOOLTIP_TIMEOUT = 8.0  # s without mouse events (the pointer may have left the window)
_BLANK = np.full((2, 2), 40, np.uint8)  # stands in for a frame before the first one
Tool = overlays.Tool
TOOLS = [
    Tool("fit", "fit", "Fit spots", ("F",), "Ctrl-drag on the image to fit only inside"),
    Tool("stats", "stats", "Spot statistics", ("G",)),
    Tool("grid", "grid", "Rows & columns", ("H",), "For beam arrays"),
    None,
    Tool("auto", "auto", "Auto exposure", ("A",)),
    Tool("exp_down", "exp_down", "Exposure −10%", ("Left",), "Shift-click or Down key: ÷10"),
    Tool("exp_up", "exp_up", "Exposure +10%", ("Right",), "Shift-click or Up key: ×10"),
    None,
    Tool("clear", "clear", "Clear rectangles", ("C", "V"), "C clears the white one, V the green"),
    Tool("cameras", "cameras", "Cameras", ("K",), "Every camera found; switch, set pixel size"),
    Tool("next", "next", "Next camera", ("T",)),
    None,
    Tool("sync", "sync", "Sync output (Line3)", ("Y",)),
    Tool("web", "web", f"HTTP API on port {WEB_SERVER_PORT}", ("W",),
         "Also shows the white rectangle's pixel statistics"),
    Tool("save", "save", "Save frame", ("S",), "JPEG + raw .npy.  Shift-click or D: save as…"),
    None,
    Tool("info", "info", "Info panel", ("I",)),
    Tool("settings", "settings", "Settings", ("O",), "Spot fit and interface zoom (+ / -)"),
    Tool("help", "help", "Keyboard & mouse", ("?",)),
]


class Readout:
    """Numbers as displayed: refreshed every `overlays.READOUT_PERIOD` (and
    at once when one appears or goes).  Digits that change 60 times a second
    cannot be read, and every change costs a text render."""

    def __init__(self, clock=time.monotonic):
        self._values: dict = {}
        self._clock = clock

    def __call__(self, key, value):
        now = self._clock()
        at, old = self._values.get(key, (-1e9, None))
        if now - at >= overlays.READOUT_PERIOD or (old is None) != (value is None):
            self._values[key] = (now, value)
            return value
        return old


def initial_window_size() -> tuple[int, int]:
    """The size the live view opened at last time, else WINDOW_SIZE square;
    either way no bigger than the screen (minus room for panels and the
    title bar) when the screen size can be found."""
    w, h = uiconfig.active().window or (WINDOW_SIZE, WINDOW_SIZE)
    screen = desktop.screen_size()
    if screen:
        w = max(uiconfig.MIN_WINDOW, min(w, screen[0] - 40))
        h = max(uiconfig.MIN_WINDOW, min(h, screen[1] - 100))
    return int(w), int(h)


def to_8bit(frame: np.ndarray) -> np.ndarray:
    """A frame as shown: "16Bit" is Mono12 MSB-aligned in uint16, so >> 8."""
    return frame if frame.dtype == np.uint8 else cv2.convertScaleAbs(frame, alpha=1 / 256.0)


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
        self._shown_frame = None  # last frame on screen: kept, dimmed, while offline
        self.latest_spots = SpotArray.empty()
        self.latest_stats: dict = {}
        self.current_rect_stats = None
        self.web_server_enabled = False
        self.show_rect_stats = False
        self._web_thread = None

        self.toasts: deque = deque(maxlen=4)
        self._unclaimed_frames = 120  # see desktop.claim_window
        self.show_info = True
        self.show_help = False
        self._camera_state = None
        self._last_disp = None
        self._error_log: dict[str, float] = {}
        self._fps_shown, self._fps_at = float(FPS_LIMIT), 0.0
        self._readout = Readout()

        self.cams = CameraSession(
            manager, self._capture_state, self._restore_state, self._notify
        )
        self.cams.start(prefer=prefer, simulate=simulate)

        # The picture is composed at the window's real size (`canvas`, followed
        # every frame in run()) and the overlays at the UI scale; `_layout`
        # rebuilds what depends on either when one changes.
        self.canvas: tuple[int, int] = initial_window_size()
        self.ui = overlays.Metrics.at(uiconfig.scale())
        self._layout_key = None
        self._layout()
        self._hover_button = None
        self._hover_since = self._mouse_at = 0.0

        # WINDOW_GUI_NORMAL drops the Qt toolbar and the pixel-hover overlay,
        # which render as black/garbled boxes (OpenCV's bundled Qt has no fonts).
        # FREERATIO: the picture fills the window instead of being letterboxed
        # by Qt, so getWindowImageRect reports the window's real shape.
        gpu.warm_up()  # pay the CuPy import off the critical path
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(WINDOW, *self.canvas)
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
        return ViewTransform(self.camera.ROI, self.canvas)

    def _layout(self):
        """Re-derive everything sized by the window or the UI scale - called
        before each frame is composed, so the toolbar the mouse is tested
        against is always the one on screen."""
        scale = uiconfig.scale()
        key = (self.canvas, scale)
        if key == self._layout_key:
            return
        if self._layout_key is not None and scale != self._layout_key[1]:
            self._notify(f"Interface zoom {scale:.0%}")
        self._layout_key = key
        self.ui = overlays.Metrics.at(scale)
        self.toolbar = overlays.Toolbar(TOOLS, self.canvas, self.ui)
        self._hover_button = None

    def _follow_window(self):
        """Adopt the window's current image size for the next frame.  Junk
        (no window yet, a closed one, a backend that cannot tell) keeps the
        last good size."""
        try:
            _x, _y, w, h = cv2.getWindowImageRect(WINDOW)
        except (cv2.error, ValueError, TypeError):
            return
        if uiconfig.valid_window((w, h)) and (w, h) != self.canvas:
            self.canvas = (int(w), int(h))

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
        self.current_frame = self._shown_frame = None
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
                self._follow_window()
                try:
                    disp = self.step()
                except Exception as e:  # a bug must not take the profiler down
                    self._report_error(e)
                    disp = self._last_disp
                if disp is None:
                    disp = np.zeros((self.canvas[1], self.canvas[0], 3), np.uint8)
                self._last_disp = disp
                cv2.imshow(WINDOW, disp)
                if not self._handle_key(cv2.waitKeyEx(1)):
                    break
                if self._unclaimed_frames:  # until the window exists: give it our
                    self._unclaimed_frames -= 1  # WM_CLASS so the dock shows our icon
                    if desktop.claim_window(WINDOW):
                        self._unclaimed_frames = 0
                if not self._window_open():
                    break
                self._sleep_for_fps(time.time() - t0)
        finally:
            uiconfig.set_window(self.canvas)  # open at this size next time
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
            self.current_frame = self._shown_frame = frame
            self._process(frame)
        elif self.offline:
            self.current_frame = None  # never serve a dead camera's last frame as current
        return self._compose(self._shown_frame,
                             SpotArray.empty() if self.offline else self.latest_spots)

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

    def _compose(self, frame, spots) -> np.ndarray:
        """Resize + pad the frame to the window and draw all overlays.

        Everything is drawn in display space after the resize, so text and
        line widths do not scale with the ROI zoom.  A 16-bit frame is
        reduced to 8 bit *after* the resize: on a 25 MP sensor that is the
        difference between converting 25 MP and 1.2 MP every frame."""
        self._layout()
        m, (win_w, win_h) = self.ui, self.canvas
        view = self.view
        resized = to_8bit(cv2.resize(_BLANK if frame is None else frame,
                                     (view.view_w, view.view_h)))
        if self.offline:
            resized //= 3  # the last picture, dimmed, under the banner
        padded = cv2.copyMakeBorder(
            resized,
            view.pad_t,
            win_h - view.view_h - view.pad_t,
            view.pad_l,
            win_w - view.view_w - view.pad_l,
            cv2.BORDER_CONSTANT,
            value=PAD_COLOR[0],
        )
        disp = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        live = not self.offline
        if len(spots) and not self.row_col_fitting:
            overlays.draw_spots(disp, spots, view, self.camera.pixel_size, m=m)
        if self.row_col_fitting and self.do_fitting and live:
            overlays.draw_row_col(disp, view, self.rows, self.columns, m=m)
        self._draw_rectangles(disp, view)

        hud = self._hud()
        overlays.draw_top_bar(disp, hud, m)
        self._draw_cards(disp, hud, len(spots), live)
        self._draw_camera_status(disp)
        now = time.monotonic()
        self.toolbar.draw(
            disp,
            self._tool_states(),
            hover=self._hover_button,
            disabled=self._tool_disabled(),
            tooltip=(TOOLTIP_DELAY <= now - self._hover_since
                     and now - self._mouse_at < TOOLTIP_TIMEOUT),
        )
        if self.show_help:
            overlays.draw_help(disp, WEB_SERVER_PORT, m)
        return disp

    def _draw_cards(self, disp, hud, n_spots: int, live: bool):
        """Left column from the top-left corner (what the view shows), right
        column from the top-right (what the fit found); at the largest scale
        up to the UI scale at which they fit above the toolbar - side by side,
        or the right one under the left on a narrow window."""
        m, win_w = self.ui, self.canvas[0]
        left, right = [], []
        if self.show_info:
            left.append(overlays.Card(overlays.info_rows(hud)))
        stats = self._readout("rect", self.current_rect_stats) if self.show_rect_stats else None
        if stats is not None:
            left.append(overlays.rect_card(stats))
        um = self.camera.pixel_size * 1e6
        if self.show_stats and live:
            numbers = self._readout("spots", (
                n_spots,
                self.std_dx_ema,
                self.std_dy_ema,
                float(np.mean(self.stats_dx)) * um,
                float(np.mean(self.stats_dy)) * um,
                self._curvature(self.stats_dx),
                self._curvature(self.stats_dy),
                self.sigma_std_ema,
            ))
            right.append(overlays.spot_card(*numbers, show_curvature=not self.row_col_fitting))
            if self.row_col_fitting:
                rows, columns, grid = self._readout(
                    "grid", (self.rows, self.columns, self.grid_stats))
                if grid:
                    right.append(overlays.grid_card(rows, columns, grid, um))
        if not left and not right:
            return
        top = m.top_h + m.px(10)
        cm, stacked = overlays.fit_cards(m, win_w, top, self.toolbar.box[1] - m.gap,
                                         left, right)
        y = top
        for card in left:
            y = card.draw(disp, cm, cm.edge, y) + cm.gap
        y = y if stacked else top
        for card in right:
            y = card.draw(disp, cm, win_w - cm.edge - cm.card_w, y) + cm.gap

    def _tool_states(self) -> dict[str, bool]:
        return {
            "fit": self.do_fitting,
            "stats": self.show_stats,
            "grid": self.row_col_fitting,
            "auto": self.auto_exp,
            "cameras": cameras_window.is_open(),
            "sync": self.camera.userdefined_line_enabled,
            "web": self.web_server_enabled,
            "info": self.show_info,
            "settings": settings.is_open(),
            "help": self.show_help,
        }

    def _tool_disabled(self) -> set[str]:
        """Buttons that would do nothing right now, drawn dimmed."""
        off = set()
        if self.auto_exp or self.offline:
            off |= {"exp_down", "exp_up"}
        if self.offline:
            off |= {"save", "sync"}
        if self.rect_sensor is None and self.fit_rect_sensor is None:
            off.add("clear")
        return off

    def _draw_camera_status(self, disp):
        if self.offline:
            label = self.camera.label
            overlays.draw_banner(
                disp,
                "Camera disconnected",
                [
                    f"{label} stopped responding.",
                    "Plug it back in - it reconnects by itself and keeps its settings.",
                    "Or switch camera:  K  Cameras window,  T  next camera.",
                ],
                "error",
                icon="camera_off",
                m=self.ui,
            )
        connecting = self.cams.connecting
        if connecting:
            if self.offline:  # under the banner
                overlays.draw_pill(disp, f"Connecting to {connecting}...", "warn",
                                   y=self.canvas[1] // 2 + self.ui.px(90), m=self.ui)
            else:  # the old camera is still live: stay out of the picture and the cards
                overlays.draw_pill(disp, f"Connecting to {connecting}...", "warn", m=self.ui)
        overlays.draw_toasts(  # a copy: the Tk thread may append meanwhile
            disp, list(self.toasts), time.monotonic(),
            bottom=self.toolbar.box[1] - self.ui.px(12),
            lifetime=TOAST_LIFETIME, m=self.ui,
        )

    def _hud(self) -> overlays.Hud:
        cam = self.camera
        um = cam.pixel_size * 1e6
        rect = self.rect_sensor
        entry = self.manager.entry(self.cams.uid)
        others = sum(
            1 for d in self.manager.snapshot()
            if d["present"] and not d["active"] and d["backend"] != "sim"
        )
        now = time.monotonic()
        if now - self._fps_at > 0.5:  # a number that changes 60x a second is unreadable
            self._fps_shown, self._fps_at = 1 / self.avg_dt, now
        # Without a configured pixel_size every "um" is really a pixel count;
        # scale_known=False makes the status bar say so.
        return overlays.Hud(
            status="offline" if self.offline else "live",  # a switch shows as a pill
            camera=cam.label,
            source=entry.backend.title if entry is not None else cam.backend,
            others=others,
            scale_known=cam.pixel_size_known,
            fps=self._fps_shown,
            exposure=self._readout("exposure", cam.ExposureTime) if self.auto_exp
            else cam.ExposureTime,
            auto=self.auto_exp,
            fitting=self.do_fitting,
            backend=gpu.backend_name(),
            web=self.web_server_enabled,
            web_port=WEB_SERVER_PORT,
            sync=cam.userdefined_line_enabled,
            roi=cam.ROI,
            mouse=self.last_mouse,
            rect=rect,
            rect_um=((rect[2] - rect[0]) * um, (rect[3] - rect[1]) * um) if rect else None,
            fit_rect=self.fit_rect_sensor,
            fit=fitconfig.describe(),
        )

    def _draw_rectangles(self, disp, view: ViewTransform):
        um = self.camera.pixel_size * 1e6
        if self.rect_sensor is not None:
            r = self.rect_sensor
            overlays.draw_rect(disp, view.to_display(r[0], r[1]), view.to_display(r[2], r[3]),
                               overlays.OV.RECT,
                               f"{(r[2] - r[0]) * um:.1f} × {(r[3] - r[1]) * um:.1f} µm",
                               self.ui)
        if self.fit_rect_sensor is not None:
            r = self.fit_rect_sensor
            overlays.draw_rect(disp, view.to_display(r[0], r[1]), view.to_display(r[2], r[3]),
                               overlays.OV.FIT_RECT, "fit area", self.ui)
        if self.rect_disp is not None:  # being dragged: show its size live
            x1, y1, x2, y2 = self.rect_disp
            w, h = abs(x2 - x1) / view.scale * um, abs(y2 - y1) / view.scale * um
            overlays.draw_rect(disp, (x1, y1), (x2, y2), overlays.OV.RECT,
                               f"{w:.1f} × {h:.1f} µm", self.ui)
        if self.fit_rect_disp is not None:
            x1, y1, x2, y2 = self.fit_rect_disp
            overlays.draw_rect(disp, (x1, y1), (x2, y2), overlays.OV.FIT_RECT, "fit area", self.ui)

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
        now = time.monotonic()
        self._mouse_at = now
        dragging = self.rect_disp is not None or self.fit_rect_disp is not None
        on_toolbar = not dragging and self.toolbar.contains(x, y)
        button = self.toolbar.hit(x, y) if on_toolbar else None
        if button != self._hover_button:
            self._hover_since = now
        self._hover_button = button
        if on_toolbar:  # the dock swallows clicks and wheel, gaps included
            if event == cv2.EVENT_LBUTTONDOWN and button is not None:
                self._toolbar_click(button, flags)
            return
        view = self.view
        if view.contains(x, y):
            sx, sy = view.to_sensor(x, y)
            self.last_mouse = (int(sx), int(sy))
        ctrl = flags & cv2.EVENT_FLAG_CTRLKEY

        if event == cv2.EVENT_LBUTTONDOWN and self.show_help:
            self.show_help = False  # a click anywhere dismisses the help card
            return

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

    def _toolbar_click(self, name: str, flags: int = 0):
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        action = {
            "exp_down": "exposure_div10" if shift else "exposure_down",
            "exp_up": "exposure_x10" if shift else "exposure_up",
            "save": "save_as" if shift else "save",
            "clear": "clear_both",
            "settings": "toggle_settings",
        }.get(name, name)
        self._act(action)
        write_status(self)

    def _act(self, action: str):
        """Everything a key or a toolbar button can do, by name."""
        if action == "fit":
            self.do_fitting = not self.do_fitting
        elif action == "stats":
            self.show_stats = not self.show_stats
        elif action == "grid":
            self.row_col_fitting = not self.row_col_fitting
        elif action == "auto":
            self.auto_exp = not self.auto_exp
        elif action in ("exposure_down", "exposure_up", "exposure_div10", "exposure_x10"):
            self._adjust_exposure({"exposure_down": 0.9, "exposure_up": 1.1,
                                   "exposure_div10": 0.1, "exposure_x10": 10}[action])
        elif action == "clear_rect":
            self.rect_sensor = None
        elif action == "clear_fit_rect":
            self.fit_rect_sensor = None
        elif action == "clear_both":
            self.rect_sensor = self.fit_rect_sensor = None
        elif action == "save":
            self._quick_save()
        elif action == "save_as":
            self._dialog_save()
        elif action == "next":
            self.cams.next()
        elif action == "cameras":
            self._toggle_cameras_window()
        elif action == "web":
            self._toggle_web_server()
        elif action == "open_settings":
            settings.open_settings()
        elif action == "toggle_settings":
            if settings.is_open():
                settings.close_settings()
            else:
                settings.open_settings()
        elif action == "sync":
            cam = self.camera
            self.cams.guard(cam.set_userdefined_line, not cam.userdefined_line_enabled)
            logging.info(f"userdefined line sync mode set to {cam.userdefined_line_enabled}")
        elif action == "info":
            self.show_info = not self.show_info
        elif action == "help":
            self.show_help = not self.show_help
        elif action == "zoom_in":
            uiconfig.zoom(+1)  # the next frame re-lays out and says the new %
        elif action == "zoom_out":
            uiconfig.zoom(-1)
        elif action == "zoom_reset":
            uiconfig.reset_scale()

    @staticmethod
    def _finish_rect(rect_disp, view: ViewTransform) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect_disp
        s1 = view.to_sensor(min(x1, x2), min(y1, y2))
        s2 = view.to_sensor(max(x1, x2), max(y1, y2))
        return (int(s1[0]), int(s1[1]), int(s2[0]), int(s2[1]))

    # ------------------------------ keys ---------------------------------- #
    def _handle_key(self, key: int) -> bool:
        """False ends the program (Esc) - unless the help card is up, which
        Esc closes first."""
        if key == 27:  # ESC
            if self.show_help:
                self.show_help = False
                return True
            return False
        if key in KEY_FACTORS:
            self._adjust_exposure(KEY_FACTORS[key])
        elif key in KEY_ACTIONS:
            self._act(KEY_ACTIONS[key])
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
    def _save_frame(self, jpg_path: str, frame=None):
        """8-bit JPEG of the frame (as displayed) plus the raw frame as .npy."""
        frame = self.current_frame if frame is None else frame
        cv2.imwrite(jpg_path, to_8bit(frame))
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
        frame = self.current_frame

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
                self._save_frame(fp, frame)

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
