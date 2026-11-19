"""Procedural line icons for the toolbar, the status bar and the Tk windows.

Drawn with OpenCV primitives on a 24-unit grid (stroke 2, round caps - the
style of Lucide/Feather), 4x supersampled and area-downscaled, so they are
crisp at any size and need neither Pillow nor any asset file: the
PyInstaller one-file exe ships nothing extra.  `mask(name, size)` is the
coverage (uint8), coloured at blend time; `png(...)` encodes one for Tk.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .gfx import SpriteCache

SS = 4  # supersampling factor
_SHIFT = 4  # cv2 fixed-point bits
_masks = SpriteCache(512)


class _Pen:
    def __init__(self, size: int, stroke: float = 2.0):
        self.size = size
        self.n = size * SS
        self.k = self.n / 24.0
        self.img = np.zeros((self.n, self.n), np.uint8)
        self.stroke = stroke

    def _p(self, x, y):
        f = self.k * (1 << _SHIFT)
        return int(round(x * f)), int(round(y * f))

    def _r(self, r):
        return int(round(r * self.k * (1 << _SHIFT)))

    def _t(self, w):
        return max(1, int(round((self.stroke if w is None else w) * self.k)))

    def line(self, x0, y0, x1, y1, w=None, color=255):
        cv2.line(self.img, self._p(x0, y0), self._p(x1, y1), color, self._t(w), cv2.LINE_AA,
                 _SHIFT)

    def poly(self, pts, closed=False, w=None, color=255):
        arr = np.array([self._p(x, y) for x, y in pts], np.int32)
        cv2.polylines(self.img, [arr], closed, color, self._t(w), cv2.LINE_AA, _SHIFT)

    def fill(self, pts, color=255):
        arr = np.array([self._p(x, y) for x, y in pts], np.int32)
        cv2.fillPoly(self.img, [arr], color, cv2.LINE_AA, _SHIFT)

    def circle(self, cx, cy, r, w=None, fill=False, color=255):
        cv2.circle(self.img, self._p(cx, cy), self._r(r), color,
                   -1 if fill else self._t(w), cv2.LINE_AA, _SHIFT)

    def arc(self, cx, cy, rx, ry, start, end, rot=0.0, w=None, color=255):
        cv2.ellipse(self.img, self._p(cx, cy), (self._r(rx), self._r(ry)), rot, start, end,
                    color, self._t(w), cv2.LINE_AA, _SHIFT)

    def rrect(self, x0, y0, x1, y1, r, w=None, color=255):
        self.line(x0 + r, y0, x1 - r, y0, w, color)
        self.line(x0 + r, y1, x1 - r, y1, w, color)
        self.line(x0, y0 + r, x0, y1 - r, w, color)
        self.line(x1, y0 + r, x1, y1 - r, w, color)
        for cx, cy, a in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270),
                          (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
            self.arc(cx, cy, r, r, a, a + 90, w=w, color=color)

    def result(self) -> np.ndarray:
        return cv2.resize(self.img, (self.size, self.size), interpolation=cv2.INTER_AREA)


def _polar(cx, cy, r, deg):
    a = math.radians(deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


# ------------------------------- the icons --------------------------------- #
def _fit(p):  # a fitted spot in a reticle
    p.arc(12, 12, 6.8, 4.6, 0, 360, rot=-35)
    p.circle(12, 12, 1.6, fill=True)
    for a in (0, 90, 180, 270):
        p.line(*_polar(12, 12, 8.8, a), *_polar(12, 12, 11.5, a))


def _stats(p):  # bar chart
    for x, top in ((5, 13), (10, 6), (15, 10), (20, 15)):
        p.line(x, 20, x, top, w=2.6)


def _grid(p):  # rows and columns of spots
    for x in (6, 12, 18):
        for y in (6, 12, 18):
            p.circle(x, y, 1.9, fill=True)
    p.line(2.5, 12, 21.5, 12, w=1.3)


def _sun(p, cx, cy, r, ray0, ray1):
    p.circle(cx, cy, r)
    for i in range(8):
        p.line(*_polar(cx, cy, ray0, i * 45), *_polar(cx, cy, ray1, i * 45))


def _exposure(p):
    _sun(p, 12, 12, 4, 7, 9.5)


def _auto(p):  # "A" in a ring: automatic
    p.circle(12, 12, 9.5)
    p.poly([(8.2, 16.8), (12, 6.8), (15.8, 16.8)])
    p.line(9.6, 13.4, 14.4, 13.4)


def _exp_down(p):
    _sun(p, 9.5, 9.5, 3.2, 5.6, 7.6)
    p.line(14.5, 19.5, 21.5, 19.5)


def _exp_up(p):
    _sun(p, 9.5, 9.5, 3.2, 5.6, 7.6)
    p.line(14.5, 19.5, 21.5, 19.5)
    p.line(18, 16, 18, 23)


def _clear(p):  # dashed rectangle, crossed out
    x0, y0, x1, y1 = 3, 4, 18, 16
    for a, b in ((x0, x1), ):
        for t in np.arange(a, b, 4.0):
            p.line(t, y0, min(t + 2, b), y0, w=1.8)
            p.line(t, y1, min(t + 2, b), y1, w=1.8)
    for t in np.arange(y0, y1, 4.0):
        p.line(x0, t, x0, min(t + 2, y1), w=1.8)
        p.line(x1, t, x1, min(t + 2, y1), w=1.8)
    p.circle(18, 18, 5.5, fill=True, color=0)
    p.line(15, 15, 21, 21)
    p.line(21, 15, 15, 21)


def _camera_body(p):
    p.rrect(2.5, 7, 21.5, 19.5, 2.5)
    p.poly([(7.8, 7), (9.6, 4.3), (14.4, 4.3), (16.2, 7)])


def _camera(p):
    _camera_body(p)
    p.circle(12, 13.2, 3.6)


def _next(p):  # camera with an arrow: the next camera
    _camera_body(p)
    p.line(7.8, 13.2, 16, 13.2)
    p.poly([(13, 10.2), (16, 13.2), (13, 16.2)])


def _camera_off(p):
    _camera(p)
    p.line(2.5, 2.5, 21.5, 21.5, w=5, color=0)
    p.line(3, 3, 21, 21)


def _web(p):  # globe
    p.circle(12, 12, 9.5)
    p.arc(12, 12, 4.2, 9.5, 0, 360)
    p.line(2.5, 12, 21.5, 12)


def _sync(p):  # square wave: the trigger line
    p.poly([(2, 17), (6, 17), (6, 7), (12, 7), (12, 17), (18, 17), (18, 7), (22, 7)])


def _save(p):  # floppy disk
    p.poly([(5, 3.5), (16.5, 3.5), (20.5, 7.5), (20.5, 19), (19, 20.5), (5, 20.5),
            (3.5, 19), (3.5, 5)], closed=True)
    p.poly([(7.5, 3.5), (7.5, 8.5), (14.5, 8.5), (14.5, 3.5)])
    p.poly([(7, 20.5), (7, 13.5), (17, 13.5), (17, 20.5)])


def _settings(p):  # gear
    pts = []
    for i in range(8):
        c = i * 45
        pts += [_polar(12, 12, 7.0, c - 15), _polar(12, 12, 9.8, c - 8),
                _polar(12, 12, 9.8, c + 8), _polar(12, 12, 7.0, c + 15)]
    p.poly(pts, closed=True, w=1.8)
    p.circle(12, 12, 3.2, w=1.8)


def _info(p):
    p.circle(12, 12, 9.5)
    p.circle(12, 7.6, 1.4, fill=True)
    p.line(12, 11, 12, 16.8)


def _help(p):
    p.circle(12, 12, 9.5)
    p.arc(12, 9.4, 3.1, 3.1, 180, 400)
    x, y = _polar(12, 9.4, 3.1, 40)
    p.poly([(x, y), (12, 12.6), (12, 13.8)])
    p.circle(12, 17.2, 1.35, fill=True)


def _warning(p):
    p.poly([(12, 3), (21.5, 20), (2.5, 20)], closed=True)
    p.line(12, 9.5, 12, 14)
    p.circle(12, 17, 1.3, fill=True)


def _error(p):
    p.circle(12, 12, 9.5)
    p.line(8.5, 8.5, 15.5, 15.5)
    p.line(15.5, 8.5, 8.5, 15.5)


def _ok(p):
    p.circle(12, 12, 9.5)
    p.poly([(7.5, 12.5), (10.8, 15.6), (16.5, 9)])


def _check(p):
    p.poly([(4.5, 12.5), (9.5, 17.5), (19.5, 6.5)])


def _close(p):
    p.line(6, 6, 18, 18)
    p.line(18, 6, 6, 18)


def _arrowhead(p, cx, cy, r, deg, clockwise=True, size=4.0):
    """Head for an arc of radius r ending at angle `deg` (cv2 angles run
    clockwise on screen)."""
    tip = _polar(cx, cy, r, deg)
    a = math.radians(deg)
    dx, dy = (-math.sin(a), math.cos(a)) if clockwise else (math.sin(a), -math.cos(a))
    back = math.atan2(-dy, -dx)
    wings = [(tip[0] + size * math.cos(back + s), tip[1] + size * math.sin(back + s))
             for s in (0.7, -0.7)]
    p.poly([wings[0], tip, wings[1]])


def _refresh(p):  # rescan
    p.arc(12, 12, 8, 8, -60, 230)
    _arrowhead(p, 12, 12, 8, 230)


def _reset(p):  # back to defaults (counter-clockwise arrow)
    p.arc(12, 12, 8, 8, -150, 140)
    _arrowhead(p, 12, 12, 8, -150, clockwise=False)


def _file(p):  # reload from file
    p.poly([(6, 2.5), (14, 2.5), (19, 7.5), (19, 21.5), (6, 21.5)], closed=True)
    p.poly([(14, 2.5), (14, 7.5), (19, 7.5)])
    p.line(9, 13, 16, 13, w=1.6)
    p.line(9, 17, 16, 17, w=1.6)


def _app(p):  # beam spot in a reticle
    p.circle(12, 12, 9.5)
    p.circle(12, 12, 4.8)
    p.circle(12, 12, 1.8, fill=True)
    for a in (0, 90, 180, 270):
        p.line(*_polar(12, 12, 9.5, a), *_polar(12, 12, 12, a))


def _ruler(p):  # rectangle size
    p.rrect(3, 6, 21, 18, 1.5)
    for x in (7, 11, 15):
        p.line(x, 6, x, 10 if x == 11 else 9, w=1.6)


DRAW = {
    "fit": _fit, "stats": _stats, "grid": _grid, "exposure": _exposure, "auto": _auto,
    "exp_down": _exp_down, "exp_up": _exp_up, "clear": _clear, "cameras": _camera,
    "camera": _camera, "next": _next, "camera_off": _camera_off, "web": _web, "sync": _sync,
    "save": _save, "settings": _settings, "info": _info, "help": _help, "warning": _warning,
    "error": _error, "ok": _ok, "check": _check, "close": _close, "refresh": _refresh,
    "reset": _reset, "file": _file, "app": _app, "ruler": _ruler,
}
NAMES = tuple(DRAW)


def mask(name: str, size: int) -> np.ndarray:
    """Coverage of icon `name` at `size` x `size` px (uint8), cached."""

    def render():
        pen = _Pen(size)
        DRAW[name](pen)
        return pen.result()

    return _masks.get((name, size), render)


def png(name: str, size: int, rgb_hex: str) -> bytes:
    """The icon in one colour on transparency, PNG-encoded (for Tk)."""
    h = rgb_hex.lstrip("#")
    m = mask(name, size)
    bgra = np.empty((size, size, 4), np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2] = int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)
    bgra[..., 3] = m
    ok, buf = cv2.imencode(".png", bgra)
    return buf.tobytes()


def dot_png(size: int, rgb_hex: str) -> bytes:
    """A filled status dot (Tk lists)."""
    pen = _Pen(size)
    pen.circle(12, 12, 5.5, fill=True)
    h = rgb_hex.lstrip("#")
    bgra = np.empty((size, size, 4), np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2] = int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)
    bgra[..., 3] = pen.result()
    return cv2.imencode(".png", bgra)[1].tobytes()
