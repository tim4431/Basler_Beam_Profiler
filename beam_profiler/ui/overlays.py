"""Everything drawn over the live frame: the status bar, the info and
statistics cards, the icon toolbar with its tooltips, spot ellipses,
row/column lines, the rectangles, and the camera banner, pill and toasts.

All drawing happens in display space (see `roi.ViewTransform`), so text and
lines keep their size at every zoom of the picture.  Their size follows the
UI scale instead: every length goes through a `Metrics` (`m.px(40)`,
`m.font(13)`), never a raw pixel constant, and every cached rendering has the
scale in its key - so a new scale re-renders text and icons at the new size
rather than resampling them.  Text, icons and composite widgets go through
`gfx`'s sprite caches - a frame blends finished sprites and renders only what
changed - and the per-spot graphics are batched into a handful of cv2 calls,
never one call per spot.  Colours are BGR (imshow).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..processing.spots import SpotArray
from ..roi import ViewTransform
from . import gfx
from .theme import LEVELS, OV

MAX_LABELS = 200  # spot labels per frame, a bound on the worst case
READOUT_PERIOD = 0.15  # s between refreshes of changing numbers (cards, labels)
MIN_MARKER_PITCH = 16  # design px between spots below which grid markers are left out


class Metrics:
    """Every overlay size at one UI scale.  The numbers written here are the
    design (100 %) sizes; `px` and `font` scale anything else."""

    _cache: dict = {}

    def __init__(self, scale: float = 1.0):
        self.scale = s = float(scale)
        px = self.px
        self.top_h = px(40)  # status bar
        self.edge = px(12)  # margin to the window edge
        self.gap = px(8)  # between stacked cards
        self.card_w = px(330)
        self.card_pad = px(12)
        self.row_h = px(21)
        self.title_h = px(27)
        self.label_w = px(74)  # label column inside a card
        self.gauge_x = px(74)  # gauge start, after the value column
        self.gauge_h = max(2, px(6))
        self.radius = px(10)  # card corners
        self.stroke = max(1, round(2 * s))  # the "2 px" line
        self.hair = max(1, round(s))  # the "1 px" line

    def px(self, v: float) -> int:
        return int(round(v * self.scale))

    def font(self, size: float, bold: bool = False) -> gfx.Font:
        return gfx.font(max(7, int(round(size * self.scale))), bold)

    @classmethod
    def at(cls, scale: float) -> Metrics:
        scale = round(float(scale), 3)
        m = cls._cache.get(scale)
        if m is None:
            m = cls._cache[scale] = cls(scale)
        return m


def _m(m: Metrics | None) -> Metrics:
    return Metrics.at(1.0) if m is None else m


def _fit_scale(m: Metrics, size: tuple, room: tuple) -> Metrics:
    """`m`, shrunk (in 5 % steps) so a widget of `size` at m's scale fits in
    `room` - a big zoom on a small window must not push things off screen."""
    ratio = min(room[0] / max(size[0], 1), room[1] / max(size[1], 1))
    if ratio >= 1:
        return m
    return Metrics.at(max(0.25, math.floor(m.scale * ratio * 20) / 20))


# --------------------------------------------------------------------------- #
#  status bar
# --------------------------------------------------------------------------- #
@dataclass
class Hud:
    """What the status bar and the info card show, gathered by the viewer."""

    status: str  # "live" | "offline": the camera on screen
    camera: str
    source: str
    others: int  # further cameras connected
    scale_known: bool
    fps: float
    exposure: float
    auto: bool
    fitting: bool
    backend: str
    web: bool
    web_port: int
    sync: bool
    update: str | None  # chip text while an update waits for a restart
    roi: tuple
    mouse: tuple
    rect: tuple | None
    rect_um: tuple | None
    fit_rect: tuple | None
    fit: str


_chips = gfx.cache(256)
_dots = gfx.cache(16)


def _chip(m: Metrics, text: str, icon: str | None, fg, bg=None, bg_alpha=0.0,
          badge: str | None = None):
    """Pill with an optional icon and badge, cached by its content."""

    def render():
        f, fb = m.font(13), m.font(11, True)
        h, pad, isz, gap = m.px(26), m.px(9), m.px(16), m.px(6)
        w = pad + (isz + gap if icon else 0) + f.width(text) + pad
        if badge:
            w += fb.width(badge) + m.px(14)
        c = gfx.Canvas(w, h)
        if bg is not None:
            c.rrect(0, 0, w, h, h // 2, bg, bg_alpha)
        x = pad
        if icon:
            c.icon(icon, x, (h - isz) // 2, isz, fg)
            x += isz + gap
        x += c.text(text, x, h // 2, f, fg, anchor="lm")
        if badge:
            bw = fb.width(badge) + m.px(10)
            c.rrect(x + m.px(5), m.px(5), x + m.px(5) + bw, h - m.px(5), m.px(4), OV.ACCENT, 1.0)
            c.text(badge, x + m.px(10), h // 2, fb, (255, 255, 255), anchor="lm")
        return c.sprite()

    return _chips.get((m.scale, text, icon, fg, bg, bg_alpha, badge), render)


def _status_dot(m: Metrics, color) -> gfx.Sprite:
    def render():
        d = m.px(20)
        c = gfx.Canvas(d, d)
        c.rrect(0, 0, d, d, d / 2, color, 0.22)
        c.rrect(d / 4, d / 4, d - d // 4, d - d // 4, d / 4, color, 1.0)
        return c.sprite()

    return _dots.get((m.scale, color), render)


def draw_top_bar(img, hud: Hud, m: Metrics | None = None):
    """Camera and connection state on the left; exposure, frame rate and the
    switched-on features (fit backend, HTTP API, sync line) on the right."""
    m = _m(m)
    W = img.shape[1]
    gfx.shade(img, 0, 0, W, m.top_h, OV.PANEL, 0.84)
    if m.top_h < img.shape[0]:
        img[m.top_h, :] = OV.HAIRLINE
    cy = m.top_h // 2

    x = W - m.edge
    if hud.status == "offline":
        chips = [_chip(m, "Disconnected", "camera_off", OV.BAD_TEXT, OV.BAD, 0.16)]
    else:
        x -= gfx.draw_text(img, f"{hud.fps:.0f} fps", x, cy, m.font(12), OV.SUBTEXT,
                           anchor="rm") + m.px(10)
        chips = []
    chips.append(_chip(m, f"{hud.exposure:,.0f} µs", "exposure", OV.TEXT, OV.TEXT, 0.08,
                       badge="AUTO" if hud.auto else None))
    if hud.fitting:
        chips.append(_chip(m, f"Fit · {hud.backend.upper()}", "fit", OV.GOOD_TEXT, OV.GOOD, 0.14))
    if hud.web:
        chips.append(_chip(m, f"API :{hud.web_port}", "web", OV.ACCENT_HI, OV.ACCENT, 0.18))
    if hud.sync:
        chips.append(_chip(m, "Sync", "sync", OV.ACCENT_HI, OV.ACCENT, 0.18))
    for chip in chips:
        if x - chip.w < W // 3:  # no room left on a narrow window
            break
        x -= chip.w
        chip.draw(img, x, cy - chip.h // 2)
        x -= m.gap
    right_limit = x - m.gap

    x = m.edge
    dot_color = OV.BAD if hud.status == "offline" else OV.GOOD
    dot = _status_dot(m, dot_color)
    dot.draw(img, x - m.px(2), cy - dot.h // 2)
    x += m.px(24)
    extras = []
    if hud.others:
        extras.append(_chip(m, f"+{hud.others} more · K", "camera", OV.SUBTEXT, OV.TEXT, 0.08))
    if not hud.scale_known:
        extras.append(_chip(m, "pixel size unset: µm = px", "warning", OV.WARN_TEXT, OV.WARN,
                            0.16))
    if hud.update:
        extras.append(_chip(m, hud.update, "refresh", OV.ACCENT_HI, OV.ACCENT, 0.18))
    fb, fs = m.font(14, True), m.font(12)
    room = right_limit - x - sum(c.w + m.gap for c in extras) - m.px(10) - fs.width(hud.source)
    label = gfx.ellipsize(hud.camera, fb, max(room, m.px(60)))
    x += gfx.draw_text(img, label, x, cy, fb, OV.TEXT, anchor="lm") + m.px(10)
    if x + fs.width(hud.source) < right_limit:
        x += gfx.draw_text(img, hud.source, x, cy, fs, OV.SUBTEXT, anchor="lm") + m.px(12)
    for chip in extras:
        if x + chip.w > right_limit:
            break
        chip.draw(img, x, cy - chip.h // 2)
        x += chip.w + m.gap


# --------------------------------------------------------------------------- #
#  cards
# --------------------------------------------------------------------------- #
@dataclass
class Bar:
    """A card row with a value and a gauge.  `frac` is 0..1, or -1..1 for a
    `centered` gauge (green to the right, red to the left)."""

    label: str
    text: str
    frac: float
    color: tuple = OV.ACCENT_HI
    centered: bool = False


def card_height(m: Metrics, rows: int, title: bool) -> int:
    return 2 * m.card_pad + m.row_h * rows - m.px(4) + (m.title_h if title else 0)


def _card(img, m: Metrics, x, y, rows, title=None, icon=None, badge=None) -> int:
    """Translucent card of label/value rows (tuples) and `Bar`s; returns the
    y below it."""
    w = m.card_w
    h = card_height(m, len(rows), bool(title))
    gfx.panel(img, x, y, x + w, y + h, OV.PANEL, OV.PANEL_ALPHA, radius=m.radius)
    fl, fv = m.font(12), m.font(13)
    yy = y + m.card_pad
    mid = m.px(7)  # cap middle of a row, from its top
    if title:
        tx = x + m.card_pad
        tmid = m.px(8)
        if icon:
            isz = m.px(16)
            gfx.icon_sprite(icon, isz, OV.ACCENT_HI).draw(img, tx, yy + tmid - isz // 2)
            tx += isz + m.px(7)
        gfx.draw_text(img, title, tx, yy + tmid, m.font(13, True), OV.TEXT, anchor="lm")
        if badge:
            gfx.draw_text(img, badge, x + w - m.card_pad, yy + tmid, fl, OV.SUBTEXT, anchor="rm")
        yy += m.title_h
    vx = x + m.card_pad + m.label_w
    vmax = x + w - m.card_pad - vx
    for row in rows:
        cy = yy + mid
        if isinstance(row, Bar):
            gfx.draw_text(img, row.label, x + m.card_pad, cy, fl, OV.SUBTEXT, anchor="lm")
            gfx.draw_text(img, row.text, vx, cy, fv, OV.TEXT, anchor="lm")
            _gauge(img, m, vx + m.gauge_x, x + w - m.card_pad, cy, row)
        else:
            label, value = row[0], row[1]
            color = row[2] if len(row) > 2 and row[2] is not None else OV.TEXT
            if label:
                gfx.draw_text(img, label, x + m.card_pad, cy, fl, OV.SUBTEXT, anchor="lm")
            gfx.draw_text(img, gfx.ellipsize(value, fv, vmax), vx, cy, fv, color, anchor="lm")
        yy += m.row_h
    return y + h


def _gauge(img, m: Metrics, x0, x1, cy, bar: Bar):
    half = m.gauge_h // 2
    y0, y1 = cy - half, cy + half
    cv2.rectangle(img, (x0, y0), (x1, y1), OV.TRACK, -1)
    if bar.centered:
        mid = (x0 + x1) // 2
        f = float(np.clip(bar.frac, -1, 1))
        end = int(round(mid + f * (x1 - mid)))
        if end != mid:
            cv2.rectangle(img, (min(mid, end), y0), (max(mid, end), y1),
                          OV.GOOD if f > 0 else OV.BAD, -1)
        cv2.line(img, (mid, y0 - half), (mid, y1 + half), OV.SUBTEXT, m.hair)
    else:
        f = float(np.clip(bar.frac, 0, 1))
        end = int(round(x0 + f * (x1 - x0)))
        if end > x0:
            cv2.rectangle(img, (x0, y0), (end, y1), bar.color, -1)


def _pt(p) -> str:
    return f"{p[0]}, {p[1]}"


def info_rows(hud: Hud) -> list:
    w, h, ox, oy = hud.roi
    rows = [("ROI", f"{w} × {h}   at  {ox}, {oy}"), ("Cursor", _pt(hud.mouse))]
    if hud.rect:
        rows.append(("Rect", f"({_pt(hud.rect[:2])}) – ({_pt(hud.rect[2:])})"))
        rows.append(("", f"{hud.rect_um[0]:.1f} × {hud.rect_um[1]:.1f} µm", OV.ACCENT_HI))
    else:
        rows.append(("Rect", "drag on the image", OV.MUTED))
    if hud.fit_rect:
        rows.append(("Fit area", f"({_pt(hud.fit_rect[:2])}) – ({_pt(hud.fit_rect[2:])})",
                     OV.GOOD_TEXT))
    else:
        rows.append(("Fit area", "whole frame  (Ctrl-drag to limit)", OV.MUTED))
    rows.append(("Fit", hud.fit))
    return rows


@dataclass
class Card:
    rows: list
    title: str | None = None
    icon: str | None = None
    badge: str | None = None

    def height(self, m: Metrics) -> int:
        return card_height(m, len(self.rows), bool(self.title))

    def draw(self, img, m: Metrics, x: int, y: int) -> int:
        return _card(img, m, x, y, self.rows, self.title, self.icon, self.badge)


def fit_cards(m: Metrics, width: int, top: int, bottom: int, left: list, right: list):
    """The largest scale (<= m's, in 5 % steps) at which the left column (from
    the top-left corner) and the right column (from the top-right) fit between
    `top` and `bottom`: side by side, or else the right one under the left.
    Returns (metrics, stacked).  `top` and `bottom` are at m's scale."""
    def column(mm, cards):
        return sum(c.height(mm) for c in cards) + mm.gap * max(len(cards) - 1, 0)

    scale, floor = m.scale, max(0.4, m.scale * 0.4)
    while True:
        mm = Metrics.at(scale)
        room = bottom - top
        hl, hr = column(mm, left), column(mm, right)
        one_column = mm.card_w + 2 * mm.edge <= width
        if left and right:
            if 2 * mm.card_w + 2 * mm.edge + mm.gap <= width and max(hl, hr) <= room:
                return mm, False
            if one_column and hl + mm.gap + hr <= room:
                return mm, True
        elif one_column and max(hl, hr) <= room:
            return mm, False
        if scale <= floor:
            return mm, bool(left and right and 2 * mm.card_w + 2 * mm.edge + mm.gap > width)
        scale = round(max(floor, scale - 0.05), 2)


def draw_info(img, hud: Hud, x: int, y: int, m: Metrics | None = None) -> int:
    """ROI, cursor, rectangles and fit settings; returns the y below."""
    return _card(img, _m(m), x, y, info_rows(hud))


def draw_rect_stats(img, stats: dict | None, x: int, y: int, m: Metrics | None = None) -> int:
    """Live pixel statistics of the white rectangle; returns the y below."""
    if stats is None:
        return y
    return rect_card(stats).draw(img, _m(m), x, y)


def rect_card(stats: dict) -> Card:
    rows = [
        ("Sum", f"{stats['sum']:,}"),
        ("Mean", f"{stats['mean']:.1f}    std {stats['std']:.1f}"),
        ("Min / max", f"{stats['min']} / {stats['max']}"),
        ("Median", f"{stats['median']:.1f}"),
        ("Size", f"{stats['width_pixels']} × {stats['height_pixels']} px"),
        ("", f"{stats['width_um']:.1f} × {stats['height_um']:.1f} µm", OV.ACCENT_HI),
        ("Pixels", f"{stats['total_pixels']:,}"),
    ]
    return Card(rows, "Rectangle", "ruler")


def draw_spot_stats(
    img,
    x: int,
    y: int,
    n: int,
    std_dx: float,
    std_dy: float,
    mean_dx_um: float,
    mean_dy_um: float,
    curv_x: float,
    curv_y: float,
    sigma_std: float,
    show_curvature: bool,
    m: Metrics | None = None,
) -> int:
    """Spot spacing statistics: mean spacings, spacing scatter, curvature and
    width scatter (pixels unless marked um); returns the y below."""
    return spot_card(n, std_dx, std_dy, mean_dx_um, mean_dy_um, curv_x, curv_y, sigma_std,
                     show_curvature).draw(img, _m(m), x, y)


def spot_card(n, std_dx, std_dy, mean_dx_um, mean_dy_um, curv_x, curv_y, sigma_std,
              show_curvature) -> Card:
    rows = [
        ("Dx", f"{mean_dx_um:.1f} µm"),
        ("Dy", f"{mean_dy_um:.1f} µm"),
        Bar("std(x)", f"{std_dx:.2f} px", std_dx * 0.05),
        Bar("std(y)", f"{std_dy:.2f} px", std_dy * 0.05),
    ]
    if show_curvature:
        rows += [
            Bar("Curv X", f"{curv_x:+.2f}", curv_x * 0.05, centered=True),
            Bar("Curv Y", f"{curv_y:+.2f}", curv_y * 0.05, centered=True),
            Bar("std(s)", f"{sigma_std:.3f} px", sigma_std * 0.2, OV.SPOT),
        ]
    return Card(rows, "Spot statistics", "stats", f"{n} spot{'s' if n != 1 else ''}")


def draw_grid_stats(img, x: int, y: int, rows, columns, grid_stats: dict,
                    pixel_to_um: float, m: Metrics | None = None) -> int:
    """Row/column counts, spacings and straightness (beam-array mode)."""
    if not grid_stats:
        return y
    return grid_card(rows, columns, grid_stats, pixel_to_um).draw(img, _m(m), x, y)


def grid_card(rows, columns, grid_stats: dict, pixel_to_um: float) -> Card:
    r, c = grid_stats["rows"], grid_stats["columns"]
    lines = [("Rows", f"{len(rows)}" + (f"     Dx {r['avg_mean_dx'] * pixel_to_um:.1f} µm"
                                        if "avg_mean_dx" in r else ""))]
    if "avg_std_x" in r:
        lines.append(("", f"std(x) {r['avg_std_x'] * pixel_to_um:.1f}  ·  "
                          f"std(y) {r['avg_std_y'] * pixel_to_um:.1f} µm", OV.SUBTEXT))
    lines.append(("Columns", f"{len(columns)}" + (f"     Dy {c['avg_mean_dy'] * pixel_to_um:.1f} µm"
                                                  if "avg_mean_dy" in c else "")))
    if "avg_std_y" in c:
        lines.append(("", f"std(y) {c['avg_std_y'] * pixel_to_um:.1f}  ·  "
                          f"std(x) {c['avg_std_x'] * pixel_to_um:.1f} µm", OV.SUBTEXT))
    return Card(lines, "Rows & columns", "grid")


# --------------------------------------------------------------------------- #
#  toolbar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tool:
    name: str
    icon: str
    label: str
    keys: tuple = ()
    hint: str = ""


class Toolbar:
    """Icon dock centred along the bottom edge, in groups split by `None`.

    Layout is fixed at construction for one window size and scale, in display
    coordinates (the viewer makes a new one when either changes); the viewer
    asks `hit` on mouse events and `draw` every frame.  The dock is one cached
    sprite per (active, hovered, disabled) combination, so drawing it costs a
    single blend; a hovered button gets a tooltip with its shortcut.  If the
    dock would not fit the window's width at the requested scale it shrinks."""

    # design (100 %) sizes
    BUTTON = 40
    ICON = 22
    GAP = 4
    SEP = 17
    PAD = 6
    MARGIN = 12
    HEIGHT = BUTTON + 2 * PAD

    def __init__(self, items: list, window, m: Metrics | None = None):
        self.items = [t for t in items if t is not None]
        self.tools = {t.name: t for t in self.items}
        win_w, win_h = (window, window) if np.isscalar(window) else window
        self.window = (int(win_w), int(win_h))
        m = _m(m)
        n = len(self.items)
        groups = sum(1 for t in items if t is None)
        design_w = 2 * self.PAD + n * self.BUTTON + (n - 1 - groups) * self.GAP + groups * self.SEP
        m = _fit_scale(m, (m.px(design_w), 1), (win_w - 2 * m.edge, 2))
        self.m = m
        self.button, self.icon = m.px(self.BUTTON), m.px(self.ICON)
        self.gap, self.sep, self.pad = m.px(self.GAP), m.px(self.SEP), m.px(self.PAD)
        self.margin = m.px(self.MARGIN)
        self.height = self.button + 2 * self.pad

        widths = []
        for i, t in enumerate(items):
            if t is None:
                widths.append(self.sep - self.gap)
            else:
                widths.append(self.button + (self.gap if i else 0))
        total = 2 * self.pad + sum(widths)
        x0 = (self.window[0] - total) // 2
        y1 = self.window[1] - self.margin
        y0 = y1 - self.height
        self.box = (x0, y0, x0 + total, y1)
        self.rects: dict[str, tuple] = {}
        self.separators: list[int] = []
        x = x0 + self.pad
        for i, t in enumerate(items):
            if t is None:
                self.separators.append(x + (self.sep - self.gap) // 2 - self.gap // 2)
                x += self.sep - self.gap
                continue
            if i:
                x += self.gap
            self.rects[t.name] = (x, y0 + self.pad, x + self.button, y0 + self.pad + self.button)
            x += self.button
        self._docks = gfx.cache(64)
        self._tips = gfx.cache(32)

    def hit(self, x: int, y: int) -> str | None:
        for name, (x0, y0, x1, y1) in self.rects.items():
            if x0 <= x < x1 and y0 <= y < y1:
                return name
        return None

    def contains(self, x: int, y: int) -> bool:
        """Anywhere on the dock, gaps included (mouse input stops here)."""
        x0, y0, x1, y1 = self.box
        return x0 <= x < x1 and y0 <= y < y1

    def draw(self, img, states: dict, hover: str | None = None, disabled=(),
             tooltip: bool = True):
        active = tuple(sorted(n for n, on in states.items() if on and n in self.tools))
        off = tuple(sorted(n for n in disabled if n in self.tools))
        key = (active, hover, off, gfx.pillow_active())
        self._docks.get(key, lambda: self._render(set(active), hover, set(off))).draw(
            img, self.box[0], self.box[1])
        if hover in self.tools and tooltip:
            self._draw_tip(img, hover)

    def _render(self, active, hover, disabled) -> gfx.Sprite:
        m = self.m
        bx0, by0, bx1, by1 = self.box
        c = gfx.Canvas(bx1 - bx0, by1 - by0)
        r_dock, r_btn = m.px(14), m.px(10)
        c.rrect(0, 0, c.w, c.h, r_dock, OV.PANEL, 0.86, border=(255, 255, 255),
                border_opacity=0.10)
        for sx in self.separators:
            c.paint(np.ones((self.button - m.px(16), m.hair), np.float32), sx - bx0,
                    self.pad + m.px(8), (255, 255, 255), 0.14)
        off = (self.button - self.icon) // 2
        for t in self.items:
            x0, y0, x1, y1 = (v - o for v, o in zip(self.rects[t.name], (bx0, by0, bx0, by0)))
            if t.name in active:
                c.rrect(x0, y0, x1, y1, r_btn, OV.ACCENT, 0.95)
                color, opacity = (255, 255, 255), 1.0
            elif t.name in disabled:
                color, opacity = OV.MUTED, 0.6
            elif t.name == hover:
                c.rrect(x0, y0, x1, y1, r_btn, (255, 255, 255), 0.13)
                color, opacity = (255, 255, 255), 1.0
            else:
                color, opacity = OV.ICON, 1.0
            if t.name == hover and t.name in active:
                c.rrect(x0, y0, x1, y1, r_btn, (255, 255, 255), 0.12)
            c.icon(t.icon, x0 + off, y0 + off, self.icon, color, opacity)
        return c.sprite()

    def _draw_tip(self, img, name: str):
        sprite = self._tips.get((name, gfx.pillow_active()),
                                lambda: _tooltip(self.m, self.tools[name]))
        x0, _y0, x1, _y1 = self.rects[name]
        edge = self.m.edge
        x = int(np.clip((x0 + x1) // 2 - sprite.w // 2, edge,
                        max(edge, self.window[0] - edge - sprite.w)))
        sprite.draw(img, x, self.box[1] - self.m.gap - sprite.h)


def _keycaps(c: gfx.Canvas, m: Metrics, keys, x, cy, f) -> int:
    """Draw keys as little rounded caps from x; returns the new x."""
    half = m.px(10)
    for i, k in enumerate(keys):
        if i:
            x += c.text("/", x + m.px(4), cy, f, OV.MUTED, anchor="lm") + m.gap
        w = max(f.width(k) + m.px(12), m.px(22))
        c.rrect(x, cy - half, x + w, cy + half, m.px(5), (255, 255, 255), 0.08,
                border=(255, 255, 255), border_opacity=0.28)
        c.text(k, x + w // 2, cy, f, OV.TEXT, anchor="mm")
        x += w
    return x


def _keycaps_width(m: Metrics, keys, f) -> int:
    return sum(max(f.width(k) + m.px(12), m.px(22)) for k in keys) + sum(
        f.width("/") + m.px(12) for _ in keys[1:])


def _tooltip(m: Metrics, tool: Tool) -> gfx.Sprite:
    fb, fk, fh = m.font(13, True), m.font(12), m.font(12)
    pad, line = m.px(12), m.px(20)
    keys_w = m.px(14) + _keycaps_width(m, tool.keys, fk) if tool.keys else 0
    w = 2 * pad + max(fb.width(tool.label) + keys_w, fh.width(tool.hint) if tool.hint else 0)
    h = 2 * pad + line + (line if tool.hint else 0)
    c = gfx.Canvas(w, h)
    c.rrect(0, 0, w, h, m.px(8), OV.TOOLTIP, 0.95, border=(255, 255, 255), border_opacity=0.12)
    cy = pad + line // 2
    x = pad + c.text(tool.label, pad, cy, fb, OV.TEXT, anchor="lm")
    if tool.keys:
        _keycaps(c, m, tool.keys, x + m.px(14), cy, fk)
    if tool.hint:
        c.text(tool.hint, pad, cy + m.px(21), fh, OV.SUBTEXT, anchor="lm")
    return c.sprite()


# --------------------------------------------------------------------------- #
#  help card
# --------------------------------------------------------------------------- #
_help = gfx.cache(8)


def help_sections(port: int):
    return (
        (
            ("Analysis", (
                (("F",), "Fit spots"),
                (("G",), "Spot statistics"),
                (("H",), "Rows & columns (beam arrays)"),
                (("O",), "Settings window"),
            )),
            ("Exposure", (
                (("A",), "Auto exposure on / off"),
                (("Left", "Right"), "Exposure −10% / +10%"),
                (("Down", "Up"), "Exposure ÷10 / ×10"),
            )),
            ("Mouse", (
                (("Drag",), "White rectangle: pixel statistics"),
                (("Ctrl-drag",), "Green rectangle: fit only inside"),
                (("C", "V"), "Clear white / green rectangle"),
                (("Wheel",), "Zoom the camera ROI (Ctrl + wheel too)"),
                (("Shift", "Wheel"), "ROI aspect ratio"),
            )),
        ),
        (
            ("Cameras", (
                (("K",), "Cameras window"),
                (("T",), "Next camera"),
                (("Y",), "Sync output (Line3)"),
            )),
            ("Output", (
                (("S",), "Quick save (JPEG + raw .npy)"),
                (("D",), "Save as…"),
                (("W",), f"HTTP API on port {port} + rect stats"),
            )),
            ("View", (
                (("+", "−"), "Interface zoom in / out"),
                (("0",), "Interface zoom 100%"),
                (("I",), "Info panel"),
                (("U",), "Update: check now / restart into it"),
                (("?",), "This help"),
                (("Esc",), "Close the help, else quit"),
            )),
        ),
    )


def draw_help(img, port: int, m: Metrics | None = None):
    """Centred card listing every shortcut (toggled with ?), shrunk if the
    window is too small for it at this scale."""
    m = _m(m)
    H, W = img.shape[:2]
    sprite = _help.get((port, m.scale, gfx.pillow_active()), lambda: _render_help(m, port))
    room = (W - 2 * m.edge, H - m.top_h - 2 * m.edge)
    fit = _fit_scale(m, (sprite.w, sprite.h), room)
    if fit is not m:
        sprite = _help.get((port, fit.scale, gfx.pillow_active()), lambda: _render_help(fit, port))
    sprite.draw(img, (W - sprite.w) // 2,
                max(m.top_h + m.edge, (H - sprite.h) // 2 - m.px(30)))


def _render_help(m: Metrics, port: int) -> gfx.Sprite:
    ft, fh, fk, fv = m.font(19, True), m.font(12, True), m.font(12), m.font(13)
    sections = help_sections(port)
    key_w = m.px(118)
    col_w = key_w + max(fv.width(text) for col in sections for _, items in col
                        for _, text in items) + m.px(12)
    pad, gap, row_h, head_h, after = m.px(28), m.px(36), m.px(30), m.px(30), m.px(14)
    col_h = [sum(head_h + row_h * len(items) + after for _, items in col) for col in sections]
    w = 2 * pad + 2 * col_w + gap
    h = pad + m.px(46) + max(col_h) + pad - after
    c = gfx.Canvas(w, h)
    c.rrect(0, 0, w, h, m.px(14), OV.PANEL, 0.94, border=(255, 255, 255), border_opacity=0.12)
    isz = m.px(24)
    c.icon("help", pad, pad, isz, OV.ACCENT_HI)
    c.text("Keyboard & mouse", pad + isz + m.px(10), pad + isz // 2, ft, OV.TEXT, anchor="lm")
    c.text("Esc or ? to close", w - pad, pad + isz // 2, fk, OV.SUBTEXT, anchor="rm")
    for ci, col in enumerate(sections):
        x = pad + ci * (col_w + gap)
        y = pad + m.px(46)
        for title, items in col:
            c.text(title.upper(), x, y + head_h * 2 // 5, fh, OV.ACCENT_HI, anchor="lm")
            y += head_h
            for keys, text in items:
                _keycaps(c, m, keys, x, y + row_h // 3, fk)
                c.text(text, x + key_w, y + row_h // 3, fv, OV.TEXT, anchor="lm")
                y += row_h
            y += after
    return c.sprite()


# --------------------------------------------------------------------------- #
#  spots and grid
# --------------------------------------------------------------------------- #
_BASIS: dict = {}


def _ellipse_polys(cx, cy, a0, a1, angle_deg, k: int) -> np.ndarray:
    """(n, k, 2) int32 outlines in 1/8 px (cv2 shift=3): one (k, 3) @ (n, 3, 2)
    matrix product instead of a dozen (n, k) temporaries."""
    basis = _BASIS.get(k)
    if basis is None:
        t = np.linspace(0.0, 2 * np.pi, k, endpoint=False)
        basis = _BASIS[k] = np.stack([np.ones(k), np.cos(t), np.sin(t)], 1) * 8
    rad = np.radians(angle_deg)
    ca, sa = np.cos(rad), np.sin(rad)
    coef = np.empty((len(cx), 3, 2))
    coef[:, 0, 0], coef[:, 1, 0], coef[:, 2, 0] = cx, a0 * ca, -a1 * sa
    coef[:, 0, 1], coef[:, 1, 1], coef[:, 2, 1] = cy, a0 * sa, a1 * ca
    return np.rint(basis @ coef).astype(np.int32)


def _draw_ellipses(img, cx, cy, a0, a1, angle, color, m: Metrics):
    """All ellipses in a few batched polylines calls (antialiased unless
    there are thousands).  Line width follows the scale as concentric 1 px
    rings: 2 px at 100 % where a spot is big enough, more when zoomed."""
    line = cv2.LINE_AA if len(cx) <= 2000 else cv2.LINE_8
    r = np.maximum(a0, a1)
    k = np.where(r < 3, 8, np.where(r < 10, 16, np.where(r < 30, 32, 64)))
    for kk in np.unique(k):
        i = k == kk
        cv2.polylines(img, _ellipse_polys(cx[i], cy[i], a0[i], a1[i], angle[i], int(kk)),
                      True, color, 1, line, 3)
    big = r >= 6 * m.stroke
    if 0 < np.count_nonzero(big) <= 400:  # extra rings where there is room for them
        for ring in range(1, m.stroke):
            d = 0.9 * ring
            cv2.polylines(img, _ellipse_polys(cx[big], cy[big], a0[big] + d, a1[big] + d,
                                              angle[big], 64), True, color, 1, cv2.LINE_AA, 3)


class _LabelMemo:
    """Spot label texts, recomputed READOUT_PERIOD apart: keyed by where the
    spot is, so a label follows its spot every frame while its digits change
    only at a readable rate (and each change costs a text render)."""

    def __init__(self):
        self._at, self._texts = -1e9, {}

    def tick(self):
        now = time.monotonic()
        if now - self._at >= READOUT_PERIOD:
            self._at, self._texts = now, {}

    def get(self, key, make):
        texts = self._texts.get(key)
        if texts is None:
            texts = self._texts[key] = make()
        return texts


_labels = _LabelMemo()


def draw_spots(img, spots, view: ViewTransform, pixel_size: float,
               min_label_radius: int = 15, m: Metrics | None = None):
    """Fitted ellipses, and for spots big enough on screen their axes (red =
    s0, blue = s1) and a label: centre (ROI px) and widths s0 x s1 (um).
    Everything is in display space, so line widths and text keep their size
    at every zoom of the picture; labels (and the room a spot needs for one,
    `min_label_radius`) follow the UI scale."""
    m = _m(m)
    if not isinstance(spots, SpotArray):
        spots = SpotArray.from_dicts(spots)
    if not len(spots):
        return
    cx = spots.x * view.scale + view.pad_l
    cy = spots.y * view.scale + view.pad_t
    a0, a1 = spots.sigma_0 * view.scale, spots.sigma_1 * view.scale
    angle = spots.angle
    _draw_ellipses(img, cx, cy, a0, a1, angle, OV.SPOT, m)

    labelled = np.flatnonzero(a0 >= m.px(min_label_radius))[:MAX_LABELS]
    if not len(labelled):
        return
    rad = np.radians(angle[labelled])
    lx, ly, l0, l1 = cx[labelled], cy[labelled], a0[labelled], a1[labelled]
    centre = np.stack([lx, ly], -1)
    for length, phase, color in ((l0, 0.0, OV.AXIS_MAJOR), (l1, np.pi / 2, OV.AXIS_MINOR)):
        tip = np.stack([lx + length * np.cos(rad + phase), ly + length * np.sin(rad + phase)], -1)
        segs = np.rint(np.stack([centre, tip], 1) * 8).astype(np.int32)
        cv2.polylines(img, segs, False, color, m.stroke, cv2.LINE_AA, 3)

    f = m.font(12)
    um = pixel_size * 1e6
    top = ly + np.sqrt(l0 * l1) + m.px(8)
    step = m.px(15)
    _labels.tick()
    for j, i in enumerate(labelled):
        x, y = int(lx[j]), int(top[j])
        coord, width = _labels.get(
            (round(spots.x[i] / 4), round(spots.y[i] / 4), um),
            lambda i=i: (f"({spots.x[i]:.0f}, {spots.y[i]:.0f})",
                         f"{spots.sigma_0[i] * um:.1f} × {spots.sigma_1[i] * um:.1f} µm"))
        label = gfx.text_block(((coord, OV.SPOT_COORD), (width, OV.SPOT_LABEL)), f, OV.HALO, step)
        label.draw(img, x - label.tw // 2, y)


def _hue_color(i: int, saturation: int) -> tuple:
    hsv = np.uint8([[[(i * 30) % 180, saturation, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])


_ROW_COLORS = [_hue_color(i, 255) for i in range(6)]
_COL_COLORS = [_hue_color(i, 180) for i in range(6)]
_COL_PIX = [np.array(c, np.uint8) for c in _COL_COLORS]  # for slice assignment
_SQUARE = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], np.float64)


def _markers(img, x, y, r, color, square: bool, thick: int):
    if not len(x):
        return
    if square:
        pts = np.stack([x[:, None] + r * _SQUARE[:, 0], y[:, None] + r * _SQUARE[:, 1]], -1)
        polys = np.rint(pts * 8).astype(np.int32)
    else:
        n = len(x)
        polys = _ellipse_polys(x, y, np.full(n, r), np.full(n, r), np.zeros(n), 8)
    cv2.polylines(img, polys, True, color, thick, cv2.LINE_8, 3)


def _dashed_column(img, x, y0, y1, pix, dash=8, period=12, width=2):
    """`width` px wide vertical dashes from y0 to y1: the strip reshaped to
    (periods, period) - splitting one axis is always a view - and filled in
    one assignment."""
    x0 = max(x - width // 2, 0)
    strip = img[y0:y1, x0:x0 + width]
    n, rest = divmod(y1 - y0, period)
    if n:
        strip[:n * period].reshape(n, period, *strip.shape[1:])[:, :dash] = pix
    if rest:
        strip[n * period:n * period + min(rest, dash)] = pix


def _flatten(groups):
    """Rows (or columns) with 2+ spots as flat columns plus per-group
    index/length/start, so per-group means and extents are one reduceat."""
    keep = [i for i, g in enumerate(groups) if len(g) >= 2]
    if not keep:
        return None
    lens = np.array([len(groups[i]) for i in keep])
    xs = np.concatenate([groups[i].x for i in keep])
    ys = np.concatenate([groups[i].y for i in keep])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return np.array(keep), lens, xs, ys, starts


def draw_row_col(img, view: ViewTransform, rows, columns, m: Metrics | None = None):
    """Hue-coded rows (solid lines, circles) and columns (dashed lines,
    squares).  Rows and columns are axis-aligned, so their lines are array
    slices; the markers are one batched polylines call per colour."""
    m = _m(m)
    H, W = img.shape[:2]
    s, pl, pt = view.scale, view.pad_l, view.pad_t
    R, C = _flatten(rows), _flatten(columns)
    n = (len(R[2]) if R else 0) + (len(C[2]) if C else 0)
    if not n:
        return
    pitch = float(np.median(np.diff(R[2][:R[1][0]]))) * s if R else 20.0
    # Markers only where there is room: on a dense array they would bury the
    # lines, which already show each spot's row and column by colour.
    markers = pitch >= m.px(MIN_MARKER_PITCH)
    r = float(np.clip(0.3 * pitch, 2 * m.scale, 5 * m.scale))
    thick = m.stroke if n <= 1500 else m.hair
    lw = m.stroke  # line width

    if R:
        idx, lens, xs, ys, starts = R
        yline = (np.add.reduceat(ys, starts) / lens * s).astype(int) + pt
        x0 = np.maximum((np.minimum.reduceat(xs, starts) * s).astype(int) + pl, 0)
        x1 = np.minimum((np.maximum.reduceat(xs, starts) * s).astype(int) + pl, W - 1)
        for i, y, a, b in zip(idx.tolist(), yline.tolist(), x0.tolist(), x1.tolist()):
            if 0 < y < H and b > a:
                cv2.rectangle(img, (a, y - lw // 2), (b, y - lw // 2 + lw - 1),
                              _ROW_COLORS[i % 6], -1)
        cid = np.repeat(idx % 6, lens) if markers else ()
        for c in np.unique(cid):
            sel = cid == c
            _markers(img, xs[sel] * s + pl, ys[sel] * s + pt, r, _ROW_COLORS[c], False, thick)
    if C:
        idx, lens, xs, ys, starts = C
        xline = (np.add.reduceat(xs, starts) / lens * s).astype(int) + pl
        y0 = np.maximum((np.minimum.reduceat(ys, starts) * s).astype(int) + pt, 0)
        y1 = np.minimum((np.maximum.reduceat(ys, starts) * s).astype(int) + pt, H - 1)
        dash, period = max(2, m.px(8)), max(3, m.px(12))
        for i, x, a, b in zip(idx.tolist(), xline.tolist(), y0.tolist(), y1.tolist()):
            if 0 < x < W and b > a:
                _dashed_column(img, x, a, b + 1, _COL_PIX[i % 6], dash, period, lw)
        cid = np.repeat(idx % 6, lens) if markers else ()
        for c in np.unique(cid):
            sel = cid == c
            _markers(img, xs[sel] * s + pl, ys[sel] * s + pt, r, _COL_COLORS[c], True, thick)


def draw_rect(img, p1, p2, color, label: str | None = None, m: Metrics | None = None):
    """A measurement rectangle: a thin colour line on a dark outline, readable
    over bright and dark picture alike, with an optional size label."""
    m = _m(m)
    (x0, y0), (x1, y1) = p1, p2
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), m.hair + 2)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, m.hair)
    if label:  # below the bottom-left corner, clear of the cards; inside if no room
        f = m.font(12)
        gap = m.px(4)
        ly = y1 + gap if y1 + gap + f.height < img.shape[0] else y1 - gap - f.height
        gfx.draw_text(img, label, x0 + m.px(2), ly, f, color, halo=OV.HALO)


# --------------------------------------------------------------------------- #
#  camera status: banner, "connecting" pill, toasts
# --------------------------------------------------------------------------- #
_banners = gfx.cache(16)
_pills = gfx.cache(16)
_toasts = gfx.cache(64)


def draw_banner(img, title: str, lines: list[str], level: str = "error",
                icon: str | None = None, m: Metrics | None = None):
    """Centred card over the frame, for a state that blocks the picture
    (camera disconnected); shrunk if the window is too small for it."""
    m = _m(m)
    H, W = img.shape[:2]

    def sprite_at(mm):
        return _banners.get((title, tuple(lines), level, icon, mm.scale, gfx.pillow_active()),
                            lambda: _render_banner(mm, title, lines, level, icon))

    sprite = sprite_at(m)
    fit = _fit_scale(m, (sprite.w, sprite.h), (W - 2 * m.edge, H - 2 * m.edge))
    if fit is not m:
        sprite = sprite_at(fit)
    sprite.draw(img, (W - sprite.w) // 2, max(0, (H - sprite.h) // 2 - m.px(40)))


def _render_banner(m: Metrics, title, lines, level, icon):
    color, level_icon = LEVELS.get(level, LEVELS["info"])
    ft, fl = m.font(20, True), m.font(14)
    pad, disc, gap, line = m.px(28), m.px(56), m.px(22), m.px(25)
    text_w = max([ft.width(title)] + [fl.width(s) for s in lines])
    w = pad + disc + gap + text_w + pad
    h = pad + m.px(34) + line * len(lines) + pad - m.px(6)
    c = gfx.Canvas(w, h)
    c.rrect(0, 0, w, h, m.px(16), OV.PANEL, 0.92, border=color, border_opacity=0.45)
    c.rrect(pad, pad, pad + disc, pad + disc, disc // 2, color, 0.18)
    isz = m.px(30)
    c.icon(icon or level_icon, pad + (disc - isz) // 2, pad + (disc - isz) // 2, isz, color)
    x = pad + disc + gap
    c.text(title, x, pad + m.px(12), ft, OV.TEXT, anchor="lm")
    for i, text in enumerate(lines):
        c.text(text, x, pad + m.px(44) + line * i, fl, OV.SUBTEXT if i else OV.TEXT, anchor="lm")
    return c.sprite()


def draw_pill(img, message: str, level: str = "warn", y: int | None = None,
              right: bool = False, m: Metrics | None = None):
    """Small notice that does not hide the picture (connecting), with a
    spinner: centred under the status bar (or at `y`), or at the right edge."""
    m = _m(m)
    color = LEVELS.get(level, LEVELS["info"])[0]
    h = m.px(32)

    def render():
        f = m.font(13)
        w = m.px(44) + f.width(message) + m.px(16)
        c = gfx.Canvas(w, h)
        c.rrect(0, 0, w, h, h / 2, OV.PANEL, 0.9, border=color, border_opacity=0.5)
        c.text(message, m.px(40), h // 2, f, OV.TEXT, anchor="lm")
        return c.sprite()

    sprite = _pills.get((message, level, m.scale, gfx.pillow_active()), render)
    W = img.shape[1]
    y = m.top_h + m.px(10) if y is None else y
    x = W - m.edge - sprite.w if right else (W - sprite.w) // 2
    sprite.draw(img, x, y)
    angle = (time.monotonic() * 400) % 360
    r = m.px(7)
    cv2.ellipse(img, (x + m.px(21), y + h // 2), (r, r), angle, 0, 270, color, m.stroke,
                cv2.LINE_AA)


def draw_toasts(img, toasts, now: float, bottom: int, lifetime: float = 6.0,
                m: Metrics | None = None):
    """Recent (time, text, level) messages, newest at the bottom right:
    sliding in, fading out over their last second."""
    m = _m(m)
    W = img.shape[1]
    y = bottom
    for t0, message, level in reversed(toasts):
        age = now - t0
        if age > lifetime or age < 0:
            continue
        fade = min(1.0, lifetime - age, 0.25 + age / 0.2)
        slide = int(max(0.0, 1 - age / 0.2) * m.px(30))
        sprite = _toasts.get((message, level, W, m.scale, gfx.pillow_active()),
                             lambda: _render_toast(m, message, level, W - 2 * m.edge))
        sprite.draw(img, W - m.edge - sprite.w + slide, y - sprite.h, opacity=fade)
        y -= sprite.h + m.gap


def _render_toast(m: Metrics, message, level, max_w):
    color, icon = LEVELS.get(level, LEVELS["info"])
    f = m.font(13)
    text_x, isz = m.px(44), m.px(18)
    message = gfx.ellipsize(message, f, max_w - text_x - m.px(16))
    w, h = text_x + f.width(message) + m.px(16), m.px(38)
    c = gfx.Canvas(w, h)
    c.rrect(0, 0, w, h, m.px(10), OV.PANEL, 0.92, border=color, border_opacity=0.35)
    c.icon(icon, m.px(14), (h - isz) // 2, isz, color)
    c.text(message, text_x, h // 2, f, OV.TEXT, anchor="lm")
    return c.sprite()
