"""Drawing primitives for the live-view overlays: antialiased text, icons and
translucent panels, blended onto the frame as cached sprites.

Why sprites: rendering a string with FreeType costs 30-140 us and a
supersampled icon far more, while blending a finished sprite costs a few us.
So everything is rendered once and kept in an LRU `SpriteCache` keyed by its
*content* (text, colour, size, state...); a frame only blends what it shows.
A value that changes simply misses the cache and is rendered anew - there is
no invalidation to get wrong, and the LRU bound keeps a stream of changing
numbers from growing memory.  Panels of uniform alpha skip even that: one
cv2.addWeighted against a solid image, plus four tiny corner sprites.

Text goes through Pillow (TrueType: the platform UI font, else the font that
Pillow bundles), one glyph at a time - each glyph is rendered once and strings
are assembled from the glyph masks with the font's kerning, so new numbers
cost ~15 us instead of a full FreeType layout.  Without Pillow, OpenCV's
Hershey font stands in behind the same interface: the overlays look plainer
but work the same.  Colours are BGR, like everything handed to cv2.imshow.
"""

from __future__ import annotations

import math
import sys
import threading
from collections import OrderedDict

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    HAVE_PIL = True
except ImportError:  # optional: the overlays fall back to Hershey text
    HAVE_PIL = False

_use_pil = HAVE_PIL

# Characters the UI uses that a font may lack (Pillow's bundled font has no
# Greek or maths symbols, Hershey has nothing beyond ASCII).
SUBSTITUTES = {
    "µ": "u", "σ": "s", "×": "x", "−": "-", "–": "-", "—": "-", "Δ": "d",
    "·": "-", "…": "...", "÷": "/", "→": "->", "←": "<-", "↑": "^", "↓": "v",
    "±": "+/-", "≈": "~", "²": "2", "°": "deg",
}
_PROBE_MISSING = "\U000f0000"  # private-use plane: renders as the .notdef box

# (regular, bold candidates) by preference - the platform UI font first, so
# the overlays match the Tk windows; Pillow's own font when none is found.
_FACES = (
    [("segoeui.ttf", ("seguisb.ttf", "segoeuib.ttf")), ("arial.ttf", ("arialbd.ttf",))]
    if sys.platform == "win32"
    else [
        ("NotoSans-Regular.ttf", ("NotoSans-SemiBold.ttf", "NotoSans-Bold.ttf")),
        ("DejaVuSans.ttf", ("DejaVuSans-Bold.ttf",)),
        ("LiberationSans-Regular.ttf", ("LiberationSans-Bold.ttf",)),
        ("Arial.ttf", ("Arial Bold.ttf",)),
    ]
)


# --------------------------------------------------------------------------- #
#  cache
# --------------------------------------------------------------------------- #
class SpriteCache:
    """Bounded LRU map from a content key to whatever rendering it produced.

    `get(key, render)` returns the cached value or calls `render()` once and
    keeps the result; the least recently used entry goes when full.  Keys
    must capture everything the rendering depends on - then a change of
    content is a different key and nothing ever needs invalidating."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.hits = self.misses = 0
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()  # icons are also built on the Tk thread

    def get(self, key, render):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
                self.hits += 1
                return value
        value = render()
        with self._lock:
            self.misses += 1
            self._items[key] = value
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return value

    def __contains__(self, key) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()


_caches: list[SpriteCache] = []


def cache(capacity: int) -> SpriteCache:
    """A SpriteCache that `reset()` empties along with the fonts."""
    c = SpriteCache(capacity)
    _caches.append(c)
    return c


# --------------------------------------------------------------------------- #
#  fonts
# --------------------------------------------------------------------------- #
_faces_found = None


def _faces():
    """(regular, bold) font files to use, or None for Pillow's bundled font."""
    global _faces_found
    if _faces_found is None:
        _faces_found = (None, None)
        for regular, bolds in _FACES:
            if _loads(regular):
                bold = next((b for b in bolds if _loads(b)), regular)
                _faces_found = (regular, bold)
                break
    return _faces_found if _faces_found[0] else None


def _loads(name: str) -> bool:
    try:
        ImageFont.truetype(name, 12)
        return True
    except OSError:
        return False


class Font:
    """One face at one pixel size.  `mask(text)` is the antialiased coverage
    (uint8, `height` rows: ascender top to descender bottom), cached."""

    def __init__(self, size: int, bold: bool = False):
        self.size, self.bold = size, bold
        self._masks = SpriteCache(4096)
        self._ft = None
        if _use_pil:
            self._ft = self._load_pil()
        if self._ft is not None:
            ascent, descent = self._ft.getmetrics()
            self.ascent, self.height = ascent, ascent + descent
            self._glyphs: dict = {}
            self._pairs: dict = {}
            self._has: dict = {}
            self._notdef = self._render_glyph(_PROBE_MISSING)
            cap_top = self._ft.getbbox("H")[1]
        else:
            self._scale = size / 28.0
            self._thick = 2 if bold else 1
            (_, th), base = cv2.getTextSize("Hg", cv2.FONT_HERSHEY_SIMPLEX,
                                            self._scale, self._thick)
            self.ascent, self.height = th + 2, th + base + 4
            cap_top = 2
        self.cap_mid = (cap_top + self.ascent) / 2  # visual middle of capitals/digits
        self.key = (size, bold, self._ft is not None)

    def _load_pil(self):
        faces = _faces()
        try:
            if faces:
                return ImageFont.truetype(faces[1] if self.bold else faces[0], self.size)
            return ImageFont.load_default(size=self.size)
        except (OSError, TypeError):  # Pillow < 10.1 has no sized default font
            return None

    def mask(self, text: str) -> np.ndarray:
        return self._masks.get(text, lambda: self._compose(text))

    def width(self, text: str) -> int:
        return self.mask(text).shape[1]

    # ------------------------------------------------------------------ #
    def _compose(self, text: str) -> np.ndarray:
        if self._ft is None:
            return self._hershey(text)
        if not text.isascii():
            text = "".join(c if c.isascii() or self._covers(c) else SUBSTITUTES.get(c, "?")
                           for c in text)
        glyphs, pairs = self._glyphs, self._pairs
        placed, pen, prev = [], 0.0, ""
        for ch in text:
            g = glyphs.get(ch) or self._glyph(ch)
            if prev:
                k = pairs.get(prev + ch)
                pen += self._kern(prev, ch) if k is None else k
            if g[1] is not None:
                placed.append((int(pen + 0.5) + g[0], g[1]))
            pen += g[2]
            prev = ch
        if not placed:
            return np.zeros((self.height, max(1, int(round(pen)))), np.uint8)
        x0 = min(0, min(x for x, _ in placed))
        x1 = max(int(math.ceil(pen)), max(x + g.shape[1] for x, g in placed))
        out = np.zeros((self.height, x1 - x0), np.uint8)
        for x, glyph in placed:
            dst = out[:, x - x0:x - x0 + glyph.shape[1]]
            np.maximum(dst, glyph, out=dst)
        return out

    def _glyph(self, ch):
        g = self._glyphs.get(ch)
        if g is None:
            g = self._glyphs[ch] = self._render_glyph(ch)
        return g

    def _render_glyph(self, ch):
        left, _top, right, _bottom = self._ft.getbbox(ch)
        advance = self._ft.getlength(ch)
        if right <= left:  # blank (space)
            return 0, None, advance
        img = Image.new("L", (right - left, self.height))
        ImageDraw.Draw(img).text((-left, 0), ch, font=self._ft, fill=255)
        return left, np.asarray(img), advance

    def _kern(self, a: str, b: str) -> float:
        k = self._pairs.get(a + b)
        if k is None:
            k = self._pairs[a + b] = (self._ft.getlength(a + b) - self._ft.getlength(a)
                                      - self._ft.getlength(b))
        return k

    def _covers(self, ch: str) -> bool:
        has = self._has.get(ch)
        if has is None:
            left, glyph, advance = self._glyph(ch)
            nl, ng, na = self._notdef
            has = not (advance == na and left == nl and glyph is not None and ng is not None
                       and glyph.shape == ng.shape and np.array_equal(glyph, ng))
            self._has[ch] = has
        return has

    def _hershey(self, text: str) -> np.ndarray:
        text = "".join(c if c.isascii() else SUBSTITUTES.get(c, "?") for c in text)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self._scale, self._thick)
        out = np.zeros((self.height, tw + 2), np.uint8)
        cv2.putText(out, text, (1, self.ascent), cv2.FONT_HERSHEY_SIMPLEX, self._scale,
                    255, self._thick, cv2.LINE_AA)
        return out


_fonts: dict = {}


def font(size: int = 14, bold: bool = False) -> Font:
    f = _fonts.get((size, bold))
    if f is None:
        f = _fonts[(size, bold)] = Font(size, bold)
    return f


def use_pillow(enabled: bool):
    """Switch between TrueType and the Hershey fallback (tests, or Pillow
    missing) and drop everything rendered with the other."""
    global _use_pil
    _use_pil = bool(enabled) and HAVE_PIL
    reset()


def reset():
    _fonts.clear()
    for c in _caches:
        c.clear()


def pillow_active() -> bool:
    return _use_pil


# --------------------------------------------------------------------------- #
#  sprites
# --------------------------------------------------------------------------- #
class Sprite:
    """A rendered image ready to blend: premultiplied colour and inverse
    alpha in 8.8 fixed point, so blending is `dst = (dst*inv + premul) >> 8`
    in uint16 over the sprite's own pixels only.  (ox, oy) shifts it from
    the requested position (e.g. a text halo that extends past the text)."""

    __slots__ = ("premul", "inv", "w", "h", "ox", "oy", "tw", "th")

    def __init__(self, premul: np.ndarray, inv: np.ndarray, ox: int = 0, oy: int = 0):
        self.premul, self.inv = premul, inv
        self.h, self.w = premul.shape[:2]
        self.ox, self.oy = ox, oy
        self.tw, self.th = self.w, self.h  # content size (text without its halo)

    @classmethod
    def from_premul(cls, rgb: np.ndarray, alpha: np.ndarray, ox: int = 0, oy: int = 0):
        """rgb: premultiplied float (h, w, 3) in 0..255; alpha: float 0..1
        (compositing keeps both in range, so no clipping is needed)."""
        a256 = (alpha * 256 + 0.5).astype(np.uint16)
        premul = (rgb * 256 + 0.5).astype(np.uint16)
        return cls(premul, (256 - a256)[..., None], ox, oy)

    @classmethod
    def from_mask(cls, mask: np.ndarray, color, halo=None, opacity: float = 1.0):
        """A solid colour through a coverage mask, optionally outlined by a
        soft `halo` colour so it reads over any picture.  Integer maths: a
        text that changes every frame renders this anew."""
        if halo is None:
            a = mask.astype(np.uint16)
            a += a >> 7  # 0..255 -> 0..256
            if opacity < 0.999:
                a = (a * int(opacity * 256)) >> 8
            return cls(a[..., None] * np.asarray(color, np.uint16), (256 - a)[..., None])
        pad = 2
        m = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        ring = cv2.GaussianBlur(cv2.dilate(m, _DISC3), (3, 3), 0).astype(np.uint32)
        a = m.astype(np.uint32)
        under = (ring * (255 - a) * 230) >> 16  # halo at 90 %, only outside the glyphs
        alpha = a + under  # 0..255
        a256 = (alpha + (alpha >> 7)).astype(np.uint16)
        premul = (a[..., None] * np.asarray(color, np.uint32)
                  + under[..., None] * np.asarray(halo, np.uint32))
        premul = (premul + (premul >> 8) + 128).astype(np.uint16)  # x256/255, rounded
        s = cls(premul, (256 - a256)[..., None], -pad, -pad)
        s.tw, s.th = s.w - 2 * pad, s.h - 2 * pad
        return s

    def draw(self, img: np.ndarray, x: int, y: int, opacity: float = 1.0):
        x, y = int(x) + self.ox, int(y) + self.oy
        H, W = img.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + self.w, W), min(y + self.h, H)
        if x1 <= x0 or y1 <= y0:
            return
        sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
        premul, inv = self.premul[sl], self.inv[sl]
        if opacity < 0.999:
            k = max(0, int(opacity * 256))
            premul = ((premul.astype(np.uint32) * k) >> 8).astype(np.uint16)
            inv = (256 - (((256 - inv.astype(np.uint32)) * k) >> 8)).astype(np.uint16)
        region = img[y0:y1, x0:x1]
        acc = region.astype(np.uint16)
        acc *= inv
        acc += premul
        acc >>= 8
        region[...] = acc


_DISC3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_text_sprites = cache(1024)  # ~10-20 KB each: bounded to ~16 MB


def text_sprite(text: str, fnt: Font, color, halo=None) -> Sprite:
    return _text_sprites.get(
        (fnt.key, text, color, halo), lambda: Sprite.from_mask(fnt.mask(text), color, halo)
    )


_blocks = cache(512)  # spot labels: > MAX_LABELS, so one frame always hits


def text_block(lines, fnt: Font, halo, line_step: int) -> Sprite:
    """Several centred lines of (text, colour) under one halo, as one sprite:
    a spot label is one blend, not one per line."""

    def render():
        masks = [fnt.mask(t) for t, _ in lines]
        w = max(m.shape[1] for m in masks)
        h = line_step * (len(lines) - 1) + fnt.height
        pad = 2
        c = Canvas(w + 2 * pad, h + 2 * pad)
        for i, ((_, color), m) in enumerate(zip(lines, masks)):
            c.paint(m, pad + (w - m.shape[1]) // 2, pad + i * line_step, color)
        ring = cv2.GaussianBlur(cv2.dilate(c.a, _DISC3), (3, 3), 0) * 0.9
        under = ring * (1 - c.a)
        c.rgb += under[..., None] * np.asarray(halo, np.float32)
        c.a += under
        sprite = c.sprite()
        sprite.ox = sprite.oy = -pad
        sprite.tw, sprite.th = w, h
        return sprite

    return _blocks.get((fnt.key, tuple(lines), halo, line_step), render)


_icon_sprites = cache(256)


def icon_sprite(name: str, size: int, color) -> Sprite:
    from . import icons  # noqa: PLC0415  (icons imports this module)

    return _icon_sprites.get((name, size, color),
                             lambda: Sprite.from_mask(icons.mask(name, size), color))


def draw_text(img, text: str, x, y, fnt: Font, color, halo=None, anchor: str = "lt") -> int:
    """Blend `text` with its box anchored at (x, y): anchor[0] in l/m/r,
    anchor[1] in t (top of the line box), m (middle of the capitals), b
    (bottom of the line box).  Returns the text width."""
    s = text_sprite(text, fnt, color, halo)
    if anchor[0] == "m":
        x -= s.tw // 2
    elif anchor[0] == "r":
        x -= s.tw
    if anchor[1] == "m":
        y -= int(round(fnt.cap_mid))
    elif anchor[1] == "b":
        y -= fnt.height
    s.draw(img, x, y)
    return s.tw


def ellipsize(text: str, fnt: Font, max_w: int) -> str:
    """`text`, shortened with an ellipsis to fit `max_w` pixels."""
    if fnt.width(text) <= max_w:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fnt.width(text[:mid] + "…") <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


# --------------------------------------------------------------------------- #
#  shapes and panels
# --------------------------------------------------------------------------- #
def rrect_alpha(w: int, h: int, r: float) -> np.ndarray:
    """Exact antialiased coverage of a w x h rounded rectangle (signed
    distance), float32 0..1."""
    r = max(0.0, min(r, w / 2, h / 2))
    xs = np.abs(np.arange(w, dtype=np.float32) + 0.5 - w / 2) - (w / 2 - r)
    ys = np.abs(np.arange(h, dtype=np.float32) + 0.5 - h / 2) - (h / 2 - r)
    qx, qy = xs[None, :], ys[:, None]
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return np.clip(0.5 - (outside + inside - r), 0, 1).astype(np.float32)


_rrects = cache(128)
_solids: dict = {}
_corners = cache(64)


def _solid(color, h: int, w: int) -> np.ndarray:
    """A view of a solid colour image at least h x w, grown on demand."""
    img = _solids.get(color)
    if img is None or img.shape[0] < h or img.shape[1] < w:
        hh, ww = max(h, 64 if img is None else img.shape[0]), max(w, 64 if img is None else img.shape[1])
        img = _solids[color] = np.full((hh, ww, 3), color, np.uint8)
    return img[:h, :w]


def shade(img, x0, y0, x1, y1, color, alpha: float):
    """Blend a uniform translucent rectangle (one SIMD pass)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), W), min(int(y1), H)
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1]
    cv2.addWeighted(_solid(color, y1 - y0, x1 - x0), alpha, region, 1 - alpha, 0, dst=region)


def panel(img, x0, y0, x1, y1, color, alpha: float, radius: int = 10):
    """Translucent rounded rectangle: three uniform bands plus four corner
    sprites, so the cost is that of `shade`, not of a full sprite blend."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    if r <= 0:
        shade(img, x0, y0, x1, y1, color, alpha)
        return
    shade(img, x0 + r, y0, x1 - r, y1, color, alpha)
    shade(img, x0, y0 + r, x0 + r, y1 - r, color, alpha)
    shade(img, x1 - r, y0 + r, x1, y1 - r, color, alpha)
    tl, tr, bl, br = _corners.get((r, color, alpha), lambda: _corner_sprites(r, color, alpha))
    tl.draw(img, x0, y0)
    tr.draw(img, x1 - r, y0)
    bl.draw(img, x0, y1 - r)
    br.draw(img, x1 - r, y1 - r)


def _corner_sprites(r, color, alpha):
    a = rrect_alpha(2 * r, 2 * r, r) * alpha
    rgb = a[..., None] * np.asarray(color, np.float32)
    quads = ((slice(0, r), slice(0, r)), (slice(0, r), slice(r, 2 * r)),
             (slice(r, 2 * r), slice(0, r)), (slice(r, 2 * r), slice(r, 2 * r)))
    return tuple(Sprite.from_premul(rgb[q], a[q]) for q in quads)


class Canvas:
    """Premultiplied float BGRA scratch image for composing a sprite once
    (toolbar, tooltip, chips, cards with static content).  Only ever used on
    a cache miss, so clarity wins over speed here."""

    def __init__(self, w: int, h: int):
        self.w, self.h = int(w), int(h)
        self.rgb = np.zeros((self.h, self.w, 3), np.float32)
        self.a = np.zeros((self.h, self.w), np.float32)

    def paint(self, alpha: np.ndarray, x: int, y: int, color, opacity: float = 1.0):
        """Composite `color` through coverage `alpha` (float 0..1 or uint8)
        with its top-left at (x, y), clipped to the canvas."""
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0
        h, w = alpha.shape
        x, y = int(x), int(y)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, self.w), min(y + h, self.h)
        if x1 <= x0 or y1 <= y0:
            return
        a = alpha[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
        rgb, dst_a = self.rgb[y0:y1, x0:x1], self.a[y0:y1, x0:x1]
        keep = 1 - a
        rgb *= keep[..., None]
        rgb += a[..., None] * np.asarray(color, np.float32)
        dst_a *= keep
        dst_a += a

    def rrect(self, x0, y0, x1, y1, r, color, opacity=1.0, border=None, border_opacity=1.0):
        w, h = int(x1 - x0), int(y1 - y0)
        if w <= 0 or h <= 0:
            return
        fill = _rrects.get((w, h, r), lambda: rrect_alpha(w, h, r))
        self.paint(fill, x0, y0, color, opacity)
        if border is not None:
            inner = _rrects.get((w - 2, h - 2, max(r - 1, 0)),
                                lambda: rrect_alpha(w - 2, h - 2, max(r - 1, 0)))
            ring = fill.copy()
            ring[1:-1, 1:-1] -= inner
            self.paint(np.clip(ring, 0, 1), x0, y0, border, border_opacity)

    def text(self, text, x, y, fnt: Font, color, opacity=1.0, anchor="lt") -> int:
        m = fnt.mask(text)
        if anchor[0] == "m":
            x -= m.shape[1] // 2
        elif anchor[0] == "r":
            x -= m.shape[1]
        if anchor[1] == "m":
            y -= int(round(fnt.cap_mid))
        self.paint(m, x, y, color, opacity)
        return m.shape[1]

    def icon(self, name: str, x, y, size: int, color, opacity=1.0):
        from . import icons  # noqa: PLC0415  (icons imports this module)

        self.paint(icons.mask(name, size), x, y, color, opacity)

    def sprite(self) -> Sprite:
        return Sprite.from_premul(self.rgb, self.a)
