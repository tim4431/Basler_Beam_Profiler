"""One palette for every window: the overlays on the live view (BGR, for
cv2.imshow) and the Tk windows (hex strings).  Dark, flat, one blue accent,
green/amber/red only for state - so a colour means the same thing everywhere.
"""

from __future__ import annotations


def bgr(hex_rgb: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (b, g, r), the order OpenCV draws in."""
    h = hex_rgb.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


# --- shared --------------------------------------------------------------- #
BG = "#111318"  # window background
SURFACE = "#1a1d24"  # cards
SURFACE_2 = "#242832"  # fields, buttons
SURFACE_3 = "#2e3340"  # hover
BORDER = "#323846"
TEXT = "#e8ebf0"
SUBTEXT = "#9aa3b2"
MUTED = "#6b7383"
ACCENT = "#3b82f6"
ACCENT_HI = "#60a5fa"
ACCENT_DARK = "#2563eb"
SELECTION = "#1e3a64"
GOOD = "#22c55e"
WARN = "#f59e0b"
BAD = "#ef4444"
GOOD_TEXT = "#4ade80"  # the state colours, light enough to read on SURFACE
WARN_TEXT = "#fbbf24"
BAD_TEXT = "#f87171"


class OV:
    """Overlay colours, BGR."""

    PANEL = bgr("#0c0e12")
    PANEL_ALPHA = 0.86
    TOOLTIP = bgr("#0a0b0f")
    HAIRLINE = bgr("#2a2f3a")
    TEXT = bgr(TEXT)
    SUBTEXT = bgr(SUBTEXT)
    MUTED = bgr(MUTED)
    ICON = bgr("#c3cad6")
    ACCENT = bgr(ACCENT)
    ACCENT_HI = bgr(ACCENT_HI)
    GOOD = bgr(GOOD)
    WARN = bgr(WARN)
    BAD = bgr(BAD)
    GOOD_TEXT = bgr(GOOD_TEXT)
    WARN_TEXT = bgr(WARN_TEXT)
    BAD_TEXT = bgr(BAD_TEXT)
    TRACK = bgr("#2a2f3a")  # empty part of a bar
    HALO = bgr("#000000")  # outline behind text drawn straight on the picture

    # the measurement graphics keep their documented colours
    SPOT = bgr("#4ade80")  # fitted ellipse
    AXIS_MAJOR = bgr("#f87171")  # s0 direction
    AXIS_MINOR = bgr("#60a5fa")  # s1 direction
    SPOT_LABEL = bgr("#bbf7d0")
    SPOT_COORD = bgr("#d6dbe3")
    RECT = (255, 255, 255)  # white rectangle: pixel statistics
    FIT_RECT = bgr("#22e05a")  # green rectangle: fit area


LEVELS = {  # toast / banner level -> (colour, icon)
    "info": (OV.ACCENT_HI, "info"),
    "warn": (OV.WARN, "warning"),
    "error": (OV.BAD, "error"),
}
