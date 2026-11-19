"""The one Tk interpreter behind every auxiliary window (fit settings, cameras).

Tk is not thread-safe, and a second `tk.Tk()` in a process whose first root
was destroyed fails ("tk wasn't installed properly").  So the app makes exactly
one root - hidden, on a daemon thread that lives as long as the process - and
every window is a Toplevel of it.  Other threads never call into Tk: `call(fn)`
queues `fn(root)` and the Tk thread runs it within POLL_MS, so the OpenCV loop
never waits on a widget.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk

from .. import uiconfig
from . import appicon, desktop, icons, theme

POLL_MS = 50

# The palette is shared with the overlays (see theme.py); these names are what
# the windows import.  State colours are the variants readable on dark cards.
BG = theme.BG
CARD = theme.SURFACE
BORDER = theme.BORDER
ACCENT = theme.ACCENT
ACCENT_HI = theme.ACCENT_HI
ACCENT_DARK = theme.ACCENT_DARK
TEXT = theme.TEXT
SUBTEXT = theme.SUBTEXT
FIELD = theme.SURFACE
GOOD = theme.GOOD_TEXT
WARN = theme.WARN_TEXT
BAD = theme.BAD_TEXT
MUTED = theme.MUTED
UI_FAMILIES = ("Segoe UI", "Noto Sans", "Ubuntu", "Cantarell", "DejaVu Sans")


def S(n: float) -> int:
    """`n` design pixels (or points) at the current UI scale.  Every size in a
    Tk window goes through this - fonts, paddings, icons, minsize - so that
    `uiconfig` zooms the windows like the live view."""
    return int(round(n * uiconfig.scale()))


_rescalers: list = []


def on_rescale(fn):
    """Run `fn()` on the Tk thread after every UI scale change, once the
    theme has been re-applied (windows rebuild themselves at the new size)."""
    if fn not in _rescalers:
        _rescalers.append(fn)
    return fn


def _rescale(root):
    apply_theme(root)
    for fn in list(_rescalers):
        try:
            fn()
        except Exception:
            logging.exception("could not rescale a Tk window")


def bind_zoom(win):
    """Ctrl + / Ctrl - / Ctrl 0 in a Tk window zoom the whole interface."""
    for seq, step in (("<Control-plus>", 1), ("<Control-equal>", 1), ("<Control-KP_Add>", 1),
                      ("<Control-minus>", -1), ("<Control-KP_Subtract>", -1),
                      ("<Control-0>", 0), ("<Control-KP_0>", 0)):
        win.bind(seq, lambda _e, st=step: uiconfig.zoom(st) if st else uiconfig.reset_scale())

_queue: queue.SimpleQueue = queue.SimpleQueue()
_thread: threading.Thread | None = None
_lock = threading.Lock()
_failed: str | None = None


def call(fn):
    """Run `fn(root)` on the Tk thread, starting it on first use."""
    global _thread
    if _failed is not None:
        logging.warning(f"window unavailable: {_failed}")
        return
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=_run, name="tk", daemon=True)
            _thread.start()
    _queue.put(fn)


def _run():
    global _failed
    try:
        # className sets WM_CLASS, which the dock matches to beam-profiler.desktop
        root = tk.Tk(className=desktop.WM_INSTANCE)
        root.withdraw()
        apply_theme(root)
        # a zoom from any thread (keys in the live view, the HTTP API, a button
        # here) re-themes and rebuilds the windows, on this thread
        uiconfig.on_change(lambda _cfg: call(_rescale))
    except Exception as e:  # no display, broken Tk install
        _failed = str(e)
        logging.error(f"Tk windows unavailable: {e}")
        return

    def pump():
        while True:
            try:
                fn = _queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn(root)
            except Exception:
                logging.exception("error in a Tk window")
        root.after(POLL_MS, pump)

    root.after(0, pump)
    root.mainloop()


def apply_theme(root: tk.Tk):
    """Dark, flat theme on top of clam, in the overlays' palette and font, at
    the current UI scale.  Styles belong to the interpreter, so this runs once
    for every window - and again after every zoom (named fonts and styles
    update the widgets that use them in place)."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    from tkinter import font as tkfont  # noqa: PLC0415  (needs a root first)

    families = set(tkfont.families(root))
    family = next((f for f in UI_FAMILIES if f in families), None)
    for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
        try:
            f = tkfont.nametofont(name)
            f.configure(size=S(10), **({"family": family} if family else {}))
        except tk.TclError:
            pass
    family = tkfont.nametofont("TkDefaultFont").actual("family")
    root.configure(background=BG)
    # classic Tk widgets (tk.Text, the Linux file dialog) read the option database
    for pattern, value in (
        ("*Background", CARD), ("*Foreground", TEXT), ("*selectBackground", theme.SELECTION),
        ("*selectForeground", TEXT), ("*insertBackground", TEXT), ("*troughColor", BG),
        ("*highlightBackground", BG), ("*highlightColor", ACCENT),
        ("*activeBackground", theme.SURFACE_3), ("*activeForeground", TEXT),
    ):
        root.option_add(pattern, value)

    S2, S3 = theme.SURFACE_2, theme.SURFACE_3
    style.configure(
        ".", background=BG, foreground=TEXT, fieldbackground=S2, bordercolor=BORDER,
        lightcolor=S2, darkcolor=S2, troughcolor=CARD, selectbackground=theme.SELECTION,
        selectforeground=TEXT, insertcolor=TEXT, arrowcolor=SUBTEXT, focuscolor=ACCENT,
    )
    style.map(".", foreground=[("disabled", MUTED)])
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Title.TLabel", font=(family, S(15), "bold"))
    style.configure("Sub.TLabel", foreground=SUBTEXT)
    style.configure("Card.TLabel", background=CARD)
    style.configure("CardSub.TLabel", background=CARD, foreground=SUBTEXT)
    style.configure("CardWarn.TLabel", background=CARD, foreground=WARN)
    style.configure("CardGood.TLabel", background=CARD, foreground=GOOD)
    style.configure("CardBad.TLabel", background=CARD, foreground=BAD)
    style.configure("CardBold.TLabel", background=CARD, font=(family, S(11), "bold"))
    style.configure(
        "Card.TLabelframe", background=CARD, relief="solid", borderwidth=1,
        bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD,
    )
    style.configure(
        "Card.TLabelframe.Label", background=BG, foreground=ACCENT_HI,
        font=(family, S(10), "bold"),
    )
    style.configure("Card.TFrame", background=CARD)
    style.configure(
        "Card.TCheckbutton", background=CARD, foreground=TEXT, indicatorbackground=S2,
        indicatorforeground="#ffffff", upperbordercolor=BORDER, lowerbordercolor=BORDER,
        indicatorsize=S(15),
    )
    style.map(
        "Card.TCheckbutton", background=[("active", CARD)],
        indicatorbackground=[("selected", ACCENT), ("active", S3)],
    )
    style.configure(
        "TEntry", fieldbackground=S2, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER,
        lightcolor=S2, darkcolor=S2, padding=S(4),
    )
    style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])
    style.configure(
        "Horizontal.TScale", troughcolor=S3, background=ACCENT, bordercolor=CARD,
        lightcolor=ACCENT, darkcolor=ACCENT, gripcount=0, sliderlength=S(16),
    )
    style.map("Horizontal.TScale", background=[("active", ACCENT_HI)])
    style.configure(
        "TScrollbar", background=S2, troughcolor=CARD, bordercolor=CARD, arrowcolor=SUBTEXT,
        arrowsize=S(14),
        lightcolor=S2, darkcolor=S2,
    )
    style.map("TScrollbar", background=[("active", S3)])
    style.configure(
        "TButton", padding=(S(12), S(7)), background=S2, foreground=TEXT, bordercolor=BORDER,
        lightcolor=S2, darkcolor=S2, focusthickness=0, focuscolor=S2,
    )
    style.map(
        "TButton", background=[("disabled", CARD), ("pressed", BORDER), ("active", S3)],
        foreground=[("disabled", MUTED)],
    )
    style.configure(
        "Accent.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT,
        lightcolor=ACCENT, darkcolor=ACCENT, focuscolor=ACCENT,
    )
    style.map(
        "Accent.TButton",
        background=[("disabled", "#1f3357"), ("pressed", ACCENT_DARK), ("active", ACCENT_DARK)],
        foreground=[("disabled", "#7d93b8")],
        bordercolor=[("disabled", "#1f3357")],
        lightcolor=[("disabled", "#1f3357")],
        darkcolor=[("disabled", "#1f3357")],
    )
    style.configure(
        "Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT,
        bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD, rowheight=S(30),
    )
    style.map("Treeview", background=[("selected", theme.SELECTION)],
              foreground=[("selected", TEXT)])
    style.configure(
        "Treeview.Heading", background=S2, foreground=SUBTEXT, bordercolor=BORDER,
        lightcolor=S2, darkcolor=S2, relief="flat", font=(family, S(9), "bold"),
        padding=(S(6), S(5)),
    )
    style.map("Treeview.Heading", background=[("active", S3)])
    sizes = [app_icon(size, root) for size in sorted(appicon.PNG, reverse=True)]
    sizes = [img for img in sizes if img is not None]
    if sizes:
        try:
            root.iconphoto(True, *sizes)  # every Toplevel inherits it
        except tk.TclError:
            pass


_images: dict = {}


class _Image(tk.PhotoImage):
    """A PhotoImage that only deletes itself on the thread that made it.
    The cache below is torn down by the main thread at interpreter exit,
    and a Tcl call from there would wait forever on the (daemon) Tk thread."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread = threading.get_ident()

    def __del__(self):
        if getattr(self, "_thread", None) == threading.get_ident():
            super().__del__()


def icon(name: str, size: int = 16, color: str = TEXT, master=None):
    """One of `icons` as a Tk image (Tk thread only); None if this Tk cannot
    decode PNG, in which case buttons simply show their text."""
    return _image(("icon", name, size, color), lambda: icons.png(name, size, color), master)


def app_icon(size: int, master=None):
    """The app icon (docs/icon.svg, pre-rendered in appicon.py) as a Tk image."""
    return _image(("appicon", size), lambda: base64.b64decode(appicon.PNG[size]), master)


def dot(color: str, size: int = 14, master=None):
    """A status dot for lists."""
    return _image(("dot", color, size), lambda: icons.dot_png(size, color), master)


def _image(key, make_png, master):
    master = master or getattr(tk, "_default_root", None)
    if master is None:
        return None
    key = (*key, str(master.tk))
    img = _images.get(key)
    if img is None:
        try:
            img = _Image(master=master, data=base64.b64encode(make_png()), format="png")
        except tk.TclError:
            return None
        _images[key] = img  # Tk images vanish when their last Python reference goes
    return img
