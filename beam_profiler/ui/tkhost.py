"""The one Tk interpreter behind every auxiliary window (fit settings, cameras).

Tk is not thread-safe, and a second `tk.Tk()` in a process whose first root
was destroyed fails ("tk wasn't installed properly").  So the app makes exactly
one root - hidden, on a daemon thread that lives as long as the process - and
every window is a Toplevel of it.  Other threads never call into Tk: `call(fn)`
queues `fn(root)` and the Tk thread runs it within POLL_MS, so the OpenCV loop
never waits on a widget.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk

POLL_MS = 50

# palette shared by every window (light, flat, single accent)
BG = "#eceff4"
CARD = "#ffffff"
BORDER = "#d8dee9"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
TEXT = "#1f2430"
SUBTEXT = "#5b6472"
FIELD = "#f6f8fb"
GOOD = "#15803d"
WARN = "#b45309"
BAD = "#b91c1c"
MUTED = "#8a93a3"

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
        root = tk.Tk()
        root.withdraw()
        apply_theme(root)
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
    """Flat, light theme with one accent colour, on top of clam.  Styles
    belong to the interpreter, so this runs once for every window."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    from tkinter import font as tkfont  # noqa: PLC0415  (needs a root first)

    base = tkfont.nametofont("TkDefaultFont")
    base.configure(size=10)
    family = base.actual("family")
    style.configure(".", background=BG, foreground=TEXT)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Title.TLabel", font=(family, 14, "bold"))
    style.configure("Sub.TLabel", foreground=SUBTEXT)
    style.configure("Card.TLabel", background=CARD)
    style.configure("CardSub.TLabel", background=CARD, foreground=SUBTEXT)
    style.configure("CardWarn.TLabel", background=CARD, foreground=WARN)
    style.configure("CardGood.TLabel", background=CARD, foreground=GOOD)
    style.configure("CardBad.TLabel", background=CARD, foreground=BAD)
    style.configure("CardBold.TLabel", background=CARD, font=(family, 11, "bold"))
    style.configure(
        "Card.TLabelframe", background=CARD, relief="solid",
        borderwidth=1, bordercolor=BORDER,
    )
    style.configure(
        "Card.TLabelframe.Label", background=BG, foreground=ACCENT,
        font=(family, 10, "bold"),
    )
    style.configure("Card.TFrame", background=CARD)
    style.configure("Card.TCheckbutton", background=CARD)
    style.map("Card.TCheckbutton", background=[("active", CARD)])
    style.configure(
        "TEntry", fieldbackground=FIELD, bordercolor=BORDER,
        lightcolor=BORDER, darkcolor=BORDER, padding=3,
    )
    style.configure(
        "Horizontal.TScale", troughcolor=BORDER, background=CARD,
        lightcolor=CARD, darkcolor=CARD,
    )
    style.configure(
        "TButton", padding=(14, 7), background="#e2e7ef", bordercolor=BORDER,
        lightcolor="#e2e7ef", darkcolor="#e2e7ef", focusthickness=0,
    )
    style.map("TButton", background=[("active", "#d3dae6"), ("pressed", "#c6cedd")])
    style.configure(
        "Accent.TButton", background=ACCENT, foreground="#ffffff",
        bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_DARK), ("pressed", ACCENT_DARK),
                    ("disabled", "#9db4ea")],
    )
    style.configure(
        "Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT,
        bordercolor=BORDER, rowheight=28,
    )
    style.map("Treeview", background=[("selected", "#dbe6fd")],
              foreground=[("selected", TEXT)])
    style.configure(
        "Treeview.Heading", background="#e2e7ef", foreground=SUBTEXT,
        bordercolor=BORDER, relief="flat", font=(family, 9, "bold"),
    )
