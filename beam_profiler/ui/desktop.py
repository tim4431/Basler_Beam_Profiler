"""Make the desktop recognise the app's windows as "Beam Profiler".

GNOME (and most Linux docks) group windows into apps by WM_CLASS and match it
against the `.desktop` file - setup.sh installs `beam-profiler.desktop` with
`StartupWMClass=beam-profiler`.  The Tk windows get that class from tkhost.
OpenCV's bundled Qt, though, starts with an empty application name and then
sets no WM_CLASS at all, so the live view showed up as "Unknown" with a
generic icon.  `claim_window` sets it afterwards through Xlib (ctypes, no new
dependency); the shell picks up the change and re-associates the window.  It
also sets _NET_WM_ICON, for window managers and switchers that use the
window's own icon rather than the launcher's.

It is a no-op off X11 (Windows, macOS, a native Wayland session): there the
window is matched another way or not at all, and nothing breaks.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import logging
import os
import sys

import numpy as np

WM_INSTANCE = "beam-profiler"  # = StartupWMClass in the .desktop file
WM_CLASS = "Beam-profiler"  # Tk's capitalisation of the same name


class _ClassHint(ctypes.Structure):  # for XSetClassHint: we own the strings
    _fields_ = [("res_name", ctypes.c_char_p), ("res_class", ctypes.c_char_p)]


class _ClassHintOut(ctypes.Structure):  # for XGetClassHint: Xlib owns them, we XFree
    _fields_ = [("res_name", ctypes.c_void_p), ("res_class", ctypes.c_void_p)]


def _xlib():
    if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
        return None
    path = ctypes.util.find_library("X11")
    if not path:
        return None
    x = ctypes.cdll.LoadLibrary(path)
    x.XOpenDisplay.restype = ctypes.c_void_p
    x.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x.XDefaultRootWindow.restype = ctypes.c_ulong
    x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x.XQueryTree.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
        ctypes.POINTER(ctypes.c_uint),
    ]
    x.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
    x.XGetClassHint.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_ClassHintOut)]
    x.XSetClassHint.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_ClassHint)]
    x.XFree.argtypes = [ctypes.c_void_p]
    x.XFlush.argtypes = [ctypes.c_void_p]
    x.XInternAtom.restype = ctypes.c_ulong
    x.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x.XChangeProperty.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
    ]
    x.XCloseDisplay.argtypes = [ctypes.c_void_p]
    return x


def claim_window(title: str) -> bool:
    """Give the top-level window called `title` our WM_CLASS, unless it
    already has one.  True once the window was found (whether or not it
    needed the change), so the caller can stop asking; False if it does not
    exist yet or this is not X11."""
    try:
        x = _xlib()
        if x is None:
            return True  # nothing to do on this platform; stop asking
        dpy = x.XOpenDisplay(None)
        if not dpy:
            return True
        try:
            return _claim(x, dpy, title.encode())
        finally:
            x.XCloseDisplay(dpy)
    except Exception as e:  # never let a cosmetic fix disturb the app
        logging.debug(f"could not set WM_CLASS: {e}")
        return True


def _claim(x, dpy, title: bytes) -> bool:
    root = x.XDefaultRootWindow(dpy)
    root_ret, parent = ctypes.c_ulong(), ctypes.c_ulong()
    children, count = ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
    if not x.XQueryTree(dpy, root, ctypes.byref(root_ret), ctypes.byref(parent),
                        ctypes.byref(children), ctypes.byref(count)):
        return False
    try:
        for i in range(count.value):
            window = children[i]
            # a window manager reparents clients into frames: look one level down too
            for candidate in (window, *_children(x, dpy, window)):
                if _name(x, dpy, candidate) != title:
                    continue
                current = _ClassHintOut()
                if x.XGetClassHint(dpy, candidate, ctypes.byref(current)):
                    has_class = bool(current.res_class and ctypes.string_at(current.res_class))
                    for ptr in (current.res_name, current.res_class):
                        if ptr:
                            x.XFree(ptr)
                    if has_class:
                        return True  # someone set one already; leave it alone
                hint = _ClassHint(WM_INSTANCE.encode(), WM_CLASS.encode())
                x.XSetClassHint(dpy, candidate, ctypes.byref(hint))
                _set_icon(x, dpy, candidate)
                x.XFlush(dpy)
                return True
        return False
    finally:
        if children:
            x.XFree(children)


def _children(x, dpy, window) -> list[int]:
    root_ret, parent = ctypes.c_ulong(), ctypes.c_ulong()
    children, count = ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
    if not x.XQueryTree(dpy, window, ctypes.byref(root_ret), ctypes.byref(parent),
                        ctypes.byref(children), ctypes.byref(count)):
        return []
    try:
        return [children[i] for i in range(count.value)]
    finally:
        if children:
            x.XFree(children)


def _name(x, dpy, window) -> bytes | None:
    name = ctypes.c_char_p()
    if not x.XFetchName(dpy, window, ctypes.byref(name)) or not name.value:
        return None
    value = name.value
    x.XFree(ctypes.cast(name, ctypes.c_void_p))
    return value


def _icon_cardinals() -> np.ndarray | None:
    """_NET_WM_ICON payload: for each size, width, height, then ARGB pixels -
    each as a C long, which is what Xlib wants for 32-bit properties."""
    import cv2

    from . import appicon

    parts = []
    for size in sorted(appicon.PNG):
        png = np.frombuffer(base64.b64decode(appicon.PNG[size]), np.uint8)
        bgra = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            continue
        b, g, r, a = (bgra[..., i].astype(np.uint64) for i in range(4))
        argb = (a << 24) | (r << 16) | (g << 8) | b
        parts += [np.array([bgra.shape[1], bgra.shape[0]], np.uint64), argb.ravel()]
    if not parts:
        return None
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.dtype(f"u{ctypes.sizeof(ctypes.c_ulong)}"))


def _set_icon(x, dpy, window):
    data = _icon_cardinals()
    if data is None:
        return
    net_wm_icon = x.XInternAtom(dpy, b"_NET_WM_ICON", 0)
    cardinal = x.XInternAtom(dpy, b"CARDINAL", 0)
    x.XChangeProperty(dpy, window, net_wm_icon, cardinal, 32, 0,  # PropModeReplace
                      data.ctypes.data_as(ctypes.c_void_p), len(data))


def screen_size() -> tuple[int, int] | None:
    """Usable size of the main screen in pixels (Windows: the work area, minus
    the taskbar), for sizing the first window; None when it cannot be found
    cheaply.  Never raises."""
    try:
        if sys.platform == "win32":
            user32 = ctypes.windll.user32

            class _Rect(ctypes.Structure):
                _fields_ = [(n, ctypes.c_long) for n in ("left", "top", "right", "bottom")]

            area = _Rect()
            if user32.SystemParametersInfoW(0x30, 0, ctypes.byref(area), 0):  # SPI_GETWORKAREA
                return area.right - area.left, area.bottom - area.top
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        x = _xlib()
        if x is None:
            return None
        x.XDefaultScreen.argtypes = [ctypes.c_void_p]
        x.XDisplayWidth.argtypes = x.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dpy = x.XOpenDisplay(None)
        if not dpy:
            return None
        try:
            screen = x.XDefaultScreen(dpy)
            return x.XDisplayWidth(dpy, screen), x.XDisplayHeight(dpy, screen)
        finally:
            x.XCloseDisplay(dpy)
    except Exception as e:
        logging.debug(f"could not read the screen size: {e}")
        return None
