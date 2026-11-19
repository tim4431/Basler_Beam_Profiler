"""The live view's X11 window: who it belongs to, and who gets its wheel.

OpenCV's bundled Qt viewer misbehaves on Linux in two ways this module fixes
from the outside, over the app's own X connection (xcb through ctypes: no new
dependency, and unlike Xlib an X error cannot terminate the process):

* **Identity.**  It starts with an empty application name and so sets no
  WM_CLASS; GNOME's dock then shows the window as "Unknown" with a generic
  icon.  `LiveWindow` sets WM_CLASS (matching `StartupWMClass` in the
  `.desktop` file setup.sh installs, and the Tk windows' class) and
  _NET_WM_ICON.

* **Zoom.**  Qt magnifies the displayed image on every mouse-wheel event and
  on Ctrl +/-, overlays and all, with no way to turn that off.  The app uses
  the wheel to zoom the camera ROI, so each scroll zoomed twice.
  `LiveWindow` takes passive grabs on the wheel buttons and on Ctrl +/-/0 for
  that window, so the events come to the app instead of to Qt.  `poll()`
  hands them to the viewer, which zooms the ROI (wheel) or the interface
  (Ctrl +/-/0).  Clicks and motion are not grabbed and still reach OpenCV.
  A real mouse or touchpad on XWayland also reports scrolling as XInput2
  "smooth scroll" motion, which no button grab intercepts, and Qt zooms on
  that too - so `prepare_qt()` turns XInput2 off for Qt before it starts
  (QT_XCB_NO_XI2), leaving it the plain wheel-button events the grab takes.

Everything here is a no-op off X11 (Windows, macOS, native Wayland): the
Windows backend has no magnifier, and the viewer then takes the wheel from
OpenCV's own mouse callback.
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

# X protocol constants
_ATOM_CARDINAL, _ATOM_STRING, _ATOM_WM_NAME, _ATOM_WM_CLASS = 6, 31, 39, 67
_BUTTON_PRESS, _KEY_PRESS = 4, 2
_EVENT_MASK_BUTTONS = 0x4 | 0x8  # ButtonPress | ButtonRelease
_GRAB_ASYNC = 1
_MOD_ANY = 0x8000
SHIFT, LOCK, CONTROL, NUMLOCK = 0x1, 0x2, 0x4, 0x10
WHEEL_BUTTONS = (4, 5, 6, 7)  # up, down, left, right

# Ctrl + these keysyms -> viewer action.  Qt uses Ctrl +/- for its magnifier.
_ZOOM_KEYS = {
    0x2B: "zoom_in", 0x3D: "zoom_in", 0xFFAB: "zoom_in",  # plus, equal, KP_Add
    0x2D: "zoom_out", 0xFFAD: "zoom_out",  # minus, KP_Subtract
    0x30: "zoom_reset", 0xFFB0: "zoom_reset", 0xFF9E: "zoom_reset",  # 0, KP_0, KP_Insert
}


def prepare_qt():
    """Call before OpenCV opens its first window.  Qt then reads the mouse
    through core X events only, so scrolling reaches it solely as wheel-button
    presses - which LiveWindow grabs.  It costs Qt nothing the live view uses
    (XInput2 adds touch, tablets and smooth scrolling)."""
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_XCB_NO_XI2", "1")


class _Cookie(ctypes.Structure):
    _fields_ = [("sequence", ctypes.c_uint)]


class _ScreenIter(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("rem", ctypes.c_int), ("index", ctypes.c_int)]


_xcb_lib = None


def _xcb():
    """libxcb with the signatures used here, or None off X11."""
    global _xcb_lib
    if _xcb_lib is not None:
        return _xcb_lib or None
    _xcb_lib = False
    if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
        return None
    path = ctypes.util.find_library("xcb")
    if not path:
        return None
    lib = ctypes.cdll.LoadLibrary(path)
    vp, u8, u16, u32, cookie = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                                ctypes.c_uint32, _Cookie)
    for name, res, args in (
        ("xcb_connect", vp, [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]),
        ("xcb_connection_has_error", ctypes.c_int, [vp]),
        ("xcb_disconnect", None, [vp]),
        ("xcb_flush", ctypes.c_int, [vp]),
        ("xcb_get_setup", vp, [vp]),
        ("xcb_setup_roots_iterator", _ScreenIter, [vp]),
        ("xcb_screen_next", None, [ctypes.POINTER(_ScreenIter)]),
        ("xcb_query_tree", cookie, [vp, u32]),
        ("xcb_query_tree_reply", vp, [vp, cookie, vp]),
        ("xcb_query_tree_children", ctypes.POINTER(u32), [vp]),
        ("xcb_query_tree_children_length", ctypes.c_int, [vp]),
        ("xcb_intern_atom", cookie, [vp, u8, u16, ctypes.c_char_p]),
        ("xcb_intern_atom_reply", vp, [vp, cookie, vp]),
        ("xcb_get_property", cookie, [vp, u8, u32, u32, u32, u32, u32]),
        ("xcb_get_property_reply", vp, [vp, cookie, vp]),
        ("xcb_get_property_value", vp, [vp]),
        ("xcb_get_property_value_length", ctypes.c_int, [vp]),
        ("xcb_change_property_checked", cookie, [vp, u8, u32, u32, u32, u8, u32, vp]),
        ("xcb_grab_button_checked", cookie, [vp, u8, u32, u16, u8, u8, u32, u32, u8, u16]),
        ("xcb_grab_key_checked", cookie, [vp, u8, u32, u16, u8, u8, u8]),
        ("xcb_get_keyboard_mapping", cookie, [vp, u8, u8]),
        ("xcb_get_keyboard_mapping_reply", vp, [vp, cookie, vp]),
        ("xcb_get_keyboard_mapping_keysyms", ctypes.POINTER(u32), [vp]),
        ("xcb_get_keyboard_mapping_keysyms_length", ctypes.c_int, [vp]),
        ("xcb_request_check", vp, [vp, cookie]),
        ("xcb_poll_for_event", vp, [vp]),
    ):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.free.argtypes = [vp]
    lib.free = libc.free
    _xcb_lib = lib
    return lib


class LiveWindow:
    """Our own X connection to the live view's window.  Call `attach()` each
    frame until it returns True (the window exists once Qt has shown it),
    then `poll()` each frame for the wheel and zoom-key events it took."""

    def __init__(self, title: str):
        self.title = title.encode()
        self.window: int | None = None
        self.wheel_grabbed = False
        self._keys: dict[int, str] = {}  # keycode -> action
        self._conn = None
        self._x = None
        self._done = False

    # ------------------------------------------------------------------ #
    def attach(self) -> bool:
        """True once there is nothing more to do: the window was found and
        set up, or this is not X11.  Never raises."""
        if self._done:
            return True
        try:
            if self._conn is None and not self._connect():
                self._done = True
                return True
            window = self._find()
            if window is None:
                return False
            self.window = window
            self._claim(window)
            self.wheel_grabbed = self._grab_wheel(window)
            self._grab_zoom_keys(window)
            self._x.xcb_flush(self._conn)
            if self.wheel_grabbed:
                logging.info(
                    "live window: dock identity set; mouse wheel and Ctrl +/-/0 taken "
                    f"from Qt (XInput2 for Qt: "
                    f"{'off' if os.environ.get('QT_XCB_NO_XI2') else 'ON - smooth scroll may still magnify'})")
            else:
                logging.warning("could not take the mouse wheel from the Qt viewer; "
                                "scrolling will also magnify the window")
        except Exception as e:  # a cosmetic fix must never disturb the app
            logging.debug(f"live window setup failed: {e}")
        self._done = True
        return True

    def poll(self) -> list[tuple[str, int | str, int]]:
        """Events taken from Qt since the last call: ("wheel", button, state)
        with button 4-7 = up/down/left/right, or ("key", action, state)."""
        if self._conn is None or not self.window:
            return []
        out = []
        x = self._x
        try:
            if x.xcb_connection_has_error(self._conn):
                self.close()
                return []
            while True:
                ev = x.xcb_poll_for_event(self._conn)
                if not ev:
                    break
                raw = ctypes.string_at(ev, 32)
                x.free(ev)
                decoded = decode_event(raw, self._keys)
                if decoded is not None:
                    out.append(decoded)
        except Exception as e:
            logging.debug(f"live window poll failed: {e}")
        return out

    def close(self):
        if self._conn is not None:
            try:
                self._x.xcb_disconnect(self._conn)
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------------ #
    def _connect(self) -> bool:
        self._x = _xcb()
        if self._x is None:
            return False
        screen = ctypes.c_int(0)
        conn = self._x.xcb_connect(None, ctypes.byref(screen))
        if not conn or self._x.xcb_connection_has_error(conn):
            if conn:
                self._x.xcb_disconnect(conn)
            return False
        self._conn = conn
        it = self._x.xcb_setup_roots_iterator(self._x.xcb_get_setup(conn))
        for _ in range(screen.value):
            self._x.xcb_screen_next(ctypes.byref(it))
        self._root = ctypes.cast(it.data, ctypes.POINTER(ctypes.c_uint32))[0]
        self._atoms = {name: self._atom(name) for name in
                       (b"_NET_WM_NAME", b"_NET_WM_PID", b"_NET_WM_ICON")}
        return True

    def _atom(self, name: bytes) -> int:
        x = self._x
        reply = x.xcb_intern_atom_reply(self._conn, x.xcb_intern_atom(self._conn, 0, len(name), name), None)
        if not reply:
            return 0
        atom = ctypes.cast(reply, ctypes.POINTER(ctypes.c_uint32))[2]  # after type/pad/seq/length
        x.free(reply)
        return atom

    def _children(self, window: int) -> list[int]:
        x = self._x
        reply = x.xcb_query_tree_reply(self._conn, x.xcb_query_tree(self._conn, window), None)
        if not reply:
            return []
        try:
            ptr = x.xcb_query_tree_children(reply)
            return [ptr[i] for i in range(x.xcb_query_tree_children_length(reply))]
        finally:
            x.free(reply)

    def _property(self, window: int, prop: int, length: int = 256) -> bytes | None:
        if not prop:
            return None
        x = self._x
        cookie = x.xcb_get_property(self._conn, 0, window, prop, 0, 0, length)  # type: any
        reply = x.xcb_get_property_reply(self._conn, cookie, None)
        if not reply:
            return None
        try:
            n = x.xcb_get_property_value_length(reply)  # bytes
            return ctypes.string_at(x.xcb_get_property_value(reply), n) if n else b""
        finally:
            x.free(reply)

    def _find(self) -> int | None:
        """Our process's top-level window called `title`.  A window manager
        reparents clients into frames, so look one level below the root's
        children too."""
        for top in self._children(self._root):
            for window in (top, *self._children(top)):
                name = (self._property(window, self._atoms[b"_NET_WM_NAME"])
                        or self._property(window, _ATOM_WM_NAME))
                if name != self.title:
                    continue
                pid = self._property(window, self._atoms[b"_NET_WM_PID"], 1)
                if pid and len(pid) >= 4 and int.from_bytes(pid[:4], sys.byteorder) != os.getpid():
                    continue  # another instance's window
                return window
        return None

    def _check(self, cookie) -> bool:
        err = self._x.xcb_request_check(self._conn, cookie)
        if err:
            self._x.free(err)
            return False
        return True

    def _claim(self, window: int):
        current = self._property(window, _ATOM_WM_CLASS) or b""
        if current.split(b"\0")[1:2] not in ([b""], []):
            return  # someone gave it a class already; leave it alone
        value = f"{WM_INSTANCE}\0{WM_CLASS}\0".encode()
        self._check(self._x.xcb_change_property_checked(
            self._conn, 0, window, _ATOM_WM_CLASS, _ATOM_STRING, 8, len(value), value))
        icon = icon_cardinals()
        if icon is not None and self._atoms[b"_NET_WM_ICON"]:
            self._check(self._x.xcb_change_property_checked(
                self._conn, 0, window, self._atoms[b"_NET_WM_ICON"], _ATOM_CARDINAL, 32,
                len(icon), icon.ctypes.data_as(ctypes.c_void_p)))

    def _grab_wheel(self, window: int) -> bool:
        x = self._x
        return all([  # a list, not a generator: try every button even if one fails
            self._check(x.xcb_grab_button_checked(
                self._conn, 0, window, _EVENT_MASK_BUTTONS, _GRAB_ASYNC, _GRAB_ASYNC,
                0, 0, button, _MOD_ANY))
            for button in WHEEL_BUTTONS
        ])

    def _grab_zoom_keys(self, window: int):
        x = self._x
        setup = ctypes.string_at(x.xcb_get_setup(self._conn), 36)
        first, last = setup[34], setup[35]  # min_keycode, max_keycode
        reply = x.xcb_get_keyboard_mapping_reply(
            self._conn, x.xcb_get_keyboard_mapping(self._conn, first, last - first + 1), None)
        if not reply:
            return
        try:
            per_code = ctypes.string_at(reply, 2)[1]  # keysyms_per_keycode
            syms = x.xcb_get_keyboard_mapping_keysyms(reply)
            n = x.xcb_get_keyboard_mapping_keysyms_length(reply)
            for i in range(n):
                action = _ZOOM_KEYS.get(syms[i])
                if action and per_code:
                    self._keys.setdefault(first + i // per_code, action)
        finally:
            x.free(reply)
        # Ctrl, whatever the state of Shift, Caps Lock and Num Lock
        mods = [CONTROL | s | c | n for s in (0, SHIFT) for c in (0, LOCK) for n in (0, NUMLOCK)]
        for keycode in self._keys:
            for mod in mods:
                self._check(x.xcb_grab_key_checked(
                    self._conn, 0, window, mod, keycode, _GRAB_ASYNC, _GRAB_ASYNC))


def decode_event(raw: bytes, keys: dict[int, str]) -> tuple[str, int | str, int] | None:
    """One 32-byte X event -> ("wheel", button, state) / ("key", action, state),
    or None for anything else (releases, mapping changes)."""
    kind = raw[0] & 0x7F
    detail = raw[1]
    state = int.from_bytes(raw[28:30], "little")
    if kind == _BUTTON_PRESS and detail in WHEEL_BUTTONS:
        return ("wheel", detail, state)
    if kind == _KEY_PRESS and detail in keys:
        return ("key", keys[detail], state)
    return None


def icon_cardinals() -> np.ndarray | None:
    """_NET_WM_ICON payload as uint32 (xcb's format-32 unit): for each size,
    width, height, then ARGB pixels."""
    import cv2

    from . import appicon

    parts = []
    for size in sorted(appicon.PNG):
        png = np.frombuffer(base64.b64decode(appicon.PNG[size]), np.uint8)
        bgra = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            continue
        b, g, r, a = (bgra[..., i].astype(np.uint32) for i in range(4))
        parts += [np.array([bgra.shape[1], bgra.shape[0]], np.uint32),
                  ((a << 24) | (r << 16) | (g << 8) | b).ravel()]
    return np.ascontiguousarray(np.concatenate(parts)) if parts else None


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
        if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
            return None
        path = ctypes.util.find_library("X11")
        if not path:
            return None
        x = ctypes.cdll.LoadLibrary(path)
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
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
