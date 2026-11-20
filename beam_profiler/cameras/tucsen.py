"""Tucsen scientific cameras (Dhyana, FL, Aries, Libra, ...) through Tucsen's
TUCam SDK, over USB 3.0 or CameraLink (Active Silicon FireBird card).

Built on the `tucsen` package from the tucsen-camera-driver repository
(github.com/tim4431/tucsen-camera-driver), which wraps libTUCam's C++ ABI in a
small C shim and fixes what the SDK gets wrong - above all, that
`TUCAM_Buf_WaitForFrame` ignores its timeout, so a grab here is bounded by the
package's watchdog instead of hanging the main loop.  Setup, in that
repository: `sudo ./linux-setup/install_tucsen_linux.sh`, then
`.venv/bin/pip install <repo>/python` here (or `./setup.sh --tucsen=<repo>`).

Discovery.  The SDK counts cameras only in `Api_Init`, which takes a second,
prints a screenful of probe messages and must not run while a camera is open.
So it is asked at start-up, on Rescan, after an open fails, and when the set of
Tucsen devices on the USB bus changes - read from sysfs (vendor 0x5453), which
is free and sees hot-plugging.  While one of these cameras is open, the last
list is repeated.  CameraLink has no hot-plug signal: a CameraLink camera that
drops out gets one reopen attempt, then needs Rescan.

Identity.  The serial number is only readable after the ~5 s open, and not at
all over CameraLink, so devices are listed by SDK index; `serials:` entries in
camera_config.yaml still apply, because the opened camera resolves its
settings with the serial it reads.

Mapped onto the app:

* Exposure in us (the SDK takes ms; 10.32 us steps on the Dhyana 4040 V2).
* Gain is the gain *mode* (GLOBALGAIN), e.g. 0 HDR, 1 High gain, 2 Low gain on
  the Dhyana 4040 V2.  Every mode delivers full-scale 16-bit data, used as is;
  8-bit mode keeps the top byte.
* ROI offsets and sizes go in multiples of 4 and are read back, so the app
  always works with what the SDK really applied (it widens anything under 48
  px, among others).  Offsets are in the delivered image's own frame (the SDK
  flips vertically by default), so sensor coordinates are full-frame image
  coordinates.
* Binning is reset to 1x1 at open: pixel_size is per sensor pixel.
* Software trigger unless `trigger_mode: "off"`; I/O lines are not driven.
* Unplugging: over USB, sysfs decides (a timeout is never a disconnect); over
  CameraLink, a run of failed grabs does.
"""

from __future__ import annotations

import logging
import math
import os
import threading

import numpy as np

# The tucsen-camera-driver package - a top-level import, not this module.
import tucsen as tucam

from .. import cameraconfig
from .base import Backend, Camera, DeviceInfo

USB_SYSFS = "/sys/bus/usb/devices"
USB_DEVFS = "/dev/bus/usb"  # the device nodes libusb opens, <bus>/<device>
PROC = "/proc"
USB_VENDOR = "5453"  # Tucsen's USB vendor id
USB_RETRIES = 3  # scans spent waiting for a new USB device to reach the SDK
ROI_STEP = 4  # the SDK wants ROI offsets and sizes in multiples of 4
MIN_ROI = 48  # the Dhyana 4040 V2 widens any narrower ROI to this


def _read(path: str) -> str:
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def _listdir(path: str) -> list[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _usb_ports(sysfs: str) -> list[str]:
    """sysfs names ('10-2') of the Tucsen devices on the USB bus."""
    return sorted(
        name for name in _listdir(sysfs)
        if ":" not in name  # interfaces, not devices
        and _read(os.path.join(sysfs, name, "idVendor")).lower() == USB_VENDOR
    )


def usb_devices(sysfs: str = USB_SYSFS) -> frozenset[str]:
    """Tucsen devices on the USB bus now, as 'port#devnum' (the device number
    changes on every replug, so a quick unplug-replug is still a change)."""
    return frozenset(
        f"{port}#{_read(os.path.join(sysfs, port, 'devnum'))}" for port in _usb_ports(sysfs)
    )


def usb_holder(sysfs: str = USB_SYSFS) -> str:
    """Who has the Tucsen USB camera, for an open that failed - '' unless every
    Tucsen device this process does not hold is claimed by another program.

    The SDK reports only TUCAMRET_FAILOPEN_CAMERA (and prints "Error in
    claiming interface") when another process has the camera - most often a
    second copy of this app, started while the first was still opening.  A
    claimed interface is bound to `usbfs`; the claimant is the process with
    the device node open (only this user's processes are visible)."""
    holders, claimed = [], False
    for port in _usb_ports(sysfs):
        bus, dev = (_read(os.path.join(sysfs, port, f)) for f in ("busnum", "devnum"))
        node = f"{USB_DEVFS}/{int(bus):03d}/{int(dev):03d}" if bus.isdigit() and dev.isdigit() else ""
        pids = _pids_using(node) if node else []
        if os.getpid() in pids:
            continue  # one this process has open: not the one that failed
        drivers = [
            os.path.basename(os.path.realpath(os.path.join(sysfs, name, "driver")))
            for name in _listdir(sysfs) if name.startswith(port + ":")
        ]
        if "usbfs" not in drivers:
            return ""  # a free Tucsen: the open failed for another reason
        claimed = True
        holders += [p for p in pids if p not in holders]
    if not claimed:
        return ""
    return ", ".join(_program(pid) for pid in holders) or "another program"


def _pids_using(node: str) -> list[int]:
    pids = []
    for pid in _listdir(PROC):
        if not pid.isdigit():
            continue
        fds = os.path.join(PROC, pid, "fd")
        for fd in _listdir(fds):
            try:
                if os.readlink(os.path.join(fds, fd)) == node:
                    pids.append(int(pid))
                    break
            except OSError:
                continue
    return sorted(pids)


def _program(pid: int) -> str:
    argv = [a for a in _read(os.path.join(PROC, str(pid), "cmdline")).split("\0") if a]
    cmd = " ".join([os.path.basename(argv[0]), *argv[1:]]) if argv else ""
    if "beam_profiler" in cmd or "pylon_camera" in cmd:
        return f"another copy of the beam profiler (PID {pid})"
    return f"{cmd[:60]} (PID {pid})" if cmd else f"PID {pid}"


def _align(value: int) -> int:
    return value - value % ROI_STEP


class TucsenBackend(Backend):
    name = "tucsen"
    title = "Tucsen (TUCam)"
    install_hint = (
        "install libTUCam (tucsen-camera-driver: sudo ./linux-setup/install_tucsen_linux.sh), "
        "then `pip install <tucsen-camera-driver>/python` into this venv"
    )

    def __init__(self, sysfs: str = USB_SYSFS):
        self.sysfs = sysfs
        self._lock = threading.Lock()
        self._open = 0  # cameras held or being opened: the SDK must not be re-initialised
        self._cached: list[DeviceInfo] = []
        self._bus: dict[str, str] = {}  # address -> bus, from the last SDK scan
        self._usb_seen: frozenset[str] | None = None
        self._retries = 0
        self._stale = False  # an open failed: the list may be out of date

    def available(self) -> tuple[bool, str]:
        return True, f"tucsen {getattr(tucam, '__version__', '?')}"

    def enumerate(self, full: bool = False) -> list[DeviceInfo]:
        if not self._lock.acquire(blocking=False):  # an open is starting
            return list(self._cached)
        try:
            if self._open:
                return list(self._cached)
            usb = usb_devices(self.sysfs)
            if usb != self._usb_seen:
                self._usb_seen, self._retries = usb, USB_RETRIES
            if full or self._retries or self._stale:
                self._retries = max(0, self._retries - 1)
                self._stale = False
                self._scan()
                # Right after plugging in, the SDK may not see the device yet
                # (udev still setting permissions): look again next scan.
                if sum(b.startswith("USB") for b in self._bus.values()) >= len(usb):
                    self._retries = 0
            return list(self._cached)
        finally:
            self._lock.release()

    def _scan(self):
        try:
            found = tucam.list_cameras()
        except tucam.TucamError as e:  # no camera is a list, not a failure
            if getattr(e, "name", "") != "TUCAMRET_NO_CAMERA":
                raise
            found = []
        self._cached = [
            DeviceInfo(self.name, d["model"], "", "Tucsen", str(d["index"])) for d in found
        ]
        self._bus = {str(d["index"]): d["bus"] for d in found}

    def open(self, info: DeviceInfo, mode: str) -> Camera:
        with self._lock:
            self._open += 1
            bus = self._bus.get(info.address, "")
        try:
            return TucsenCamera(mode, info, bus, self.sysfs, on_close=self._released)
        except Exception:
            with self._lock:
                self._stale = True
            self._released()
            raise

    def _released(self):
        with self._lock:
            self._open = max(0, self._open - 1)

    def probe(self, info: DeviceInfo) -> dict:
        out = {**super().probe(info), "backend": self.name}
        try:
            cam = self.open(info, "16Bit")
        except Exception as e:
            return {**out, "error": str(e)}
        try:
            out.update(
                serial=cam.serial,
                bus=cam.bus,
                sensor_size=(cam.W, cam.H),
                exposure_range=cam.exposure_range,
                lines=[],  # none driven: the stub nulls exposure/user lines
            )
        finally:
            cam.close()
        return out


class TucsenCamera(Camera):
    backend = "tucsen"

    def __init__(self, mode: str, info: DeviceInfo, bus: str = "", sysfs: str = USB_SYSFS,
                 on_close=None):
        self.device = info
        self._sysfs = sysfs
        self._on_close = None  # a failed open is released by the backend, once
        try:
            self.cam = tucam.Camera(int(info.address))
        except tucam.TucamError as e:  # the SDK's code does not say "taken"; say by whom
            holder = usb_holder(sysfs) if not bus or bus.startswith("USB") else ""
            if holder:
                raise RuntimeError(f"in use by {holder}") from e
            raise
        try:
            self._setup(mode, info, bus)
        except Exception:
            self.close()
            raise
        self._on_close = on_close

    def _setup(self, mode: str, info: DeviceInfo, bus: str):
        cam = self.cam
        self.model_name = cam.model
        if info.model and self.model_name != info.model:
            raise RuntimeError(
                f"SDK camera {info.address} is now a {self.model_name}, not a {info.model}"
            )
        self.serial = cam.serial or ""
        self.bus = bus or cam.bus
        try:
            if cam.binning != 1:
                logging.info(f"{self.model_name}: binning {cam.binning}x reset to 1x1")
                cam.binning = 1
        except (tucam.TucamError, KeyError) as e:  # a model without binning modes
            logging.debug(f"{self.model_name}: binning not read: {e}")
        cam.roi = None  # full sensor, whatever the SDK's settings file restored
        _, _, self.W, self.H = cam.roi
        self._roi = (self.W, self.H, 0, 0)
        exp = cam.describe("EXPOSURETM")
        self._exp_ms = (float(exp["min"]), float(exp["max"]))
        self._exp_step_ms = float(exp.get("step") or 0.0)
        try:
            gain = cam.describe("GLOBALGAIN")
            self._gain_modes: tuple[int, int] | None = (int(gain["min"]), int(gain["max"]))
            self._gain = float(gain["value"])
        except (tucam.TucamError, KeyError):
            self._gain_modes, self._gain = None, 0.0
        self._exposure = float(exp["value"]) * 1000.0

        settings = cameraconfig.resolve(self.model_name, self.serial)
        super().__init__(mode, settings)
        logging.info(cameraconfig.describe(settings, self.model_name))
        self.Gain = settings.default_gain
        self.ExposureTime = settings.default_exposure
        self._start_capture()
        logging.info(
            f"Opened {self.model_name} ({self.serial or 'no serial'}) over {self.bus}: "
            f"{self.W} x {self.H} px, {'software trigger' if self.software_trigger else 'free-running'}"
        )

    # ------------------------------ setup -------------------------------- #
    @property
    def exposure_range(self) -> tuple[float, float]:
        """What the SDK accepts, in us."""
        return (self._exp_ms[0] * 1000.0, self._exp_ms[1] * 1000.0)

    def apply_settings(self, settings):
        """Keep the configured exposure window inside the SDK's, on every
        reload as well as at open."""
        super().apply_settings(settings)
        if not hasattr(self, "_exp_ms"):
            return
        lo, hi = self.exposure_range
        narrowed = (max(settings.min_exposure, lo), min(settings.max_exposure, hi))
        if narrowed[0] >= narrowed[1]:
            narrowed = (lo, hi)
        self.set_exposure_limits(*narrowed)

    def _start_capture(self):
        self.software_trigger = self.settings.trigger_mode != "off"
        if self.software_trigger:
            try:
                self.cam.start("software")
                return
            except tucam.TucamError as e:
                log = logging.error if self.settings.trigger_mode == "software" else logging.warning
                log(f"{self.model_name}: software trigger unavailable, free-running ({e})")
                self.software_trigger = False
        self.cam.start("sequence")

    # ------------------------------ properties --------------------------- #
    # Getters answer from the value last read back, like every backend: the
    # HUD reads them each frame, and they must survive an unplug.
    @property
    def ExposureTime(self):
        return self._exposure

    @ExposureTime.setter
    def ExposureTime(self, value):
        us = float(np.clip(value, self.min_exposure, self.max_exposure))
        # clip in ms too: the SDK's limits are not exact in binary (0.01032...)
        self.cam.exposure_ms = float(np.clip(self._on_grid(us), *self._exp_ms))
        self._exposure = self.cam.exposure_ms * 1000.0

    def _on_grid(self, us: float) -> float:
        """`us` as SDK milliseconds on its exposure step, inside the window.

        The SDK rounds to the nearest step (10.32 us), which swallows
        auto-exposure's 10% moves at short exposures - 31 us * 1.1 rounds back
        to 31 - so round towards the request instead: every change moves."""
        step = self._exp_step_ms
        if step <= 0:
            return us / 1000.0
        n = us / 1000.0 / step
        if abs(n - round(n)) < 1e-6:
            n = round(n)
        else:
            n = math.ceil(n) if us > self._exposure else math.floor(n)
        lo = math.ceil(self.min_exposure / 1000.0 / step - 1e-6)
        hi = math.floor(self.max_exposure / 1000.0 / step + 1e-6)
        if lo <= hi:
            n = min(max(n, lo), hi)
        return n * step

    @property
    def Gain(self):
        return self._gain

    @Gain.setter
    def Gain(self, value):
        if self._gain_modes is None:
            self._gain = float(value)
            return
        self.cam.set("GLOBALGAIN", int(np.clip(round(float(value)), *self._gain_modes)))
        self._gain = float(self.cam.get("GLOBALGAIN"))

    @property
    def ROI(self):
        return self._roi

    @ROI.setter
    def ROI(self, value):
        w, h, ox, oy = (int(v) for v in value)
        w = _align(max(MIN_ROI, min(w, self.W)))
        h = _align(max(MIN_ROI, min(h, self.H)))
        ox = _align(max(0, min(ox, self.W - w)))
        oy = _align(max(0, min(oy, self.H - h)))
        try:
            self.cam.roi = (ox, oy, w, h)  # pauses and restarts the capture
        except Exception as e:
            logging.error(f"Failed to set ROI: {e}")
        self._roi = self._read_roi()

    def _read_roi(self):
        try:
            x, y, w, h = self.cam.roi
            return (int(w), int(h), int(x), int(y))
        except Exception:  # unplugged mid-change: keep what we had
            return self._roi

    # ------------------------------ acquisition -------------------------- #
    def is_connected(self) -> bool:
        if self.bus.startswith("USB"):
            return bool(usb_devices(self._sysfs))
        return super().is_connected()

    def _grab(self) -> np.ndarray:
        if self.software_trigger:
            self.cam.software_trigger()
        # Long exposures must not read as missed frames: wait for the exposure
        # plus the configured margin.
        timeout = self.settings.grab_timeout_ms / 1000.0 + self._exposure / 1e6
        return self._convert(self.cam.grab(timeout=timeout))

    def _convert(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:  # colour models
            img = img.mean(axis=2).astype(img.dtype)
        if self.mode == "8Bit":
            return (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img
        if img.dtype == np.uint8:
            return img.astype(np.uint16) << 8  # MSB-aligned, like Mono12 in uint16
        return np.ascontiguousarray(img)

    def close(self):
        try:
            self.cam.close()  # stops capture; the SDK is uninitialised with the last camera
        except Exception as e:
            logging.debug(f"Tucsen close: {e}")
        if self._on_close is not None:
            self._on_close()
            self._on_close = None
