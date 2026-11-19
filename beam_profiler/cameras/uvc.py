"""USB video-class cameras (webcams, USB microscopes, cheap beam cameras)
through OpenCV, which is already a dependency - so this backend always works.

These cameras are 8-bit, colour as often as not, and their exposure control is
whatever the OS driver exposes, so treat them as a convenience rather than a
measurement instrument: frames are converted to mono, 16-bit mode shifts the
8 bits to the top of a uint16 (as the other backends do for Mono12), and the
ROI is cropped in software.  `pixel_size` still comes from camera_config.yaml,
keyed by the name the OS reports (`--list-cameras` prints it).

They are never opened on their own (`auto_open = False`): a laptop webcam must
not replace the profiler camera because it happened to be found first.

Enumeration on Linux reads /sys/class/video4linux, which is free and sees hot
plugging.  Elsewhere there is no cheap way to list them without opening each
index, so all indices are tried only at start-up and on Rescan; in between,
only cameras this app has closed are re-tried, so one that was unplugged is
seen to be gone - and seen again when it comes back.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading

import cv2
import numpy as np

from .. import cameraconfig
from .base import Backend, Camera, DeviceInfo

SYSFS = "/sys/class/video4linux"
MAX_PROBE_INDEX = 4  # indices tried on platforms without sysfs


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def _api() -> int:
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform == "win32":
        return cv2.CAP_DSHOW  # MSMF misbehaves when opened on one thread and read on another
    return cv2.CAP_ANY


class UVCBackend(Backend):
    name = "uvc"
    title = "USB video (UVC)"
    auto_open = False

    def __init__(self, sysfs: str = SYSFS, use_sysfs: bool | None = None):
        self.sysfs = sysfs
        self.use_sysfs = sys.platform.startswith("linux") if use_sysfs is None else use_sysfs
        self._lock = threading.Lock()
        self._probed: list[DeviceInfo] = []
        self._open: set[str] = set()  # addresses we hold, never probed
        self._watch: set[str] = set()  # closed by us: re-tried on every scan

    def enumerate(self, full: bool = False) -> list[DeviceInfo]:
        if self.use_sysfs:
            return self._enumerate_sysfs()  # no sysfs folder = no video devices
        if full:
            found = self._probe_indices()
            with self._lock:
                self._probed, self._watch = found, set()
        else:
            for address in list(self._watch):
                if address in self._open or not self._probe(int(address)):
                    continue
                with self._lock:
                    self._watch.discard(address)
                    self._probed.append(self._index_info(int(address)))
        with self._lock:
            return list(self._probed)

    def _closed(self, address: str):
        self._open.discard(address)
        if not self.use_sysfs:  # gone from the list until a probe finds it again
            with self._lock:
                self._probed = [d for d in self._probed if d.address != address]
                self._watch.add(address)

    def _enumerate_sysfs(self) -> list[DeviceInfo]:
        """Capture nodes only: a UVC camera registers a second /dev/video node
        for metadata, which has index 1."""
        out = []
        try:
            nodes = sorted(os.listdir(self.sysfs), key=lambda n: (len(n), n))
        except OSError:
            return out
        for node in nodes:
            base = os.path.join(self.sysfs, node)
            if not node.startswith("video") or _read(os.path.join(base, "index")) not in ("", "0"):
                continue
            usb = os.path.join(base, "device", "..")  # the USB device above the interface
            name = _read(os.path.join(base, "name")) or node
            out.append(DeviceInfo(
                backend=self.name,
                model=name.split(":")[0].strip() or name,
                serial=_read(os.path.join(usb, "serial")) or node,
                vendor=_read(os.path.join(usb, "manufacturer")),
                address=f"/dev/{node}",
            ))
        return out

    def _probe_indices(self) -> list[DeviceInfo]:
        out = []
        try:  # an empty index is expected here; keep OpenCV from warning about it
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass
        for i in range(MAX_PROBE_INDEX):
            if str(i) in self._open or self._probe(i):
                out.append(self._index_info(i))
        return out

    def _index_info(self, i: int) -> DeviceInfo:
        return DeviceInfo(self.name, f"USB camera {i}", f"index{i}", "", str(i))

    @staticmethod
    def _probe(i: int) -> bool:
        cap = cv2.VideoCapture(i, _api())
        try:
            return bool(cap.isOpened())
        finally:
            cap.release()

    def open(self, info: DeviceInfo, mode: str) -> Camera:
        cam = UVCCamera(mode, info, on_close=lambda: self._closed(info.address))
        self._open.add(info.address)
        return cam

    def probe(self, info: DeviceInfo) -> dict:
        out = {**super().probe(info), "backend": self.name}
        try:
            cam = UVCCamera("8Bit", info)
        except Exception as e:
            return {**out, "error": str(e)}
        try:
            out["sensor_size"] = (cam.W, cam.H)
        finally:
            cam.close()
        return out


class UVCCamera(Camera):
    backend = "uvc"

    def __init__(self, mode: str, info: DeviceInfo, on_close=None):
        self.model_name, self.serial, self.device = info.model, info.serial, info
        self._on_close = on_close
        source = int(info.address) if info.address.isdigit() else info.address
        self.cap = cv2.VideoCapture(source, _api())
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"could not open {info.address} (in use by another program?)")
        # Ask for far more than any webcam has; the driver settles on its largest mode.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError(f"{info.address} opened but delivers no frames")
        self.H, self.W = frame.shape[:2]
        self._roi = (self.W, self.H, 0, 0)
        self._api_name = self.cap.getBackendName()

        settings = cameraconfig.resolve(self.model_name, self.serial)
        super().__init__(mode, settings)
        # Manual exposure: V4L2 calls it 1, DirectShow 0.25.
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if self._api_name == "V4L2" else 0.25)
        self._exposure = float(settings.default_exposure)
        self._gain = float(settings.default_gain)
        self.ExposureTime = settings.default_exposure
        self.Gain = settings.default_gain
        logging.info(f"Opened {self.model_name} at {info.address}: {self.W} x {self.H} px")

    # Exposure units are driver-specific: V4L2 counts 100 us steps,
    # DirectShow and MSMF take log2(seconds).
    def _to_driver(self, us: float) -> float:
        if self._api_name == "V4L2":
            return max(1.0, round(us / 100.0))
        return math.log2(max(us, 1.0) * 1e-6)

    @property
    def ExposureTime(self):
        return self._exposure

    @ExposureTime.setter
    def ExposureTime(self, value):
        value = float(np.clip(value, self.min_exposure, self.max_exposure))
        if not self.cap.set(cv2.CAP_PROP_EXPOSURE, self._to_driver(value)):
            logging.debug(f"{self.model_name}: driver ignored the exposure setting")
        self._exposure = value

    @property
    def Gain(self):
        return self._gain

    @Gain.setter
    def Gain(self, value):
        self.cap.set(cv2.CAP_PROP_GAIN, float(value))
        self._gain = float(value)

    @property
    def ROI(self):
        return self._roi

    @ROI.setter
    def ROI(self, value):
        w, h, ox, oy = (int(v) for v in value)
        w = max(10, min(w, self.W))
        h = max(10, min(h, self.H))
        self._roi = (w, h, max(0, min(ox, self.W - w)), max(0, min(oy, self.H - h)))

    def is_connected(self) -> bool:
        address = self.device.address if self.device else ""
        if address.startswith("/dev/") and not os.path.exists(address):
            return False
        return super().is_connected()

    def _grab(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("no frame from the camera")
        return self._convert(frame)

    def _convert(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w, h, ox, oy = self._roi
        frame = frame[oy : oy + h, ox : ox + w]
        if frame.dtype == np.uint16:  # a few drivers hand out 16-bit mono
            return frame if self.mode == "16Bit" else (frame >> 8).astype(np.uint8)
        frame = frame.astype(np.uint8, copy=False)
        if self.mode == "16Bit":
            return frame.astype(np.uint16) << 8  # MSB-aligned, like Mono12 in uint16
        return np.ascontiguousarray(frame)

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        if self._on_close is not None:
            self._on_close()
            self._on_close = None
