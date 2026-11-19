"""Which cameras exist, which one is in use, and what changed since last frame.

A background thread enumerates every backend every couple of seconds and
publishes the result; the main thread merges it in `poll()`, which turns the
difference into events: a camera appeared, went away, or came back.  Opening
runs on a worker thread so a slow GigE connect never freezes the window;
grabbing and closing stay on the main thread.

The Cameras window and the HTTP server never touch a `Camera`: they read
`snapshot()` and `post()` requests, which the main loop collects with
`take_requests()`.  Every mutation of an entry happens on the main thread,
under `_lock` so that a snapshot taken from another thread is consistent.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from .. import cameraconfig
from ..cameraconfig import CameraSettings
from .base import Backend, Camera, DeviceInfo

ABSENT_AFTER = 2  # scans in a row without a device before it counts as unplugged
ERROR_HOLD_S = 10.0  # a camera that would not open is shown as such this long, then retried

# Entry states, in the words the UI uses for them.
STATES = {
    "active": "Live",
    "connecting": "Connecting...",
    "available": "Ready",
    "lost": "Disconnected",
    "absent": "Unplugged",
    "error": "Error",
}


@dataclass
class Entry:
    info: DeviceInfo
    backend: Backend
    order: int
    settings: CameraSettings = field(default_factory=CameraSettings)
    present: bool = True
    missed: int = 0
    state: str = "available"
    error: str = ""
    sensor: tuple[int, int] | None = None
    lost_at: float = 0.0
    error_at: float = 0.0


class CameraManager:
    def __init__(
        self,
        backends: list[Backend],
        mode: str = "16Bit",
        scan_interval: float | None = 2.0,
    ):
        """`backends` in priority order: when two find the same serial (a
        Basler seen by both pylon and a GenTL producer), the first one wins.
        `scan_interval=None` disables the background thread - tests call
        `scan_now()` themselves."""
        self.mode = mode
        self.scan_interval = scan_interval
        self.backends = list(backends)
        self.status: dict[str, tuple[bool, str]] = {}
        for b in self.backends:
            try:
                self.status[b.name] = b.available()
            except Exception as e:
                self.status[b.name] = (False, str(e))
        self._usable = [b for b in self.backends if self.status[b.name][0]]
        self._priority = {b.name: i for i, b in enumerate(self.backends)}

        self._lock = threading.Lock()
        self._entries: dict[str, Entry] = {}
        self._latest: tuple[float, list[DeviceInfo]] | None = None
        self._applied = -1.0
        self._by_backend: dict[str, list[DeviceInfo]] = {}
        self._errors: dict[str, str] = {}
        self._requests: queue.SimpleQueue = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-open")
        self.active_uid: str | None = None

    # ------------------------------ scanning ------------------------------ #
    def start(self):
        """Scan once synchronously (so the first frame has a camera list), then
        keep scanning in the background."""
        self.scan_now(full=True)
        self.poll()
        if self.scan_interval and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="camera-scan", daemon=True)
            self._thread.start()

    def rescan(self):
        """Ask for an immediate, thorough scan (the Rescan button)."""
        if self._thread is None:
            self.scan_now(full=True)
        else:
            self._wake.set()

    def _run(self):
        while not self._stop.is_set():
            full = self._wake.wait(self.scan_interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            self.scan_now(full=full)

    def scan_now(self, full: bool = False):
        started = time.monotonic()
        found, serials = [], set()
        for backend in self._usable:
            try:
                devices = backend.enumerate(full)
                self._by_backend[backend.name] = devices
                self._errors.pop(backend.name, None)
            except Exception as e:
                # keep the previous list: one failed scan is not an unplug
                devices = self._by_backend.get(backend.name, [])
                if self._errors.get(backend.name) != str(e):
                    self._errors[backend.name] = str(e)
                    logging.warning(f"{backend.title}: enumeration failed: {e}")
            for d in devices:
                if d.serial and d.serial in serials:
                    continue
                serials.add(d.serial)
                found.append(d)
        with self._lock:
            self._latest = (started, found)

    def poll(self) -> list[tuple[str, str]]:
        """Merge the newest scan; return (event, uid) pairs, event being
        'added', 'removed', 'returned' (a lost camera is back) or 'ready' (a
        camera that would not open is worth another try)."""
        with self._lock:
            if self._latest is None or self._latest[0] <= self._applied:
                return []
            started, devices = self._latest
            self._applied = started
            events: list[tuple[str, str]] = []
            seen = {d.uid: d for d in devices}
            for uid, info in seen.items():
                entry = self._entries.get(uid)
                if entry is None:
                    if info.serial and any(  # a Basler also seen through a GenTL
                        e.info.serial == info.serial and e.info.backend != info.backend
                        for e in self._entries.values()  # producer stays one entry
                    ):
                        continue
                    self._entries[uid] = Entry(
                        info=info,
                        backend=self._backend(info.backend),
                        order=len(self._entries),
                        settings=_resolve(info),
                    )
                    events.append(("added", uid))
                    continue
                entry.info, entry.missed = info, 0
                if entry.present:
                    if entry.state == "error" and started - entry.error_at > ERROR_HOLD_S:
                        entry.state = "available"
                        events.append(("ready", uid))
                    continue
                if entry.state == "lost":
                    if started < entry.lost_at:  # this scan predates the loss
                        continue
                    entry.present = True
                    events.append(("returned", uid))
                else:
                    entry.present, entry.state, entry.error = True, "available", ""
                    events.append(("added", uid))
            for uid, entry in self._entries.items():
                if uid in seen or not entry.present or entry.state in ("active", "connecting"):
                    continue  # the grab path, not a scan, decides that a live camera is gone
                entry.missed += 1
                if entry.missed >= ABSENT_AFTER:
                    entry.present = False
                    if entry.state != "lost":
                        entry.state, entry.error = "absent", ""
                    events.append(("removed", uid))
            return events

    def _backend(self, name: str) -> Backend:
        return next(b for b in self.backends if b.name == name)

    # ------------------------------ opening ------------------------------- #
    def open_async(self, uid: str) -> Future:
        """Open on the worker thread; the Future resolves to the Camera."""
        entry = self._entries[uid]
        self.set_state(uid, "connecting")
        return self._executor.submit(entry.backend.open, entry.info, self.mode)

    def open_sync(self, uid: str) -> Camera:
        return self.open_async(uid).result()

    def set_active(self, uid: str, camera: Camera):
        with self._lock:
            old = self._entries.get(self.active_uid) if self.active_uid != uid else None
            if old is not None and old.state == "active":
                old.state = "available"
            entry = self._entries[uid]
            entry.state, entry.error, entry.present = "active", "", True
            entry.sensor = (camera.W, camera.H)
            entry.settings = camera.settings  # what is in force, fallbacks included
            self.active_uid = uid

    def release(self, uid: str, camera: Camera | None):
        """Close a camera the app has stopped using."""
        if camera is not None:
            camera.close()
        with self._lock:
            entry = self._entries.get(uid)
            if entry is None:
                return
            if entry.state == "lost":  # gone and no longer waited for
                entry.state, entry.present = "absent", False
            elif entry.state in ("active", "connecting"):
                entry.state = "available"
            if self.active_uid == uid:
                self.active_uid = None

    def mark_lost(self, uid: str, reason: str):
        """The open camera stopped answering.  It counts as absent until a scan
        that started after this moment sees it again."""
        with self._lock:
            entry = self._entries.get(uid)
            if entry is not None:
                entry.state, entry.error = "lost", reason
                entry.present, entry.lost_at = False, time.monotonic()

    def set_state(self, uid: str, state: str, error: str = ""):
        with self._lock:
            entry = self._entries.get(uid)
            if entry is not None:
                entry.state, entry.error = state, error
                if state == "error":
                    entry.error_at = time.monotonic()

    def refresh_settings(self):
        """Re-resolve labels and pixel sizes after camera_config.yaml changed."""
        with self._lock:
            for entry in self._entries.values():
                entry.settings = _resolve(entry.info)

    # ------------------------------ queries ------------------------------- #
    def entry(self, uid: str | None) -> Entry | None:
        return self._entries.get(uid) if uid else None

    def label(self, uid: str) -> str:
        entry = self._entries.get(uid)
        if entry is None:
            return uid
        return entry.settings.label or entry.info.model

    def uids(self) -> list[str]:
        with self._lock:
            return [uid for uid, _ in self._sorted()]

    def _sorted(self):
        return sorted(
            self._entries.items(),
            key=lambda kv: (self._priority.get(kv[1].info.backend, 99), kv[1].order),
        )

    def find(self, query: str) -> str | None:
        """A uid from what a user would type: the uid itself, a serial, a
        label or a model name (case-insensitive, exact before substring)."""
        q = query.strip().lower()
        if not q:
            return None
        with self._lock:
            items = self._sorted()
        for uid, e in items:
            if q in (uid.lower(), e.info.serial.lower(), e.info.model.lower(),
                     e.settings.label.lower()):
                return uid
        for uid, e in items:
            if q in e.info.model.lower() or (e.settings.label and q in e.settings.label.lower()):
                return uid
        return None

    def snapshot(self) -> list[dict]:
        """Plain dicts for the UI and the HTTP API; safe from any thread."""
        with self._lock:
            out = []
            for uid, e in self._sorted():
                out.append({
                    "uid": uid,
                    "backend": e.info.backend,
                    "backend_title": e.backend.title,
                    "model": e.info.model,
                    "serial": e.info.serial,
                    "vendor": e.info.vendor,
                    "address": e.info.address,
                    "label": e.settings.label or e.info.model,
                    "configured_label": e.settings.label,
                    "state": e.state,
                    "state_text": STATES.get(e.state, e.state),
                    "error": e.error,
                    "present": e.present,
                    "active": uid == self.active_uid,
                    "auto_open": e.backend.auto_open,
                    "sensor": list(e.sensor) if e.sensor else None,
                    "pixel_size": e.settings.pixel_size,
                    "pixel_size_known": e.settings.pixel_size_known,
                })
            return out

    def backend_status(self) -> list[dict]:
        return [
            {
                "name": b.name,
                "title": b.title,
                "available": self.status[b.name][0],
                "detail": self.status[b.name][1] or self._errors.get(b.name, ""),
                "install_hint": b.install_hint,
            }
            for b in self.backends
        ]

    # ------------------------------ requests ------------------------------ #
    def post(self, request: tuple):
        """From any thread: ('switch', uid) | ('next',) | ('rescan',) |
        ('reload_config',).  Handled on the main loop's next frame."""
        self._requests.put(request)

    def take_requests(self) -> list[tuple]:
        out = []
        while True:
            try:
                out.append(self._requests.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------ shutdown ------------------------------ #
    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._executor.shutdown(wait=False, cancel_futures=True)
        for b in self._usable:
            try:
                b.close()
            except Exception as e:
                logging.debug(f"{b.title} close: {e}")


def _resolve(info: DeviceInfo) -> CameraSettings:
    try:
        return cameraconfig.resolve(info.model, info.serial, warn=False)
    except Exception:
        return CameraSettings()
