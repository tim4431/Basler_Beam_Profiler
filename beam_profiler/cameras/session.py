"""The camera the viewer is showing, and what to do when that changes.

`CameraSession` sits between the `CameraManager` (which devices exist) and the
viewer (what is on screen).  Once per frame `service()` turns manager events
and UI requests into actions:

* the live camera stops answering -> go offline, keep its settings, and reopen
  it as soon as a scan sees it again (ROI, exposure, rectangles restored);
* a camera is plugged in -> say so, and switch to it if the app is only
  showing the fallback simulator and the user has not chosen anything yet;
* the user picks a camera (Cameras window, `t`, HTTP API) -> open it on the
  worker thread while the old one keeps running, then swap.

The viewer supplies three callbacks: `capture()` returns its per-camera state
as a dict, `restore(camera, state_or_None)` applies one, and `notify(text,
level)` shows a message.  Nothing here touches OpenCV or Tk.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from typing import Callable

import numpy as np

from .. import cameraconfig
from .base import Camera, CameraDisconnected
from .manager import CameraManager

RETRY_S = 3.0  # between attempts to reopen a camera that is present but will not open


class CameraSession:
    def __init__(
        self,
        manager: CameraManager,
        capture: Callable[[], dict],
        restore: Callable[[Camera, dict | None], None],
        notify: Callable[[str, str], None] = lambda text, level: None,
    ):
        self.manager = manager
        self._capture = capture
        self._restore = restore
        self._notify = notify
        self.camera: Camera | None = None
        self.uid: str | None = None
        self.offline: str | None = None  # why there is no live picture, or None
        # Follow newly plugged cameras - only while the fallback simulator is
        # showing and the user has not chosen anything.
        self.auto_select = True
        self._states: dict[str, dict] = {}
        self._pending: tuple[str, Future, bool] | None = None  # (uid, future, quiet)
        self._target: str | None = None  # the camera the session should end up on
        self._restoring = False
        self._retry_at = 0.0
        self._last_error_log = 0.0

    # ------------------------------ start-up ------------------------------ #
    def start(self, prefer: str | None = None, simulate: bool = False):
        """Scan, then open the first camera that works: the one asked for, else
        the first real camera, else the simulator."""
        self.manager.start()
        snapshot = self.manager.snapshot()
        sim = [d["uid"] for d in snapshot if d["backend"] == "sim"]
        real = [d["uid"] for d in snapshot if d["auto_open"] and d["present"]]
        candidates = sim if simulate else real + sim
        wanted = self.manager.find(prefer) if prefer else None
        if prefer and wanted is None:
            logging.warning(f"--camera {prefer!r} matches no connected camera")
        if wanted:
            candidates = [wanted] + [u for u in candidates if u != wanted]
        for uid in candidates:
            try:
                camera = self.manager.open_sync(uid)
            except Exception as e:
                self.manager.set_state(uid, "error", str(e))
                self._notify(f"Could not open {self.manager.label(uid)}: {e}", "error")
                continue
            self._activate(uid, camera)
            self.auto_select = not (simulate or wanted) and uid in sim
            return
        raise RuntimeError("no camera could be opened, not even the simulator")

    # ------------------------------ per frame ----------------------------- #
    @property
    def connecting(self) -> str | None:
        """Label of the camera being opened, if any."""
        return self.manager.label(self._pending[0]) if self._pending else None

    def service(self):
        for event, uid in self.manager.poll():
            self._on_event(event, uid)
        for request in self.manager.take_requests():
            self._on_request(request)
        self._check_pending()
        if (self.offline and self._pending is None and self._target == self.uid
                and time.monotonic() >= self._retry_at):
            entry = self.manager.entry(self.uid)
            if entry is not None and entry.present:
                self._begin_open(self.uid, quiet=True)

    def grab(self) -> np.ndarray | None:
        if self.offline or self.camera is None:
            return None
        try:
            return self.camera.grab_image()
        except CameraDisconnected as e:
            self.lost(str(e))
        except Exception as e:  # e.g. the auto-exposure write after a good grab
            self._camera_error(e)
        return None

    def guard(self, fn, *args, **kwargs):
        """Run a call that talks to the hardware (exposure, ROI, sync line).
        Offline it is skipped; a failure that means the camera is gone takes
        the session offline instead of propagating."""
        if self.offline or self.camera is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self._camera_error(e)
            return None

    def _camera_error(self, e: Exception):
        if self.camera is not None and not self.camera.is_connected():
            self.lost(str(e))
            return
        now = time.monotonic()
        if now - self._last_error_log > 5:
            self._last_error_log = now
            logging.warning(f"camera call failed: {e}")

    def lost(self, reason: str):
        """The live camera is gone: remember its state, release it, and wait."""
        if self.offline or self.camera is None:
            return
        label = self.camera.label
        if not self._restoring:  # half-restored values would overwrite the good ones
            self._save_state(self.uid, self.camera)
        self.offline = reason or "disconnected"
        self.manager.mark_lost(self.uid, self.offline)
        self.camera.close()  # free the handle so the device can be found and reopened
        self._retry_at = 0.0
        logging.error(f"Camera disconnected: {label} ({reason})")
        self._notify(f"{label} disconnected - waiting for it to come back", "error")

    # ------------------------------ switching ----------------------------- #
    def switch(self, uid: str, explicit: bool = True, quiet: bool = False):
        """Make `uid` the camera to show.  The latest request wins: an open
        still in flight for another camera is released when it finishes."""
        if explicit:
            self.auto_select = False
        if self.manager.entry(uid) is None:
            self._notify(f"Unknown camera {uid}", "error")
            return
        self._target = uid
        if self._pending is not None:
            return  # settled in _check_pending
        if uid == self.uid and not self.offline:
            return
        self._begin_open(uid, quiet)

    def next(self):
        """Cycle through connected real cameras (the simulator only when it is
        the one showing)."""
        snapshot = self.manager.snapshot()
        uids = [d["uid"] for d in snapshot
                if d["present"] and d["backend"] != "sim" and d["state"] != "error"]
        if not uids:
            if self.uid is not None and self.manager.entry(self.uid).info.backend != "sim":
                self._notify("No other camera connected", "info")
            return
        target = uids[(uids.index(self.uid) + 1) % len(uids)] if self.uid in uids else uids[0]
        if target != self.uid:
            self.switch(target)

    def _begin_open(self, uid: str, quiet: bool = False):
        self._pending = (uid, self.manager.open_async(uid), quiet)

    def _check_pending(self):
        if self._pending is None or not self._pending[1].done():
            return
        uid, future, quiet = self._pending
        self._pending = None
        try:
            camera = future.result()
        except Exception as e:
            if uid == self.uid and self.offline:  # a reconnect attempt: keep waiting
                self.manager.set_state(uid, "lost", str(e))
                self._retry_at = time.monotonic() + RETRY_S
                logging.info(f"reconnect to {self.manager.label(uid)} failed: {e}")
            else:
                self.manager.set_state(uid, "error", str(e))
                message = f"Could not open {self.manager.label(uid)}: {e}"
                logging.warning(message)
                if not quiet:
                    self._notify(message, "error")
                if self._target == uid:
                    self._target = self.uid  # stay where we are
        else:
            if uid == self._target:
                self._activate(uid, camera)
            else:  # the user picked another camera while this one was opening
                self.manager.release(uid, camera)
        if self._target is not None and self._target != self.uid:
            self._begin_open(self._target)

    def _activate(self, uid: str, camera: Camera):
        old_uid, old = self.uid, self.camera
        reconnect = uid == old_uid
        if old is not None and not reconnect:
            if not self.offline:  # an offline camera's state was saved when it was lost
                self._save_state(old_uid, old)
            self.manager.release(old_uid, old)
        self.camera, self.uid, self.offline = camera, uid, None
        self._target = uid
        if camera.backend != "sim":
            self.auto_select = False  # a real camera is showing: stop following
        self.manager.set_active(uid, camera)
        state = self._states.get(uid)
        if state is not None and state.get("sensor") != (camera.W, camera.H):
            state = None  # different binning / sensor: old ROI means nothing
        self._restoring = True
        try:
            self.guard(self._restore, camera, state)
        finally:
            self._restoring = False
        if self.offline:  # dropped again while being set up
            return
        logging.info(f"{'Reconnected' if reconnect else 'Using'} {camera.label} ({uid})")
        self._notify(f"{'Reconnected' if reconnect else 'Now showing'} {camera.label}", "info")

    def _save_state(self, uid: str, camera: Camera):
        """Remember the viewer's state for this camera, to restore on return."""
        try:
            state = dict(self._capture())
        except Exception as e:
            logging.debug(f"could not capture state of {camera.label}: {e}")
            return
        state["sensor"] = (camera.W, camera.H)
        self._states[uid] = state

    # ------------------------------ events -------------------------------- #
    def _on_event(self, event: str, uid: str):
        label = self.manager.label(uid)
        if event == "added":
            if self.manager.entry(uid).info.backend == "sim":
                return
            self._notify(f"Camera connected: {label}", "info")
            if self.auto_select and self.manager.entry(uid).backend.auto_open:
                self.switch(uid, explicit=False)
        elif event == "ready":  # its open failed a while ago; worth another try
            if self.auto_select and self.manager.entry(uid).backend.auto_open:
                self.switch(uid, explicit=False, quiet=True)
        elif event == "removed":
            self._notify(f"Camera unplugged: {label}", "warn")
        elif event == "returned" and uid == self.uid:
            self._retry_at = 0.0  # service() reopens it this frame

    def _on_request(self, request: tuple):
        kind = request[0]
        if kind == "switch":
            self.switch(request[1])
        elif kind == "next":
            self.next()
        elif kind == "rescan":
            self.manager.rescan()
        elif kind == "reload_config":
            self.reload_config()
        else:
            logging.warning(f"unknown camera request {request!r}")

    def reload_config(self):
        """camera_config.yaml changed (the Cameras window saved it)."""
        cameraconfig.reload()
        self.manager.refresh_settings()
        if self.camera is not None:
            self.guard(self.camera.reload_settings)
            if not self.offline:
                self.manager.set_active(self.uid, self.camera)

    # ------------------------------ shutdown ------------------------------ #
    def close(self):
        if self._pending is not None:  # do not leave a half-open device behind
            try:
                self._pending[1].result(timeout=10).close()
            except Exception:
                pass
        if self.camera is not None:
            self.camera.close()
        self.manager.stop()
