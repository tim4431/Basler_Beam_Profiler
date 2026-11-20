"""Camera discovery, hot-plugging and the camera session - without hardware.

`FakeBackend` hands out `FakeCamera`s that can be unplugged: they then raise on
every hardware call and report is_connected() False, which is what a Basler
does when its cable is pulled.  A camera with a genuine SDK behind it is
exercised by the pylon-emulator test at the end (skipped without pypylon).
"""

import threading
import time

import numpy as np
import pytest

from beam_profiler.cameras import Backend, Camera, CameraManager, DeviceInfo
from beam_profiler.cameras import manager as manager_mod
from beam_profiler.cameras import session as session_mod
from beam_profiler.cameras.session import CameraSession
from beam_profiler.cameras.simulated import SimulatedBackend


class FakeCamera(Camera):
    def __init__(self, info: DeviceInfo, mode: str = "16Bit"):
        self.model_name, self.serial, self.device = info.model, info.serial, info
        self.backend = info.backend
        self.W, self.H = 64, 48
        super().__init__(mode)
        self._roi, self._exposure, self._gain = (64, 48, 0, 0), 200.0, 0.0
        self.plugged, self.closed = True, False
        self.grabs = 0

    def _check(self):
        if not self.plugged:
            raise RuntimeError("device removed")

    @property
    def ROI(self):
        return self._roi

    @ROI.setter
    def ROI(self, value):
        self._check()
        self._roi = tuple(int(v) for v in value)

    @property
    def ExposureTime(self):
        return self._exposure

    @ExposureTime.setter
    def ExposureTime(self, value):
        self._check()
        self._exposure = float(value)

    @property
    def Gain(self):
        return self._gain

    @Gain.setter
    def Gain(self, value):
        self._check()
        self._gain = float(value)

    def is_connected(self):
        return self.plugged

    def _grab(self):
        self._check()
        self.grabs += 1
        w, h, _, _ = self._roi
        return np.full((h, w), 1000, np.uint16)

    def close(self):
        self.closed = True


class FakeBackend(Backend):
    title = "Fake"

    def __init__(self, name="fake"):
        self.name = name
        self.devices: dict[str, DeviceInfo] = {}
        self.opened: list[FakeCamera] = []
        self.refuse: set[str] = set()  # serials whose open fails (in use elsewhere)
        self.gate: threading.Event | None = None  # block opens until set

    def plug(self, serial, model="FakeCam"):
        self.devices[serial] = DeviceInfo(self.name, model, serial, "Fake Inc.")
        return f"{self.name}:{serial}"

    def unplug(self, serial):
        self.devices.pop(serial, None)
        for cam in self.opened:
            if cam.serial == serial:
                cam.plugged = False

    def enumerate(self, full=False):
        return list(self.devices.values())

    def open(self, info, mode):
        if self.gate is not None:
            self.gate.wait(5)
        if info.serial not in self.devices or info.serial in self.refuse:
            raise RuntimeError(f"cannot open {info.serial}")
        cam = FakeCamera(info, mode)
        self.opened.append(cam)
        return cam


SIM = "sim:SIM-0001"


class Harness:
    """A manager + session wired to a minimal stand-in for the viewer."""

    def __init__(self, *backends):
        self.fake = backends[0] if backends else FakeBackend()
        self.manager = CameraManager(
            [*(backends or (self.fake,)), SimulatedBackend(width=64, height=48, grid=(1, 1))],
            scan_interval=None,
        )
        self.notes: list[tuple[str, str]] = []
        self.restored: list[dict | None] = []
        self.session = CameraSession(self.manager, self.capture, self.restore,
                                     lambda text, level: self.notes.append((level, text)))

    def capture(self):
        cam = self.session.camera
        return {"roi": cam.ROI, "exposure": cam.ExposureTime}

    def restore(self, cam, state):
        self.restored.append(state)
        if state:
            cam.ROI = state["roi"]
            cam.ExposureTime = state["exposure"]

    def settle(self):
        """Scan, then run frames until no open is in flight."""
        self.manager.scan_now()
        self.session.service()
        for _ in range(100):
            if self.session._pending is None:
                return
            self.session._pending[1].exception(timeout=5)
            self.session.service()
        raise AssertionError("camera open never finished")

    @property
    def uid(self):
        return self.session.uid

    def state(self, uid):
        return self.manager.entry(uid).state


# --------------------------------------------------------------------------- #
#  the manager
# --------------------------------------------------------------------------- #
def test_same_serial_from_two_backends_is_listed_once():
    """A Basler is found by pylon and by its GenTL producer; pylon wins."""
    first, second = FakeBackend("first"), FakeBackend("second")
    first.plug("123")
    second.plug("123")
    second.plug("456")
    h = Harness(first, second)
    h.manager.start()
    uids = h.manager.uids()
    assert "first:123" in uids and "second:123" not in uids and "second:456" in uids


def test_a_missed_scan_is_not_an_unplug():
    h = Harness()
    uid = h.fake.plug("A")
    h.manager.start()
    h.fake.unplug("A")
    h.manager.scan_now()
    assert h.manager.poll() == []  # one miss: GigE discovery drops a reply now and then
    assert h.manager.entry(uid).present
    h.manager.scan_now()
    assert h.manager.poll() == [("removed", uid)]
    assert h.state(uid) == "absent"
    h.fake.plug("A")
    h.manager.scan_now()
    assert h.manager.poll() == [("added", uid)]
    assert h.state(uid) == "available"


def test_find_accepts_what_a_user_would_type(tmp_path, monkeypatch):
    from beam_profiler import cameraconfig

    path = tmp_path / "camera_config.yaml"
    path.write_text('serials:\n  "B-7":\n    label: west arm\n')
    monkeypatch.setattr(cameraconfig, "_cached", cameraconfig.load(str(path)))
    h = Harness()
    h.fake.plug("A-1", model="acA1920-40um")
    h.fake.plug("B-7", model="a2A5060-15umBAS")
    h.manager.start()
    assert h.manager.find("A-1") == "fake:A-1"
    assert h.manager.find("west arm") == "fake:B-7"
    assert h.manager.find("a2a5060") == "fake:B-7"  # model substring, any case
    assert h.manager.find("fake:A-1") == "fake:A-1"
    assert h.manager.find("nothing like it") is None


def test_backend_that_cannot_load_is_reported_not_hidden():
    from beam_profiler.cameras import load_backends

    names = [b.name for b in load_backends()]
    assert names[-1] == "sim" and set(names) >= {"basler", "genicam", "uvc", "sim"}
    manager = CameraManager(load_backends(["genicam"]), scan_interval=None)
    status = {s["name"]: s for s in manager.backend_status()}
    assert "basler" not in status  # not asked for
    if not status["genicam"]["available"]:
        assert status["genicam"]["detail"]  # says why
        assert "harvesters" in status["genicam"]["install_hint"]


# --------------------------------------------------------------------------- #
#  start-up and following new cameras
# --------------------------------------------------------------------------- #
def test_starts_on_a_real_camera_before_the_simulator():
    h = Harness()
    h.fake.plug("A")
    h.session.start()
    assert h.uid == "fake:A" and h.state("fake:A") == "active"
    assert h.restored == [None]  # first time shown: configured defaults


def test_follows_a_camera_plugged_in_after_start():
    """No camera at start-up -> simulator; plugging one in switches to it."""
    h = Harness()
    h.session.start()
    assert h.uid == SIM and h.session.auto_select
    h.fake.plug("A")
    h.settle()
    assert h.uid == "fake:A"
    assert any("connected" in text for _, text in h.notes)


def test_an_explicit_choice_stops_the_following():
    h = Harness()
    h.session.start(simulate=True)  # --sim
    h.fake.plug("A")
    h.settle()
    assert h.uid == SIM  # announced, not switched to
    assert any("connected" in text for _, text in h.notes)


def test_prefer_picks_the_named_camera():
    h = Harness()
    h.fake.plug("A")
    h.fake.plug("B")
    h.session.start(prefer="B")
    assert h.uid == "fake:B" and not h.session.auto_select


def test_a_camera_that_will_not_open_falls_through_to_the_next():
    h = Harness()
    h.fake.plug("A")
    h.fake.refuse.add("A")
    h.session.start()
    assert h.uid == SIM
    assert h.state("fake:A") == "error" and "cannot open" in h.manager.entry("fake:A").error


def test_a_slow_camera_finishes_opening_behind_the_window(monkeypatch):
    """A Tucsen takes ~7 s to open.  The window must not wait for it: that
    long with nothing on screen looks like a failed launch, and a second click
    starts another copy of the app, which then finds the camera taken."""
    monkeypatch.setattr(session_mod, "STARTUP_WAIT_S", 0.05)
    h = Harness()
    h.fake.plug("A")
    h.fake.gate = threading.Event()  # A takes its time
    h.session.start()
    assert h.uid == SIM and h.session.connecting == "FakeCam"
    assert h.notes == []  # the stand-in is not news
    h.fake.plug("B")  # plugged in meanwhile: must not cut in
    h.manager.scan_now()
    h.session.service()
    h.fake.gate.set()
    h.settle()
    assert h.uid == "fake:A" and not h.session.auto_select
    assert h.state("fake:B") == "available"
    assert ("info", "Now showing FakeCam") in h.notes


def test_a_slow_camera_that_fails_at_start_falls_through_to_the_next(monkeypatch):
    monkeypatch.setattr(session_mod, "STARTUP_WAIT_S", 0.05)
    h = Harness()
    h.fake.plug("A")
    h.fake.plug("B")
    h.fake.refuse.add("A")
    h.fake.gate = threading.Event()
    h.session.start()
    h.fake.gate.set()
    h.settle()
    assert h.uid == "fake:B" and h.state("fake:A") == "error"
    assert any(level == "error" and "Could not open" in text for level, text in h.notes)


def test_a_slow_camera_that_fails_alone_leaves_the_simulator_following(monkeypatch):
    monkeypatch.setattr(session_mod, "STARTUP_WAIT_S", 0.05)
    h = Harness()
    h.fake.plug("A")
    h.fake.refuse.add("A")
    h.fake.gate = threading.Event()
    h.session.start()
    assert not h.session.auto_select  # nothing may cut in while A opens ...
    h.fake.gate.set()
    h.settle()
    assert h.uid == SIM and h.session.auto_select  # ... but now, as after a quick failure
    h.fake.plug("C")
    h.settle()
    assert h.uid == "fake:C"


# --------------------------------------------------------------------------- #
#  unplug and reconnect
# --------------------------------------------------------------------------- #
def test_unplug_goes_offline_instead_of_raising():
    h = Harness()
    h.fake.plug("A")
    h.session.start()
    assert h.session.grab() is not None
    h.fake.unplug("A")
    assert h.session.grab() is None  # no exception reaches the viewer
    assert h.session.offline and h.state("fake:A") == "lost"
    assert h.session.camera.closed  # handle released so it can be reopened
    assert h.session.camera.ROI == (64, 48, 0, 0)  # cached getters still answer
    assert h.session.grab() is None and h.session.guard(lambda: 1 / 0) is None


def test_replug_reconnects_and_restores_the_state():
    h = Harness()
    h.fake.plug("A")
    h.session.start()
    first = h.session.camera
    first.ROI = (32, 24, 8, 8)
    first.ExposureTime = 1234
    h.fake.unplug("A")
    h.session.grab()
    for _ in range(3):  # absent: no attempt to open it while it is gone
        h.settle()
    assert len(h.fake.opened) == 1 and h.session.offline

    h.fake.plug("A")
    h.settle()
    assert not h.session.offline and h.uid == "fake:A"
    second = h.session.camera
    assert second is not first and len(h.fake.opened) == 2
    assert second.ROI == (32, 24, 8, 8) and second.ExposureTime == 1234
    assert h.session.grab().shape == (24, 32)
    assert any("Reconnected" in text for _, text in h.notes)


def test_a_reconnect_that_fails_is_retried(monkeypatch):
    monkeypatch.setattr(session_mod, "RETRY_S", 0.05)
    h = Harness()
    h.fake.plug("A")
    h.session.start()
    h.fake.unplug("A")
    h.session.grab()
    h.fake.plug("A")
    h.fake.refuse.add("A")  # back, but still held by something else
    h.settle()
    assert h.session.offline and h.state("fake:A") == "lost"
    h.settle()  # within RETRY_S: no new attempt
    assert h.session._pending is None and h.session.offline
    h.fake.refuse.clear()
    time.sleep(0.06)
    h.settle()
    assert not h.session.offline


def test_a_scan_from_before_the_unplug_does_not_count_as_return():
    h = Harness()
    h.fake.plug("A")
    h.session.start()
    h.manager.scan_now()  # taken while A is still there ...
    h.fake.unplug("A")
    h.session.grab()  # ... and the loss is noticed afterwards
    h.session.service()  # merges that stale scan
    assert h.session._pending is None and h.manager.entry("fake:A").present is False


def test_a_dead_camera_inside_a_hardware_call_goes_offline():
    h = Harness()
    h.fake.plug("A")
    h.session.start()
    cam = h.session.camera
    h.fake.unplug("A")
    assert h.session.guard(setattr, cam, "ExposureTime", 500) is None
    assert h.session.offline


# --------------------------------------------------------------------------- #
#  switching
# --------------------------------------------------------------------------- #
def test_switching_keeps_the_old_camera_live_until_the_new_one_is_open():
    h = Harness()
    h.fake.plug("A")
    h.fake.plug("B")
    h.session.start()
    old = h.session.camera
    old.ROI = (20, 10, 0, 0)
    h.fake.gate = threading.Event()  # B takes its time to open
    h.session.switch("fake:B")
    h.session.service()
    assert h.session.connecting == "FakeCam" and h.state("fake:B") == "connecting"
    assert h.session.grab() is not None and h.uid == "fake:A"  # still showing A
    h.fake.gate.set()
    h.settle()
    assert h.uid == "fake:B" and old.closed and h.state("fake:A") == "available"

    h.session.switch("fake:A")  # and back: A's own state comes with it
    h.settle()
    assert h.session.camera.ROI == (20, 10, 0, 0)


def test_a_failed_switch_stays_on_the_current_camera():
    h = Harness()
    h.fake.plug("A")
    h.fake.plug("B")
    h.session.start()
    h.fake.refuse.add("B")
    h.session.switch("fake:B")
    h.settle()
    assert h.uid == "fake:A" and h.session.grab() is not None
    assert h.state("fake:B") == "error"
    assert any(level == "error" and "Could not open" in text for level, text in h.notes)


def test_next_cycles_real_cameras_only():
    h = Harness()
    h.fake.plug("A")
    h.fake.plug("B")
    h.session.start()
    h.session.next()
    h.settle()
    assert h.uid == "fake:B"
    h.session.next()
    h.settle()
    assert h.uid == "fake:A"  # wraps, skipping the simulator


def test_requests_posted_from_another_thread_run_on_the_next_frame():
    h = Harness()
    h.fake.plug("A")
    h.fake.plug("B")
    h.session.start()
    t = threading.Thread(target=h.manager.post, args=(("switch", "fake:B"),))
    t.start()
    t.join()
    h.settle()
    assert h.uid == "fake:B" and not h.session.auto_select


def test_background_scanner_sees_hot_plugging(monkeypatch):
    fake = FakeBackend()
    manager = CameraManager([fake], scan_interval=0.02)
    manager.start()
    try:
        fake.plug("A")
        deadline = time.monotonic() + 3
        events = []
        while time.monotonic() < deadline and not events:
            events = manager.poll()
            time.sleep(0.01)
        assert events == [("added", "fake:A")]
    finally:
        manager.stop()


# --------------------------------------------------------------------------- #
#  the viewer, driven headlessly
# --------------------------------------------------------------------------- #
@pytest.fixture
def viewer(monkeypatch, tmp_path):
    import cv2

    from beam_profiler import status
    from beam_profiler.processing import gpu
    from beam_profiler.ui import viewer as viewer_mod

    for name in ("namedWindow", "resizeWindow", "setMouseCallback"):
        monkeypatch.setattr(cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(gpu, "warm_up", lambda: None)
    monkeypatch.setattr(status, "STATUS_JSON_PATH", str(tmp_path / "status.json"))
    fake = FakeBackend()
    fake.plug("A")
    manager = CameraManager(
        [fake, SimulatedBackend(width=64, height=48, grid=(1, 1))], scan_interval=None
    )
    v = viewer_mod.Viewer(manager)
    v.fake = fake
    yield v
    v.cams.close()


def _frames(viewer, n=1):
    for _ in range(n):
        viewer.manager.scan_now()
        img = viewer.step()
        if viewer.cams._pending is not None:
            viewer.cams._pending[1].exception(timeout=5)
    return img


def test_viewer_survives_an_unplug_and_restores_everything(viewer):
    from beam_profiler.config import WINDOW_SIZE

    assert _frames(viewer).shape == (WINDOW_SIZE, WINDOW_SIZE, 3)
    viewer.rect_sensor = (1, 2, 30, 20)
    viewer.do_fitting = True
    viewer._adjust_exposure(10)
    exposure = viewer.camera.ExposureTime
    viewer.fake.unplug("A")
    img = _frames(viewer, 3)  # the loop keeps going, showing the banner
    assert viewer.offline and viewer.current_frame is None
    assert img.shape == (WINDOW_SIZE, WINDOW_SIZE, 3)
    viewer._adjust_exposure(10)  # ignored while offline, not an error
    viewer._quick_save()  # refused with a message, not a crash
    assert any("No live frame" in t for _, t, _ in viewer.toasts)

    viewer.fake.plug("A")
    _frames(viewer, 3)
    assert not viewer.offline and viewer.current_frame is not None
    assert viewer.rect_sensor == (1, 2, 30, 20) and viewer.do_fitting
    assert viewer.camera.ExposureTime == exposure


def test_viewer_status_and_http_api_report_the_cameras(viewer):
    from beam_profiler.server import create_app
    from beam_profiler.status import viewer_status

    _frames(viewer)
    assert viewer_status(viewer)["camera_connected"] is True
    client = create_app(viewer).test_client()
    body = client.get("/api/cameras").get_json()
    assert body["active"] == "fake:A" and body["connected"]
    assert {c["uid"] for c in body["cameras"]} == {"fake:A", SIM}

    assert client.post("/api/select_camera", json={"camera": "nope"}).status_code == 404
    reply = client.post("/api/select_camera", json={"camera": "SIM-0001"})
    assert reply.status_code == 202
    _frames(viewer, 2)
    assert viewer.cams.uid == SIM

    viewer.fake.unplug("A")
    viewer.fake.plug("A")
    viewer.cams.switch("fake:A")
    _frames(viewer, 2)
    viewer.fake.unplug("A")
    _frames(viewer)
    status = viewer_status(viewer)
    assert status["camera_connected"] is False and status["camera_offline_reason"]


def test_a_bug_in_one_frame_does_not_end_the_loop(viewer, monkeypatch):
    """run() reports the exception and carries on; only Esc quits."""
    import cv2

    from beam_profiler.ui import viewer as viewer_mod

    calls = {"n": 0}
    real_step = viewer.step

    def flaky_step():
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("boom")
        return real_step()

    keys = iter([-1, -1, -1, 27])
    monkeypatch.setattr(viewer, "step", flaky_step)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKeyEx", lambda _ms: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(viewer_mod.Viewer, "_window_open", staticmethod(lambda: True))
    viewer.run()
    assert calls["n"] == 4
    assert any("boom" in t for _, t, _ in viewer.toasts)


# --------------------------------------------------------------------------- #
#  USB / UVC
# --------------------------------------------------------------------------- #
def test_uvc_lists_capture_nodes_from_sysfs(tmp_path):
    from beam_profiler.cameras.uvc import UVCBackend

    usb = tmp_path / "usb1"
    for node, index in (("video0", "0"), ("video1", "1"), ("video10", "0")):
        d = tmp_path / "v4l" / node
        d.mkdir(parents=True)
        (d / "name").write_text("HD Webcam: HD Webcam\n" if node == "video0" else "Scope\n")
        (d / "index").write_text(index)
        (usb / node / "intf").mkdir(parents=True)
        (d / "device").symlink_to(usb / node / "intf")
    (usb / "video0" / "serial").write_text("SN42\n")
    (usb / "video0" / "manufacturer").write_text("Acme\n")
    devices = UVCBackend(sysfs=str(tmp_path / "v4l"), use_sysfs=True).enumerate()
    assert [(d.model, d.serial, d.address) for d in devices] == [
        ("HD Webcam", "SN42", "/dev/video0"),  # video1 is the metadata node
        ("Scope", "video10", "/dev/video10"),  # no USB serial: the node names it
    ]
    assert devices[0].vendor == "Acme" and devices[0].uid == "uvc:SN42"


def test_uvc_frames_are_mono_cropped_and_msb_aligned(monkeypatch):
    import cv2

    from beam_profiler.cameras import uvc

    class FakeCapture:
        def __init__(self, *_):
            self.props = {}
            self.alive = True

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.props[prop] = value
            return True

        def read(self):
            if not self.alive:
                return False, None
            frame = np.zeros((48, 64, 3), np.uint8)
            frame[10:20, 30:40] = 200
            return True, frame

        def getBackendName(self):
            return "V4L2"

        def release(self):
            pass

    monkeypatch.setattr(uvc.cv2, "VideoCapture", FakeCapture)
    info = DeviceInfo("uvc", "HD Webcam", "SN42", "", "/dev/null")
    cam = uvc.UVCCamera("16Bit", info)
    assert (cam.W, cam.H) == (64, 48)
    cam.ExposureTime = 1000
    assert cam.cap.props[cv2.CAP_PROP_EXPOSURE] == 10  # V4L2 counts 100 us
    cam.ROI = (20, 20, 25, 5)
    frame = cam.grab_image()
    assert frame.shape == (20, 20) and frame.dtype == np.uint16
    assert frame.max() == 200 << 8 and frame[5, 5] == frame.max()

    cam.cap.alive = False  # unplugged: missed frames pile up into a disconnect
    from beam_profiler.cameras.base import MAX_CONSECUTIVE_FAILURES, CameraDisconnected

    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        assert cam.grab_image() is None
    with pytest.raises(CameraDisconnected):
        cam.grab_image()


# --------------------------------------------------------------------------- #
#  a real SDK: two emulated Baslers (PYLON_CAMEMU)
# --------------------------------------------------------------------------- #
def test_basler_backend_against_the_pylon_emulator(monkeypatch):
    pytest.importorskip("pypylon")
    monkeypatch.setenv("PYLON_CAMEMU", "2")
    from beam_profiler.cameras.basler import BaslerBackend

    backend = BaslerBackend()
    devices = [d for d in backend.enumerate() if d.model == "Emulation"]
    if not devices:
        pytest.skip("pylon camera emulator not available")
    cam = backend.open(devices[0], "16Bit")
    try:
        assert cam.uid == devices[0].uid and cam.is_connected()
        cam.ROI = (640, 480, 64, 32)  # restarts grabbing: the first trigger must not be lost
        assert cam.ROI == (640, 480, 64, 32)
        for _ in range(3):
            frame = cam.grab_image()
            assert frame is not None and frame.shape == (480, 640)
        cam.ExposureTime = 500
        assert cam.ExposureTime == pytest.approx(500, abs=1)
    finally:
        cam.close()
    cam.close()  # twice, as after an unplug: must not raise


# --------------------------------------------------------------------------- #
#  regressions from review
# --------------------------------------------------------------------------- #
def test_following_stops_once_a_real_camera_is_showing():
    h = Harness()
    h.session.start()  # nothing plugged in: simulator
    h.fake.plug("A")
    h.settle()
    assert h.uid == "fake:A" and not h.session.auto_select
    h.fake.plug("B")  # a second camera must not take the view away from A
    h.settle()
    assert h.uid == "fake:A"


def test_a_camera_that_was_busy_at_start_is_retried(monkeypatch):
    """Started while pylon Viewer held the camera: once it is free, the app
    picks it up by itself (quietly - no error toast every few seconds)."""
    monkeypatch.setattr(manager_mod, "ERROR_HOLD_S", 0.0)
    h = Harness()
    h.fake.plug("A")
    h.fake.refuse.add("A")
    h.session.start()
    assert h.uid == SIM and h.state("fake:A") == "error"
    errors_before = sum(level == "error" for level, _ in h.notes)
    h.settle()  # still busy: retried and failed, silently
    assert h.uid == SIM and h.state("fake:A") == "error"
    assert sum(level == "error" for level, _ in h.notes) == errors_before
    h.fake.refuse.clear()  # the other program lets go
    h.settle()
    h.settle()
    assert h.uid == "fake:A"


def test_the_latest_choice_wins_over_an_open_in_flight():
    h = Harness()
    for serial in "ABC":
        h.fake.plug(serial)
    h.session.start()
    h.fake.gate = threading.Event()
    h.session.switch("fake:B")  # slow to open ...
    h.session.switch("fake:C")  # ... a misclick ...
    h.session.switch("fake:B")  # ... corrected
    h.fake.gate.set()
    h.settle()
    assert h.uid == "fake:B"

    h.fake.gate = threading.Event()
    h.session.switch("fake:C")
    h.session.switch("fake:B")  # back to the one already showing
    h.fake.gate.set()
    h.settle()
    assert h.uid == "fake:B"
    c = [cam for cam in h.fake.opened if cam.serial == "C"][-1]
    assert c.closed and h.state("fake:C") == "available"  # opened, then let go


def test_a_drop_during_restore_keeps_the_saved_state():
    h = Harness()
    h.fake.plug("A")
    h.session.start()
    h.session.camera.ExposureTime = 1234
    h.fake.unplug("A")
    h.session.grab()
    real_restore = h.restore

    def flaky_restore(cam, state):  # cable drops again half-way through
        if state:
            h.fake.unplug("A")
        real_restore(cam, state)

    h.restore = flaky_restore
    h.session._restore = flaky_restore
    h.fake.plug("A")
    h.settle()
    assert h.session.offline
    h.session._restore = h.restore = real_restore
    h.fake.plug("A")
    h.settle()
    assert not h.session.offline and h.session.camera.ExposureTime == 1234


def test_a_missed_scan_does_not_create_a_phantom_from_another_backend():
    first, second = FakeBackend("first"), FakeBackend("second")
    first.plug("123")
    second.plug("123")
    h = Harness(first, second)
    h.manager.start()
    first.devices.clear()  # pylon misses the GigE camera once
    h.manager.scan_now()
    assert h.manager.poll() == []
    assert "second:123" not in h.manager.uids()


def test_uvc_without_sysfs_notices_an_unplugged_camera(monkeypatch):
    from beam_profiler.cameras.uvc import UVCBackend

    present = {0}
    monkeypatch.setattr(UVCBackend, "_probe", staticmethod(lambda i: i in present))
    backend = UVCBackend(use_sysfs=False)
    assert [d.address for d in backend.enumerate(full=True)] == ["0"]
    backend._open.add("0")
    present.clear()  # pulled while open; the session closes it
    backend._closed("0")
    assert backend.enumerate() == []  # gone - not "present but failing" forever
    present.add(0)
    assert [d.address for d in backend.enumerate()] == ["0"]  # and back


# --------------------------------------------------------------------------- #
#  Tucsen (TUCam SDK, through the `tucsen` package)
# --------------------------------------------------------------------------- #
class FakeTucam:
    """Stands in for the `tucsen` package: cameras by SDK index, a sysfs tree
    for the USB ones, and a count of SDK scans (each one an Api_Init)."""

    W, H, STEP_MS = 256, 192, 0.01032

    class TucamError(RuntimeError):
        def __init__(self, name):
            super().__init__(name)
            self.name = name

    def __init__(self, sysfs):
        self.sysfs = sysfs
        self.devices: list[dict] = []
        self.scans = 0
        self.opened: list = []
        fake = self

        class Camera:
            def __init__(cam, index):
                visible = fake.visible()
                if index >= len(visible):
                    raise fake.TucamError("TUCAMRET_OUT_OF_RANGE")
                d = visible[index]
                if d.get("held"):  # another process has it
                    raise fake.TucamError("TUCAMRET_FAILOPEN_CAMERA")
                cam.model, cam.bus, cam.serial = d["model"], d["bus"], d["serial"] or None
                cam.binning = 2  # left over in the SDK's settings file
                cam._roi = (0, 0, fake.W // 2, fake.H // 2)
                cam.exposure_ms_ = 10.0
                cam.gain, cam.mode, cam.triggers = 0, None, 0
                cam.timeouts, cam.closed, cam.device = False, False, d
                fake.opened.append(cam)

            def describe(cam, name):
                if name == "EXPOSURETM":
                    return {"min": fake.STEP_MS, "max": 3.6e6, "step": fake.STEP_MS,
                            "value": cam.exposure_ms_}
                return {"min": 0, "max": 5, "step": 1, "value": cam.gain}

            exposure_ms = property(
                lambda cam: cam.exposure_ms_,
                lambda cam, ms: setattr(cam, "exposure_ms_", round(ms / fake.STEP_MS) * fake.STEP_MS),
            )

            def get(cam, name):
                return float(cam.gain)

            def set(cam, name, value):
                cam.gain = int(value)

            @property
            def roi(cam):
                return cam._roi

            @roi.setter
            def roi(cam, value):
                x, y, w, h = value if value is not None else (0, 0, fake.W, fake.H)
                if any(v % 4 for v in (x, y, w, h)) or x + w > fake.W or y + h > fake.H:
                    raise ValueError(f"bad ROI {value}")  # what the real package raises
                cam._roi = (x, y, w, h)

            def start(cam, mode="sequence"):
                cam.mode = mode

            def software_trigger(cam):
                cam.triggers += 1

            def grab(cam, timeout=5.0):
                cam.last_timeout = timeout
                if cam.timeouts or cam.device not in fake.devices:
                    raise fake.TucamError("TUCAMRET_TIMEOUT")
                _, _, w, h = cam._roi
                return np.full((h, w), 1000, np.uint16)

            def close(cam):
                cam.closed = True

        self.Camera = Camera
        (sysfs / "usb1").mkdir(parents=True)  # a root hub: not Tucsen
        (sysfs / "usb1" / "idVendor").write_text("1d6b\n")

    def visible(self):
        return [d for d in self.devices if d["visible"]]

    def list_cameras(self):
        self.scans += 1
        return [{"index": i, "model": d["model"], "bus": d["bus"], "product": 0}
                for i, d in enumerate(self.visible())]

    def plug(self, bus="USB 3.0", model="FakeDhyana", serial="KB1", visible=True):
        d = {"model": model, "bus": bus, "serial": serial if bus != "CameraLink" else "",
             "visible": visible, "node": None}
        if bus.startswith("USB"):
            d["node"] = self.sysfs / f"10-{len(self.devices) + 1}"
            d["node"].mkdir()
            (d["node"] / "idVendor").write_text("5453\n")
            (d["node"] / "busnum").write_text("10\n")
            (d["node"] / "devnum").write_text(f"{len(self.devices) + 6}\n")
            (self.sysfs / f"{d['node'].name}:1.0").mkdir()  # an interface: skipped
        self.devices.append(d)
        return d

    def unplug(self, d):
        self.devices.remove(d)
        if d["node"] is not None:
            for f in d["node"].iterdir():
                f.unlink()
            d["node"].rmdir()


@pytest.fixture
def tucsen(monkeypatch, tmp_path):
    import sys

    fake = FakeTucam(tmp_path / "usb")
    monkeypatch.setitem(sys.modules, "tucsen", fake)  # importable without the SDK
    from beam_profiler.cameras import tucsen as mod

    monkeypatch.setattr(mod, "tucam", fake)
    return mod, fake, mod.TucsenBackend(sysfs=str(fake.sysfs))


def test_tucsen_asks_the_sdk_only_when_something_changed(tucsen):
    mod, fake, backend = tucsen
    fake.plug()
    assert [d.uid for d in backend.enumerate(full=True)] == ["tucsen:0"]
    for _ in range(3):
        backend.enumerate()
    assert fake.scans == 1  # nothing changed on the bus: no Api_Init every 2 s

    cam = backend.open(backend.enumerate()[0], "16Bit")
    fake.plug()  # a second one arrives while the first is open
    assert len(backend.enumerate(full=True)) == 1 and fake.scans == 1  # SDK left alone
    cam.close()
    assert len(backend.enumerate()) == 2 and fake.scans == 2  # bus changed: looked again


def test_tucsen_waits_a_few_scans_for_a_usb_camera_the_sdk_cannot_see_yet(tucsen):
    mod, fake, backend = tucsen
    d = fake.plug(visible=False)  # on the bus, but udev is still setting permissions
    assert backend.enumerate(full=True) == []
    d["visible"] = True
    assert len(backend.enumerate()) == 1 and fake.scans == 2

    fake.plug(visible=False)  # one that never shows up stops costing scans
    for _ in range(10):
        backend.enumerate()
    assert fake.scans == 2 + mod.USB_RETRIES


def test_tucsen_maps_exposure_gain_and_roi_onto_the_sdk(tucsen):
    mod, fake, backend = tucsen
    fake.plug()
    cam = backend.open(backend.enumerate(full=True)[0], "16Bit")
    sdk = fake.opened[-1]
    assert (cam.W, cam.H) == (fake.W, fake.H) and sdk.binning == 1  # full sensor, unbinned
    assert cam.serial == "KB1" and cam.software_trigger and sdk.mode == "software"

    cam.ExposureTime = 1234.5  # us in, ms to the SDK, on its 10.32 us grid
    assert cam.ExposureTime == pytest.approx(1238.4)
    cam.ExposureTime = 31
    cam.ExposureTime = 31 * 1.1  # an auto-exposure step smaller than one SDK step
    assert cam.ExposureTime == pytest.approx(41.28)  # still moves
    cam.ExposureTime = 5
    assert cam.ExposureTime == pytest.approx(30.96)  # never under min_exposure (30)
    cam.Gain = 1.4
    assert cam.Gain == 1 and sdk.gain == 1  # a mode index
    cam.Gain = 9
    assert cam.Gain == 5

    cam.ROI = (101, 51, 13, 7)  # (w, h, ox, oy) -> multiples of 4 as (x, y, w, h)
    assert sdk.roi == (12, 4, 100, 48) and cam.ROI == (100, 48, 12, 4)
    frame = cam.grab_image()
    assert frame.shape == (48, 100) and frame.dtype == np.uint16 and sdk.triggers == 1
    assert sdk.last_timeout == pytest.approx(1.0 + cam.ExposureTime / 1e6)
    cam.ROI = (10, 10, 250, 190)  # too small, past the corner
    assert cam.ROI == (mod.MIN_ROI, mod.MIN_ROI, fake.W - mod.MIN_ROI, fake.H - mod.MIN_ROI)
    cam.close()
    assert sdk.closed and backend._open == 0

    cam8 = backend.open(backend.enumerate()[0], "8Bit")
    assert cam8.grab_image().dtype == np.uint8 and cam8.grab_image().max() == 1000 >> 8
    cam8.close()


def test_tucsen_usb_unplug_is_a_disconnect_but_a_timeout_is_not(tucsen):
    from beam_profiler.cameras.base import MAX_CONSECUTIVE_FAILURES, CameraDisconnected

    mod, fake, backend = tucsen
    usb = fake.plug()
    cam = backend.open(backend.enumerate(full=True)[0], "16Bit")
    fake.opened[-1].timeouts = True  # missed frames, camera still on the bus
    for _ in range(MAX_CONSECUTIVE_FAILURES + 2):
        assert cam.grab_image() is None
    assert cam.is_connected()
    fake.unplug(usb)
    with pytest.raises(CameraDisconnected):
        cam.grab_image()
    cam.close()

    fake.plug(bus="CameraLink")  # no bus to ask: a run of failures decides
    cam = backend.open(backend.enumerate()[0], "16Bit")
    fake.opened[-1].timeouts = True
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        assert cam.grab_image() is None
    with pytest.raises(CameraDisconnected):
        cam.grab_image()


def test_tucsen_failed_open_makes_the_next_scan_look_again(tucsen):
    mod, fake, backend = tucsen
    d = fake.plug(bus="CameraLink")
    info = backend.enumerate(full=True)[0]
    fake.unplug(d)  # switched off: nothing changes on the USB bus
    assert backend.enumerate() == [info] and fake.scans == 1
    with pytest.raises(fake.TucamError):
        backend.open(info, "16Bit")
    assert backend._open == 0
    assert backend.enumerate() == [] and fake.scans == 2


def test_tucsen_open_names_the_program_that_has_the_camera(tucsen, tmp_path, monkeypatch):
    """The SDK answers a second copy of the app with TUCAMRET_FAILOPEN_CAMERA
    only; the error says who has the camera instead."""
    mod, fake, backend = tucsen
    d = fake.plug()
    info = backend.enumerate(full=True)[0]
    d["held"] = True
    with pytest.raises(fake.TucamError):  # nothing on the bus says it is taken: the SDK's word
        backend.open(info, "16Bit")

    # What Linux shows while another process has it: the interface bound to
    # usbfs, and that process holding the device node open.
    (tmp_path / "drivers" / "usbfs").mkdir(parents=True)
    (fake.sysfs / f"{d['node'].name}:1.0" / "driver").symlink_to(tmp_path / "drivers" / "usbfs")
    node = tmp_path / "dev" / "010" / "006"
    node.parent.mkdir(parents=True)
    node.touch()
    proc = tmp_path / "proc"
    (proc / "4242" / "fd").mkdir(parents=True)
    (proc / "4242" / "fd" / "9").symlink_to(node)
    (proc / "4242" / "cmdline").write_text("/lab/.venv/bin/python\0-m\0beam_profiler\0")
    (proc / "self").mkdir()
    monkeypatch.setattr(mod, "USB_DEVFS", str(tmp_path / "dev"))
    monkeypatch.setattr(mod, "PROC", str(proc))
    with pytest.raises(RuntimeError, match=r"^in use by another copy of the beam profiler \(PID 4242\)$"):
        backend.open(info, "16Bit")
    assert backend._open == 0

    (proc / "4242" / "cmdline").write_text("python3\0grab_frames.py\0")
    assert mod.usb_holder(str(fake.sysfs)) == "python3 grab_frames.py (PID 4242)"
    (proc / "4242" / "fd" / "9").unlink()  # another user's process: only usbfs is visible
    assert mod.usb_holder(str(fake.sysfs)) == "another program"
    (fake.sysfs / f"{d['node'].name}:1.0" / "driver").unlink()  # free: some other failure
    assert mod.usb_holder(str(fake.sysfs)) == ""
