"""Contract/error-path tests for SDK buffer ownership and camera lifecycle."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from beam_profiler.cameras import open_cameras
from beam_profiler.cameras.flir import FlirCamera, open_flir_cameras


class Node:
    def __init__(self, value=0, lo=0, hi=10000, inc=1, writable=True):
        self.value, self.lo, self.hi, self.inc = value, lo, hi, inc
        self.writable = writable
        self.history = []

    def GetValue(self): return self.value
    def GetIntValue(self): return self.value
    def GetMin(self): return self.lo
    def GetMax(self): return self.hi
    def GetInc(self): return self.inc
    def GetEntryByName(self, name): return Node(name)
    def SetIntValue(self, value): self.SetValue(value)
    def SetValue(self, value):
        assert self.writable
        self.value = value
        self.history.append(value)
    def Execute(self): self.history.append('execute')


class FakeCamera:
    def __init__(self, model='Blackfly S BFS-U3-31S4M'):
        self.nodes = {
            'DeviceModelName': Node(model), 'DeviceSerialNumber': Node('123'),
            'Width': Node(2048, 16, 2048, 4), 'Height': Node(1536, 16, 1536, 2),
            'OffsetX': Node(0, 0, 2048, 4), 'OffsetY': Node(0, 0, 1536, 2),
            'ExposureTime': Node(200., 10., 1e6), 'Gain': Node(0., 0., 40.),
        }
        for name in ('AcquisitionMode', 'ExposureAuto', 'ExposureMode', 'GainAuto',
                     'GammaEnable', 'PixelFormat', 'TriggerMode', 'TriggerSelector',
                     'TriggerSource', 'StreamBufferHandlingMode', 'LineSelector',
                     'LineMode', 'LineInverter', 'LineSource', 'UserOutputSelector',
                     'UserOutputValue', 'TriggerSoftware'):
            self.nodes[name] = Node()
        self.initialized = False
        self.running = False
        self.fail_grab = False
        self.incomplete = False
        self.releases = 0
        self.timeout = None
        self.buffer = np.full((20, 20), 32000, np.uint16)

    def Init(self): self.initialized = True
    def DeInit(self): self.initialized = False
    def GetNodeMap(self): return self
    def GetTLStreamNodeMap(self): return self
    def GetNode(self, name):
        node = self.nodes.get(name)
        if name == 'OffsetX': node.hi = 2048 - self.nodes['Width'].value
        if name == 'OffsetY': node.hi = 1536 - self.nodes['Height'].value
        return node
    def BeginAcquisition(self): self.running = True
    def EndAcquisition(self): self.running = False
    def GetNextImage(self, timeout):
        self.timeout = timeout
        if self.fail_grab: raise RuntimeError('disconnected')
        return self
    def IsIncomplete(self): return self.incomplete
    def GetImageStatus(self): return 9
    def GetNDArray(self): return self.buffer
    def Release(self):
        self.releases += 1
        self.buffer.fill(0)  # SDK is allowed to overwrite the returned buffer


def sdk_for(*raw):
    class DeviceList:
        def GetSize(self): return len(raw)
        def GetByIndex(self, index): return raw[index]
        def Clear(self): pass
    class System:
        releases = 0
        def GetCameras(self): return DeviceList()
        def ReleaseInstance(self): self.releases += 1
    system = System()
    sdk = SimpleNamespace(System=SimpleNamespace(GetInstance=lambda: system),
                          IsReadable=lambda n: n is not None,
                          IsWritable=lambda n: n is not None and n.writable)
    for kind in ('String', 'Integer', 'Float', 'Enumeration', 'Boolean', 'Command'):
        setattr(sdk, f'C{kind}Ptr', lambda n: n)
    return sdk, system


@pytest.fixture
def camera():
    raw = FakeCamera()
    sdk, _ = sdk_for(raw)
    camera = FlirCamera(raw, sdk)
    yield camera, raw
    camera.close()


def test_frame_owns_memory_and_releases_sdk_buffer(camera):
    c, raw = camera
    frame = c.grab_image()
    assert frame.dtype == np.uint16
    assert np.all(frame == 32000) and np.all(raw.buffer == 0)
    assert raw.releases == 1


def test_incomplete_frame_released(camera):
    c, raw = camera
    raw.incomplete = True
    assert c.grab_image() is None
    assert raw.releases == 1


def test_timeout_restores_user_output(camera):
    c, raw = camera
    assert c.set_userdefined_line(True)
    raw.fail_grab = True
    with pytest.raises(RuntimeError, match='disconnected'): c.grab_image()
    assert raw.nodes['UserOutputValue'].history[-2:] == [False, True]
    assert raw.releases == 0


def test_roi_alignment_clamps_and_restarts(camera):
    c, raw = camera
    c.ROI = (501, 303, 9999, -20)
    assert c.ROI == (500, 302, 1548, 0)
    assert raw.running
    c.ROI = (2048, 1536, 0, 0)
    assert c.ROI == (2048, 1536, 0, 0)


def test_exposure_gain_and_long_timeout(camera):
    c, raw = camera
    c.ExposureTime = 9e9
    c.Gain = -20
    assert c.ExposureTime == 1e6 and c.Gain == 0
    c.grab_image()
    assert raw.timeout == 2000
    with pytest.raises(ValueError): c.ExposureTime = float('nan')


def test_trigger_fallback_disarms():
    raw = FakeCamera()
    raw.nodes['TriggerSource'].writable = False
    sdk, _ = sdk_for(raw)
    c = FlirCamera(raw, sdk)
    try:
        assert not c.software_trigger
        assert raw.nodes['TriggerMode'].value == 'Off'
        c.grab_image()
    finally: c.close()


def test_multicamera_sdk_lifetime(monkeypatch):
    raw = [FakeCamera(), FakeCamera()]
    sdk, system = sdk_for(*raw)
    monkeypatch.setitem(sys.modules, 'PySpin', sdk)
    cameras = open_flir_cameras()
    cameras[0].close()
    assert system.releases == 0 and raw[1].running
    cameras[1].close()
    cameras[1].close()
    assert system.releases == 1
    assert not any(c.initialized for c in raw)


def test_partial_discovery_failure_closes_all(monkeypatch):
    raw = [FakeCamera(), FakeCamera('unknown calibration')]
    sdk, system = sdk_for(*raw)
    monkeypatch.setitem(sys.modules, 'PySpin', sdk)
    with pytest.raises(RuntimeError, match='pixel_size'):
        open_flir_cameras()
    assert system.releases == 1
    assert not any(c.initialized or c.running for c in raw)


def test_explicit_flir_never_silently_simulates(monkeypatch):
    sdk, system = sdk_for()
    monkeypatch.setitem(sys.modules, 'PySpin', sdk)
    with pytest.raises(RuntimeError, match='No FLIR cameras'):
        open_cameras(backend='flir')
    assert system.releases == 1


def test_simulation_does_not_load_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, 'PySpin', None)
    camera = open_cameras(simulate=True)[0]
    assert camera.grab_image().dtype == np.uint16
