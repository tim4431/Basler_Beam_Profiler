"""Camera backends and discovery.

`BACKENDS` lists every supported family in priority order.  Each is imported
lazily, so a missing SDK (no pypylon, harvesters or tucsen) only disables that one
backend - it is still listed, with the reason, in the Cameras window and in
`--list-cameras`.

Adding a vendor: write `cameras/<name>.py` with a `Backend` and a `Camera`
subclass (see `base.py`; `uvc.py` is the shortest example), then add a line
here.  Nothing else in the app needs to know.
"""

from __future__ import annotations

import logging

from .base import Backend, Camera, CameraDisconnected, DeviceInfo
from .manager import CameraManager
from .simulated import SimulatedBackend, SimulatedCamera

__all__ = [
    "BACKENDS",
    "Backend",
    "Camera",
    "CameraDisconnected",
    "CameraManager",
    "DeviceInfo",
    "SimulatedCamera",
    "load_backends",
    "open_cameras",
    "probe_devices",
]

# Order is priority: a camera reachable through two backends (a Basler is also
# a GenICam camera) is opened by the first.
BACKENDS = ("basler", "genicam", "tucsen", "uvc", "sim")


def _backend_class(name: str):
    """Import one backend module.  Written out as plain `from . import x`
    statements, not importlib, so PyInstaller sees and bundles them."""
    if name == "basler":
        from . import basler

        return basler.BaslerBackend
    if name == "genicam":
        from . import genicam

        return genicam.GenICamBackend
    if name == "tucsen":
        from . import tucsen

        return tucsen.TucsenBackend
    if name == "uvc":
        from . import uvc

        return uvc.UVCBackend
    raise KeyError(name)


class UnavailableBackend(Backend):
    """Stands in for a backend whose module would not import."""

    def __init__(self, name: str, reason: str, title: str = "", hint: str = ""):
        self.name, self.title = name, title or name
        self.install_hint = hint
        self._reason = reason

    def available(self):
        return False, self._reason

    def enumerate(self, full=False):
        return []


_TITLES = {
    "basler": ("Basler (pylon)", "install the pylon SDK and `pip install pypylon`"),
    "genicam": ("GenICam (harvesters)", "`pip install harvesters`, plus your vendor's SDK"),
    "tucsen": (
        "Tucsen (TUCam)",
        "install libTUCam and `pip install <tucsen-camera-driver>/python` (docs/cameras.md)",
    ),
}


def load_backends(names: list[str] | None = None, **sim_opts) -> list[Backend]:
    """Instantiate the named backends (all of them by default), in priority
    order.  The simulator is always included - it is the fallback."""
    wanted = [n for n in BACKENDS if names is None or n in names or n == "sim"]
    unknown = set(names or ()) - set(BACKENDS)
    if unknown:
        logging.warning(f"unknown camera backends ignored: {sorted(unknown)}")
    out: list[Backend] = []
    for name in wanted:
        if name == "sim":
            out.append(SimulatedBackend(**sim_opts))
            continue
        try:
            backend = _backend_class(name)()
        except ImportError as e:
            title, hint = _TITLES.get(name, (name, ""))
            backend = UnavailableBackend(name, f"{e.name or e} is not installed", title, hint)
        except Exception as e:  # an SDK that imports but fails to initialise
            title, hint = _TITLES.get(name, (name, ""))
            backend = UnavailableBackend(name, str(e), title, hint)
        out.append(backend)
    return out


def open_cameras(mode: str = "16Bit", simulate: bool = False, **sim_opts) -> list[Camera]:
    """Open every camera that may be opened unattended; fall back to the
    simulator when there are none (or when `simulate` is set).

    Kept for scripts; the viewer uses a `CameraManager`, which opens one
    camera at a time and follows hot-plugging."""
    cameras: list[Camera] = []
    if not simulate:
        for backend in load_backends(**sim_opts):
            if backend.name == "sim" or not backend.auto_open or not backend.available()[0]:
                continue
            try:
                for info in backend.enumerate(full=True):
                    cameras.append(backend.open(info, mode))
            except Exception as e:
                logging.warning(f"{backend.title}: {e}")
        if cameras:
            return cameras
        logging.warning("No cameras detected - using the simulated camera.")
    sim = SimulatedBackend(**sim_opts)
    return [sim.open(sim.enumerate()[0], mode)]


def probe_devices(names: list[str] | None = None) -> list[dict]:
    """What every connected camera reports about itself, for --list-cameras.
    The simulator is left out; backends that cannot load are logged."""
    probes = []
    for backend in load_backends(names):
        if backend.name == "sim":
            continue
        ok, reason = backend.available()
        if not ok:
            logging.info(f"{backend.title}: unavailable ({reason})")
            continue
        try:
            infos = backend.enumerate(full=True)
        except Exception as e:
            logging.error(f"{backend.title}: enumeration failed: {e}")
            continue
        for info in infos:
            try:
                probe = backend.probe(info)
            except Exception as e:
                probe = {**Backend.probe(backend, info), "error": str(e)}
            probes.append({"backend": backend.name, **probe})
        backend.close()
    return probes
