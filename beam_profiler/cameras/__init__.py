"""Camera discovery with optional vendor SDKs and an explicit simulation mode."""
from __future__ import annotations

import logging

from .base import Camera
from .simulated import SimulatedCamera

__all__ = ["Camera", "SimulatedCamera", "open_cameras"]


def _open_basler(mode):
    from pypylon import pylon
    from .basler import BaslerCamera
    cameras = []
    try:
        for i, _ in enumerate(pylon.TlFactory.GetInstance().EnumerateDevices()):
            cameras.append(BaslerCamera(mode=mode, device_idx=i))
        return cameras
    except Exception:
        for camera in cameras:
            camera.close()
        raise


def open_cameras(mode="16Bit", simulate=False, backend="auto", **sim_opts):
    if backend not in ("auto", "flir", "basler", "sim"):
        raise ValueError(f"Unknown camera backend: {backend}")
    if simulate or backend == "sim":
        return [SimulatedCamera(mode=mode, **sim_opts)]
    from .flir import open_flir_cameras
    cameras = []
    for name, discover in (("flir", open_flir_cameras), ("basler", _open_basler)):
        if backend not in ("auto", name):
            continue
        try:
            found = discover(mode)
            if not found and backend == name:
                raise RuntimeError(f"No {name.upper()} cameras detected. Check USB and the vendor viewer.")
            cameras.extend(found)
        except Exception as exc:
            if backend == name:
                raise RuntimeError(f"Cannot open {name.upper()} camera: {exc}") from exc
            logging.warning("%s discovery: %s", name.upper(), exc)
    if cameras:
        return cameras
    logging.warning("NO HARDWARE CONNECTED: using SIMULATED data. Use --camera flir to require FLIR.")
    return [SimulatedCamera(mode=mode, **sim_opts)]
