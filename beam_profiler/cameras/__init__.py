"""Camera discovery: real Basler cameras with a simulated fallback."""

from __future__ import annotations

import logging

from .base import Camera
from .simulated import SimulatedCamera

__all__ = ["Camera", "SimulatedCamera", "open_cameras", "probe_devices"]


def open_cameras(mode: str = "16Bit", simulate: bool = False, **sim_opts) -> list[Camera]:
    """Open every connected Basler camera; fall back to the simulator when
    none are found (or when `simulate` is set)."""
    if not simulate:
        try:
            from pypylon import pylon

            from .basler import BaslerCamera

            devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            if devices:
                return [BaslerCamera(mode=mode, device_idx=i) for i in range(len(devices))]
            logging.warning("No Basler cameras detected - using the simulated camera.")
        except ImportError:
            logging.warning("pypylon not installed - using the simulated camera.")
    return [SimulatedCamera(mode=mode, **sim_opts)]


def probe_devices() -> list[dict]:
    """What every connected camera reports about itself, for --list-cameras.
    Empty when pypylon is missing or nothing is plugged in."""
    try:
        from .basler import list_devices
    except ImportError:
        logging.error("pypylon is not installed, so no cameras can be enumerated.")
        return []
    return list_devices()
