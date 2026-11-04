"""Paths and UI constants.

Per-camera settings live in `cameraconfig.py`, which owns camera_config.yaml."""

from __future__ import annotations

import os
import sys


def app_dir() -> str:
    """Next to the .exe when frozen by PyInstaller, the repo root otherwise.
    Everything the user is meant to see or edit (data/, camera_config.yaml,
    pylon_camera.json) lives here."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


APP_DIR = app_dir()
DATA_DIR = os.path.join(APP_DIR, "data")
STATUS_JSON_PATH = os.path.join(APP_DIR, "pylon_camera.json")

# Exposure limits are per camera - see `cameraconfig.CameraSettings`.
FPS_LIMIT = 60  # UI update limit, frames/s
WINDOW_SIZE = 1100  # longest display edge, px
PAD_COLOR = (128, 128, 128)
WEB_SERVER_PORT = 5000
