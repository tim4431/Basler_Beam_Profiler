"""Paths, UI constants and camera_config.yaml loading."""

from __future__ import annotations

import logging
import os
import sys
import shutil

import yaml


def app_dir() -> str:
    """Writable data directory: macOS Application Support, beside a Windows
    executable, or the repository root when running from source."""
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            path = os.path.expanduser("~/Library/Application Support/BeamProfiler")
            os.makedirs(path, exist_ok=True)
            config = os.path.join(path, "camera_config.yaml")
            bundled = os.path.join(sys._MEIPASS, "camera_config.yaml")
            if not os.path.exists(config) and os.path.isfile(bundled):
                shutil.copyfile(bundled, config)
            return path
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


APP_DIR = app_dir()
DATA_DIR = os.path.join(APP_DIR, "data")
STATUS_JSON_PATH = os.path.join(APP_DIR, "pylon_camera.json")

MIN_EXPOSURE_US = 30
MAX_EXPOSURE_US = 100_000
FPS_LIMIT = 60  # UI update limit, frames/s
WINDOW_SIZE = 1100  # longest display edge, px
PAD_COLOR = (128, 128, 128)
WEB_SERVER_PORT = 5000


def load_camera_config() -> dict[str, dict]:
    """Per-model settings: {model_name: {"default_roi": tuple, "pixel_size": float}}."""
    path = os.path.join(APP_DIR, "camera_config.yaml")
    if not os.path.exists(path):
        path = "./camera_config.yaml"
    try:
        with open(path, encoding="utf-8") as f:
            cameras = yaml.safe_load(f)["cameras"]
        for cfg in cameras.values():
            cfg["default_roi"] = tuple(cfg["default_roi"])
        return cameras
    except Exception as e:
        logging.error(f"Error loading camera config: {e}")
        return {}
