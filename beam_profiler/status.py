"""Mirror the live viewer state to pylon_camera.json so other processes can
read it without starting the HTTP server."""

from __future__ import annotations

import json
import logging

from .config import STATUS_JSON_PATH


def viewer_status(viewer) -> dict:
    return {
        "camera_model": viewer.camera.model_name,
        "camera_serial": viewer.camera.serial,
        "camera_label": viewer.camera.label,
        # False means no pixel_size is configured for this model, so every
        # *_um figure in the API is really a pixel count - see cameraconfig.
        "pixel_size": viewer.camera.pixel_size,
        "pixel_size_known": viewer.camera.pixel_size_known,
        "exposure_time": viewer.camera.ExposureTime,
        "exposure_limits": [viewer.camera.min_exposure, viewer.camera.max_exposure],
        "roi": viewer.camera.ROI,
        "auto_exposure": viewer.auto_exp,
        "fitting_enabled": viewer.do_fitting,
        "rect_sensor": viewer.rect_sensor,
        "fit_rect_sensor": viewer.fit_rect_sensor,
    }


def write_status(viewer):
    try:
        with open(STATUS_JSON_PATH, "w") as f:
            json.dump(viewer_status(viewer), f, indent=4)
    except Exception as e:
        logging.error(f"Error saving status to {STATUS_JSON_PATH}: {e}")
