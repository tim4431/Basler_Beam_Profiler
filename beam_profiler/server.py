"""HTTP API exposing camera status, frames and rectangle statistics.

Endpoints are documented in docs/web_api.md and listed on the index page."""

from __future__ import annotations

import time
from dataclasses import asdict
from io import BytesIO

import cv2
from flask import Flask, jsonify, request, send_file

from . import fitconfig
from .processing import crop_rect, region_stats
from .status import viewer_status

ENDPOINTS = [
    ("GET", "/api/status", "Camera status (model, exposure, ROI, rectangles)."),
    ("GET", "/api/rect_stats", "Pixel statistics of the white rectangle."),
    ("GET", "/api/fit_rect_stats", "Pixel statistics of the green fitting rectangle."),
    ("GET", "/api/spots", "Detected spots and their statistics."),
    ("GET", "/api/fit_config", "Current spot-fitting parameters."),
    ("PUT", "/api/fit_config", "Update spot-fitting parameters (JSON body)."),
    ("GET", "/api/image", "Current frame as JPEG."),
    ("GET", "/api/image_within_rect", "Current frame cropped to the white rectangle, as JPEG."),
    ("POST", "/api/set_rect", 'Set the white rectangle. JSON: {"coords": [x1, y1, x2, y2]}.'),
    ("POST", "/api/set_fit_rect", 'Set the fitting rectangle. JSON: {"coords": [x1, y1, x2, y2]}.'),
    ("POST", "/api/clear_rect", "Clear the white rectangle."),
    ("POST", "/api/clear_fit_rect", "Clear the fitting rectangle."),
]


def create_app(viewer) -> Flask:
    app = Flask(__name__)

    def rect_stats_response(rect, label):
        if viewer.current_frame is None:
            return jsonify({"error": "No frame available"}), 400
        if rect is None:
            return jsonify({"error": f"No {label} defined"}), 400
        stats = region_stats(
            viewer.current_frame, viewer.camera.ROI, rect, viewer.camera.pixel_size
        )
        if stats is None:
            return jsonify({"error": f"{label} outside of ROI"}), 400
        bounds_keys = (
            "sensor_coords", "roi_coords",
            "width_pixels", "height_pixels", "width_um", "height_um",
        )
        return jsonify(
            {
                "rectangle_bounds": {k: stats[k] for k in bounds_keys},
                "pixel_stats": {k: v for k, v in stats.items() if k not in bounds_keys},
                "timestamp": time.time(),
            }
        )

    def jpeg_response(img):
        if viewer.camera.mode == "16Bit":
            img = cv2.convertScaleAbs(img, alpha=1 / 256.0)
        _, buffer = cv2.imencode(".jpg", img)
        return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")

    def parse_coords():
        data = request.get_json()
        if not data or "coords" not in data:
            return None, (jsonify({"error": "Invalid request data"}), 400)
        coords = data["coords"]
        if len(coords) != 4:
            return None, (jsonify({"error": "Coordinates must be [x1, y1, x2, y2]"}), 400)
        return tuple(int(c) for c in coords), None

    @app.route("/api/status")
    def get_status():
        return jsonify(viewer_status(viewer))

    @app.route("/api/rect_stats")
    def get_rect_stats():
        return rect_stats_response(viewer.rect_sensor, "rectangle")

    @app.route("/api/fit_rect_stats")
    def get_fit_rect_stats():
        return rect_stats_response(viewer.fit_rect_sensor, "fitting rectangle")

    @app.route("/api/spots")
    def get_spots():
        spots = viewer.latest_spots  # SpotArray; serialised here so the main
        return jsonify(  # loop never pays for it when nobody is polling
            {
                "spots": spots.to_json(),
                "spot_count": len(spots),
                "stats": viewer.latest_stats,
                "timestamp": time.time(),
            }
        )

    @app.route("/api/fit_config", methods=["GET", "PUT"])
    def fit_config():
        if request.method == "PUT":
            body = request.get_json(silent=True) or {}
            unknown = set(body) - {s.name for s in fitconfig.SETTINGS}
            if unknown:
                return jsonify({"error": f"unknown settings: {sorted(unknown)}"}), 400
            fitconfig.set_active(fitconfig.active().replace(**body))
        return jsonify(
            {
                "config": asdict(fitconfig.active()),
                "settings": [asdict(s) for s in fitconfig.SETTINGS],
            }
        )

    @app.route("/api/image")
    def get_image():
        if viewer.current_frame is None:
            return jsonify({"error": "No frame available"}), 400
        return jpeg_response(viewer.current_frame)

    @app.route("/api/image_within_rect")
    def get_image_within_rect():
        if viewer.current_frame is None:
            return jsonify({"error": "No frame available"}), 400
        if viewer.rect_sensor is None:
            return jsonify({"error": "No rectangle defined"}), 400
        cropped = crop_rect(viewer.current_frame, viewer.camera.ROI, viewer.rect_sensor)
        if cropped is None:
            return jsonify({"error": "Rectangle outside of ROI"}), 400
        return jpeg_response(cropped[0])

    @app.route("/api/set_rect", methods=["POST"])
    def set_rect():
        coords, error = parse_coords()
        if error:
            return error
        viewer.rect_sensor = coords
        return jsonify({"message": "Rectangle set successfully", "coords": coords})

    @app.route("/api/set_fit_rect", methods=["POST"])
    def set_fit_rect():
        coords, error = parse_coords()
        if error:
            return error
        viewer.fit_rect_sensor = coords
        return jsonify({"message": "Fitting rectangle set successfully", "coords": coords})

    @app.route("/api/clear_rect", methods=["POST"])
    def clear_rect():
        viewer.rect_sensor = None
        return jsonify({"message": "Rectangle cleared"})

    @app.route("/api/clear_fit_rect", methods=["POST"])
    def clear_fit_rect():
        viewer.fit_rect_sensor = None
        return jsonify({"message": "Fitting rectangle cleared"})

    @app.route("/")
    def index():
        rows = "\n".join(
            f'<div class="endpoint"><span class="method">{m}</span> '
            f"<code>{path}</code><p>{desc}</p></div>"
            for m, path, desc in ENDPOINTS
        )
        return f"""<!DOCTYPE html>
<html><head><title>Beam Profiler Server</title><style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .endpoint {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
  .method {{ color: #007acc; font-weight: bold; }}
  pre {{ background: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }}
</style></head><body>
<h1>Beam Profiler Web Server</h1>
<h2>Available Endpoints:</h2>
{rows}
<h2>Examples:</h2>
<pre>
curl http://localhost:5000/api/status
curl -X POST -H "Content-Type: application/json" \\
     -d '{{"coords": [100, 100, 300, 300]}}' http://localhost:5000/api/set_rect
curl http://localhost:5000/api/rect_stats
curl http://localhost:5000/api/image -o current_image.jpg
</pre>
</body></html>"""

    return app
