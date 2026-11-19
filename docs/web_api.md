# HTTP API

Press `w` in the profiler window to start the server on port `5000`
(`http://localhost:5000` serves an index page listing the endpoints). Pressing
`w` again hides the on-screen rectangle statistics; the Flask thread is a daemon
and shuts down with the application.

All rectangle coordinates are **sensor** coordinates `[x1, y1, x2, y2]`, not
display coordinates. The white rectangle (`rect`) and the green fitting
rectangle (`fit_rect`) are handled by separate endpoints.

## Read

| Endpoint | Description |
| --- | --- |
| `GET /api/status` | Camera model/serial/label, whether it is connected, pixel size, exposure and its limits, ROI, auto-exposure and fitting flags, both rectangles |
| `GET /api/cameras` | Every camera found (all backends), its state, and which one is in use |
| `GET /api/rect_stats` | Pixel statistics inside the white rectangle |
| `GET /api/fit_rect_stats` | Pixel statistics inside the green fitting rectangle |
| `GET /api/spots` | Detected spots plus the aggregated row/column statistics |
| `GET /api/fit_config` | Current spot-fitting parameters, plus each setting's range and help text |
| `PUT /api/fit_config` | Update spot-fitting parameters; JSON body of `{name: value}`, unknown names are rejected |
| `GET /api/image` | Current frame as JPEG |
| `GET /api/image_within_rect` | White-rectangle crop of the current frame as JPEG |

## Write

| Endpoint | Body | Description |
| --- | --- | --- |
| `POST /api/set_rect` | `{"coords": [x1, y1, x2, y2]}` | Set the white rectangle |
| `POST /api/set_fit_rect` | `{"coords": [x1, y1, x2, y2]}` | Set the green fitting rectangle |
| `POST /api/clear_rect` | - | Clear the white rectangle |
| `POST /api/clear_fit_rect` | - | Clear the green fitting rectangle |
| `POST /api/select_camera` | `{"camera": "40615333"}` | Switch camera; matches a serial, label, model or uid. Answers `202` at once - the switch happens on the next frame - or `404` if nothing matches |

## Examples

```bash
# camera state
curl http://localhost:5000/api/status

# define the region of interest to integrate over
curl -X POST -H "Content-Type: application/json" \
     -d '{"coords": [100, 100, 300, 300]}' \
     http://localhost:5000/api/set_rect

# read back its pixel statistics
curl http://localhost:5000/api/rect_stats

# grab the current frame
curl http://localhost:5000/api/image -o current_image.jpg

# list the cameras, then switch to one by serial (or label, or model)
curl http://localhost:5000/api/cameras
curl -X POST -H "Content-Type: application/json" \
     -d '{"camera": "40615333"}' http://localhost:5000/api/select_camera
```

`/api/status` responds with:

```json
{
  "camera_model": "a2A5060-15umBAS",
  "camera_serial": "40615333",
  "camera_label": "profiler head, west arm",
  "camera_uid": "basler:40615333",
  "camera_backend": "basler",
  "camera_connected": true,
  "camera_offline_reason": null,
  "pixel_size": 2.5e-06,
  "pixel_size_known": true,
  "exposure_time": 2866.0,
  "exposure_limits": [30.0, 100000.0],
  "roi": [252, 252, 2502, 1774],
  "auto_exposure": false,
  "fitting_enabled": false,
  "rect_sensor": [2476, 1740, 2837, 2099],
  "fit_rect_sensor": null
}
```

`/api/rect_stats` responds with:

```json
{
  "rectangle_bounds": {
    "sensor_coords": [100, 100, 300, 300],
    "roi_coords": [100, 100, 300, 300],
    "width_pixels": 200,
    "height_pixels": 200,
    "width_um": 500.0,
    "height_um": 500.0
  },
  "pixel_stats": {
    "sum": 2560000,
    "mean": 64.0,
    "std": 12.5,
    "min": 32,
    "max": 255,
    "median": 63.0,
    "total_pixels": 40000
  },
  "timestamp": 1692960000.0
}
```

Statistics are computed on the **raw** frame (12-bit data in a 16-bit array when
the camera runs in `16Bit` mode), while `/api/image` returns an 8-bit JPEG
rendering of it.

`/api/cameras` responds with one entry per camera; `state` is one of `active`,
`connecting`, `available`, `lost` (was in use, unplugged, will reconnect by
itself), `absent` (unplugged) or `error` (would not open; see `error`):

```json
{
  "active": "basler:40615333",
  "connected": true,
  "cameras": [
    {"uid": "basler:40615333", "backend": "basler", "model": "a2A5060-15umBAS",
     "serial": "40615333", "label": "profiler head, west arm", "state": "active",
     "present": true, "active": true, "sensor": [5060, 5060],
     "pixel_size": 2.5e-06, "pixel_size_known": true, "...": "..."},
    {"uid": "sim:SIM-0001", "backend": "sim", "model": "Simulated-4x4",
     "state": "available", "...": "..."}
  ],
  "backends": [{"name": "basler", "title": "Basler (pylon)", "available": true, "...": "..."}]
}
```

While the camera is unplugged, `/api/status` keeps answering with
`"camera_connected": false` and the last known exposure and ROI, and
`/api/image` returns 400 (there is no current frame).

## Notes

- Errors are returned as `{"error": "..."}` with HTTP 400 - e.g. when no frame
  has been grabbed yet, or when no rectangle has been defined.
- The server binds `0.0.0.0`, so it is reachable from other machines on the
  network. There is no authentication; keep it on a trusted network.
- The same status payload as `/api/status` is written to `pylon_camera.json`
  next to the application on every change, and does not require the server.
