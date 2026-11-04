# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup on a new machine

```bash
pip install -r requirements.txt        # needs Python 3.10+
python -m beam_profiler --sim          # verify everything works without hardware
```

For real cameras the Basler pylon SDK must be installed system-wide. Any
enumerated Basler opens whether or not it is in `camera_config.yaml`; the entry
(keyed by exact model name, a glob, or a serial under `serials:`) only supplies
`pixel_size` and any model quirks — `python -m beam_profiler --list-cameras`
prints a paste-ready block for whatever is plugged in. With no camera
connected, the app falls back to the simulator. `run.sh` contains a
machine-specific interpreter path — adjust or ignore it on other PCs.

## Test signal / debugging without hardware

- `python -m beam_profiler --sim` — full GUI on a simulated camera: a fixed
  grid of Gaussian spots with noise/jitter. Options: `--sim-grid 80x80`,
  `--sim-size 4096x4096`, `--sim-jitter 0` (static spots), `--sim-sigma <px>`
  (default pitch/10). Auto-exposure, ROI, fitting and the HTTP API all work
  against it.
- `python tools/make_test_images.py` — writes fixed spot pictures with
  ground-truth JSON to `test_images/`.
- `beam_profiler/synthetic.py` renders ground-truth frames programmatically —
  use it to benchmark or debug CV changes headlessly (see tests for examples).
- The HTTP API (press `w` in the GUI, port 5000) and the `pylon_camera.json`
  status mirror are the hooks for scripting against a running instance.

## Commands

```bash
python -m pytest tests/                              # run tests
python -m pytest tests/test_processing.py -k grid    # single test
python -m beam_profiler                              # run (hardware, sim fallback)
./compile                                            # local exe (needs `pip install pyinstaller`)
```

Work happens on the `dev` branch. CI (`.github/workflows/build.yml`) runs only
on push to `main` and every main push builds a Windows exe **and publishes a
GitHub release** — don't merge to main casually.

## Architecture

All code lives in the `beam_profiler/` package; `pylon_camera.py` is only a
compatibility shim for run.sh/PyInstaller.

- `cameras/` — `base.Camera` defines the interface (grab_image, ROI,
  ExposureTime, Gain properties) plus shared PID auto-exposure and
  `apply_settings`; `basler.py` (pypylon) and `simulated.py` implement it.
  `open_cameras()` does discovery + sim fallback, `probe_devices()` backs
  `--list-cameras`. Anything testable without hardware must not import pypylon
  at module level.
- `cameraconfig.py` — the same idea as `fitconfig.py` for everything that
  differs between camera models: pixel size, exposure limits, pixel formats,
  trigger mode, I/O lines. `SETTINGS` holds each field's type and help text,
  `resolve(model, serial)` merges `defaults:` < glob < model < serial from
  `camera_config.yaml`, and nothing else may read that file. `docs/cameras.md`
  is the user-facing reference.
- `fitconfig.py` — every tunable of the fit, its range and its help text, in
  one registry. `ui/settings.py` builds the settings window from it, the HTTP
  API exposes it, and it persists to `fit_config.json`. Add a knob here and it
  appears everywhere; do not put fit constants anywhere else.
- `processing/` — pure CV, no camera/UI dependencies: `blobs.py` (detection:
  blob search on a ≤1024 px downscale), `fit.py` (sub-pixel *moment* fits on
  full-resolution crops with local background subtraction, batched over all
  candidates), `gpu.py` (optional CuPy backend for that batch),
  `spots.py` (`SpotArray`, the columnar result container), `grid.py`
  (row/column classification), `stats.py` (rectangle pixel stats — the single
  source used by both UI and HTTP server).
- `ui/viewer.py` — single-threaded main loop (grab → process → draw → keys);
  owns all mutable state, which `server.py` (Flask, daemon thread) and
  `status.py` read. `ui/overlays.py` draws HUD/stats/spots.

### Conventions that bite

- **Three coordinate systems**: *sensor* coords (full chip; rectangles are
  stored in these), *ROI* coords (grabbed frame; spot fits are in these;
  sensor = ROI + offset), and *display* coords (1100 px window;
  `roi.ViewTransform` maps between them). All overlay drawing happens in
  display space so text/lines don't scale with zoom.
- Reported `sigma_0/sigma_1` are **2×** the Gaussian σ (≈ beam waist per
  axis); tests encode this (`sigma_0 ≈ 2 * true_sigma`).
- "16Bit" mode is Mono12 wrapped MSB-aligned in uint16 (saturation 65535).
- Keyboard codes in `viewer.KEY_FACTORS` must cover both Windows and Linux
  `waitKeyEx` values.
- **Spots are columnar.** `detect_spots` returns a `SpotArray` (one numpy
  array per quantity), not a list of dicts — building 6400 dicts costs more
  than the fit does. Indexing/iterating still yields the old dicts, so legacy
  `s["x"]` code works, but anything per-frame must use the columns
  (`spots.x`, `spots.sigma_0`, …) or the win is thrown away. `to_json()` is
  called by `server.py` on request, never in the main loop.
- **The crop is derived from the spot, not from the detector.** The blob
  radius is only a seed: `fit_spots` re-crops at `crop_sigma` x the fitted
  sigma (per axis) and refits, `crop_stages` times. The blob radius spans
  `r/sigma` = 0.75-1.7 depending on brightness, so using it directly made
  widths brightness-dependent (up to 30% error). A stage whose crop did not
  move is skipped — the fit depends only on the crop.
- **Never clip the crop at zero.** Clipping rectifies zero-mean noise into a
  positive pedestal that the second moment multiplies by distance^2; the error
  grows like L^4 and converges on the width of the *crop*. `clip_negative`
  exists only to reproduce pre-2026 numbers. A Gaussian weight
  (`adaptive_iters` > 0) suppresses the same noise without biasing it, because
  it is linear in the pixel value. See `docs/fitting-calibration.md`.
- **Three implementations of one estimator.** `fit.fit_spot` (scalar
  reference), `fit.fit_spots` (numpy, batched) and the fused kernel in
  `gpu.py` must agree to 1e-9 — `tests/test_processing.py` checks this over
  five configurations, so a change to one must be mirrored in all three. The
  kernel runs every stage and every adaptive iteration in a *single* launch,
  each block resizing its own crop.
- The fit defaults are measured optima and `tests/test_fit_config.py` re-runs
  the measurements. If one fails, change the default and
  `docs/fitting-calibration.md` — do not loosen the test.
- `BEAM_PROFILER_GPU=0` forces the CPU path, `=force` uses the GPU below the
  spot-count threshold; otherwise `gpu_enabled`/`gpu_min_spots` decide. The
  HUD shows which path actually ran. `max_crop` above `gpu.MAX_CROP` (512)
  silently moves everything to the CPU.
- **An unlisted camera is not an error.** `resolve()` never raises and never
  needs the model to be configured; a missing feature (no software trigger, no
  Line2) is a warning and the driver carries on. The one unknowable is
  `pixel_size`: unset, it falls back to `UNKNOWN_PIXEL_SIZE` (1 µm/px, so
  microns read as pixel counts) and `Camera.pixel_size_known` is False — check
  that flag before presenting a micron figure as calibrated. Never substitute a
  plausible-looking pitch for a missing one; a wrong scale is silent, a flagged
  one is not.
- `camera_config.yaml` is user-edited, so be forgiving: a bad value is logged
  and skipped while the rest of the block applies, `pixel_size <= 0` counts as
  unset, and `trigger_mode` accepts YAML's bare `off` (which parses as the
  boolean false). `tests/test_camera_config.py` pins all of this.
- CV changes must keep `tests/` passing — they check detection count,
  sub-pixel position error, widths, orientation and grid classification
  against synthetic ground truth.
