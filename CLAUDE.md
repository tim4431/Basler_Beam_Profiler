# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup on a new machine

```bash
./setup.sh --dev                       # .venv + requirements (+CuPy on NVIDIA) + desktop icon
./run.sh --sim                         # verify everything works without hardware
```

The project runs from `.venv/` (gitignored); use `.venv/bin/python` for tests
and scripts. `setup.sh` takes `python3` from PATH (override with `PYTHON=`);
tkinter and venv are apt packages (`python3.X-tk`, `python3.X-venv`), not pip;
setup.sh detects them and offers to install. Without tkinter the app still runs
(settings window and save dialog log a warning instead).

For Basler cameras the pylon SDK must be installed system-wide; other GenICam
cameras need `pip install harvesters` plus the vendor SDK (GenTL producer),
Tucsen cameras need libTUCam plus the `tucsen` package from a
tucsen-camera-driver checkout (`./setup.sh --tucsen=<checkout>`; on the lab PC
it is `~/Code/Tucsen camera/SamplePro_General_V1.0.34.0_ubuntu22.04`), USB
webcams need nothing. Any enumerated camera opens whether or not it is in
`camera_config.yaml`; the entry (keyed by exact model name, a glob, or a serial
under `serials:`) only supplies `pixel_size` and any model quirks —
`python -m beam_profiler --list-cameras` prints a paste-ready block, and the
Cameras window (`k`) writes label/pixel size for you. With no camera connected,
the app shows the simulator and switches to the first camera plugged in.
`PYLON_CAMEMU=2` gives two emulated Baslers for testing the real driver. `run.sh` runs `.venv/bin/python -m beam_profiler` and,
without a terminal (desktop icon), appends output to `data/launcher.log`.

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
python -m pytest tests/                              # run tests (inside .venv)
python -m pytest tests/test_processing.py -k grid    # single test
python -m beam_profiler                              # run (hardware, sim fallback)
./compile                                            # local exe (needs `pip install pyinstaller`)
```

Work happens on the `dev` branch. CI (`.github/workflows/build.yml`) runs only
on push to `main` and every main push builds a Windows exe **and publishes a
GitHub release** — don't merge to main casually: every release exe and every
checkout on `main` installs it by itself (`updater.py`), and a checkout on
`dev` installs every push to `dev`.

## Architecture

All code lives in the `beam_profiler/` package; `pylon_camera.py` is only a
compatibility shim for run.sh/PyInstaller.

- `cameras/` — `base.py` defines `Backend` (enumerate/open/probe a family of
  devices) and `Camera` (grab_image, ROI, ExposureTime, Gain properties, plus
  shared PID auto-exposure and `apply_settings`). Backends: `basler.py`
  (pypylon), `genicam.py` (harvesters, any GenTL producer; untested on real
  non-Basler hardware), `tucsen.py` (Tucsen TUCam SDK through the `tucsen`
  package; Linux, USB 3.0 or CameraLink), `uvc.py` (OpenCV webcams),
  `simulated.py`; registered
  in `cameras/__init__.BACKENDS`, imported lazily so a missing SDK disables
  only that backend. `manager.CameraManager` scans every backend on a
  background thread and turns the diff into added/removed/returned events;
  `session.CameraSession` decides what the viewer shows: opens on a worker
  thread, goes offline on disconnect, reopens and restores state on return.
  At start-up it waits `STARTUP_WAIT_S` (2 s) for the first camera, then opens
  the window on the simulator while a slower one (a Tucsen, ~7 s) finishes.
  Anything testable without hardware must not import pypylon at module level.
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
- `updater.py` — self-update. A git checkout fetches its branch's upstream on
  a background thread and fast-forwards at the *next* start
  (`apply_at_startup`, called in `main()` before the rest of the package is
  imported, then `relaunch` re-execs); the Windows exe compares the release
  tag CI stamps into `_build.py` (gitignored) with the latest GitHub release,
  downloads a newer one as `<exe>.<tag>.new` and renames it into place at the
  next start. U restarts into a ready update (`Viewer.restart_requested` →
  `updater.restart`).
- `ui/viewer.py` — main loop (`step()`: service camera events → grab →
  process → draw, then keys); owns all mutable state, which `server.py` (Flask,
  daemon thread) and `status.py` read. `ui/overlays.py` draws the status bar,
  info/statistics cards, icon toolbar + tooltips, spots, grid lines, the
  disconnect banner and toasts, all through `ui/gfx.py` (text, panels, sprite
  caches) and `ui/icons.py` (procedural icons, also used by Tk). `ui/theme.py`
  is the one palette for overlays and Tk. `ui/settings.py` and
  `ui/cameras_window.py` are Tk Toplevels on the single Tk thread in
  `ui/tkhost.py`. Toolbar buttons and keys both go through `Viewer._act`.

### Conventions that bite

- **Three coordinate systems**: *sensor* coords (full chip; rectangles are
  stored in these), *ROI* coords (grabbed frame; spot fits are in these;
  sensor = ROI + offset), and *display* coords (the window's real image
  size, `Viewer.canvas`, followed every frame - any w x h, not a fixed square;
  `roi.ViewTransform` maps between them). All overlay drawing happens in
  display space so text/lines don't scale with zoom. The canvas only changes
  right before a frame is composed (`_follow_window` → `step` → `_layout`), so
  mouse events always map through the view and toolbar of the image on screen.
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
- **Only the main thread touches a Camera.** The scanner thread only
  enumerates; opening runs on the manager's worker thread and the result is
  handed over. The Tk windows and the HTTP server read `manager.snapshot()`
  and `manager.post(...)` requests that `CameraSession.service()` carries out
  each frame. Hardware calls from the viewer go through `cams.guard(...)`, which
  turns a dead camera into the offline state instead of an exception.
- **Camera getters are cached.** `ROI`/`ExposureTime`/`Gain` return the last
  value written (read back once), never a node read: the HUD calls them every
  frame, and they must keep answering after an unplug so the session can save
  the state it restores on reconnect. `close()` must never raise.
- **The wheel belongs to the ROI, not to Qt.** OpenCV's Qt viewer magnifies
  the displayed image on every wheel event and on Ctrl +/- (overlays and all;
  no switch to turn it off). `ui/desktop.LiveWindow` takes passive X11 grabs
  (xcb via ctypes) on the wheel buttons and Ctrl +/-/0 for the live window and
  feeds them to `Viewer._wheel` / `_act`; clicks and motion still go to
  OpenCV. `desktop.prepare_qt()` (before the first cv2 window) sets
  `QT_XCB_NO_XI2=1`: a real mouse/touchpad on XWayland also sends XInput2
  smooth-scroll motion, which no button grab intercepts and Qt zooms on.
  XTest on Xvfb only produces button presses, so Xvfb tests cannot show that
  case. Off X11 the cv2 mouse callback delivers the wheel to the same
  `_wheel`. Use xcb, not Xlib, for anything X11 here: Xlib's default error
  handler exits the process on a stray BadWindow.
- **One Tk root per process** (`tkhost`). Never call `tk.Tk()` elsewhere; queue
  work with `tkhost.call(fn)`, which runs `fn(root)` on the Tk thread. Make Tk
  images with `tkhost.icon()`/`dot()`: a plain `PhotoImage` garbage-collected
  by the main thread at exit calls into Tcl and hangs the shutdown.
- **Overlays blend cached sprites; they do not draw text per frame.** Text,
  icons, chips, the toolbar and cards are rendered once into `gfx.SpriteCache`s
  keyed by their full content (text, colour, state) - a change is a new key, so
  nothing is ever invalidated by hand, and the LRU bound caps memory. Put every
  input of a rendering into its key. Numbers that change every frame go through
  `Viewer._readout` / the spot-label memo (`overlays.READOUT_PERIOD`) so they
  re-render a few times a second, not 60. Per-spot graphics must stay batched
  (one `cv2.polylines` per colour, never a loop of cv2 calls per spot) - an
  80x80 array is 6400 spots. Panels use `gfx.panel`/`gfx.shade` (addWeighted
  on a solid image), which beat a full-sprite blend by ~5x.
- **All UI sizes go through the scale; no raw pixel constants.** The UI zoom
  (`uiconfig.scale()`, 50-250 %, persisted in ui_config.json) sizes every
  overlay through an `overlays.Metrics` (`m.px(40)`, `m.font(13)`, `m.stroke`)
  and every Tk size through `tkhost.S()`. A cached rendering must have the
  scale (or the scaled font/size) in its key, so a zoom re-renders text and
  icons rather than resampling them. Widgets that would not fit the window at
  the chosen zoom (toolbar, cards, help card, banner) shrink to fit
  (`overlays._fit_scale`, `fit_cards`); nothing else may override the scale.
  Tk windows rebuild on a zoom (`rescale`, via `tkhost.on_rescale`): register
  new windows there, and keep their `class_=desktop.WM_CLASS`.
  `uiconfig` is not a fit parameter - never put it in `fitconfig.SETTINGS`.
- **Pillow is optional at runtime.** `gfx` falls back to Hershey text behind the
  same `Font` interface (`gfx.use_pillow(False)` forces it; tests run both).
  Icons are drawn with cv2, never from files - the one-file exe ships no assets.
  Non-ASCII characters (µ, σ, ×) are substituted when the font lacks them.
- **Nothing a camera does may end the loop.** `Viewer.run` catches per-frame
  exceptions (logged once per 10 s, shown as a toast); only Esc/close quits.
  Basler timeouts are not disconnects — `IsCameraDeviceRemoved()` decides; for
  a Tucsen on USB, sysfs does; other backends (and Tucsen over CameraLink) give
  up after `MAX_CONSECUTIVE_FAILURES` failed grabs.
- **The TUCam SDK is scanned only when something changed.** Its `Api_Init`
  takes ~1 s, prints a screenful to stdout and must not run while a Tucsen is
  open, so `tucsen.py` asks it at start-up, on Rescan, after a failed open and
  when the Tucsen devices in `/sys/bus/usb/devices` change - never on the
  2-second timer by itself. Serials are unreadable before the ~5 s open, so
  Tucsen uids are SDK indices (`tucsen:0`).
- `cameraconfig.save_entry` patches camera_config.yaml as text and refuses to
  write unless the parsed result equals the old file plus exactly the change.
  Keep that check; the file is hand-edited and its comments matter.
- **The updater never changes files under a running copy.** A one-file exe
  reads every lazy import from its own file by path, and a checkout imports
  `server` and the Tk windows on demand, so a swap or merge under a running
  process breaks it later. Both happen only at start-up and only when
  `others_running()` (locks on `data/.instances/<pid>`) is empty; never over
  local changes or unpushed commits; a pip failure rolls the merge back.
- CV changes must keep `tests/` passing — they check detection count,
  sub-pixel position error, widths, orientation and grid classification
  against synthetic ground truth.
