# Adding and configuring cameras

Any camera the app can enumerate will open: Basler cameras through pylon,
other industrial (GenICam) cameras through `harvesters`, and USB webcams
through OpenCV — see [Camera types](#camera-types). The only thing the
application cannot work out for itself is the **pixel size**, because cameras
do not report it — it has to come from the datasheet, and every micron figure
in the app scales with it.

## Adding a camera from the GUI

Plug the camera in while the app is running. Within a couple of seconds a
"Camera connected" message appears and the camera is listed in the **Cameras**
window (`k`, or the *Cameras* button in the toolbar):

- double-click it (or *Switch to this camera*) to show it; `t` cycles through
  the connected cameras;
- fill in **Pixel size** (µm, from the sensor datasheet) and optionally a
  **Label**, then *Save to camera_config.yaml*.

The pixel size is saved under the model name, so every unit of that model
gets it; the label is saved under the serial number, because it names one
physical camera. The file is edited in place — comments, order and
formatting are kept — and the change applies immediately.

If the app started with no camera attached it shows the simulator, and
switches to the first real camera you plug in. Once you have picked a camera
yourself (or started with `--sim` or `--camera`), new cameras are announced
but not switched to.

## Adding a camera from the command line

Plug it in and ask:

```bash
python -m beam_profiler --list-cameras
```

```
a2A2590-22umBAS  (serial 40123456, Basler)
  sensor                 2592, 1944
  pixel formats          Mono8, Mono10, Mono10p, Mono12, Mono12p
  exposure range (us)    20.0, 10000000.0
  I/O lines              Line1, Line2, Line3, Line4
  trigger sources        Software, Line1
  status                 NOT in camera_config.yaml
  resolved               a2A2590-22umBAS: pixel size UNSET, exposure 30-100000 us, trigger auto

Paste into the `cameras:` block of /opt/profiler/camera_config.yaml:

  a2A2590-22umBAS:
    # sensor 2592 x 1944 px, serial 40123456
    pixel_size: null   # TODO: pixel pitch in METRES from the datasheet, e.g. 2.74e-6
    pixel_format_16bit: [Mono12, Mono12p, Mono10, Mono10p]
    # camera allows 20 - 10000000 us exposure
```

Paste that block, replace `null` with the pitch in metres, done. On a machine
with no `camera_config.yaml` yet, `--write-camera-config` creates the whole
file from the connected cameras instead (it never overwrites an existing one).

Both flags work on the detected hardware only — they do not start the viewer.

## Running before you have the datasheet

A model that is not in the file still opens, on the defaults. It logs

```
WARNING camera model 'a2A2590-22umBAS' is not in camera_config.yaml ...
WARNING no pixel_size for 'a2A2590-22umBAS': micron readings will equal pixel counts.
```

and the status bar shows an amber *pixel size unset: µm = px* chip next to the
camera name. Pixel
counts, fits, widths and the grid statistics are all still correct — only the
conversion to microns is a placeholder of 1 µm/px, and `/api/status` reports
`"pixel_size_known": false` so scripts can refuse to trust `*_um` fields.

## How the file resolves

```yaml
defaults:                  # every camera
  max_exposure: 50000

cameras:
  "a2A*BAS":               # glob over a whole family
    trigger_mode: auto
  a2A5060-15umBAS:         # exact model name, as the camera reports it
    pixel_size: 2.5e-6

serials:
  "40615333":              # one physical unit
    label: "profiler head, west arm"
    default_roi: [2048, 2048, 1500, 1500]
```

Each step overrides the one before it, **field by field**, so a glob can set
the family's I/O lines while each model sets only its own pixel size:

```
built-in defaults  <  defaults:  <  cameras: <glob>  <  cameras: <model>  <  serials: <serial>
```

Among globs, the one with more literal characters wins (`a2A5060*` beats
`a2A*`). Matching is case-sensitive.

Bad values do not stop the application: an unparseable setting is logged and
skipped, and the rest of the block still applies.

## The settings

`beam_profiler/cameraconfig.py` is the registry — every setting, its type and
its help text live in `SETTINGS` there, and nothing else reads
`camera_config.yaml`. Adding a knob means adding a field plus its `Setting`.

| Setting | Default | What it does |
| --- | --- | --- |
| `pixel_size` | *unset* | Pixel pitch in **metres**. The one value worth setting. |
| `label` | model name | Free text for the HUD, the logs and `/api/status`. |
| `default_roi` | full sensor | `[width, height, offset_x, offset_y]` to open on, clamped to the chip. |
| `pixel_format_8bit` | `[Mono8]` | Candidates for `--mode 8Bit`, tried in order. |
| `pixel_format_16bit` | `[Mono12, Mono12p, Mono16, Mono10]` | Candidates for 16-bit mode, tried in order. |
| `min_exposure` / `max_exposure` | 30 / 100000 µs | The window the UI and auto-exposure stay inside, further narrowed to what the camera reports. |
| `default_exposure` / `default_gain` | 200 µs / 0 | Applied once at start-up. |
| `trigger_mode` | `auto` | `auto` (software trigger if available), `software` (require it), `"off"` (free-run). |
| `grab_timeout_ms` | 1000 | Raise it above your longest exposure plus readout. |
| `exposure_line` | `Line2` | Line driven with `ExposureActive`; `null` disables. Basler only. |
| `exposure_line_invert` | `true` | Makes that line active-low. |
| `user_output_line` | `Line3` | Line pulsed around each grab (`y` in the viewer); `null` disables. Basler only. |
| `user_output_selector` | `UserOutput1` | Which `UserOutput` bit drives it. |

Quote `"off"`: YAML reads a bare `off` as the boolean false. (It is accepted
either way, but quoting is what the file means.)

A feature the camera does not have is a warning, never a crash — a model with
no software trigger free-runs, and a missing output line is simply not driven.

## Multiple cameras, unplugging and reconnecting

The app looks for cameras every two seconds in the background, so cameras can
be plugged in and out while it runs. One camera is open at a time; switching
(Cameras window, `t`, `--camera NAME` at start-up, or `POST /api/select_camera`)
opens the new one on a worker thread while the old one keeps streaming, then
swaps. Each camera keeps its own ROI, exposure, gain, rectangles and toggles,
and gets them back when you return to it.

When the camera in use is unplugged or stops responding, the app does **not**
exit: the picture dims under a *Camera disconnected* banner, `/api/status`
reports `"camera_connected": false`, and as soon as the camera is seen again it
is reopened with everything as it was — ROI, exposure, rectangles, fitting.
Meanwhile you can switch to another camera. A camera that is present but will
not open (another program has it) is retried every few seconds.

Use the `serials:` block for anything that belongs to the installation rather
than to the model.

## Camera types

| Type | Covers | Needs |
| --- | --- | --- |
| **Basler (pylon)** | every Basler camera | the pylon SDK and `pip install pypylon` (in `requirements.txt`) |
| **GenICam (harvesters)** | FLIR/Teledyne, Allied Vision, IDS, Lucid, Daheng, Hikrobot, Baumer, ... | `pip install "harvesters>=1.4"` plus the vendor's SDK, which installs the GenTL producer (`.cti`) and puts it on `GENICAM_GENTL64_PATH` |
| **USB video (UVC)** | webcams, USB microscopes, cheap USB beam cameras | nothing extra (OpenCV) |
| **Simulator** | a synthetic spot grid | nothing |

The Cameras window lists each type as *ready* or *not available*, with the
reason. `--backends basler,uvc` restricts the search; `--list-cameras` probes
every available type.

Notes per type:

- **GenICam** uses the standard feature names (`ExposureTime`, `Gain`,
  `PixelFormat`, `TriggerMode`, `Width`/`OffsetX`, ...), the same ones pylon
  uses, so the pixel-format and trigger settings below mean the same thing.
  It does not drive I/O lines. Extra `.cti` files or folders can be listed in
  `BEAM_PROFILER_CTI`. Producers inside a pylon installation are skipped when
  pypylon is installed, so Basler cameras are not listed twice. This backend
  follows the harvesters 1.4 API but has not yet been run against real
  non-Basler hardware — report what happens on yours.
- **USB video** cameras are 8-bit and often colour; frames are converted to
  mono, 16-bit mode shifts them to the top byte, and the ROI is cropped in
  software. Exposure control depends on the OS driver. They are never opened
  automatically (a laptop webcam must not take over from the profiler camera)
  — pick one in the Cameras window. Their model name is whatever the OS
  reports, and that is the key for `pixel_size`.
- **Another SDK** (Thorlabs, Andor, ...) is one file: a `Backend` and a
  `Camera` subclass in `beam_profiler/cameras/` (see `base.py`; `uvc.py` is the
  shortest example) and a line in `cameras/__init__.py`.

## Without hardware

The simulated camera (`--sim`) resolves through the same path under the model
name `Simulated-<grid>`, so a `"Simulated-*"` entry exercises the file with no
camera attached. See `CLAUDE.md` for the rest of the no-hardware workflow.
