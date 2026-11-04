# Adding and configuring cameras

Any Basler that pylon can enumerate will open. The only thing the application
cannot work out for itself is the **pixel size**, because no Basler reports it
over GenICam — it has to come from the datasheet, and every micron figure in
the app scales with it.

## Adding a camera

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

and the HUD reads `Camera: a2A2590-22umBAS [px size unset: um = px]`. Pixel
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
  a2A5060-15umBAS:         # exact model name, as pylon reports it
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
| `min_exposure` / `max_exposure` | 30 / 100000 µs | The window the UI and auto-exposure stay inside, further narrowed to what pylon reports. |
| `default_exposure` / `default_gain` | 200 µs / 0 | Applied once at start-up. |
| `trigger_mode` | `auto` | `auto` (software trigger if available), `software` (require it), `"off"` (free-run). |
| `grab_timeout_ms` | 1000 | Raise it above your longest exposure plus readout. |
| `exposure_line` | `Line2` | Line driven with `ExposureActive`; `null` disables. |
| `exposure_line_invert` | `true` | Makes that line active-low. |
| `user_output_line` | `Line3` | Line pulsed around each grab (`y` in the viewer); `null` disables. |
| `user_output_selector` | `UserOutput1` | Which `UserOutput` bit drives it. |

Quote `"off"`: YAML reads a bare `off` as the boolean false. (It is accepted
either way, but quoting is what the file means.)

A feature the camera does not have is a warning, never a crash — a model with
no software trigger free-runs, and a missing output line is simply not driven.

## Multiple cameras

`open_cameras` opens every enumerated device, and `t` cycles through them. Each
keeps its own ROI, exposure, gain and rectangles, so two cameras with different
`default_roi` and `label` settings behave independently. Use the `serials:`
block for anything that belongs to the installation rather than to the model.

## Without hardware

The simulated camera (`--sim`) resolves through the same path under the model
name `Simulated-<grid>`, so a `"Simulated-*"` entry exercises the file with no
camera attached. See `CLAUDE.md` for the rest of the no-hardware workflow.
