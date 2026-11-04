"""Per-camera settings: the registry, `camera_config.yaml` and its resolution.

Everything that differs from one camera model to the next lives here rather
than as a literal in `cameras/basler.py`, so supporting another Basler is a few
lines of YAML instead of a code change.  Add a field to `CameraSettings` plus
its `SETTINGS` entry and it is loadable, type-checked and documented
everywhere; do not read `camera_config.yaml` anywhere else.

A model that is not in the file is **not** an error: every field falls back,
and the only one with no sensible default is `pixel_size` (see
`UNKNOWN_PIXEL_SIZE`).  `python -m beam_profiler --list-cameras` prints a
ready-to-paste block for whatever is plugged in, including which pixel formats
and I/O lines that model actually exposes.

Lookup order, each step overriding the previous field by field:

1. the defaults below,
2. the `defaults:` block of the file,
3. every matching glob in `cameras:`, least specific first,
4. the exact model name in `cameras:`,
5. the exact serial number in `serials:`.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, fields, replace

import yaml

from .config import APP_DIR

CONFIG_NAME = "camera_config.yaml"

# Stand-in scale when a model has no `pixel_size`: 1 um/px makes every micron
# reading numerically equal to the pixel count, so an uncalibrated number is
# obvious rather than plausible.  `Camera.pixel_size_known` is False and the
# HUD says so.
UNKNOWN_PIXEL_SIZE = 1.0e-6

TRIGGER_MODES = ("auto", "software", "off")


@dataclass(frozen=True)
class CameraSettings:
    # --- geometry and scale ------------------------------------------------
    label: str = ""
    pixel_size: float | None = None
    default_roi: tuple[int, int, int, int] | None = None
    # --- pixel format ------------------------------------------------------
    pixel_format_8bit: tuple[str, ...] = ("Mono8",)
    pixel_format_16bit: tuple[str, ...] = ("Mono12", "Mono12p", "Mono16", "Mono10")
    # --- exposure and gain -------------------------------------------------
    min_exposure: float = 30.0
    max_exposure: float = 100_000.0
    default_exposure: float = 200.0
    default_gain: float = 0.0
    # --- acquisition -------------------------------------------------------
    trigger_mode: str = "auto"
    grab_timeout_ms: int = 1000
    # --- digital I/O -------------------------------------------------------
    exposure_line: str | None = "Line2"
    exposure_line_invert: bool = True
    user_output_line: str | None = "Line3"
    user_output_selector: str = "UserOutput1"

    def replace(self, **kw) -> CameraSettings:
        return replace(self, **{k: v for k, v in kw.items() if k in _NAMES})

    @property
    def pixel_size_known(self) -> bool:
        return self.pixel_size is not None

    @property
    def effective_pixel_size(self) -> float:
        """Always a usable number, so callers never see None."""
        return UNKNOWN_PIXEL_SIZE if self.pixel_size is None else self.pixel_size


@dataclass(frozen=True)
class Setting:
    """Type and documentation for one CameraSettings field."""

    name: str
    kind: str  # float | optfloat | int | bool | str | optstr | strlist | roi | trigger
    help: str


SETTINGS: tuple[Setting, ...] = (
    Setting(
        "label", "str",
        "Free-text name for the HUD and the logs.  Purely cosmetic; the model\n"
        "name is used when it is empty.",
    ),
    Setting(
        "pixel_size", "optfloat",
        "Sensor pixel pitch in METRES (2.5e-6 = 2.5 um).  The one value that\n"
        "cannot be guessed and is not readable over pylon - take it from the\n"
        "model's datasheet.  Every micron figure in the app scales with it.\n"
        "Left out, the app still runs but reports microns == pixels and says\n"
        "'px size unset' on the HUD.",
    ),
    Setting(
        "default_roi", "roi",
        "ROI to open with, as [width, height, offset_x, offset_y] in sensor\n"
        "pixels - the same order as everywhere else in the code.  Null (the\n"
        "default) uses the whole sensor.  Oversized values are clamped, so\n"
        "[5060, 5060, 0, 0] on a 5060 px chip just means 'full frame'.",
    ),
    Setting(
        "pixel_format_8bit", "strlist",
        "Candidate pylon PixelFormat values for '8Bit' mode, tried in order\n"
        "until one is accepted.  A single string is fine.",
    ),
    Setting(
        "pixel_format_16bit", "strlist",
        "Candidate pylon PixelFormat values for '16Bit' mode, tried in order.\n"
        "Output is always converted MSB-aligned into uint16, so any Mono depth\n"
        "works and saturation stays 65535.  The default list covers ace, ace 2,\n"
        "dart and pulse; --list-cameras prints what your model really has.",
    ),
    Setting(
        "min_exposure", "float",
        "Shortest exposure the app will ask for, in microseconds.  Also the\n"
        "lower bound for auto-exposure.",
    ),
    Setting(
        "max_exposure", "float",
        "Longest exposure, in microseconds.  Lower it on a camera whose frame\n"
        "rate collapses before the pylon limit.",
    ),
    Setting(
        "default_exposure", "float",
        "Exposure in microseconds set once at start-up.",
    ),
    Setting(
        "default_gain", "float",
        "Gain set once at start-up, in the camera's own units (dB on most\n"
        "Baslers).  0 is the right answer for beam profiling.",
    ),
    Setting(
        "trigger_mode", "trigger",
        "'auto'     - use software triggering if the model has it, else free-run\n"
        "             (the safe default),\n"
        "'software' - require it, and log an error if it is missing,\n"
        "'off'      - never trigger; let the camera free-run.\n"
        "Quote it: YAML reads a bare `off` as the boolean false.",
    ),
    Setting(
        "grab_timeout_ms", "int",
        "How long to wait for a frame before pylon raises.  Raise it above the\n"
        "longest exposure you intend to use, plus readout.",
    ),
    Setting(
        "exposure_line", "optstr",
        "Output line driven with ExposureActive, for scope / DAQ sync - 'Line2'\n"
        "on most Baslers.  Null disables it.  A line the model does not have is\n"
        "a warning, not a failure.",
    ),
    Setting(
        "exposure_line_invert", "bool",
        "True makes that line active-low.",
    ),
    Setting(
        "user_output_line", "optstr",
        "Line pulsed low around each acquisition when exposure sync is on ('y'\n"
        "in the viewer) - 'Line3' on most Baslers.  Null disables it.",
    ),
    Setting(
        "user_output_selector", "str",
        "Which UserOutput bit drives `user_output_line`.",
    ),
)

_NAMES = {f.name for f in fields(CameraSettings)}
assert {s.name for s in SETTINGS} == _NAMES, "SETTINGS and CameraSettings are out of sync"
_KINDS = {s.name: s.kind for s in SETTINGS}


# --------------------------------------------------------------------------- #
#  the file
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraConfigFile:
    """Parsed `camera_config.yaml`, before any model is resolved against it."""

    defaults: dict = None  # type: ignore[assignment]
    cameras: dict = None  # type: ignore[assignment]
    serials: dict = None  # type: ignore[assignment]
    path: str = ""

    def __post_init__(self):
        for name in ("defaults", "cameras", "serials"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})

    @property
    def models(self) -> list[str]:
        """Configured keys that are plain model names, not globs."""
        return sorted(k for k in self.cameras if not _is_glob(k))


def default_config_path() -> str:
    """Where the file belongs: next to the .exe when frozen, else the repo root.
    This is what `--write-camera-config` creates."""
    return os.path.join(APP_DIR, CONFIG_NAME)


def config_path() -> str:
    """The file to read: next to the app, falling back to the working directory
    so a copy beside whatever launched the app still wins over nothing."""
    path = default_config_path()
    return path if os.path.exists(path) else os.path.join(".", CONFIG_NAME)


def load(path: str | None = None) -> CameraConfigFile:
    """Read the file.  A missing or broken file is a warning, not an error -
    the app then runs every camera on the defaults."""
    path = path or config_path()
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning(
            f"{CONFIG_NAME} not found at {path}; using default camera settings. "
            "Run `python -m beam_profiler --list-cameras` to generate one."
        )
        return CameraConfigFile(path=path)
    except Exception as e:
        logging.error(f"could not read {path} ({e}); using default camera settings")
        return CameraConfigFile(path=path)

    if not isinstance(raw, dict):
        logging.error(f"{path}: top level must be a mapping; using defaults")
        return CameraConfigFile(path=path)

    unknown = set(raw) - {"defaults", "cameras", "serials"}
    if unknown:
        logging.warning(f"{path}: ignoring unknown top-level keys {sorted(unknown)}")
    return CameraConfigFile(
        defaults=_as_mapping(raw.get("defaults"), f"{path}: defaults"),
        cameras=_as_mapping(raw.get("cameras"), f"{path}: cameras"),
        serials=_as_mapping(raw.get("serials"), f"{path}: serials"),
        path=path,
    )


_cached: CameraConfigFile | None = None


def active_file() -> CameraConfigFile:
    """The file as loaded once per process."""
    global _cached
    if _cached is None:
        _cached = load()
    return _cached


def reload(path: str | None = None) -> CameraConfigFile:
    global _cached
    _cached = load(path)
    return _cached


# --------------------------------------------------------------------------- #
#  resolution
# --------------------------------------------------------------------------- #
def resolve(
    model: str,
    serial: str = "",
    file: CameraConfigFile | None = None,
    warn: bool = True,
) -> CameraSettings:
    """Settings for one physical camera.  Never raises: a model the file says
    nothing about comes back on the defaults.  `warn=False` for callers that are
    not real hardware (the simulator) and so need no nagging about the file."""
    file = active_file() if file is None else file
    merged: dict = {}
    sources = [("defaults", file.defaults)]
    globs = sorted(
        (k for k in file.cameras if _is_glob(k) and fnmatch.fnmatchcase(model, k)),
        key=_specificity,
    )
    sources += [(f"cameras[{g}]", file.cameras[g]) for g in globs]
    if model in file.cameras:
        sources.append((f"cameras[{model}]", file.cameras[model]))
    if serial and serial in file.serials:
        sources.append((f"serials[{serial}]", file.serials[serial]))

    for where, block in sources:
        merged.update(_coerce_block(block, where))

    if warn and len(sources) == 1:  # nothing but `defaults:` matched
        known = ", ".join(file.models) or "none"
        logging.warning(
            f"camera model {model!r} is not in {os.path.basename(file.path)} "
            f"(configured: {known}); using default settings. "
            f"Add it with:\n{yaml_stub(model, serial)}"
        )
    settings = CameraSettings(**merged)
    if warn and not settings.pixel_size_known:
        logging.warning(
            f"no pixel_size for {model!r}: micron readings will equal pixel counts. "
            "Add `pixel_size: <metres>` from the datasheet to "
            f"{os.path.basename(file.path)}."
        )
    return settings


def _coerce_block(block, where: str) -> dict:
    """Type-check one YAML mapping against SETTINGS, dropping what it cannot use."""
    out: dict = {}
    if not isinstance(block, dict):
        if block is not None:
            logging.warning(f"{where}: expected a mapping, got {type(block).__name__}")
        return out
    for key, value in block.items():
        key = str(key)
        if key not in _KINDS:
            logging.warning(f"{where}: ignoring unknown setting {key!r}")
            continue
        try:
            out[key] = _coerce(value, _KINDS[key])
        except (TypeError, ValueError) as e:
            logging.warning(f"{where}: ignoring bad value for {key}: {e}")
    return out


def _coerce(value, kind: str):
    if kind in ("optfloat", "optstr", "roi") and value is None:
        return None
    if kind == "optfloat":
        number = float(value)
        if number <= 0:
            raise ValueError(f"must be positive, got {number!r} (leave it out if unknown)")
        return number
    if kind == "float":
        return float(value)
    if kind == "int":
        return int(value)
    if kind == "bool":
        if isinstance(value, str):  # yaml already handles true/false
            raise ValueError(f"{value!r} is not a boolean")
        return bool(value)
    if kind == "str":
        return str(value)
    if kind == "optstr":
        text = str(value).strip()
        return text or None
    if kind == "strlist":
        if isinstance(value, str):
            return (value,)
        items = tuple(str(v) for v in value)
        if not items:
            raise ValueError("needs at least one entry")
        return items
    if kind == "trigger":
        if value is False:  # YAML 1.1 reads a bare `off` as a boolean
            return "off"
        text = str(value).strip().lower()
        if text not in TRIGGER_MODES:
            raise ValueError(f"expected one of {TRIGGER_MODES}, got {value!r}")
        return text
    if kind == "roi":
        roi = tuple(int(v) for v in value)
        if len(roi) != 4:
            raise ValueError(f"needs 4 numbers [w, h, x, y], got {len(roi)}")
        if roi[0] <= 0 or roi[1] <= 0:
            raise ValueError(f"width and height must be positive, got {roi[:2]}")
        if roi[2] < 0 or roi[3] < 0:
            raise ValueError(f"offsets cannot be negative, got {roi[2:]}")
        return roi
    raise AssertionError(f"unhandled kind {kind}")


def _as_mapping(value, where: str) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    logging.warning(f"{where}: expected a mapping, got {type(value).__name__}")
    return {}


def _is_glob(key: str) -> bool:
    return any(c in str(key) for c in "*?[")


def _specificity(pattern: str) -> int:
    """Literal characters in a glob.  More literal = more specific = applied
    later, so `a2A5060*` wins over `a2A*` and both lose to an exact name."""
    return sum(c not in "*?[]!-" for c in pattern)


# --------------------------------------------------------------------------- #
#  generating config
# --------------------------------------------------------------------------- #
def yaml_stub(model: str, serial: str = "", probe: dict | None = None) -> str:
    """A paste-ready `cameras:` entry for one camera.

    `probe` is the optional dict from `cameras.basler.probe_device`: whatever it
    reports off the camera is written in as comments, so the only thing left to
    fill in by hand is the pixel size."""
    probe = probe or {}
    lines = [f"  {model}:"]
    detail = [f"serial {serial}"] if serial else []
    if probe.get("sensor_size"):
        detail.insert(0, "sensor %d x %d px" % probe["sensor_size"])
    if detail:
        lines.append(f"    # {', '.join(detail)}")

    pixel_size = probe.get("pixel_size")
    if pixel_size:
        lines.append(f"    pixel_size: {pixel_size:.3e}   # reported by the camera")
    else:
        lines.append(
            "    pixel_size: null   # TODO: pixel pitch in METRES from the datasheet, "
            "e.g. 2.74e-6"
        )

    formats = probe.get("pixel_formats") or []
    mono16 = [f for f in formats if _mono_depth(f) > 8]
    if mono16 and tuple(mono16) != CameraSettings().pixel_format_16bit:
        order = sorted(mono16, key=lambda f: (-_mono_depth(f), "p" in f))
        lines.append(f"    pixel_format_16bit: [{', '.join(order)}]")
    mono8 = [f for f in formats if _mono_depth(f) == 8]
    if mono8 and tuple(mono8) != CameraSettings().pixel_format_8bit:
        lines.append(f"    pixel_format_8bit: [{', '.join(mono8)}]")
    if formats and not mono16 and not mono8:
        lines.append(f"    # no Mono pixel format found; camera offers: {', '.join(formats)}")

    exposure = probe.get("exposure_range")
    if exposure:
        lines.append(f"    # camera allows {exposure[0]:.0f} - {exposure[1]:.0f} us exposure")

    lines += _line_stub(probe)
    if probe.get("trigger_sources") is not None:
        if "Software" not in probe["trigger_sources"]:
            lines.append('    trigger_mode: "off"   # no software trigger on this model')
    return "\n".join(lines)


def _line_stub(probe: dict) -> list[str]:
    """Comment or override the I/O line defaults from what the camera reports."""
    available = probe.get("lines")
    if available is None:
        return []
    out = []
    for field, default in (
        ("exposure_line", CameraSettings().exposure_line),
        ("user_output_line", CameraSettings().user_output_line),
    ):
        if default in available:
            continue
        # Disable rather than pick one: on most Baslers Line1 is an opto input,
        # and driving the wrong line is worse than driving none.
        why = f"lines: {', '.join(available)}" if available else "no I/O lines"
        out.append(f"    {field}: null   # {default} absent ({why})")
    return out


def _mono_depth(fmt: str) -> int:
    """Bit depth of a 'MonoNN[p]' pixel format, 0 if it is not mono."""
    if not fmt.startswith("Mono"):
        return 0
    digits = "".join(c for c in fmt[4:] if c.isdigit())
    return int(digits) if digits else 0


FILE_TEMPLATE = """\
# Camera settings for the Basler Beam Profiler.
#
# Keys under `cameras:` are matched against the camera's model name, exactly as
# pylon reports it, and may use globs (`a2A*BAS:`).  `defaults:` applies to
# every camera, `serials:` overrides one physical unit.  Later and more
# specific wins, field by field:
#
#   defaults  <  cameras: <glob>  <  cameras: <model>  <  serials: <serial>
#
# A model that is not listed still works, on the defaults - only `pixel_size`
# really has to be set, because it scales every micron reading in the app.
# `python -m beam_profiler --list-cameras` prints a block for whatever is
# plugged in; `beam_profiler/cameraconfig.py` documents every setting.

defaults:
  # min_exposure: 30        # us
  # max_exposure: 100000    # us
  # default_exposure: 200   # us
  # trigger_mode: auto      # auto | software | "off"  (quote it!)
  # exposure_line: Line2    # ExposureActive output, null to disable
  # user_output_line: Line3 # pulsed around each grab, null to disable

cameras:
{entries}
# serials:
#   "40615333":
#     label: "profiler head, west arm"
#     default_roi: [2048, 2048, 1500, 1500]
"""


def file_template(entries: list[str]) -> str:
    """A whole `camera_config.yaml` for the cameras described by `entries`
    (each the output of `yaml_stub`)."""
    body = "\n\n".join(entries) if entries else "  # a2A5060-15umBAS:\n  #   pixel_size: 2.5e-6"
    return FILE_TEMPLATE.format(entries=body + "\n")


def describe(settings: CameraSettings, model: str = "") -> str:
    """One-line summary for the logs."""
    name = settings.label or model
    if settings.pixel_size_known:
        scale = f"{settings.pixel_size * 1e6:g} um/px"
    else:
        scale = "pixel size UNSET"
    return (
        f"{name}: {scale}, exposure {settings.min_exposure:g}-{settings.max_exposure:g} us, "
        f"trigger {settings.trigger_mode}"
    )
