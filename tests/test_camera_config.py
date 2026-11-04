"""camera_config.yaml: resolution order, type checking and the generated stubs.

No pypylon here - everything under test is pure, which is the point of keeping
per-model settings out of the driver.
"""

import textwrap

import pytest
import yaml

from beam_profiler import cameraconfig
from beam_profiler.cameraconfig import (
    CameraConfigFile,
    CameraSettings,
    describe,
    file_template,
    load,
    resolve,
    yaml_stub,
)
from beam_profiler.cameras import SimulatedCamera
from beam_profiler.roi import ROIModel


def write(tmp_path, text: str) -> CameraConfigFile:
    path = tmp_path / "camera_config.yaml"
    path.write_text(textwrap.dedent(text))
    return load(str(path))


# --------------------------------------------------------------------------- #
#  resolution order
# --------------------------------------------------------------------------- #
def test_unknown_model_falls_back_instead_of_raising():
    """The whole point: an unlisted Basler must still open."""
    settings = resolve("a2A9999-99umBAS", "123", CameraConfigFile(), warn=False)
    assert settings == CameraSettings()
    assert not settings.pixel_size_known
    assert settings.effective_pixel_size == cameraconfig.UNKNOWN_PIXEL_SIZE


def test_precedence_defaults_glob_model_serial(tmp_path):
    cfg = write(tmp_path, """
        defaults:
          default_exposure: 100
          max_exposure: 50000
          label: from-defaults
        cameras:
          "a2A*":
            label: from-glob
            default_exposure: 200
          a2A5060-15umBAS:
            label: from-model
            pixel_size: 2.5e-6
        serials:
          "4061":
            label: from-serial
        """)
    assert resolve("other", "", cfg).label == "from-defaults"
    assert resolve("a2A1234", "", cfg).label == "from-glob"
    assert resolve("a2A5060-15umBAS", "", cfg).label == "from-model"
    assert resolve("a2A5060-15umBAS", "4061", cfg).label == "from-serial"

    # unmentioned fields keep falling through to the less specific source
    settings = resolve("a2A5060-15umBAS", "4061", cfg)
    assert settings.default_exposure == 200  # glob
    assert settings.max_exposure == 50000  # defaults
    assert settings.pixel_size == 2.5e-6  # model
    assert settings.min_exposure == CameraSettings().min_exposure  # built-in


def test_more_literal_glob_wins(tmp_path):
    cfg = write(tmp_path, """
        cameras:
          "a2A*": {label: broad}
          "a2A5060*": {label: narrow}
        """)
    assert resolve("a2A5060-15umBAS", "", cfg).label == "narrow"
    assert resolve("a2A4504-18umBAS", "", cfg).label == "broad"


def test_glob_is_case_sensitive(tmp_path):
    """fnmatchcase, so a model name is never matched by accident on Windows."""
    cfg = write(tmp_path, 'cameras:\n  "a2A*": {label: ace2}\n')
    assert resolve("a2A5060", "", cfg).label == "ace2"
    assert resolve("A2A5060", "", cfg, warn=False).label == ""


# --------------------------------------------------------------------------- #
#  type checking
# --------------------------------------------------------------------------- #
def test_bad_value_is_dropped_and_the_rest_of_the_block_survives(tmp_path):
    cfg = write(tmp_path, """
        cameras:
          cam:
            pixel_size: "not a number"
            default_exposure: 500
        """)
    settings = resolve("cam", "", cfg)
    assert settings.default_exposure == 500
    assert not settings.pixel_size_known


def test_unknown_setting_is_ignored(tmp_path):
    cfg = write(tmp_path, "cameras:\n  cam: {pixel_zise: 2.5e-6, label: typo}\n")
    settings = resolve("cam", "", cfg)
    assert settings.label == "typo"
    assert not settings.pixel_size_known


@pytest.mark.parametrize("value", ["0", "0.0", "-2.5e-6"])
def test_non_positive_pixel_size_counts_as_unset(tmp_path, value):
    """A stub pasted but not filled in must not read as a calibrated 0 um/px."""
    cfg = write(tmp_path, f"cameras:\n  cam: {{pixel_size: {value}}}\n")
    assert not resolve("cam", "", cfg).pixel_size_known


def test_bare_off_trigger_survives_yamls_booleans(tmp_path):
    """YAML 1.1 reads an unquoted `off` as False; it still has to mean off."""
    cfg = write(tmp_path, """
        cameras:
          quoted: {trigger_mode: "off"}
          bare: {trigger_mode: off}
          upper: {trigger_mode: "OFF"}
          bad: {trigger_mode: maybe}
        """)
    assert resolve("quoted", "", cfg).trigger_mode == "off"
    assert resolve("bare", "", cfg).trigger_mode == "off"
    assert resolve("upper", "", cfg).trigger_mode == "off"
    assert resolve("bad", "", cfg).trigger_mode == "auto"  # rejected, default kept


def test_pixel_format_accepts_a_string_or_a_list(tmp_path):
    cfg = write(tmp_path, """
        cameras:
          one: {pixel_format_16bit: Mono16}
          many: {pixel_format_16bit: [Mono12, Mono16]}
          empty: {pixel_format_16bit: []}
        """)
    assert resolve("one", "", cfg).pixel_format_16bit == ("Mono16",)
    assert resolve("many", "", cfg).pixel_format_16bit == ("Mono12", "Mono16")
    assert resolve("empty", "", cfg).pixel_format_16bit == CameraSettings().pixel_format_16bit


def test_line_can_be_disabled_with_null(tmp_path):
    cfg = write(tmp_path, "cameras:\n  cam: {exposure_line: null, user_output_line: null}\n")
    settings = resolve("cam", "", cfg)
    assert settings.exposure_line is None and settings.user_output_line is None


@pytest.mark.parametrize("roi", ["[1, 2, 3]", "[0, 10, 0, 0]", "[10, 10, -1, 0]", "nonsense"])
def test_malformed_roi_is_rejected(tmp_path, roi):
    cfg = write(tmp_path, f"cameras:\n  cam: {{default_roi: {roi}}}\n")
    assert resolve("cam", "", cfg).default_roi is None


def test_roi_is_read_in_camera_order(tmp_path):
    cfg = write(tmp_path, "cameras:\n  cam: {default_roi: [640, 480, 10, 20]}\n")
    assert resolve("cam", "", cfg).default_roi == (640, 480, 10, 20)


# --------------------------------------------------------------------------- #
#  the file itself
# --------------------------------------------------------------------------- #
def test_missing_file_gives_defaults(tmp_path):
    cfg = load(str(tmp_path / "absent.yaml"))
    assert cfg.cameras == {} and resolve("cam", "", cfg, warn=False) == CameraSettings()


@pytest.mark.parametrize("text", ["", "[1, 2, 3]\n", "cameras: 7\n", ":\n  bad: yaml: here\n"])
def test_broken_file_gives_defaults(tmp_path, text):
    cfg = write(tmp_path, text)
    assert resolve("cam", "", cfg, warn=False) == CameraSettings()


def test_legacy_file_still_loads(tmp_path):
    """The pre-2026 format was a bare `cameras:` map with a default_roi whose
    offsets were really padding; it must keep working, pixel sizes intact."""
    cfg = write(tmp_path, """
        cameras:
          a2A5060-15umBAS:
            default_roi: [5060, 5060, 4, 4]
            pixel_size: 2.5e-6
        """)
    settings = resolve("a2A5060-15umBAS", "", cfg)
    assert settings.pixel_size == 2.5e-6
    # an oversized ROI is clamped back to the full sensor, as before
    assert ROIModel(5060, 5060, _clamp(settings.default_roi, 5060, 5060)).tuple == (5060, 5060, 0, 0)


def _clamp(roi, w_max, h_max):
    w, h = min(roi[0], w_max), min(roi[1], h_max)
    return (w, h, min(roi[2], w_max - w), min(roi[3], h_max - h))


def test_settings_registry_matches_the_dataclass():
    """SETTINGS drives the type checking and the docs, so it cannot drift."""
    assert {s.name for s in cameraconfig.SETTINGS} == {
        f.name for f in CameraSettings.__dataclass_fields__.values()
    }
    for setting in cameraconfig.SETTINGS:
        assert setting.help.strip(), f"{setting.name} has no help text"


def test_shipped_config_is_valid():
    """The repo's own camera_config.yaml must parse and resolve cleanly."""
    cfg = load(cameraconfig.config_path())
    assert cfg.models, "no models configured"
    for model in cfg.models:
        assert resolve(model, "", cfg).pixel_size_known, f"{model} has no pixel_size"


# --------------------------------------------------------------------------- #
#  generated stubs
# --------------------------------------------------------------------------- #
ACE2_PROBE = dict(
    model="a2A2590-22umBAS", serial="40123456", vendor="Basler",
    sensor_size=(2592, 1944),
    pixel_formats=["Mono8", "Mono10", "Mono10p", "Mono12", "Mono12p"],
    exposure_range=(20.0, 1e7),
    lines=["Line1", "Line2", "Line3", "Line4"],
    trigger_sources=["Software", "Line1"],
)
DART_PROBE = dict(
    model="daA1920-160um", serial="24001122", vendor="Basler",
    sensor_size=(1920, 1200), pixel_formats=["Mono8"],
    exposure_range=(26.0, 1e6), lines=["Line1"], trigger_sources=["Line1"],
)


@pytest.mark.parametrize("probe", [ACE2_PROBE, DART_PROBE, {}])
def test_stub_is_valid_yaml(probe):
    text = "cameras:\n" + yaml_stub(probe.get("model", "cam"), probe.get("serial", ""), probe)
    assert yaml.safe_load(text)["cameras"]


def test_stub_leaves_pixel_size_unset_rather_than_zero(tmp_path):
    """Paste the stub, forget the datasheet: the app must say so, not report 0."""
    cfg = write(tmp_path, "cameras:\n" + yaml_stub("cam", "1", ACE2_PROBE))
    assert not resolve("cam", "", cfg).pixel_size_known


def test_stub_keeps_only_the_pixel_formats_the_camera_has(tmp_path):
    cfg = write(tmp_path, "cameras:\n" + yaml_stub("cam", "", ACE2_PROBE))
    formats = resolve("cam", "", cfg).pixel_format_16bit
    assert formats == ("Mono12", "Mono12p", "Mono10", "Mono10p")  # deepest first
    assert "Mono16" not in formats  # this model does not have it


def test_stub_disables_absent_lines_rather_than_guessing(tmp_path):
    """Line1 is an opto *input* on most Baslers - driving it would be worse
    than driving nothing."""
    cfg = write(tmp_path, "cameras:\n" + yaml_stub("cam", "", DART_PROBE))
    settings = resolve("cam", "", cfg)
    assert settings.exposure_line is None and settings.user_output_line is None
    assert settings.trigger_mode == "off"  # no Software trigger source


def test_generated_file_round_trips(tmp_path):
    stubs = [yaml_stub(p["model"], p["serial"], p) for p in (ACE2_PROBE, DART_PROBE)]
    path = tmp_path / "camera_config.yaml"
    path.write_text(file_template(stubs))
    cfg = load(str(path))
    assert cfg.models == ["a2A2590-22umBAS", "daA1920-160um"]
    assert resolve("daA1920-160um", "", cfg).trigger_mode == "off"


def test_file_template_without_cameras_is_still_valid():
    assert yaml.safe_load(file_template([])) is not None


def test_describe_flags_an_unset_pixel_size():
    assert "UNSET" in describe(CameraSettings(), "cam")
    assert "2.5 um/px" in describe(CameraSettings(pixel_size=2.5e-6), "cam")


# --------------------------------------------------------------------------- #
#  the camera side
# --------------------------------------------------------------------------- #
def test_simulated_camera_reads_the_config(tmp_path, monkeypatch):
    cfg = write(tmp_path, """
        cameras:
          "Simulated-*":
            label: bench sim
            pixel_size: 3.45e-6
            default_exposure: 500
            default_gain: 2
            default_roi: [400, 300, 50, 60]
        """)
    monkeypatch.setattr(cameraconfig, "_cached", cfg)
    cam = SimulatedCamera(width=1200, height=1200, grid=(3, 3), jitter=0.0)
    assert cam.label == "bench sim"
    assert cam.pixel_size == 3.45e-6 and cam.pixel_size_known
    assert cam.ExposureTime == 500 and cam.Gain == 2
    assert cam.default_roi == (400, 300, 50, 60)


def test_default_roi_is_clamped_to_the_sensor(tmp_path, monkeypatch):
    cfg = write(tmp_path, """
        cameras:
          "Simulated-*": {default_roi: [99999, 99999, 7, 7]}
        """)
    monkeypatch.setattr(cameraconfig, "_cached", cfg)
    cam = SimulatedCamera(width=640, height=480, grid=(2, 2), jitter=0.0)
    assert cam.default_roi == (640, 480, 0, 0)


def test_unconfigured_camera_still_grabs_frames():
    cam = SimulatedCamera(width=320, height=240, grid=(2, 2), jitter=0.0)
    assert cam.pixel_size == 2.5e-6  # the simulator's own fallback
    assert cam.grab_image().shape == (240, 320)


def test_auto_exposure_cannot_widen_the_configured_limits(tmp_path, monkeypatch):
    cfg = write(tmp_path, """
        cameras:
          "Simulated-*": {min_exposure: 100, max_exposure: 5000}
        """)
    monkeypatch.setattr(cameraconfig, "_cached", cfg)
    cam = SimulatedCamera(width=320, height=240, grid=(2, 2), jitter=0.0)
    cam.AutoExposureOn(range=(1, 1_000_000))
    assert (cam.min_exposure, cam.max_exposure) == (100, 5000)
    cam.AutoExposureOn(range=(200, 1000))  # narrowing is allowed
    assert (cam.min_exposure, cam.max_exposure) == (200, 1000)
    cam.AutoExposureOff()
    assert (cam.min_exposure, cam.max_exposure) == (100, 5000)


def test_auto_exposure_restores_the_hardware_narrowed_window(tmp_path, monkeypatch):
    """A driver may narrow the configured window to what the camera accepts;
    toggling auto-exposure must not widen it back to the configured values."""
    cfg = write(tmp_path, """
        cameras:
          "Simulated-*": {min_exposure: 10, max_exposure: 1000000}
        """)
    monkeypatch.setattr(cameraconfig, "_cached", cfg)
    cam = SimulatedCamera(width=320, height=240, grid=(2, 2), jitter=0.0)
    cam.set_exposure_limits(26, 20000)  # what the hardware really allows
    cam.AutoExposureOn(range=(100, 5000))
    cam.AutoExposureOff()
    assert (cam.min_exposure, cam.max_exposure) == (26, 20000)


def test_roi_model_reproduces_a_configured_roi():
    model = ROIModel(2000, 1000, (400, 300, 120, 50))
    assert model.tuple == (400, 300, 120, 50)


# --------------------------------------------------------------------------- #
#  --list-cameras
# --------------------------------------------------------------------------- #
def _fake_probes(monkeypatch, probes):
    from beam_profiler import cameras

    monkeypatch.setattr(cameras, "probe_devices", lambda: probes)


def test_list_cameras_without_hardware(monkeypatch, capsys):
    from beam_profiler.__main__ import list_cameras

    _fake_probes(monkeypatch, [])
    assert list_cameras() == 1
    assert "No Basler cameras found" in capsys.readouterr().out


def test_list_cameras_reports_and_offers_a_stub(monkeypatch, capsys, tmp_path):
    from beam_profiler.__main__ import list_cameras

    _fake_probes(monkeypatch, [ACE2_PROBE])
    monkeypatch.setattr(cameraconfig, "config_path", lambda: str(tmp_path / "absent.yaml"))
    assert list_cameras() == 0
    out = capsys.readouterr().out
    assert "a2A2590-22umBAS" in out and "40123456" in out
    assert "NOT in camera_config.yaml" in out
    assert "Mono12p" in out  # what the camera reports
    assert "pixel_size" in out  # the block to paste


def test_list_cameras_nags_until_the_pixel_size_is_filled_in(
    monkeypatch, capsys, tmp_path
):
    """Listing the model is not enough - an unset pixel_size must stay visible."""
    from beam_profiler.__main__ import list_cameras

    path = tmp_path / "camera_config.yaml"
    path.write_text("cameras:\n  a2A2590-22umBAS: {}\n")
    monkeypatch.setattr(cameraconfig, "config_path", lambda: str(path))
    _fake_probes(monkeypatch, [ACE2_PROBE])
    assert list_cameras() == 0
    out = capsys.readouterr().out
    assert "listed, but pixel_size is not set" in out
    assert "Still needs a pixel_size" in out

    path.write_text("cameras:\n  a2A2590-22umBAS: {pixel_size: 2.74e-6}\n")
    assert list_cameras() == 0
    out = capsys.readouterr().out
    assert "Every camera above is configured" in out


def test_write_camera_config_never_overwrites(monkeypatch, capsys, tmp_path):
    from beam_profiler.__main__ import list_cameras

    path = tmp_path / "camera_config.yaml"
    monkeypatch.setattr(cameraconfig, "config_path", lambda: str(path))
    monkeypatch.setattr(cameraconfig, "default_config_path", lambda: str(path))
    _fake_probes(monkeypatch, [ACE2_PROBE])

    assert list_cameras(write=True) == 0
    assert load(str(path)).models == ["a2A2590-22umBAS"]

    before = path.read_text()
    assert list_cameras(write=True) == 1  # refuses the second time
    assert path.read_text() == before
    assert "already exists" in capsys.readouterr().out
