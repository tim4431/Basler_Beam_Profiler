"""The fit defaults are measured optima, not guesses - so re-measure them here.

If one of these fails, the default has drifted away from what the data says;
change the default (and docs/fitting-calibration.md), do not loosen the test.
"""

import json

import numpy as np
import pytest

from beam_profiler import fitconfig
from beam_profiler.fitconfig import FitConfig, clamp, load_config, save_config
from beam_profiler.processing import fit_spot
from beam_profiler.synthetic import Spot, render

SIGMA = 6.0
CENTRE = 800.0
SEEDS = range(6)


def _width_error(crop_sigma, amplitude, clip=False, adaptive=0, seed=0):
    """Relative error of the fitted width for one noise realisation."""
    cfg = FitConfig(crop_sigma=crop_sigma, clip_negative=clip, adaptive_iters=adaptive)
    img = render(1600, 1600, [Spot(CENTRE, CENTRE, sigma_x=SIGMA, amplitude=amplitude)],
                 noise=0.003, background=0.01, rng=np.random.default_rng(100 + seed))
    got = fit_spot(img, CENTRE, CENTRE, SIGMA, cfg)
    return got["sigma_0"] / (2 * SIGMA) - 1


def _rms(crop_sigma, amplitude, **kw):
    errs = np.array([_width_error(crop_sigma, amplitude, seed=s, **kw) for s in SEEDS])
    return float(np.sqrt(errs.mean() ** 2 + errs.std() ** 2))


# --------------------------------------------------------------------------- #
#  the numbers
# --------------------------------------------------------------------------- #
def test_default_crop_sigma_is_the_minimum_rms_choice():
    """crop_sigma trades truncation (too small) against tail noise (too big).
    The default must sit at the minimum of that curve across brightnesses."""
    grid = [2.5, 3.0, 3.5, 4.0, 4.5]
    assert FitConfig().crop_sigma in grid
    worst = {n: max(_rms(n, 0.8), _rms(n, 0.15)) for n in grid}
    best = min(worst, key=worst.get)
    assert best == FitConfig().crop_sigma, f"minimum-RMS crop_sigma is now {best}: {worst}"


def test_three_stages_are_enough_and_two_are_not():
    """The blob radius spans r/sigma = 0.75-1.7; re-cropping has to converge
    from the worst of those."""
    img = render(1600, 1600, [Spot(CENTRE, CENTRE, sigma_x=SIGMA, amplitude=0.8)],
                 noise=0.003, background=0.01, rng=np.random.default_rng(1))
    seeds = [0.75 * SIGMA, 1.7 * SIGMA]
    spread = []
    for stages in (0, FitConfig().crop_stages):
        cfg = FitConfig(crop_stages=stages)
        widths = [fit_spot(img, CENTRE, CENTRE, s, cfg)["sigma_0"] for s in seeds]
        spread.append(abs(widths[0] - widths[1]) / np.mean(widths))
    assert spread[0] > 0.10, "seed spread should matter without re-cropping"
    assert spread[1] < 0.01, f"3 stages left {spread[1]:.1%} seed dependence"


def test_clipping_is_off_because_it_diverges_with_crop_size():
    """Clipping rectifies zero-mean noise into a positive pedestal that the
    second moment amplifies by distance^2.  This is why the default is off."""
    assert FitConfig().clip_negative is False
    clipped = _width_error(8.0, amplitude=0.15, clip=True)
    plain = _width_error(8.0, amplitude=0.15, clip=False)
    assert clipped > 0.20, f"clip pathology gone? {clipped:.1%}"
    assert abs(plain) < 0.5 * clipped


def test_adaptive_iterations_remove_the_crop_dependence():
    """With a matched Gaussian weight the answer stops depending on the window."""
    span = [_width_error(n, 0.8, adaptive=8) for n in (3.5, 5.0, 8.0)]
    assert max(span) - min(span) < 0.01, f"adaptive still window-dependent: {span}"
    plain = [_width_error(n, 0.8) for n in (3.5, 5.0, 8.0)]
    assert max(plain) - min(plain) > max(span) - min(span)


# --------------------------------------------------------------------------- #
#  the plumbing
# --------------------------------------------------------------------------- #
def test_settings_cover_every_field_with_a_sane_range():
    for s in fitconfig.SETTINGS:
        value = getattr(FitConfig(), s.name)
        assert s.kind in ("float", "int", "bool")
        assert s.lo <= float(value) <= s.hi, f"{s.name} default is outside its own range"
        assert s.help.strip(), f"{s.name} has no explanation"


def test_clamp_forces_values_into_range_and_type():
    wild = FitConfig(crop_sigma=1e6, crop_stages=-4, adaptive_iters=999, crop_quantum=0)
    got = clamp(wild)
    assert got.crop_sigma == 12.0
    assert got.crop_stages == 0
    assert got.adaptive_iters == 20
    assert got.crop_quantum == 1


def test_config_round_trips_through_json(tmp_path):
    path = str(tmp_path / "fit_config.json")
    cfg = FitConfig(crop_sigma=4.25, adaptive_iters=6, gpu_enabled=False)
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_unknown_and_broken_files_fall_back_to_defaults(tmp_path):
    good = tmp_path / "extra.json"
    good.write_text(json.dumps({"crop_sigma": 4.0, "not_a_setting": 1}), encoding="utf-8")
    assert load_config(str(good)).crop_sigma == 4.0

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_config(str(broken)) == FitConfig()
    assert load_config(str(tmp_path / "missing.json")) == FitConfig()


def test_replace_ignores_unknown_keys():
    assert FitConfig().replace(crop_sigma=5.0, bogus=1).crop_sigma == 5.0


def test_set_active_swaps_atomically_and_notifies(tmp_path, monkeypatch):
    monkeypatch.setattr(fitconfig, "CONFIG_PATH", str(tmp_path / "fit_config.json"))
    seen = []
    fitconfig.on_change(seen.append)
    before = fitconfig.active()
    try:
        fitconfig.set_active(FitConfig(crop_sigma=4.0), save=False)
        assert fitconfig.active().crop_sigma == 4.0
        assert seen and seen[-1].crop_sigma == 4.0
    finally:
        fitconfig._listeners.remove(seen.append) if seen.append in fitconfig._listeners else None
        fitconfig._listeners.clear()
        fitconfig.set_active(before, save=False)


@pytest.mark.parametrize("iters", [0, 4])
def test_describe_mentions_the_mode(iters):
    text = fitconfig.describe(FitConfig(adaptive_iters=iters))
    assert ("adaptive" in text) == (iters > 0)
    assert "crop" in text


# --------------------------------------------------------------------------- #
#  the settings window, driven without showing it
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def window(tk_root):
    from beam_profiler.ui.settings import SettingsWindow

    win = SettingsWindow()
    win.build(tk_root)
    win.win.withdraw()
    yield win
    win.destroy()


def test_settings_window_builds_a_control_for_every_setting(window):
    for s in fitconfig.SETTINGS:
        assert s.name in window._vars, f"{s.name} has no widget"


def test_settings_window_edits_apply_and_persist(tmp_path, monkeypatch, window):
    path = tmp_path / "fit_config.json"
    monkeypatch.setattr(fitconfig, "CONFIG_PATH", str(path))
    before = fitconfig.active()
    win = window
    try:
        win._vars["crop_sigma"].set("4.25")
        win._vars["adaptive_iters"].set("6")
        win._vars["clip_negative"].set(True)
        win._apply()
        assert fitconfig.active().crop_sigma == 4.25
        assert fitconfig.active().adaptive_iters == 6
        assert fitconfig.active().clip_negative is True
        assert json.loads(path.read_text(encoding="utf-8"))["crop_sigma"] == 4.25

        # out-of-range typing is clamped, not crashed on
        win._vars["crop_sigma"].set("999")
        win._apply()
        assert fitconfig.active().crop_sigma == 12.0

        # unparseable input keeps the value in force
        win._vars["crop_sigma"].set("banana")
        win._apply()
        assert fitconfig.active().crop_sigma == 12.0

        win._defaults()
        assert fitconfig.active() == FitConfig()
    finally:
        fitconfig.set_active(before, save=False)


def test_settings_window_picks_up_external_changes(tmp_path, monkeypatch, window):
    monkeypatch.setattr(fitconfig, "CONFIG_PATH", str(tmp_path / "fit_config.json"))
    before = fitconfig.active()
    try:
        fitconfig.set_active(FitConfig(crop_sigma=5.5), save=False)  # e.g. over HTTP
        window._poll_external()
        assert window._vars["crop_sigma"].get() == "5.5"
    finally:
        fitconfig.set_active(before, save=False)
