"""The overlay layer without a window: sprite caching, blending, text with
and without Pillow, icons, toolbar hit-testing, and the viewer drawing every
overlay state headlessly."""

import json

import cv2
import numpy as np
import pytest

from beam_profiler.cameras import CameraManager
from beam_profiler.cameras.simulated import SimulatedBackend
from beam_profiler.config import WINDOW_SIZE
from beam_profiler.processing.spots import SpotArray
from beam_profiler.roi import ViewTransform
from beam_profiler import uiconfig
from beam_profiler.ui import gfx, icons, overlays
from beam_profiler.ui import viewer as viewer_mod


@pytest.fixture(params=["pillow", "hershey"])
def text_backend(request):
    """Run a test with TrueType text and again with the no-Pillow fallback."""
    if request.param == "pillow" and not gfx.HAVE_PIL:
        pytest.skip("Pillow not installed")
    gfx.use_pillow(request.param == "pillow")
    yield request.param
    gfx.use_pillow(True)


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    from beam_profiler import status
    from beam_profiler.processing import gpu

    for name in ("namedWindow", "resizeWindow", "setMouseCallback"):
        monkeypatch.setattr(cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(gpu, "warm_up", lambda: None)
    monkeypatch.setattr(status, "STATUS_JSON_PATH", str(tmp_path / "status.json"))
    manager = CameraManager([SimulatedBackend(width=480, height=480, grid=(3, 3))],
                            scan_interval=None)
    v = viewer_mod.Viewer(manager, simulate=True)
    yield v
    v.cams.close()


# --------------------------------------------------------------------------- #
#  sprite cache
# --------------------------------------------------------------------------- #
def test_sprite_cache_renders_each_key_once_and_evicts_the_oldest():
    cache = gfx.SpriteCache(capacity=2)
    calls = []

    def render(key):
        calls.append(key)
        return f"sprite {key}"

    assert cache.get("a", lambda: render("a")) == "sprite a"
    assert cache.get("a", lambda: render("a")) == "sprite a"
    assert calls == ["a"] and (cache.hits, cache.misses) == (1, 1)
    cache.get("b", lambda: render("b"))
    cache.get("a", lambda: render("a"))  # touch a: b is now the oldest
    cache.get("c", lambda: render("c"))
    assert "b" not in cache and "a" in cache and "c" in cache and len(cache) == 2
    cache.get("b", lambda: render("b"))
    assert calls == ["a", "b", "c", "b"]


def test_changed_text_is_a_new_sprite_and_unchanged_text_is_reused(text_backend):
    f = gfx.font(13)
    a = gfx.text_sprite("12.3 µm", f, (255, 255, 255))
    assert gfx.text_sprite("12.3 µm", f, (255, 255, 255)) is a
    assert gfx.text_sprite("12.4 µm", f, (255, 255, 255)) is not a
    assert gfx.text_sprite("12.3 µm", f, (0, 255, 0)) is not a  # colour is content too


def test_reset_drops_everything_rendered():
    f = gfx.font(13)
    a = gfx.text_sprite("x", f, (1, 2, 3))
    gfx.reset()
    assert gfx.text_sprite("x", gfx.font(13), (1, 2, 3)) is not a


# --------------------------------------------------------------------------- #
#  blending
# --------------------------------------------------------------------------- #
def test_blend_is_exact_at_full_and_zero_coverage():
    img = np.full((10, 10, 3), 40, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[:2] = 255
    gfx.Sprite.from_mask(mask, (10, 200, 250)).draw(img, 3, 3)
    assert (img[3:5, 3:7] == (10, 200, 250)).all()  # opaque: the colour
    assert (img[5:7, 3:7] == 40).all()  # transparent: untouched
    assert (img[:3] == 40).all() and (img[7:] == 40).all()


def test_half_coverage_mixes_and_opacity_fades():
    img = np.zeros((4, 4, 3), np.uint8)
    gfx.Sprite.from_mask(np.full((4, 4), 128, np.uint8), (200, 200, 200)).draw(img, 0, 0)
    assert 98 <= img[0, 0, 0] <= 102
    img[:] = 0
    gfx.Sprite.from_mask(np.full((4, 4), 255, np.uint8), (200, 200, 200)).draw(
        img, 0, 0, opacity=0.25)
    assert 48 <= img[0, 0, 0] <= 52


def test_sprites_clip_at_every_edge():
    img = np.zeros((20, 30, 3), np.uint8)
    s = gfx.Sprite.from_mask(np.full((8, 8), 255, np.uint8), (0, 0, 255))
    for x, y in ((-4, -4), (26, 16), (-20, 5), (40, 40), (5, -10)):
        s.draw(img, x, y)  # partly or wholly outside: no exception
    assert (img[0:4, 0:4] == (0, 0, 255)).all() and (img[16:20, 26:30] == (0, 0, 255)).all()


def test_halo_text_outlines_beyond_the_glyphs(text_backend):
    f = gfx.font(14)
    s = gfx.text_sprite("8", f, (255, 255, 255), halo=(0, 0, 0))
    assert (s.ox, s.oy) == (-2, -2) and s.w == s.tw + 4
    img = np.full((40, 40, 3), 200, np.uint8)
    s.draw(img, 10, 10)
    assert img.min() < 120  # a dark rim around white text on a bright ground


def test_panel_is_a_uniform_shade_with_rounded_corners():
    a = np.full((60, 80, 3), 200, np.uint8)
    b = a.copy()
    gfx.panel(a, 10, 10, 70, 50, (20, 20, 20), 0.8, radius=8)
    gfx.shade(b, 10, 10, 70, 50, (20, 20, 20), 0.8)
    assert np.array_equal(a[20:40, 20:60], b[20:40, 20:60])  # interior
    assert a[10, 10, 0] > b[10, 10, 0] + 50  # the corner pixel is cut away
    assert (a[:10] == 200).all() and (a[50:] == 200).all()


# --------------------------------------------------------------------------- #
#  text and icons
# --------------------------------------------------------------------------- #
def test_text_renders_with_and_without_pillow(text_backend):
    f = gfx.font(14)
    m = f.mask("ROI 1200 × 1200 µm σ")
    assert m.dtype == np.uint8 and m.shape[0] == f.height and m.max() > 200
    assert f.width("WWW") > f.width("iii")  # proportional (or at least not empty)
    assert gfx.pillow_active() == (text_backend == "pillow")


@pytest.mark.skipif(not gfx.HAVE_PIL, reason="Pillow not installed")
def test_a_glyph_the_font_lacks_is_substituted(monkeypatch):
    """Pillow's bundled font has no Greek: σ must become s, not a box."""
    monkeypatch.setattr(gfx, "_faces_found", (None, None))
    gfx.reset()
    try:
        f = gfx.Font(14)
        if f._ft is None:
            pytest.skip("Pillow too old for a sized default font")
        assert np.array_equal(f.mask("σ"), f.mask("s"))
        assert f.width("1.0") > 0
    finally:
        gfx.reset()


@pytest.mark.parametrize("name", icons.NAMES)
def test_every_icon_renders(name):
    m = icons.mask(name, 22)
    assert m.shape == (22, 22) and m.dtype == np.uint8
    coverage = (m > 128).mean()
    assert 0.03 < coverage < 0.6, f"{name} is {coverage:.0%} ink"
    png = cv2.imdecode(np.frombuffer(icons.png(name, 16, "#3b82f6"), np.uint8),
                       cv2.IMREAD_UNCHANGED)
    assert png.shape == (16, 16, 4) and png[..., 3].max() > 128


# --------------------------------------------------------------------------- #
#  toolbar
# --------------------------------------------------------------------------- #
def test_toolbar_hit_testing():
    bar = overlays.Toolbar(viewer_mod.TOOLS, WINDOW_SIZE)
    x0, y0, x1, y1 = bar.box
    assert 0 < x0 and x1 < WINDOW_SIZE and y1 == WINDOW_SIZE - bar.margin
    assert abs((x0 + x1) / 2 - WINDOW_SIZE / 2) <= 1  # centred
    names = [t.name for t in viewer_mod.TOOLS if t is not None]
    assert list(bar.rects) == names
    for name, (bx0, by0, bx1, by1) in bar.rects.items():
        assert bar.hit((bx0 + bx1) // 2, (by0 + by1) // 2) == name
        assert bar.hit(bx0, by0) == name and bar.hit(bx1 - 1, by1 - 1) == name
        assert x0 <= bx0 < bx1 <= x1 and y0 <= by0 < by1 <= y1
    # between two buttons, and on a group separator: the dock, but no button
    a, b = bar.rects["fit"], bar.rects["stats"]
    gap = ((a[2] + b[0]) // 2, (a[1] + a[3]) // 2)
    sep = (bar.separators[0], gap[1])
    for point in (gap, sep, (x0 + 2, y0 + 2)):
        assert bar.hit(*point) is None and bar.contains(*point)
    for point in ((x0 - 1, gap[1]), (gap[0], y0 - 1), (WINDOW_SIZE // 2, 300)):
        assert bar.hit(*point) is None and not bar.contains(*point)
    rects = sorted(bar.rects.values())
    assert all(r1[2] <= r2[0] for r1, r2 in zip(rects, rects[1:]))  # no overlaps


def test_toolbar_sprite_is_cached_per_state():
    bar = overlays.Toolbar(viewer_mod.TOOLS, WINDOW_SIZE)
    img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8)
    bar.draw(img, {"fit": False})
    misses = bar._docks.misses
    plain = img.copy()
    for _ in range(3):
        bar.draw(img, {"fit": False})
    assert bar._docks.misses == misses  # same state: blended, not re-rendered
    img[:] = 0
    bar.draw(img, {"fit": True})
    assert bar._docks.misses == misses + 1
    x0, y0, x1, y1 = bar.rects["fit"]
    assert not np.array_equal(img[y0:y1, x0:x1], plain[y0:y1, x0:x1])  # highlighted
    bar.draw(img, {"fit": True}, hover="stats")
    bar.draw(img, {"fit": True}, disabled={"exp_up"})
    assert bar._docks.misses == misses + 3


def test_tooltip_appears_above_the_hovered_button():
    bar = overlays.Toolbar(viewer_mod.TOOLS, WINDOW_SIZE)
    a = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8)
    b = a.copy()
    bar.draw(a, {}, hover="help", tooltip=True)
    bar.draw(b, {}, hover="help", tooltip=False)
    diff = np.argwhere((a != b).any(axis=2))
    assert len(diff) and diff[:, 0].max() < bar.box[1]  # only above the dock
    assert diff[:, 1].max() < WINDOW_SIZE - bar.m.edge + 1  # kept inside the window


# --------------------------------------------------------------------------- #
#  spots, grid, readouts
# --------------------------------------------------------------------------- #
def _spots(xs, ys, sigma=10.0):
    n = len(xs)
    v = np.tile([1.0, 0.0], (n, 1))
    return SpotArray(xs, ys, np.full(n, sigma), np.full(n, sigma * 0.8), v, v[:, ::-1],
                     np.ones(n), np.ones(n))


def test_draw_spots_takes_columns_dicts_and_nothing():
    img = np.zeros((200, 200, 3), np.uint8)
    view = ViewTransform((200, 200, 0, 0), 200)
    overlays.draw_spots(img, SpotArray.empty(), view, 2.5e-6)
    assert not img.any()
    spots = _spots(np.array([50.0, 150.0]), np.array([60.0, 60.0]), sigma=20)
    overlays.draw_spots(img, spots, view, 2.5e-6)
    assert img.any()
    overlays.draw_spots(img, [spots[0], spots[1]], view, 2.5e-6)  # legacy dicts


def test_grid_markers_only_where_there_is_room():
    from beam_profiler.processing import classify_grid

    def drawn(pitch_px):
        g = np.arange(1, 6) * pitch_px
        xs, ys = (a.ravel().astype(float) for a in np.meshgrid(g, g))
        rows, cols, _ = classify_grid(_spots(xs, ys, 2.0), eps=pitch_px / 3)
        img = np.zeros((6 * pitch_px, 6 * pitch_px, 3), np.uint8)
        overlays.draw_row_col(img, ViewTransform((img.shape[1], img.shape[0], 0, 0),
                                                 img.shape[0]), rows, cols)
        return img

    sparse, dense = drawn(60), drawn(10)
    # a marker ring sits off the lines: look half a radius above a spot
    assert sparse[60 - 5, 60 + 5].any()
    assert not dense[10 - 3, 10 + 3].any()
    assert dense[10, 12].any()  # the row line is still there


def test_dashed_column_pattern():
    img = np.zeros((60, 10, 3), np.uint8)
    overlays._dashed_column(img, 5, 3, 40, np.array([9, 9, 9], np.uint8))
    on = img[:, 4, 0] > 0
    assert not on[:3].any() and on[3:11].all() and not on[11:15].any() and on[15:23].all()
    assert not on[40:].any() and (img[:, 4] == img[:, 5]).all() and not img[:, 6].any()


def test_readout_refreshes_at_most_every_period_but_at_once_on_appearing():
    now = [0.0]
    r = viewer_mod.Readout(clock=lambda: now[0])
    assert r("k", 1) == 1
    now[0] = 0.05
    assert r("k", 2) == 1  # too soon: still showing the old digits
    now[0] = 0.05 + overlays.READOUT_PERIOD
    assert r("k", 3) == 3
    assert r("k", None) is None  # gone: at once
    assert r("k", 4) == 4  # back: at once


# --------------------------------------------------------------------------- #
#  the viewer
# --------------------------------------------------------------------------- #
def _frames(v, n=3):
    for _ in range(n):
        img = v.step()
    return img


def test_viewer_draws_every_overlay_state(viewer, text_backend):
    v = viewer
    assert _frames(v).shape == (WINDOW_SIZE, WINDOW_SIZE, 3)
    v.do_fitting = v.show_stats = v.show_rect_stats = True
    v.rect_sensor, v.fit_rect_sensor = (10, 10, 200, 150), (0, 0, 400, 400)
    v.rect_disp = [100, 100, 300, 250]
    v._notify("Saved something.jpg")
    v._notify("Something went wrong", "error")
    v._hover_button, v._hover_since = "fit", 0.0
    v._mouse_at = float("inf")
    _frames(v)
    assert len(v.latest_spots) == 9
    v.row_col_fitting = True
    _frames(v)
    v.show_help = True
    v.show_info = False
    _frames(v)
    v.cams.lost("unplugged in a test")  # the banner, then the sim comes straight back
    assert _frames(v, 1).shape == (WINDOW_SIZE, WINDOW_SIZE, 3)


def test_toolbar_buttons_and_keys_share_the_actions(viewer):
    v = viewer
    _frames(v, 1)
    v._toolbar_click("fit")
    assert v.do_fitting
    v._handle_key(ord("f"))
    assert not v.do_fitting
    e = v.camera.ExposureTime
    v._toolbar_click("exp_up")
    assert v.camera.ExposureTime == pytest.approx(int(e * 1.1), abs=1)
    v._toolbar_click("exp_down", cv2.EVENT_FLAG_SHIFTKEY)  # shift: /10
    assert v.camera.ExposureTime < e
    v.rect_sensor = v.fit_rect_sensor = (1, 1, 5, 5)
    v._toolbar_click("clear")
    assert v.rect_sensor is None and v.fit_rect_sensor is None
    v._toolbar_click("info")
    assert not v.show_info
    v._handle_key(ord("i"))
    assert v.show_info


def test_esc_closes_the_help_before_it_quits(viewer):
    v = viewer
    v._handle_key(ord("?"))
    assert v.show_help
    assert v._handle_key(27) is True and not v.show_help
    assert v._handle_key(65470) is True and v.show_help  # F1 (GTK code)
    v.show_help = False
    assert v._handle_key(27) is False


def test_mouse_on_the_dock_never_starts_a_rectangle(viewer):
    v = viewer
    _frames(v, 1)
    bar = v.toolbar
    a, b = bar.rects["fit"], bar.rects["stats"]
    gap = ((a[2] + b[0]) // 2, (a[1] + a[3]) // 2)
    v.on_mouse(cv2.EVENT_LBUTTONDOWN, *gap, 0, None)
    assert v.rect_disp is None and not v.do_fitting
    v.on_mouse(cv2.EVENT_MOUSEMOVE, (a[0] + a[2]) // 2, gap[1], 0, None)
    assert v._hover_button == "fit"
    v.on_mouse(cv2.EVENT_LBUTTONDOWN, (a[0] + a[2]) // 2, gap[1], 0, None)
    assert v.do_fitting and v.rect_disp is None
    v.on_mouse(cv2.EVENT_LBUTTONDOWN, 500, 500, 0, None)  # on the picture: a rectangle
    assert v.rect_disp == [500, 500, 500, 500] and v._hover_button is None


def test_saved_jpeg_is_the_8bit_view_of_a_16bit_frame(viewer, tmp_path):
    frame = np.full((8, 8), 0x1234, np.uint16)
    viewer._save_frame(str(tmp_path / "f.jpg"), frame)
    jpg = cv2.imread(str(tmp_path / "f.jpg"), cv2.IMREAD_GRAYSCALE)
    assert abs(int(jpg.mean()) - 0x12) <= 1
    assert np.array_equal(np.load(tmp_path / "f.npy"), frame)


# --------------------------------------------------------------------------- #
#  UI scale and window size
# --------------------------------------------------------------------------- #
def test_ui_config_round_trips_and_overrides_are_not_saved(tmp_path):
    path = tmp_path / "ui.json"
    uiconfig.save_config(uiconfig.UIConfig(scale=1.7, window=(1600, 900)), str(path))
    assert uiconfig.load_config(str(path)) == uiconfig.UIConfig(1.7, (1600, 900))

    seen = []
    uiconfig.on_change(seen.append)
    try:
        assert uiconfig.set_scale(1.5) == 1.5
        saved = json.loads(open(uiconfig.CONFIG_PATH).read())
        assert saved["scale"] == 1.5
        uiconfig.set_scale(2.0, save=False)  # --ui-scale: this session only
        assert uiconfig.scale() == 2.0
        uiconfig.set_window((1300, 700))  # saves the file, with the saved scale
        saved = json.loads(open(uiconfig.CONFIG_PATH).read())
        assert saved == {"scale": 1.5, "window": [1300, 700]}
        assert [c.scale for c in seen] == [1.5, 2.0]
        uiconfig.set_scale(2.0, save=False)
        assert len(seen) == 2  # no change, no notification
    finally:
        uiconfig._listeners.remove(seen.append)


@pytest.mark.parametrize("text, expect", [
    ("{not json", uiconfig.UIConfig()),
    ("[1, 2]", uiconfig.UIConfig()),
    ('{"scale": "big", "window": [1600, 900]}', uiconfig.UIConfig(1.0, (1600, 900))),
    ('{"scale": 9, "window": [5, 5]}', uiconfig.UIConfig(2.5, None)),
    ('{"scale": 0.73, "window": "wide"}', uiconfig.UIConfig(0.7, None)),
    ('{"scale": NaN}', uiconfig.UIConfig()),
])
def test_a_bad_ui_config_falls_back_field_by_field(tmp_path, text, expect):
    path = tmp_path / "ui.json"
    path.write_text(text)
    assert uiconfig.load_config(str(path)) == expect
    assert uiconfig.load_config(str(tmp_path / "missing.json")) == uiconfig.UIConfig()


def test_zoom_steps_are_clamped_to_the_range():
    uiconfig.set_scale(2.4)
    assert uiconfig.zoom(+1) == 2.5 and uiconfig.zoom(+1) == 2.5
    uiconfig.set_scale(0.6)
    assert uiconfig.zoom(-1) == 0.5 and uiconfig.zoom(-1) == 0.5
    assert uiconfig.reset_scale() == 1.0
    with pytest.raises(ValueError):
        uiconfig.clamp_scale(float("inf"))


@pytest.mark.parametrize("roi", [(1200, 1200, 0, 0), (4096, 1000, 0, 300), (640, 2000, 32, 8)])
@pytest.mark.parametrize("window", [1100, (1600, 900), (800, 1200), (333, 777)])
def test_view_transform_round_trips_in_any_window(roi, window):
    v = ViewTransform(roi, window)
    win_w, win_h = (window, window) if np.isscalar(window) else window
    assert v.view_w <= win_w and v.view_h <= win_h
    assert v.view_w == win_w or v.view_h == win_h  # fills one direction
    assert abs(v.view_w / v.view_h - roi[0] / roi[1]) < 0.02 * roi[0] / roi[1] + 2 / v.view_h
    assert abs(2 * v.pad_l + v.view_w - win_w) <= 1 and abs(2 * v.pad_t + v.view_h - win_h) <= 1
    w, h, ox, oy = roi
    for sx, sy in ((ox, oy), (ox + w / 3, oy + h / 2), (ox + 0.99 * w, oy + 0.99 * h)):
        dx, dy = v.to_display(sx, sy)
        assert v.contains(dx, dy)
        bx, by = v.to_sensor(dx, dy)
        assert abs(bx - sx) <= 1 / v.scale + 1e-9 and abs(by - sy) <= 1 / v.scale + 1e-9
    for dx, dy in ((v.pad_l, v.pad_t), (v.pad_l + v.view_w // 2, v.pad_t + v.view_h // 3)):
        assert v.to_display(*v.to_sensor(dx, dy)) == (dx, dy)


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.5])
@pytest.mark.parametrize("window", [(1100, 1100), (1600, 900), (800, 800), (480, 700)])
def test_toolbar_hit_testing_at_every_scale_and_size(scale, window):
    m = overlays.Metrics.at(scale)
    bar = overlays.Toolbar(viewer_mod.TOOLS, window, m)
    W, H = window
    x0, y0, x1, y1 = bar.box
    assert 0 <= x0 and x1 <= W and y1 == H - bar.margin and abs((x0 + x1) - W) <= 1
    assert bar.m.scale <= scale  # shrinks only when it would not fit
    if bar.m.scale < scale:
        assert x1 - x0 <= W - 2 * m.edge
    else:
        assert bar.button == round(overlays.Toolbar.BUTTON * scale)
    for name, (bx0, by0, bx1, by1) in bar.rects.items():
        assert bar.hit((bx0 + bx1) // 2, (by0 + by1) // 2) == name
        assert x0 <= bx0 < bx1 <= x1 and y0 <= by0 < by1 <= y1
    assert bar.hit(W // 2, y0 - 1) is None and not bar.contains(W // 2, y0 - 1)
    img = np.zeros((H, W, 3), np.uint8)
    bar.draw(img, {"fit": True}, hover="help")  # renders, tooltip included, inside the image


def test_text_is_rendered_at_the_new_size_not_resampled():
    small, big = overlays.Metrics.at(1.0).font(13), overlays.Metrics.at(2.0).font(13)
    assert big.size == 2 * small.size
    a, b = small.mask("Fit 1.0 µm"), big.mask("Fit 1.0 µm")
    assert 1.7 < b.shape[0] / a.shape[0] < 2.3 and 1.7 < b.shape[1] / a.shape[1] < 2.3
    assert icons.mask("fit", 44).shape == (44, 44)


def test_compose_follows_the_window_size_and_mouse_maps_through_it(viewer):
    v = viewer
    assert _frames(v, 1).shape == (WINDOW_SIZE, WINDOW_SIZE, 3)
    v.do_fitting = v.show_stats = True
    for size in ((1600, 900), (800, 1000), (500, 400)):
        v.canvas = size
        img = _frames(v, 2)
        assert img.shape == (size[1], size[0], 3)
        x0, y0, x1, y1 = v.toolbar.box
        assert x1 <= size[0] and y1 <= size[1]
        view = v.view
        dx, dy = view.to_display(100, 200)  # a sensor point, as drawn on screen
        v.on_mouse(cv2.EVENT_MOUSEMOVE, dx, dy, 0, None)
        assert abs(v.last_mouse[0] - 100) <= 2 and abs(v.last_mouse[1] - 200) <= 2
        bx = v.toolbar.rects["info"]
        before = v.show_info
        v.on_mouse(cv2.EVENT_LBUTTONDOWN, (bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2, 0, None)
        assert v.show_info is not before


def test_zoom_keys_rescale_the_live_view_and_say_so(viewer):
    v = viewer
    _frames(v, 1)
    button = v.toolbar.button
    v._handle_key(ord("+"))
    assert uiconfig.scale() == 1.1
    _frames(v, 1)
    assert v.ui.scale == 1.1 and v.toolbar.button > button
    assert any("110%" in t for _, t, _ in v.toasts)
    v._handle_key(ord("-"))
    v._handle_key(ord("-"))
    assert uiconfig.scale() == 0.9
    v._handle_key(ord("0"))
    assert uiconfig.scale() == 1.0
    assert json.loads(open(uiconfig.CONFIG_PATH).read())["scale"] == 1.0
    for scale in (0.5, 2.5):  # every overlay at the extremes, help card and banner too
        uiconfig.set_scale(scale)
        v.do_fitting = v.show_stats = v.show_rect_stats = v.show_help = True
        v.rect_sensor = (10, 10, 200, 150)
        assert _frames(v, 2).shape == (WINDOW_SIZE, WINDOW_SIZE, 3)
        v.cams.lost("unplugged in a test")
        _frames(v, 1)


def test_window_size_is_followed_only_when_it_makes_sense(viewer, monkeypatch):
    v = viewer
    for junk in ((0, 0, 0, 0), (-1, -1, -1, -1), (0, 0, 100000, 50)):
        monkeypatch.setattr(cv2, "getWindowImageRect", lambda _n, r=junk: r)
        v._follow_window()
        assert v.canvas == (WINDOW_SIZE, WINDOW_SIZE)

    def gone(_name):
        raise cv2.error("no window")

    monkeypatch.setattr(cv2, "getWindowImageRect", gone)
    v._follow_window()
    assert v.canvas == (WINDOW_SIZE, WINDOW_SIZE)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda _n: (0, 0, 1300, 700))
    v._follow_window()
    assert v.canvas == (1300, 700)


def test_the_window_opens_at_the_remembered_size_within_the_screen(monkeypatch):
    from beam_profiler.ui import desktop

    assert viewer_mod.initial_window_size() == (WINDOW_SIZE, WINDOW_SIZE)
    uiconfig.set_window((1600, 900))
    assert viewer_mod.initial_window_size() == (1600, 900)
    monkeypatch.setattr(desktop, "screen_size", lambda: (1280, 800))
    assert viewer_mod.initial_window_size() == (1240, 700)


def test_settings_window_zooms_and_keeps_working(tk_root):
    from beam_profiler import fitconfig
    from beam_profiler.ui import settings, tkhost

    win = settings.SettingsWindow()
    win.build(tk_root)
    try:
        win.win.withdraw()
        win.visible = False
        assert win.zoom_label.cget("text") == "100%"
        uiconfig.set_scale(1.5)
        tkhost._rescale(tk_root)  # what the Tk thread runs after a change
        assert win.exists() and not win.visible  # rebuilt, still hidden
        assert win.zoom_label.cget("text") == "150%"
        assert all(s.name in win._vars for s in fitconfig.SETTINGS)
        from tkinter import font as tkfont

        assert tkfont.nametofont("TkDefaultFont").cget("size") == 15
        win._vars["crop_sigma"].set("4.5")
        win._apply()
        assert fitconfig.active().crop_sigma == 4.5
    finally:
        uiconfig.set_scale(1.0)
        tkhost.apply_theme(tk_root)
        win.destroy()


def test_cameras_window_zooms_keeping_what_is_typed(tk_root):
    from beam_profiler.ui import cameras_window, tkhost

    manager = CameraManager([SimulatedBackend(width=64, height=48, grid=(1, 1))],
                            scan_interval=None)
    manager.start()  # one synchronous scan: the list has the simulator
    win = cameras_window.CamerasWindow(manager)
    win.build(tk_root)
    try:
        win.win.withdraw()
        tk_root.update()
        assert win._form_uid == "sim:SIM-0001"
        win.label_var.set("half typed")
        uiconfig.set_scale(2.0)
        tkhost._rescale(tk_root)
        tk_root.update()
        assert win.exists() and win.label_var.get() == "half typed"
        assert win.tree.column("#0", "width") == 68
        assert win.tree.selection() == ("sim:SIM-0001",)
    finally:
        uiconfig.set_scale(1.0)
        tkhost.apply_theme(tk_root)
        win.destroy()
        manager.stop()


def test_cards_keep_the_zoom_where_they_fit_and_shrink_where_not():
    info = overlays.Card([("ROI", "x")] * 6)
    rect = overlays.Card([("Sum", "1")] * 7, "Rectangle")
    spots = overlays.Card([("Dx", "1")] * 7, "Spot statistics")

    def layout(scale, w, h):
        m = overlays.Metrics.at(scale)
        bar = overlays.Toolbar(viewer_mod.TOOLS, (w, h), m)
        top, bottom = m.top_h + m.px(10), bar.box[1] - m.gap
        cm, stacked = overlays.fit_cards(m, w, top, bottom, [info, rect], [spots])
        left = info.height(cm) + cm.gap + rect.height(cm)
        right = spots.height(cm)
        need = left + cm.gap + right if stacked else max(left, right)
        assert need <= bottom - top or cm.scale <= 0.6  # fits, unless at the floor
        return cm.scale, stacked

    assert layout(1.0, 1100, 1100) == (1.0, False)  # the design size fits as designed
    assert layout(1.5, 1100, 1100) == (1.5, False)
    assert layout(1.0, 1600, 900) == (1.0, False)
    scale, stacked = layout(1.5, 800, 800)  # too small for 150 %: shrinks, but stays big
    assert 1.0 <= scale < 1.5
    scale, stacked = layout(1.0, 420, 1000)  # one narrow column: stacked
    assert stacked
