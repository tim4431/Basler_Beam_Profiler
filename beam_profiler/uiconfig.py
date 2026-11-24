"""Interface settings that are not about the fit: the UI zoom, the last
live-view window size and whether the app updates itself (`updater`),
persisted to ui_config.json next to the app.

The same pattern as `fitconfig`: an immutable config swapped in atomically,
listeners told about every change (from whichever thread made it - a Tk
listener must hop to the Tk thread itself), and a forgiving loader: a missing
or broken file, or a nonsense value in it, gives the defaults for that value
and a log line, never an exception.

The scale multiplies every size in the live-view overlays and the Tk windows
(fonts, paddings, icons, line widths); 1.0 is the design size.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, replace

from .config import APP_DIR

CONFIG_PATH = os.path.join(APP_DIR, "ui_config.json")
MIN_SCALE, MAX_SCALE, STEP = 0.5, 2.5, 0.1
MIN_WINDOW, MAX_WINDOW = 200, 16384


@dataclass(frozen=True)
class UIConfig:
    scale: float = 1.0
    window: tuple[int, int] | None = None  # live-view image size (w, h) at last exit
    auto_update: bool = True  # see updater.py; edit the file while the app is closed


def clamp_scale(value) -> float:
    """Into [MIN_SCALE, MAX_SCALE], on the STEP grid.  Raises ValueError on
    anything that is not a finite number."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"not a finite scale: {value}")
    steps = round(value / STEP)
    return round(min(max(steps * STEP, MIN_SCALE), MAX_SCALE), 2)


def valid_window(value) -> tuple[int, int] | None:
    """(w, h) if `value` is a plausible window size, else None."""
    try:
        w, h = (int(v) for v in value)
    except (TypeError, ValueError):
        return None
    if MIN_WINDOW <= w <= MAX_WINDOW and MIN_WINDOW <= h <= MAX_WINDOW:
        return w, h
    return None


_lock = threading.Lock()
_active = UIConfig()  # in force (may hold a --ui-scale override)
_saved = UIConfig()  # what the file says, plus every change made with save=True
_listeners: list = []


def active() -> UIConfig:
    return _active


def scale() -> float:
    return _active.scale


def set_scale(value, save: bool = True) -> float:
    """Swap in a new scale (clamped) and notify; `save=False` for an override
    that should not outlive the session (--ui-scale)."""
    global _active, _saved
    s = clamp_scale(value)
    with _lock:
        changed = s != _active.scale
        _active = replace(_active, scale=s)
        if save:
            _saved = replace(_saved, scale=s)
    if save:
        save_config(_saved)
    if changed:
        for fn in list(_listeners):
            try:
                fn(_active)
            except Exception as e:
                logging.error(f"ui config listener failed: {e}")
    return s


def zoom(steps: int) -> float:
    """One (or more) STEP in or out from the current scale."""
    return set_scale(scale() + steps * STEP)


def reset_scale() -> float:
    return set_scale(1.0)


def set_window(size, save: bool = True):
    """Remember the live-view size for the next start."""
    global _active, _saved
    size = valid_window(size)
    if size is None:
        return
    with _lock:
        _active = replace(_active, window=size)
        _saved = replace(_saved, window=size)
    if save:
        save_config(_saved)


def on_change(fn):
    """Call `fn(config)` after every scale change (on the changing thread)."""
    _listeners.append(fn)
    return fn


def load_config(path: str | None = None) -> UIConfig:
    """Read ui_config.json; anything missing or malformed falls back to its
    default, field by field."""
    path = path or CONFIG_PATH  # resolved per call, so tests can redirect it
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except FileNotFoundError:
        return UIConfig()
    except Exception as e:
        logging.error(f"could not read {path} ({e}); using the default interface settings")
        return UIConfig()
    cfg = UIConfig()
    if "scale" in data:
        try:
            cfg = replace(cfg, scale=clamp_scale(data["scale"]))
        except (TypeError, ValueError):
            logging.warning(f"{path}: ignoring scale {data['scale']!r}")
    if data.get("window") is not None:
        window = valid_window(data["window"])
        if window is None:
            logging.warning(f"{path}: ignoring window {data['window']!r}")
        cfg = replace(cfg, window=window)
    if "auto_update" in data:
        if isinstance(data["auto_update"], bool):
            cfg = replace(cfg, auto_update=data["auto_update"])
        else:
            logging.warning(f"{path}: ignoring auto_update {data['auto_update']!r} "
                            "(true or false)")
    return cfg


def save_config(cfg: UIConfig, path: str | None = None):
    path = path or CONFIG_PATH
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"scale": cfg.scale,
                       "window": list(cfg.window) if cfg.window else None,
                       "auto_update": cfg.auto_update}, f, indent=4)
    except Exception as e:
        logging.error(f"could not write {path}: {e}")


def use(cfg: UIConfig):
    """Make `cfg` both active and saved, without writing or notifying (start
    up, tests)."""
    global _active, _saved
    with _lock:
        _active = _saved = cfg


use(load_config())
