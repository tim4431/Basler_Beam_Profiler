"""Any GenICam camera that has a GenTL producer, through `harvesters`:
FLIR/Teledyne, Allied Vision, IDS, Lucid, Daheng, Hikrobot, Baumer, ...

Needs `pip install harvesters` (1.4 or newer) plus the camera vendor's SDK,
whose installer provides the GenTL producer (a .cti file) and puts its folder
on GENICAM_GENTL64_PATH.  More .cti files or folders can be listed in
BEAM_PROFILER_CTI (separated like PATH).

Features are addressed by their SFNC names (ExposureTime, Gain, Width,
OffsetX, PixelFormat, TriggerMode, ...), which are the names pylon uses too,
so pixel formats and trigger mode in camera_config.yaml mean the same thing
here.  I/O lines are not driven.  Basler cameras are left to the pylon backend:
producers inside a pylon install are skipped when pypylon is available.

harvesters' `update()` (re-enumeration) invalidates every open acquirer, so
while one of these cameras is open, `enumerate` repeats the last list instead
of rescanning.  A camera that drops out is closed first, so it is still found
again when it comes back.

Status: written against the harvesters 1.4 API; not yet run against real
non-Basler hardware.
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
import sys
import threading

import numpy as np

from .. import cameraconfig
from .base import Backend, Camera, DeviceInfo

GENTL_ENV = "GENICAM_GENTL64_PATH" if sys.maxsize > 2**32 else "GENICAM_GENTL32_PATH"
EXTRA_ENV = "BEAM_PROFILER_CTI"


def find_producers(skip_pylon: bool = False) -> list[str]:
    """Every .cti named by, or inside a folder named by, the GenTL variables."""
    found = []
    for var in (GENTL_ENV, EXTRA_ENV):
        for entry in os.environ.get(var, "").split(os.pathsep):
            entry = entry.strip()
            if entry.lower().endswith(".cti") and os.path.isfile(entry):
                found.append(entry)
            elif entry and os.path.isdir(entry):
                found += sorted(glob.glob(os.path.join(entry, "*.cti")))
    out, seen = [], set()
    for path in found:
        real = os.path.realpath(path)
        if real in seen or (skip_pylon and "pylon" in real.lower()):
            continue
        seen.add(real)
        out.append(path)
    return out


def _depth(fmt: str) -> int:
    """Bits per pixel of a MonoNN[p] format; 8 for anything unrecognised."""
    digits = "".join(c for c in fmt[4:] if c.isdigit()) if fmt.startswith("Mono") else ""
    return int(digits) if digits else 8


class GenICamBackend(Backend):
    name = "genicam"
    title = "GenICam (harvesters)"
    install_hint = (
        "`pip install harvesters`, plus your camera vendor's SDK for its GenTL producer"
    )

    def __init__(self):
        self._lock = threading.RLock()
        self._harvester = None
        self._open = 0
        self._cached: list[DeviceInfo] = []
        self._producers: list[str] = []

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("harvesters") is None:
            return False, "harvesters is not installed"
        self._producers = find_producers(
            skip_pylon=importlib.util.find_spec("pypylon") is not None
        )
        if not self._producers:
            return False, f"no GenTL producer (.cti) found on {GENTL_ENV} or {EXTRA_ENV}"
        return True, ", ".join(os.path.basename(p) for p in self._producers)

    def _get_harvester(self):
        if self._harvester is None:
            from harvesters.core import Harvester

            harvester = Harvester()
            for path in self._producers:
                try:
                    harvester.add_file(path)
                except Exception as e:
                    logging.warning(f"GenTL producer {path} failed to load: {e}")
            self._harvester = harvester
        return self._harvester

    def enumerate(self, full: bool = False) -> list[DeviceInfo]:
        with self._lock:
            if self._open:  # update() would invalidate the open camera
                return list(self._cached)
            harvester = self._get_harvester()
            harvester.update()
            self._cached = [self._info(d) for d in harvester.device_info_list]
            return list(self._cached)

    def _info(self, d) -> DeviceInfo:
        props = getattr(d, "property_dict", {}) or {}
        return DeviceInfo(
            backend=self.name,
            model=str(props.get("model") or props.get("display_name") or "GenICam camera"),
            serial=str(props.get("serial_number") or ""),
            vendor=str(props.get("vendor") or ""),
            address=str(props.get("id_") or ""),
        )

    def open(self, info: DeviceInfo, mode: str) -> Camera:
        with self._lock:
            harvester = self._get_harvester()
            if not self._open:
                harvester.update()
            key = {"serial_number": info.serial} if info.serial else {"id_": info.address}
            acquirer = harvester.create(key)
            self._open += 1
        try:
            cam = GenICamCamera(mode, info, acquirer, on_close=self._release)
        except Exception:
            self._release(acquirer)
            raise
        return cam

    def _release(self, acquirer):
        with self._lock:
            try:
                acquirer.destroy()
            except Exception as e:
                logging.debug(f"GenICam destroy: {e}")
            self._open = max(0, self._open - 1)

    def probe(self, info: DeviceInfo) -> dict:
        out = {**super().probe(info), "backend": self.name}
        try:
            cam = self.open(info, "16Bit")
        except Exception as e:
            return {**out, "error": str(e)}
        try:
            out["sensor_size"] = (cam.W, cam.H)
            for key, read in (
                ("pixel_formats", lambda: list(cam.nm.PixelFormat.symbolics)),
                ("exposure_range", lambda: (float(cam._exp_node.min), float(cam._exp_node.max))),
                ("trigger_sources", lambda: list(cam.nm.TriggerSource.symbolics)),
            ):
                try:
                    out[key] = read()
                except Exception:
                    pass
        finally:
            cam.close()
        return out

    def close(self):
        with self._lock:
            if self._harvester is not None:
                try:
                    self._harvester.reset()
                except Exception:
                    pass
                self._harvester = None


class GenICamCamera(Camera):
    backend = "genicam"

    def __init__(self, mode: str, info: DeviceInfo, acquirer, on_close=None):
        self.ia = acquirer
        self.nm = acquirer.remote_device.node_map
        self.device = info
        self._on_close = on_close
        self.model_name = info.model or str(self._get("DeviceModelName", "GenICam camera"))
        self.serial = info.serial or str(self._get("DeviceSerialNumber", ""))
        self.W = int(self._get("WidthMax", 0) or self._get("SensorWidth", 0) or self.nm.Width.max)
        self.H = int(self._get("HeightMax", 0) or self._get("SensorHeight", 0) or self.nm.Height.max)

        settings = cameraconfig.resolve(self.model_name, self.serial)
        super().__init__(mode, settings)
        logging.info(cameraconfig.describe(settings, self.model_name))

        self._exp_node = self._node("ExposureTime", "ExposureTimeAbs")
        self._gain_node = self._node("Gain", "GainRaw")
        self._set("ExposureAuto", "Off")
        self._set("GainAuto", "Off")
        self._clamp_exposure_limits()
        self._select_pixel_format()
        self._setup_trigger()
        self._roi = self._read_roi()
        self._exposure = float(settings.default_exposure)
        self._gain = float(settings.default_gain)
        self.Gain = settings.default_gain
        self.ExposureTime = settings.default_exposure
        self.ia.start()

    # ------------------------------ nodes --------------------------------- #
    def _node(self, *names):
        for name in names:
            try:
                return getattr(self.nm, name)
            except AttributeError:
                continue
        return None

    def _get(self, name, default):
        node = self._node(name)
        try:
            return node.value if node is not None else default
        except Exception:
            return default

    def _set(self, name, value) -> bool:
        node = self._node(name)
        if node is None:
            return False
        try:
            node.value = value
            return True
        except Exception as e:
            logging.debug(f"{self.model_name}: {name} = {value!r} rejected: {e}")
            return False

    # ------------------------------ setup -------------------------------- #
    def _clamp_exposure_limits(self):
        node = getattr(self, "_exp_node", None)
        if node is None:
            return
        try:
            lo, hi = float(node.min), float(node.max)
        except Exception:
            return
        narrowed = (max(self.settings.min_exposure, lo), min(self.settings.max_exposure, hi))
        if narrowed[0] >= narrowed[1]:
            narrowed = (lo, hi)
        self.set_exposure_limits(*narrowed)

    def apply_settings(self, settings):
        super().apply_settings(settings)
        self._clamp_exposure_limits()

    def _select_pixel_format(self):
        candidates = (
            self.settings.pixel_format_8bit if self.mode == "8Bit" else self.settings.pixel_format_16bit
        )
        try:
            available = list(self.nm.PixelFormat.symbolics)
        except Exception:
            available = []
        for fmt in candidates:
            if available and fmt not in available:
                continue
            if self._set("PixelFormat", fmt):
                break
        else:
            logging.error(
                f"none of the configured {self.mode} formats {list(candidates)} are "
                f"available on {self.model_name} (it offers {available or 'unknown'})"
            )
        self.pixel_format = str(self._get("PixelFormat", "Mono8"))
        self._bits = _depth(self.pixel_format)

    def _setup_trigger(self):
        self.software_trigger = False
        if self.settings.trigger_mode == "off":
            self._set("TriggerMode", "Off")
            return
        ok = (
            self._set("TriggerSelector", "FrameStart")
            and self._set("TriggerMode", "On")
            and self._set("TriggerSource", "Software")
            and self._node("TriggerSoftware") is not None
        )
        if ok:
            self.software_trigger = True
        else:
            self._set("TriggerMode", "Off")
            log = logging.error if self.settings.trigger_mode == "software" else logging.warning
            log(f"{self.model_name}: software trigger unavailable, free-running")

    # ------------------------------ properties --------------------------- #
    @property
    def ExposureTime(self):
        return self._exposure

    @ExposureTime.setter
    def ExposureTime(self, value):
        value = float(np.clip(value, self.min_exposure, self.max_exposure))
        if self._exp_node is not None:
            self._exp_node.value = value
            value = float(self._exp_node.value)
        self._exposure = value

    @property
    def Gain(self):
        return self._gain

    @Gain.setter
    def Gain(self, value):
        if self._gain_node is not None:
            try:
                self._gain_node.value = type(self._gain_node.value)(value)
                value = self._gain_node.value
            except Exception as e:
                logging.debug(f"{self.model_name}: gain {value} rejected: {e}")
        self._gain = float(value)

    @property
    def ROI(self):
        return self._roi

    @ROI.setter
    def ROI(self, value):
        w, h, ox, oy = (int(v) for v in value)
        w, h = max(10, min(w, self.W)), max(10, min(h, self.H))
        ox, oy = max(0, min(ox, self.W - w)), max(0, min(oy, self.H - h))
        acquiring = self.ia.is_acquiring()
        if acquiring:
            self.ia.stop()
        try:
            # offsets first to zero so the size can grow, then size, then offsets
            for name, v in (("OffsetX", 0), ("OffsetY", 0), ("Width", w), ("Height", h),
                            ("OffsetX", ox), ("OffsetY", oy)):
                node = self._node(name)
                if node is not None:
                    inc = max(1, int(getattr(node, "inc", 1) or 1))
                    node.value = int(max(node.min, min(node.max, v - (v - node.min) % inc)))
        except Exception as e:
            logging.error(f"Failed to set ROI: {e}")
        finally:
            if acquiring:
                self.ia.start()
        self._roi = self._read_roi()

    def _read_roi(self):
        try:
            return (int(self.nm.Width.value), int(self.nm.Height.value),
                    int(self._get("OffsetX", 0)), int(self._get("OffsetY", 0)))
        except Exception:
            return getattr(self, "_roi", (self.W, self.H, 0, 0))

    # ------------------------------ acquisition -------------------------- #
    def _grab(self) -> np.ndarray | None:
        if self.software_trigger:
            self.nm.TriggerSoftware.execute()
        timeout = self.settings.grab_timeout_ms / 1000.0
        buffer = self.ia.fetch(timeout=timeout)
        if not self.software_trigger:  # free-running: skip to the newest frame
            while True:
                newer = self.ia.try_fetch(timeout=0.001)
                if newer is None:
                    break
                buffer.queue()
                buffer = newer
        try:
            component = buffer.payload.components[0]
            img = component.data.reshape(component.height, component.width).copy()
        finally:
            buffer.queue()
        return self._convert(img)

    def _convert(self, img: np.ndarray) -> np.ndarray:
        """To uint8, or MSB-aligned uint16 - what the rest of the app expects."""
        bits = self._bits
        if self.mode == "16Bit":
            img = img.astype(np.uint16, copy=False)
            return img << (16 - bits) if bits < 16 else img
        if bits > 8:
            return (img >> (bits - 8)).astype(np.uint8)
        return img.astype(np.uint8, copy=False)

    def close(self):
        try:
            self.ia.stop()
        except Exception:
            pass
        if self._on_close is not None:
            self._on_close(self.ia)
            self._on_close = None
