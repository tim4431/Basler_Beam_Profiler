"""Simulated camera: renders a fixed grid of Gaussian spots in real time so the
full CV pipeline (blob detection, Gaussian fits, grid statistics, auto-exposure)
can be exercised without hardware."""

from __future__ import annotations

import numpy as np

from .. import cameraconfig
from ..synthetic import Spot, render, spot_grid
from .base import Backend, Camera, DeviceInfo

REFERENCE_EXPOSURE_US = 200  # exposure at which spots hit their nominal amplitude
NOISE_BANK_FRAMES = 8
SIM_SERIAL = "SIM-0001"


class SimulatedBackend(Backend):
    """Always present, so there is something to look at with no hardware and
    the Cameras window always has an entry to fall back to."""

    name = "sim"
    title = "Simulator"
    auto_open = False  # used as the fallback, never chosen over real hardware

    def __init__(self, **sim_opts):
        self.sim_opts = sim_opts

    def _model(self) -> str:
        grid = self.sim_opts.get("grid", (4, 4))
        return f"Simulated-{grid[0]}x{grid[1]}"

    def enumerate(self, full: bool = False) -> list[DeviceInfo]:
        return [DeviceInfo(self.name, self._model(), SIM_SERIAL, "beam_profiler")]

    def open(self, info: DeviceInfo, mode: str) -> Camera:
        cam = SimulatedCamera(mode=mode, **self.sim_opts)
        cam.device = info
        return cam

    def probe(self, info: DeviceInfo) -> dict:
        width = self.sim_opts.get("width", 1200)
        height = self.sim_opts.get("height", 1200)
        return {**super().probe(info), "backend": self.name, "sensor_size": (width, height)}


class SimulatedCamera(Camera):
    backend = "sim"

    def __init__(
        self,
        mode: str = "16Bit",
        width: int = 1200,
        height: int = 1200,
        grid: tuple[int, int] = (4, 4),
        sigma: float | None = None,
        jitter: float = 0.3,
        noise: float = 0.003,
        pixel_size: float = 2.5e-6,
        seed: int | None = 0,
    ):
        self.model_name = f"Simulated-{grid[0]}x{grid[1]}"
        self.serial = SIM_SERIAL
        self.W, self.H = width, height
        # Resolved through camera_config.yaml like any real camera, so the file
        # can be exercised without hardware; `pixel_size` is the fallback when
        # no `Simulated*` entry configures one.
        self._fallback_pixel_size = pixel_size
        super().__init__(mode, self._resolve())
        if sigma is None:
            # beam waist (~2 sigma) defaults to 1/5 of the spot pitch
            pitch = min(width, height) / (max(grid) + 1)
            sigma = pitch / 10
        self.spots: list[Spot] = spot_grid(width, height, *grid, sigma=sigma, amplitude=0.6)
        self.jitter = jitter
        self._rng = np.random.default_rng(seed)
        self._roi = (width, height, 0, 0)
        self._exposure = float(
            np.clip(self.settings.default_exposure, self.min_exposure, self.max_exposure)
        )
        self._gain = float(self.settings.default_gain)
        # pre-generated readout noise, cycled per frame: cheaper than drawing
        # millions of fresh normal samples at 60 fps
        self._noise_bank = self._rng.normal(
            0.0, noise * self.saturation, (NOISE_BANK_FRAMES, height, width)
        ).astype(np.int16)
        self._noise_idx = 0

    def _resolve(self):
        settings = cameraconfig.resolve(self.model_name, self.serial, warn=False)
        if not settings.pixel_size_known:
            settings = settings.replace(pixel_size=self._fallback_pixel_size)
        return settings

    def reload_settings(self):
        self.apply_settings(self._resolve())

    # ------------------------------ properties --------------------------- #
    @property
    def Gain(self):
        return self._gain

    @Gain.setter
    def Gain(self, value):
        self._gain = float(value)

    @property
    def ExposureTime(self):
        return self._exposure

    @ExposureTime.setter
    def ExposureTime(self, value):
        self._exposure = float(np.clip(value, self.min_exposure, self.max_exposure))

    @property
    def ROI(self):
        return self._roi

    @ROI.setter
    def ROI(self, value):
        w, h, ox, oy = (int(v) for v in value)
        w = max(10, min(w, self.W))
        h = max(10, min(h, self.H))
        ox = max(0, min(ox, self.W - w))
        oy = max(0, min(oy, self.H - h))
        self._roi = (w, h, ox, oy)

    # ------------------------------ acquisition -------------------------- #
    def _grab(self) -> np.ndarray:
        w, h, ox, oy = self._roi
        gain = (self._exposure / REFERENCE_EXPOSURE_US) * 10 ** (self._gain / 20)
        spots = self.spots
        if self.jitter > 0:
            offsets = self._rng.normal(0.0, self.jitter, (len(spots), 2))
            flicker = self._rng.normal(1.0, 0.01, len(spots))
            spots = [
                Spot(s.x + dx, s.y + dy, s.sigma_x, s.sigma_y, s.theta, s.amplitude * f)
                for s, (dx, dy), f in zip(spots, offsets, flicker)
            ]
        frame = render(
            w, h, spots, self.saturation, background=0.01, noise=0.0,
            gain=gain, offset=(ox, oy),
        )
        self._noise_idx = (self._noise_idx + 1) % NOISE_BANK_FRAMES
        noise = self._noise_bank[self._noise_idx, oy : oy + h, ox : ox + w]
        out = frame.astype(np.int32) + noise
        return np.clip(out, 0, self.saturation).astype(frame.dtype)
