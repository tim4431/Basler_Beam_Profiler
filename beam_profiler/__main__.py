"""Entry point: python -m beam_profiler [--sim] [options]."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="beam_profiler", description="Basler Beam Profiler"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="use the simulated camera (fixed spot grid) instead of hardware",
    )
    parser.add_argument(
        "--mode", choices=["8Bit", "16Bit"], default="16Bit", help="pixel depth"
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="print what every connected camera reports plus a ready-to-paste "
             "camera_config.yaml entry for it, then exit",
    )
    parser.add_argument(
        "--write-camera-config", action="store_true",
        help="create camera_config.yaml for the connected cameras (never "
             "overwrites an existing file), then exit",
    )
    parser.add_argument(
        "--sim-grid", metavar="NxM", default="4x4",
        help="simulated spot grid, e.g. 4x4 (default) or 8x8",
    )
    parser.add_argument(
        "--sim-size", metavar="WxH", default="1200x1200", help="simulated sensor size"
    )
    parser.add_argument(
        "--sim-jitter", type=float, default=0.3,
        help="per-frame spot position jitter in px (0 = perfectly static)",
    )
    parser.add_argument(
        "--sim-sigma", type=float, default=None,
        help="spot sigma in px (default: pitch/10, i.e. waist = pitch/5)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.list_cameras or args.write_camera_config:
        return list_cameras(write=args.write_camera_config)

    from .cameras import open_cameras

    grid = tuple(int(v) for v in args.sim_grid.lower().split("x"))
    width, height = (int(v) for v in args.sim_size.lower().split("x"))
    sim_opts = dict(
        grid=grid, width=width, height=height,
        jitter=args.sim_jitter, sigma=args.sim_sigma,
    )
    cameras = open_cameras(mode=args.mode, simulate=args.sim, **sim_opts)

    from .ui import Viewer

    Viewer(cameras).run()


def list_cameras(write: bool = False) -> int:
    """Report every connected camera and the YAML that would configure it.

    This is the intended way to add a model: plug the camera in, run it, paste
    the block into `camera_config.yaml` and fill in the pixel size."""
    from . import cameraconfig
    from .cameras import probe_devices

    devices = probe_devices()
    if not devices:
        print("No Basler cameras found.")
        return 1

    config = cameraconfig.load()
    stubs, missing, unscaled = [], [], []
    for probe in devices:
        model, serial = probe["model"], probe["serial"]
        print(f"\n{model}  (serial {serial}, {probe.get('vendor', 'unknown vendor')})")
        if "error" in probe:
            print(f"  could not open: {probe['error']}")
            continue
        for key, label in (
            ("sensor_size", "sensor"),
            ("pixel_formats", "pixel formats"),
            ("exposure_range", "exposure range (us)"),
            ("lines", "I/O lines"),
            ("trigger_sources", "trigger sources"),
        ):
            if key in probe:
                value = probe[key]
                if isinstance(value, (list, tuple)):
                    value = ", ".join(str(v) for v in value)
                print(f"  {label:22} {value}")

        known = model in config.cameras or serial in config.serials
        settings = cameraconfig.resolve(model, serial, config, warn=False)
        if not known:
            status = "NOT in camera_config.yaml"
            missing.append(model)
        elif not settings.pixel_size_known:
            status = "listed, but pixel_size is not set"
            unscaled.append(model)
        else:
            status = "configured"
        print(f"  {'status':22} {status}")
        print(f"  {'resolved':22} {cameraconfig.describe(settings, model)}")
        stubs.append(cameraconfig.yaml_stub(model, serial, probe))

    if write:
        return _write_config(cameraconfig.default_config_path(), stubs)
    if missing:
        print(f"\nPaste into the `cameras:` block of {config.path}:\n")
        print("\n".join(stubs))
        print("\n(or run --write-camera-config to create the file from scratch)")
    if unscaled:
        print(
            f"\nStill needs a pixel_size in {config.path}: {', '.join(unscaled)}."
            "\nUntil then microns are reported as pixel counts."
        )
    if not missing and not unscaled:
        print(f"\nEvery camera above is configured in {config.path}.")
    return 0


def _write_config(path: str, stubs: list[str]) -> int:
    from . import cameraconfig

    if os.path.exists(path):
        print(f"\n{path} already exists - not touching it.  Paste this instead:\n")
        print("\n".join(stubs))
        return 1
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cameraconfig.file_template(stubs))
    except OSError as e:
        print(f"\ncould not write {path}: {e}")
        return 1
    print(f"\nWrote {path} - now fill in each `pixel_size` from the datasheet.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
