"""Entry point: python -m beam_profiler [--sim] [options]."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="beam_profiler", description="Basler / FLIR Beam Profiler"
    )
    parser.add_argument("--camera", choices=["auto", "flir", "basler", "sim"],
                        default="auto", help="camera driver; explicit hardware selection never simulates")
    parser.add_argument(
        "--sim", action="store_true",
        help="use the simulated camera (fixed spot grid) instead of hardware",
    )
    parser.add_argument(
        "--mode", choices=["8Bit", "16Bit"], default="16Bit", help="pixel depth"
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
    parser.add_argument("--frames", type=int, default=None,
                        help="exit after this many frames (for package smoke checks)")
    parser.add_argument("--check-drivers", action="store_true",
                        help="verify bundled Basler and FLIR libraries and exit")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.check_drivers:
        import PySpin
        from pypylon import pylon
        system = PySpin.System.GetInstance()
        try:
            version = system.GetLibraryVersion()
            print(f"FLIR runtime: {version.major}.{version.minor}.{version.type}.{version.build}")
            print(f"Basler runtime: {pylon.GetPylonVersionString()}")
        finally:
            system.ReleaseInstance()
        return

    from .cameras import open_cameras

    grid = tuple(int(v) for v in args.sim_grid.lower().split("x"))
    width, height = (int(v) for v in args.sim_size.lower().split("x"))
    sim_opts = dict(
        grid=grid, width=width, height=height,
        jitter=args.sim_jitter, sigma=args.sim_sigma,
    )
    try:
        cameras = open_cameras(mode=args.mode, simulate=args.sim, backend=args.camera, **sim_opts)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        from .ui import Viewer

        Viewer(cameras).run(max_frames=args.frames)
    finally:
        for camera in cameras:
            try:
                camera.close()
            except Exception:
                logging.exception("Error closing camera %s", camera.serial)


if __name__ == "__main__":
    main()
