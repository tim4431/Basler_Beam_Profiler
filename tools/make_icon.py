"""Render docs/icon.svg into the files that need a bitmap.

    python tools/make_icon.py

docs/icon.svg is the source; the Linux launcher uses it directly.  This writes
- beam_profiler/ui/appicon.py  PNGs as base64, for the Tk window icon (a module,
                               not a data file, so the one-file exe needs nothing extra)
- docs/icon.ico                multi-size icon for the Windows exe (pyinstaller --icon)

Needs Inkscape, rsvg-convert or ImageMagick to rasterise, and Pillow.
Re-run it after editing the SVG and commit the outputs.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG = os.path.join(ROOT, "docs", "icon.svg")
MODULE = os.path.join(ROOT, "beam_profiler", "ui", "appicon.py")
ICO = os.path.join(ROOT, "docs", "icon.ico")
TK_SIZES = (32, 64, 128)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def rasterise(size: int, out: str):
    if shutil.which("inkscape"):
        cmd = ["inkscape", SVG, "-w", str(size), "-h", str(size), "-o", out]
    elif shutil.which("rsvg-convert"):
        cmd = ["rsvg-convert", "-w", str(size), "-h", str(size), SVG, "-o", out]
    elif shutil.which("magick"):
        cmd = ["magick", "-background", "none", "-density", "384", SVG,
               "-resize", f"{size}x{size}", out]
    else:
        sys.exit("need inkscape, rsvg-convert or ImageMagick to rasterise the SVG")
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        images = {}
        for size in sorted(set(TK_SIZES) | set(ICO_SIZES)):
            path = os.path.join(tmp, f"{size}.png")
            rasterise(size, path)
            images[size] = Image.open(path).convert("RGBA")
            images[size].load()

    entries = []
    for size in TK_SIZES:
        buf = io.BytesIO()
        images[size].save(buf, format="PNG", optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        lines = [data[i:i + 76] for i in range(0, len(data), 76)]
        body = "\n".join(f'        "{line}"' for line in lines)
        entries.append(f"    {size}: (\n{body}\n    ),")
    with open(MODULE, "w", encoding="utf-8") as f:
        f.write(
            '"""The app icon as base64 PNGs, by size - generated from docs/icon.svg by\n'
            'tools/make_icon.py; do not edit by hand."""\n\n'
            "PNG = {\n" + "\n".join(entries) + "\n}\n"
        )
    images[256].save(ICO, format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                     append_images=[images[s] for s in ICO_SIZES if s != 256])
    print(f"wrote {os.path.relpath(MODULE, ROOT)} and {os.path.relpath(ICO, ROOT)}")


if __name__ == "__main__":
    main()
