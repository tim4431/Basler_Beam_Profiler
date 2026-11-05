#!/bin/bash
# One-shot setup on Ubuntu: create .venv, install requirements, add a desktop
# launcher. Safe to re-run (updates the venv and the launcher in place).
#
#   ./setup.sh            # venv + deps (+ CuPy if an NVIDIA GPU is found) + launcher
#   ./setup.sh --no-gpu   # skip CuPy
#   ./setup.sh --dev      # also install pytest/pyinstaller
#   PYTHON=python3.12 ./setup.sh   # pick the interpreter (needs 3.10+)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"
PY="${PYTHON:-python3}"
GPU=auto
DEV=0
for arg in "$@"; do
    case "$arg" in
        --no-gpu) GPU=0 ;;
        --gpu)    GPU=1 ;;
        --dev)    DEV=1 ;;
        -h|--help) sed -n 2,9p "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# --- interpreter ------------------------------------------------------------
command -v "$PY" >/dev/null || { echo "error: $PY not found" >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || { echo "error: Python 3.10+ required ($("$PY" --version))" >&2; exit 1; }
if ! "$PY" -c 'import ensurepip' 2>/dev/null; then
    ver=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    echo "error: $PY has no venv/ensurepip support. Install it with:" >&2
    echo "    sudo apt install python${ver}-venv" >&2
    exit 1
fi

# --- venv + dependencies ----------------------------------------------------
if [ ! -x .venv/bin/python ]; then
    echo ">> creating .venv with $("$PY" --version)"
    "$PY" -m venv .venv
fi
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
echo ">> installing requirements"
"$VPY" -m pip install -q -r requirements.txt

if [ "$GPU" = auto ]; then
    command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && GPU=1 || GPU=0
fi
if [ "$GPU" = 1 ]; then
    cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -1)
    pkg="cupy-cuda${cuda:-13}x"
    echo ">> NVIDIA GPU found, installing $pkg (optional)"
    "$VPY" -m pip install -q "$pkg" \
        || echo "   warning: $pkg failed to install; the CPU path will be used"
fi
if [ "$DEV" = 1 ]; then
    echo ">> installing dev tools"
    "$VPY" -m pip install -q pytest pyinstaller
fi

# --- desktop launcher -------------------------------------------------------
chmod +x run.sh
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
ENTRY="beam-profiler.desktop"
mkdir -p "$APPS"
cat > "$APPS/$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=Beam Profiler
Comment=Live laser-beam profiling with Basler cameras
Exec="$ROOT/run.sh"
Path=$ROOT
Icon=$ROOT/docs/icon.svg
Terminal=false
Categories=Science;Engineering;
EOF
chmod +x "$APPS/$ENTRY"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$APPS" || true

if [ -d "$DESKTOP_DIR" ]; then
    cp "$APPS/$ENTRY" "$DESKTOP_DIR/$ENTRY"
    chmod +x "$DESKTOP_DIR/$ENTRY"
    # GNOME only launches desktop icons that are marked trusted.
    gio set "$DESKTOP_DIR/$ENTRY" metadata::trusted true 2>/dev/null || true
fi

echo ">> checking the install"
"$VPY" -c 'import cv2, numpy, scipy, yaml, flask, pypylon; print("   imports ok")'
echo
echo "Done. Launch with the 'Beam Profiler' desktop icon / app menu, or ./run.sh"
echo "(no camera connected -> simulator). Log: $ROOT/data/launcher.log"
