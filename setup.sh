#!/bin/bash
# One-shot setup on Ubuntu: create .venv, install requirements, add a desktop
# launcher. Safe to re-run (updates the venv and the launcher in place).
#
#   ./setup.sh            # venv + deps (+ CuPy if an NVIDIA GPU is found) + launcher
#   ./setup.sh --no-gpu   # skip CuPy
#   ./setup.sh --dev      # also install pytest/pyinstaller
#   ./setup.sh --launcher # only (re)install the desktop/app-menu launcher
#   ./setup.sh --tucsen=DIR  # also the `tucsen` package from a tucsen-camera-driver
#                            # checkout (Tucsen cameras; libTUCam must be installed)
#   PYTHON=python3.12 ./setup.sh   # pick the interpreter (needs 3.10+)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"
PY="${PYTHON:-python3}"
GPU=auto
DEV=0
LAUNCHER_ONLY=0
TUCSEN=""
for arg in "$@"; do
    case "$arg" in
        --no-gpu) GPU=0 ;;
        --gpu)    GPU=1 ;;
        --dev)    DEV=1 ;;
        --launcher) LAUNCHER_ONLY=1 ;;
        --tucsen=*) TUCSEN="${arg#--tucsen=}" ;;
        -h|--help) sed -n 2,11p "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# --- desktop launcher (defined here so --launcher can run it alone) --------
# StartupWMClass must match the WM_CLASS the app gives its windows
# (beam_profiler/ui/desktop.py), or the dock shows them as "Unknown".
install_launcher() {
    chmod +x run.sh
    local apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    local entry="beam-profiler.desktop"
    local desktop_dir
    desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
    mkdir -p "$apps"
    cat > "$apps/$entry" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Beam Profiler
Comment=Live laser-beam profiling with Basler, GenICam and USB cameras
Exec="$ROOT/run.sh"
Path=$ROOT
Icon=$ROOT/docs/icon.svg
Terminal=false
Categories=Science;Engineering;
StartupWMClass=beam-profiler
DESKTOP
    chmod +x "$apps/$entry"
    command -v update-desktop-database >/dev/null && update-desktop-database -q "$apps" || true
    if [ -d "$desktop_dir" ]; then
        cp "$apps/$entry" "$desktop_dir/$entry"
        chmod +x "$desktop_dir/$entry"
        # GNOME only launches desktop icons that are marked trusted.
        gio set "$desktop_dir/$entry" metadata::trusted true 2>/dev/null || true
    fi
    echo ">> launcher installed: $apps/$entry"
}
if [ "$LAUNCHER_ONLY" = 1 ]; then
    install_launcher
    exit 0
fi

# --- interpreter ------------------------------------------------------------
command -v "$PY" >/dev/null || { echo "error: $PY not found" >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || { echo "error: Python 3.10+ required ($("$PY" --version))" >&2; exit 1; }
# venv and tkinter (settings window, save dialog) are separate apt packages on
# Ubuntu and cannot be pip-installed into the venv.
ver=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
missing=()
"$PY" -c 'import ensurepip' 2>/dev/null || missing+=("python${ver}-venv")
"$PY" -c 'import tkinter' 2>/dev/null || missing+=("python${ver}-tk")
if [ ${#missing[@]} -gt 0 ]; then
    echo ">> $PY is missing system packages: ${missing[*]}"
    if [ -t 0 ] && command -v apt-get >/dev/null; then
        read -r -p "   Install them now with sudo apt? [Y/n] " ans
        if [[ ! "$ans" =~ ^[Nn] ]]; then
            sudo apt-get update -q && sudo apt-get install -y "${missing[@]}"
        fi
    fi
    "$PY" -c 'import ensurepip' 2>/dev/null || {
        echo "error: venv support missing. Run: sudo apt install ${missing[*]}" >&2; exit 1; }
    "$PY" -c 'import tkinter' 2>/dev/null || \
        echo "   warning: no tkinter; the settings window (o) and save dialog (d) will be disabled." \
             "Fix: sudo apt install python${ver}-tk, then re-run ./setup.sh"
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
if [ -n "$TUCSEN" ]; then
    echo ">> installing the tucsen package from $TUCSEN/python"
    "$VPY" -m pip install -q "$TUCSEN/python" \
        || echo "   warning: tucsen failed to install; is libTUCam installed? (docs/cameras.md)"
fi
if [ "$DEV" = 1 ]; then
    echo ">> installing dev tools"
    "$VPY" -m pip install -q pytest pyinstaller
fi

# --- desktop launcher -------------------------------------------------------
install_launcher

echo ">> checking the install"
"$VPY" -c 'import cv2, numpy, scipy, yaml, flask, pypylon; print("   imports ok")'
echo
echo "Done. Launch with the 'Beam Profiler' desktop icon / app menu, or ./run.sh"
echo "(no camera connected -> simulator). Log: $ROOT/data/launcher.log"
