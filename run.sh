#!/bin/bash
# Launch the beam profiler from the project venv (create it with ./setup.sh).
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x .venv/bin/python ]; then
    echo "No .venv found - run ./setup.sh first." >&2
    exit 1
fi
# Started from the desktop icon there is no terminal: keep a log instead.
if [ ! -t 1 ]; then
    mkdir -p data
    exec >> data/launcher.log 2>&1
    echo "=== $(date '+%F %T') ==="
fi
exec .venv/bin/python -m beam_profiler "$@"
