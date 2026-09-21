#!/bin/bash
set -e
cd "$(dirname "$0")"
if [[ -d .vendor/spinnaker/lib ]]; then
    export SPINNAKER_GENTL64_CTI="$PWD/.vendor/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti"
    export DYLD_LIBRARY_PATH="$PWD/.vendor/spinnaker/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python -m beam_profiler "$@"
fi
exec "${PYTHON:-python3}" -m beam_profiler "$@"
