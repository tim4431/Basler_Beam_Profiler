#!/bin/bash
cd "$(dirname "$0")"
./run.sh --camera flir "$@"
status=$?
if [ "$status" -ne 0 ]; then
    echo "See docs/macos.md for driver setup. Press Return to close."
    read -r
fi
exit "$status"
