#!/bin/sh
# Flash one bring-up .c onto the HAT. Does not start a monitor.
# Usage: ./flash.sh [02-ble-monitor]
cd "$(dirname "$0")"
TEST="${1:-02-ble-monitor}"
SRC="${TEST}.c"
if [ ! -f "$SRC" ]; then
    echo "missing $SRC" >&2
    exit 1
fi
ln -sfn "../$SRC" main/app.c
. ../scripts/idf-env.sh
set -e
if [ -e /dev/ttyACM0 ]; then
    chmod 666 /dev/ttyACM0 2>/dev/null || true
fi
echo "flash $TEST  (stop idf.py monitor first if it owns ACM0)"
idf.py -p /dev/ttyACM0 flash
echo "done.  unplug USB, pack on, open /monitor/ in gibbon-console"
