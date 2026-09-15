#!/bin/sh
# Fast chip serial. Frees ACM0, no IDF env, no Django.
# Usage: ./scripts/mon.sh   or   ./scripts/mon.sh /dev/ttyACM0

set -e
PORT=${1:-/dev/ttyACM0}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

pkill -f esp_idf_monitor 2>/dev/null || true
pkill -f 'idf.py .*monitor' 2>/dev/null || true
pkill -f 'gibbon-console/.venv/bin/python' 2>/dev/null || true
pkill -f 'roarm-fw/scripts/live.py' 2>/dev/null || true
pkill -f 'roarm-fw/scripts/listen.py' 2>/dev/null || true
sleep 0.2
if command -v fuser >/dev/null 2>&1; then
    fuser -k "$PORT" 2>/dev/null || true
    sleep 0.2
fi
if [ ! -e "$PORT" ]; then
    echo "no $PORT — plug the HAT, UART switch = ESP32" >&2
    exit 1
fi
if [ ! -w "$PORT" ]; then
    sudo chmod 666 "$PORT"
fi
echo "opening $PORT (Ctrl+C quits)"
exec python3 "$ROOT/scripts/mon.py" "$PORT"
