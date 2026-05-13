#!/usr/bin/env bash
# Launch / stop the local training dashboard.
#
#   scripts/dashboard.sh start   # detach uvicorn on :8765
#   scripts/dashboard.sh stop
#   scripts/dashboard.sh status
#
# Footprint: ~30 MB RAM, single uvicorn worker. Pure HTTP — no websockets.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PID_FILE="$ROOT/runs/.dashboard.pid"
LOG_FILE="$ROOT/runs/dashboard.log"
PORT="${DASHBOARD_PORT:-8765}"

_alive() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid; pid="$(cat "$PID_FILE")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

case "${1:-status}" in
    start)
        if _alive; then
            echo "dashboard already running pid=$(cat "$PID_FILE")  → http://127.0.0.1:$PORT"
            exit 0
        fi
        mkdir -p "$ROOT/runs"
        DASHBOARD_PORT="$PORT" nohup setsid "$ROOT/.venv/bin/python" \
            "$ROOT/scripts/dashboard.py" \
            >"$LOG_FILE" 2>&1 </dev/null &
        echo $! >"$PID_FILE"
        disown || true
        sleep 1
        echo "started dashboard pid=$(cat "$PID_FILE")  → http://127.0.0.1:$PORT"
        ;;
    stop)
        if ! _alive; then
            echo "dashboard not running"
            rm -f "$PID_FILE"
            exit 0
        fi
        pid="$(cat "$PID_FILE")"
        kill -TERM "$pid" 2>/dev/null || true
        for _ in 1 2 3; do
            _alive || break
            sleep 1
        done
        _alive && kill -KILL "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "stopped"
        ;;
    status)
        if _alive; then
            echo "RUNNING pid=$(cat "$PID_FILE")  → http://127.0.0.1:$PORT"
        else
            echo "STOPPED  log=$LOG_FILE"
        fi
        ;;
    *)
        echo "usage: $0 {start|stop|status}" >&2
        exit 2
        ;;
esac
