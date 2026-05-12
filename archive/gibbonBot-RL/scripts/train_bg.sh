#!/usr/bin/env bash
# Launch / inspect / stop the MJX trainer as a *detached* background process.
# Use this instead of running `python train_mjx.py ...` directly, so neither
# Claude's Bash tool nor your shell ever gets stuck attached to the trainer.
#
# Usage:
#   scripts/train_bg.sh start [task]      # detach trainer; default task=walk
#   scripts/train_bg.sh status            # pid + last 20 log lines
#   scripts/train_bg.sh tail [N]          # last N lines (default 40)
#   scripts/train_bg.sh stop              # kill the trainer
#
# Extra args after `start <task>` are forwarded to train_mjx.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PID_FILE="$ROOT/runs/.train.pid"
LOG_FILE="$ROOT/train.log"

_alive() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid; pid="$(cat "$PID_FILE")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# Any train_mjx.py python process anywhere on this box, regardless of how it was launched.
_any_trainer_pids() {
    pgrep -f "[t]rain_mjx\.py" 2>/dev/null || true
}

cmd="${1:-status}"; shift || true

case "$cmd" in
    start)
        stray_pids="$(_any_trainer_pids)"
        if [[ -n "$stray_pids" ]]; then
            echo "trainer(s) already running — refusing to start a second one:" >&2
            ps -o pid,etime,args -p $stray_pids >&2
            echo "use '$0 stop' (kills all train_mjx processes) first" >&2
            exit 1
        fi
        # Refuse to start if another process is already on the GPU — JAX will OOM
        # silently or fight for memory, and the user almost never wants this.
        gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
        if [[ -n "$gpu_pids" ]]; then
            echo "another process is already using the GPU — refusing to start:" >&2
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
            echo "kill it first, or set ALLOW_GPU_SHARE=1 to override" >&2
            [[ "${ALLOW_GPU_SHARE:-0}" != "1" ]] && exit 1
        fi
        task="${1:-walk}"; shift || true
        mkdir -p "$ROOT/runs"
        # nohup + setsid + </dev/null detaches fully from this shell.
        # Output is redirected to train.log; no `tee` (which would keep a parent attached).
        MUJOCO_BACKEND=mjx nohup setsid "$ROOT/.venv/bin/python" \
            train_mjx.py --task "$task" --no-watch "$@" \
            >"$LOG_FILE" 2>&1 </dev/null &
        echo $! >"$PID_FILE"
        disown || true
        sleep 1
        echo "started trainer pid=$(cat "$PID_FILE") task=$task  log=$LOG_FILE"
        ;;
    status)
        if _alive; then
            echo "RUNNING  pid=$(cat "$PID_FILE")  log=$LOG_FILE"
        else
            echo "STOPPED  log=$LOG_FILE"
        fi
        echo "---"
        tail -n 20 "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
        ;;
    tail)
        n="${1:-40}"
        tail -n "$n" "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
        ;;
    stop)
        # Kill every train_mjx.py process on the box, not just ours — catches
        # trainers spawned by other Claude/Zed sessions that bypassed this wrapper.
        pids="$(_any_trainer_pids)"
        if [[ -z "$pids" ]]; then
            echo "no trainer running"
            rm -f "$PID_FILE"
            exit 0
        fi
        for pid in $pids; do
            kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done
        for _ in 1 2 3 4 5; do
            [[ -z "$(_any_trainer_pids)" ]] && break
            sleep 1
        done
        for pid in $(_any_trainer_pids); do
            kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        done
        rm -f "$PID_FILE"
        echo "stopped pids=$pids"
        ;;
    *)
        echo "usage: $0 {start [task]|status|tail [N]|stop}" >&2
        exit 2
        ;;
esac
