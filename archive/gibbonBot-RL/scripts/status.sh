#!/usr/bin/env bash
# One-shot status dump — what's training, where, and how is it going.
# Run from anywhere; reads remote + local state in one go. No daemon, no polling.
#
#   scripts/status.sh         # full report
#   scripts/status.sh quick   # just the headline (GPU, last update, age)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE="$(grep '^REMOTE=' scripts/remote.sh | head -1 | cut -d= -f2)"
TASK_DEFAULT=walk

mode="${1:-full}"
b()   { printf "\033[1m%s\033[0m\n" "$1"; }
dim() { printf "\033[2m%s\033[0m\n" "$1"; }

# --- remote H100 ----------------------------------------------------------
if [[ -n "$REMOTE" ]]; then
    b "── remote H100 ($REMOTE) ──"
    remote_out="$(ssh -o ConnectTimeout=4 -o BatchMode=yes "$REMOTE" '
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
        echo "---tmux---"
        tmux ls 2>/dev/null || echo "(no tmux)"
        echo "---health---"
        cat gibbonBot-RL/runs/ape_walk_mjx/health.json 2>/dev/null || echo "(no health.json)"
        echo "---tail---"
        tail -n 3 gibbonBot-RL/runs/ape_walk_mjx/train.log 2>/dev/null | grep -E "^\s*[0-9]+\s+[0-9,]+\s+[0-9,]+" | tail -1 || echo "(no metric row yet)"
    ' 2>/dev/null)"

    if [[ -z "$remote_out" ]]; then
        echo "  unreachable"
    else
        gpu="$(echo "$remote_out" | sed -n '1p')"
        echo "  gpu:   $gpu"
        tmux_block="$(echo "$remote_out" | awk '/---tmux---/{f=1;next} /---health---/{f=0} f')"
        echo "  tmux:  $(echo "$tmux_block" | tr '\n' '; ' | sed 's/; $//')"
        health_block="$(echo "$remote_out" | awk '/---health---/{f=1;next} /---tail---/{f=0} f')"
        tail_block="$(echo "$remote_out" | awk '/---tail---/{f=1;next} f')"
        echo "  last:  $tail_block"
        if [[ "$mode" != "quick" ]]; then
            echo "  health.json:"
            echo "$health_block" | sed 's/^/    /'
        fi
    fi
    echo
fi

# --- local trainer --------------------------------------------------------
b "── local trainer ──"
local_pids="$(pgrep -f '[t]rain_mjx.py' 2>/dev/null || true)"
if [[ -n "$local_pids" ]]; then
    echo "  running pids: $local_pids"
    ps -o pid,etime,args -p $local_pids | tail -n +2 | sed 's/^/    /'
else
    echo "  not running"
fi
echo

# --- local sync state -----------------------------------------------------
b "── sync state ──"
watch_pid_file="$ROOT/runs/.watch_remote.pid"
if [[ -f "$watch_pid_file" ]] && kill -0 "$(cat "$watch_pid_file")" 2>/dev/null; then
    echo "  watch_remote: running pid=$(cat "$watch_pid_file")"
else
    echo "  watch_remote: off"
fi
mlf_db="$ROOT/mlflow.db.remote"
if [[ -f "$mlf_db" ]]; then
    age_s=$(( $(date +%s) - $(stat -c %Y "$mlf_db") ))
    echo "  mlflow.db.remote age: ${age_s}s ($(( age_s / 60 ))m)"
fi
for d in runs/ape_*_mjx; do
    [[ -d "$d" ]] || continue
    newest_ckpt="$(ls -t "$d"/mjx_*.npz 2>/dev/null | head -1)"
    [[ -z "$newest_ckpt" ]] && continue
    age_s=$(( $(date +%s) - $(stat -c %Y "$newest_ckpt") ))
    printf "  %-22s newest ckpt: %s  age: %ds (%dm)\n" \
        "${d#runs/}" "$(basename "$newest_ckpt")" "$age_s" "$(( age_s / 60 ))"
done
echo

# --- dashboard ------------------------------------------------------------
b "── dashboard ──"
if [[ -f "$ROOT/runs/.dashboard.pid" ]] && kill -0 "$(cat "$ROOT/runs/.dashboard.pid")" 2>/dev/null; then
    echo "  running pid=$(cat "$ROOT/runs/.dashboard.pid")  → http://127.0.0.1:8765"
else
    echo "  not running.  start with:  scripts/dashboard.sh start"
fi
