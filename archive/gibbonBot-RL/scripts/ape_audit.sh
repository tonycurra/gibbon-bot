#!/usr/bin/env bash
# One-shot audit of everything that could interfere with training on this box.
# Run anytime you want to know "what's actually using my machine?"
#   scripts/ape_audit.sh
set -u

hr() { printf '\n=== %s ===\n' "$1"; }

hr "Claude Code sessions (each one could spawn its own trainer)"
pgrep -af "anthropic-ai/claude-agent-sdk" 2>/dev/null | awk '{print $1, $NF}' | head -10 \
    || echo "(none)"

hr "Trainer processes (any train_mjx.py anywhere)"
pgrep -af "[t]rain_mjx\.py" 2>/dev/null || echo "(none)"

hr "Viewer processes (view.py)"
pgrep -af "[v]iew\.py" 2>/dev/null || echo "(none)"

hr "GPU compute clients"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
    || echo "(no nvidia-smi)"

hr "GPU utilization / memory"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null \
    || echo "(no nvidia-smi)"

hr "Top 5 CPU consumers"
ps -eo pid,user,pcpu,pmem,etime,comm --sort=-pcpu --no-headers | head -5

hr "Processes with cwd in this repo"
for d in /proc/[0-9]*; do
    pid="${d##*/}"
    target="$(readlink "$d/cwd" 2>/dev/null || true)"
    [[ "$target" == /home/tony/gibbonBot-RL* ]] || continue
    comm="$(cat "$d/comm" 2>/dev/null || echo '?')"
    printf '%6s  %-20s  %s\n' "$pid" "$comm" "$target"
done

hr "Open handles on train.log"
lsof /home/tony/gibbonBot-RL/train.log 2>/dev/null | tail -n +1 || echo "(none)"

hr "Trainer pid file"
pf=/home/tony/gibbonBot-RL/runs/.train.pid
if [[ -f "$pf" ]]; then
    pid="$(cat "$pf")"
    if kill -0 "$pid" 2>/dev/null; then echo "pid=$pid (alive)"; else echo "pid=$pid (STALE — process gone)"; fi
else
    echo "(no pid file)"
fi

hr "Zombies"
ps -eo pid,stat,comm | awk '$2 ~ /Z/ {print}' | head -5
[[ $(ps -eo stat | grep -c '^Z') -eq 0 ]] && echo "(none)"

echo
