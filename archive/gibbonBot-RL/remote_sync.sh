#!/usr/bin/env bash
# remote_sync.sh — bidirectional sync with the H100 training server.
#
# Pulls:  checkpoints + logs from server → local runs/
# Pushes: envs/hints.py (reward shaping) → server, so steering takes effect immediately
#
# Usage:
#   ./remote_sync.sh          # sync loop (Ctrl+C to stop)
#   ./remote_sync.sh once     # single sync and exit

SERVER="root@51.159.147.155"
REMOTE="/root/gb2.1"
LOCAL="$(cd "$(dirname "$0")" && pwd)"
INTERVAL=30   # seconds between syncs

sync_once() {
    echo "[sync $(date '+%H:%M:%S')]"

    # push hints.py so live reward steering reaches the server
    rsync -az "$LOCAL/envs/hints.py" "$SERVER:$REMOTE/envs/hints.py"

    # pull checkpoints and logs
    rsync -az --include='*.zip' --include='*.npz' --include='*.log' \
              --include='lineage.json' --include='*/' --exclude='*' \
        "$SERVER:$REMOTE/runs/" "$LOCAL/runs/"

    # show latest log lines
    tail -3 "$LOCAL/train_mjx.log" 2>/dev/null || true
}

if [[ "${1:-}" == "once" ]]; then
    sync_once
    exit 0
fi

echo "Starting sync loop (every ${INTERVAL}s). Ctrl+C to stop."
echo "  Push: envs/hints.py → server   (edit hints.py to steer training)"
echo "  Pull: runs/ checkpoints → local (watch.py will pick them up)"
echo ""
while true; do
    sync_once
    sleep "$INTERVAL"
done
