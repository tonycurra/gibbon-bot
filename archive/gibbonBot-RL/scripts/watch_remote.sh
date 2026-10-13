#!/usr/bin/env bash
# Pull live training artifacts from a remote training box into local runs/.
# Designed to run in the background so view.py can hot-swap fresh checkpoints.
#
# Usage:
#   scripts/watch_remote.sh                # default host/path, 10s interval
#   INTERVAL=5 HOST=root@1.2.3.4 scripts/watch_remote.sh
#
# Pulls: runs/ape_<TASK>_mjx/{mjx_*.npz,*.safetensors,train.log,keepers/}
#        mlflow.db (so local mlflow ui shows live metrics)
set -euo pipefail

HOST="${HOST:?Set HOST=user@host to the GPU box you rented}"
REMOTE_DIR="${REMOTE_DIR:-/root/gibbonBot-RL}"
TASK="${TASK:-walk}"
INTERVAL="${INTERVAL:-10}"

LOCAL_RUN="runs/ape_${TASK}_mjx"
mkdir -p "$LOCAL_RUN"

echo "[watch] $HOST:$REMOTE_DIR/runs/ape_${TASK}_mjx/ -> $LOCAL_RUN/ every ${INTERVAL}s"
echo "[watch] Ctrl-C to stop"

while true; do
  rsync -az --partial \
    --include='mjx_*.npz' --include='*.safetensors' --include='*.json' \
    --include='train.log' --include='keepers/***' \
    --include='*/' --exclude='*' \
    "$HOST:$REMOTE_DIR/runs/ape_${TASK}_mjx/" "$LOCAL_RUN/" 2>/dev/null || true

  # Pull mlflow.db so local `mlflow ui` reads live metrics (optional).
  rsync -az "$HOST:$REMOTE_DIR/mlflow.db" ./mlflow.db.remote 2>/dev/null || true

  # Tail the most recent N lines of remote train log into a local mirror.
  ssh -o ConnectTimeout=5 "$HOST" "tail -200 $REMOTE_DIR/train_walk.log" \
    > "$LOCAL_RUN/train.log.remote" 2>/dev/null || true

  sleep "$INTERVAL"
done
