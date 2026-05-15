#!/usr/bin/env bash
# Sync LIANA training artifacts with Scaleway object storage.
#
# What gets synced (all under bucket `scaleway:gibbonbot-training/`):
#   runs/        →  runs/             checkpoints, keepers, train.log, library/
#   mlflow.db    →  mlflow.db         single-file SQLite store
#   mlruns/      →  mlruns/           MLflow artifact tree
#   data/brain/  →  brain/            episodic memory + policy DB
#   LIANA_LOG.md →  LIANA_LOG.md      human narrative
#
# Usage:
#   scripts/liana_sync.sh up           # push local → remote (default)
#   scripts/liana_sync.sh down         # pull remote → local (CAREFUL: overwrites)
#   scripts/liana_sync.sh status       # show diff without copying
#
# Requires rclone with `scaleway:` remote configured.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="scaleway:gibbonbot-training"

PATHS=(
    "runs/:runs/"
    "mlruns/:mlruns/"
    "data/brain/:brain/"
)
FILES=(
    "mlflow.db"
    "LIANA_LOG.md"
)

cmd="${1:-up}"

case "$cmd" in
    up)
        echo "[sync] uploading to $REMOTE/"
        for pair in "${PATHS[@]}"; do
            local_dir="${pair%%:*}"
            remote_dir="${pair##*:}"
            [[ -d "$local_dir" ]] || { echo "[sync] skip $local_dir (missing)"; continue; }
            echo "  $local_dir → $REMOTE/$remote_dir"
            rclone sync "$local_dir" "$REMOTE/$remote_dir" \
                --exclude '.jax_cache/**' \
                --exclude '__pycache__/**' \
                --exclude '*.pyc' \
                --transfers 8 --checkers 16 \
                --progress
        done
        for f in "${FILES[@]}"; do
            [[ -f "$f" ]] || { echo "[sync] skip $f (missing)"; continue; }
            echo "  $f → $REMOTE/$f"
            rclone copyto "$f" "$REMOTE/$f"
        done
        echo "[sync] up done."
        ;;

    down)
        if [[ -z "${LIANA_CONFIRM_DOWN:-}" ]]; then
            read -r -p "[sync] DOWN will overwrite local runs/, mlruns/, mlflow.db, data/brain/. Proceed? [y/N] " ans
            [[ "$ans" =~ ^[Yy]$ ]] || { echo "aborted"; exit 1; }
        fi
        echo "[sync] downloading from $REMOTE/"
        for pair in "${PATHS[@]}"; do
            local_dir="${pair%%:*}"
            remote_dir="${pair##*:}"
            mkdir -p "$local_dir"
            echo "  $REMOTE/$remote_dir → $local_dir"
            rclone sync "$REMOTE/$remote_dir" "$local_dir" \
                --transfers 8 --checkers 16 \
                --progress
        done
        for f in "${FILES[@]}"; do
            echo "  $REMOTE/$f → $f"
            rclone copyto "$REMOTE/$f" "$f" || echo "[sync] skip $f (not on remote)"
        done
        echo "[sync] down done."
        ;;

    status)
        echo "[sync] diff vs $REMOTE/"
        for pair in "${PATHS[@]}"; do
            local_dir="${pair%%:*}"
            remote_dir="${pair##*:}"
            [[ -d "$local_dir" ]] || continue
            echo "── $local_dir vs $REMOTE/$remote_dir"
            rclone check "$local_dir" "$REMOTE/$remote_dir" --one-way 2>&1 | tail -5 || true
        done
        ;;

    *)
        echo "usage: $0 {up|down|status}" >&2
        exit 2
        ;;
esac
