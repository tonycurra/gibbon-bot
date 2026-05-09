#!/bin/bash
# Cold backup of training artifacts to a path of your choice.
#
# Usage:
#   ./archive_runs.sh                       # backup to default ~/gibbonbot_backups/<date>
#   ./archive_runs.sh /mnt/usb/backups      # backup to that path/<date>
#   BACKUP_ROOT=/mnt/nas ./archive_runs.sh  # same via env var
#
# What gets backed up:
#   runs/                — all checkpoints, lineage, rollouts, history archives
#   mlruns/              — full MLflow tracking store
#   data/scene.xml + xml — current scene definition
#   envs/params_*.py     — current hyperparams
#
# Idempotent — re-running on the same day overwrites that day's snapshot.
set -e
ROOT=${1:-${BACKUP_ROOT:-$HOME/gibbonbot_backups}}
STAMP=$(date +%Y-%m-%d)
DST="$ROOT/$STAMP"

mkdir -p "$DST"
echo "[archive] backing up → $DST"

rsync -az --delete --info=progress2 \
    --exclude='__pycache__' \
    runs/ "$DST/runs/"
rsync -az --delete --info=progress2 \
    mlruns/ "$DST/mlruns/" 2>/dev/null || true
rsync -az \
    data/scene.xml data/*.xml data/meshes \
    "$DST/data/" 2>/dev/null || true
rsync -az envs/ "$DST/envs/"

cd "$ROOT" && du -sh "$STAMP"
echo "[archive] done"
