#!/usr/bin/env bash
# Push runs/ + brain checkpoints to a cloud mirror.
#
# Default backend: Hugging Face Hub (free, git-based, browsable, supports LFS).
# Set HF_REPO=user/repo-name and run `huggingface-cli login` once.
#
# Alternative: set CLOUD_BACKEND=rclone REMOTE=remote_name:bucket/path to use
# rclone instead (works with B2, R2, S3, GCS, etc.).
set -euo pipefail

CLOUD_BACKEND="${CLOUD_BACKEND:-hf}"
TARGETS=("runs" "brain")

case "$CLOUD_BACKEND" in
  hf)
    : "${HF_REPO:?set HF_REPO=user/repo-name (run huggingface-cli login first)}"
    for t in "${TARGETS[@]}"; do
      [ -d "$t" ] || continue
      echo "[cloud_push] hf upload-folder $t → $HF_REPO/$t"
      huggingface-cli upload "$HF_REPO" "$t" "$t" --repo-type=dataset
    done
    ;;
  rclone)
    : "${REMOTE:?set REMOTE=remote_name:bucket/path}"
    for t in "${TARGETS[@]}"; do
      [ -d "$t" ] || continue
      echo "[cloud_push] rclone sync $t → $REMOTE/$t"
      rclone sync "$t" "$REMOTE/$t" --progress
    done
    ;;
  *)
    echo "unknown CLOUD_BACKEND=$CLOUD_BACKEND (expected: hf, rclone)" >&2
    exit 1
    ;;
esac
echo "[cloud_push] done"
