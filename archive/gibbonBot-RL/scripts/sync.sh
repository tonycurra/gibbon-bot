#!/usr/bin/env bash
# Push training data to git remote (Codeberg). Idempotent: bails if nothing changed.
# Called by trainer on each checkpoint save when --auto-push is set, or run manually.
set -e
cd "$(dirname "$0")/.."

# Stage only training-data paths — never sweep up arbitrary working-tree changes.
git add runs/ mlruns/ mlflow.db train.log 2>/dev/null || true

# If nothing staged, exit silently.
if git diff --cached --quiet; then
    exit 0
fi

msg="auto: training data $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git commit -m "$msg" --quiet
git push --quiet 2>&1 | tail -3
echo "[sync] pushed: $msg"
