#!/usr/bin/env bash
# Push training data to ALL configured git remotes. Idempotent: bails if nothing changed.
# Called by trainer on each checkpoint save when --auto-push is set, or run manually.
#
# Add backup remotes with:
#   git remote add backup <url>     # e.g. GitLab/GitHub mirror
#   git remote add archive <url>    # e.g. self-hosted Gitea on Hetzner
# Every remote listed in `git remote` will be pushed to.
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

# Push to every configured remote. Don't fail the whole script if one remote
# is down — log it, keep going. Other remotes still get the data.
for remote in $(git remote); do
    if git push --quiet "$remote" 2>/tmp/sync_${remote}_err; then
        echo "[sync] pushed to $remote: $msg"
    else
        echo "[sync] FAILED to push to $remote: $(cat /tmp/sync_${remote}_err | tail -1)"
    fi
done
