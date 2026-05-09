#!/bin/bash
# Pulls fresh training checkpoints, mlflow runs, and log from the H100 every 30s.
# Stop with Ctrl+C.
REMOTE=root@51.159.167.125
RUN_SRC=$REMOTE:gibbonBot-RL/runs/ape_walk_mjx/
RUN_DST=/home/tony/gibbonBot-RL/runs/ape_walk_mjx/
MLF_SRC=$REMOTE:gibbonBot-RL/mlruns/
MLF_DST=/home/tony/gibbonBot-RL/mlruns/
while true; do
  rsync -az --include='mjx_*.npz' --include='*.json' --include='train.log' \
        --include='rollouts/' --include='rollouts/*.npz' \
        --exclude='*' "$RUN_SRC" "$RUN_DST"
  rsync -az "$MLF_SRC" "$MLF_DST" 2>/dev/null
  sleep 30
done
