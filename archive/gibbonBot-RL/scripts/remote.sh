#!/bin/bash
# Drive the remote H100 from your laptop. Run from repo root:
#   scripts/remote.sh start    # kick training in detached tmux at H100-max settings
#   scripts/remote.sh log      # live-tail the training log
#   scripts/remote.sh status   # last log lines + nvidia-smi snapshot
#   scripts/remote.sh attach   # ssh in and attach to the tmux session
#   scripts/remote.sh stop     # send Ctrl-C to training (graceful save)
#   scripts/remote.sh kill     # nuke the tmux session entirely
#   scripts/remote.sh gpu      # one-shot nvidia-smi
#   scripts/remote.sh sync     # one-shot pull of checkpoints + mlruns + rollouts
set -euo pipefail

REMOTE=root@51.159.167.125
REPO=gibbonBot-RL
SESSION=train
RUN_DIR="$REPO/runs/ape_walk_mjx"

# H100-max defaults — override with env vars if needed
ENVS=${ENVS:-16384}
ROLLOUT_EVERY=${ROLLOUT_EVERY:-50}
ROLLOUT_ENVS=${ROLLOUT_ENVS:-256}
RESUME=${RESUME:-1}    # default ON: force-resume past fingerprint mismatch (so param edits keep weights). Set RESUME=0 to start fresh.
BACKEND=${MUJOCO_BACKEND:-mjwarp}    # mjwarp = NVIDIA Warp kernels (2-5x faster on H100). Set MUJOCO_BACKEND=mjx to fall back.

cmd=${1:-help}

case "$cmd" in
  start)
    resume_flag=""
    [[ "$RESUME" != "0" && -n "$RESUME" ]] && resume_flag="--resume"
    # python3 -u so crashes are flushed; keep stdbuf in case tee buffers; trap the
    # exit code so a crashed run leaves a visible footer in the log instead of
    # vanishing.
    # In-process tee writes train.log line-buffered (shell tee buffered at 4KB).
    # We just truncate the file at start; python appends from there.
    # XLA_PYTHON_CLIENT_MEM_FRACTION caps VRAM so a parallel stand job can fit on the same H100.
    train_cmd="cd $REPO && export JAX_COMPILATION_CACHE_DIR=~/.jax_cache PYTHONUNBUFFERED=1 \
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 MUJOCO_BACKEND=$BACKEND \
&& mkdir -p $RUN_DIR && : > $RUN_DIR/train.log \
&& python3 -u train_walk_mjx.py --no-watch --envs $ENVS \
     --save-rollouts-every $ROLLOUT_EVERY --rollout-envs $ROLLOUT_ENVS $resume_flag; \
   ec=\$?; echo \"[remote.sh] python exited with code \$ec\" >> $RUN_DIR/train.log; \
   echo \"[remote.sh] tmux command finished — pane will close\""
    # auto-replace dead/stale sessions instead of refusing
    ssh "$REMOTE" "tmux kill-session -t $SESSION 2>/dev/null; \
                   tmux kill-session -t monitor 2>/dev/null; \
                   tmux new-session -d -s $SESSION '$train_cmd'; \
                   tmux new-session -d -s monitor 'cd $REPO && ENVS=$ENVS stdbuf -oL -eL python3 -u scripts/monitor.py'"
    echo "[remote.sh] training started in tmux session '$SESSION' on $REMOTE"
    echo "[remote.sh] watchdog started in tmux session 'monitor' (auto-restarts on stall)"
    echo "[remote.sh] envs=$ENVS  rollout_every=$ROLLOUT_EVERY  rollout_envs=$ROLLOUT_ENVS  resume=${RESUME:-auto}  backend=$BACKEND"
    echo "[remote.sh] watch with: ./remote.sh log    OR    ./remote.sh status    OR    ./remote.sh health"
    ;;

  log)
    ssh "$REMOTE" "tail -f $RUN_DIR/train.log"
    ;;

  start_stand)
    # Parallel training of stand task on the same GPU. 4096 envs ≈ 12-16GB VRAM.
    STAND_ENVS=${STAND_ENVS:-4096}
    STAND_DIR=$REPO/runs/ape_stand_mjx
    stand_cmd="cd $REPO && export JAX_COMPILATION_CACHE_DIR=~/.jax_cache_stand PYTHONUNBUFFERED=1 \
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 MUJOCO_BACKEND=$BACKEND \
&& mkdir -p $STAND_DIR && : > $STAND_DIR/train.log \
&& python3 -u archive/train_mjx.py --no-watch --envs $STAND_ENVS --resume; \
   ec=\$?; echo \"[remote.sh] stand python exited with code \$ec\" >> $STAND_DIR/train.log"
    ssh "$REMOTE" "tmux kill-session -t train_stand 2>/dev/null; \
                   tmux new-session -d -s train_stand '$stand_cmd'"
    echo "[remote.sh] stand training started in parallel tmux session 'train_stand' (envs=$STAND_ENVS, ~20% VRAM)"
    echo "[remote.sh] WARNING: walk + stand share the GPU; both will be ~30% slower"
    ;;

  stop_stand)
    ssh "$REMOTE" "tmux send-keys -t train_stand C-c 2>/dev/null"
    echo "[remote.sh] sent Ctrl-C to train_stand"
    ;;

  health)
    ssh "$REMOTE" "cat $RUN_DIR/health.json 2>/dev/null || echo '(no health.json yet — monitor not running)'"
    ;;

  status)
    echo "=== last 30 log lines ==="
    ssh "$REMOTE" "tail -n 30 $RUN_DIR/train.log 2>/dev/null || echo '(no log yet)'"
    echo
    echo "=== nvidia-smi ==="
    ssh "$REMOTE" "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                              --format=csv,noheader"
    echo
    echo "=== tmux ==="
    ssh "$REMOTE" "tmux ls 2>/dev/null || echo '(no tmux sessions)'"
    ;;

  attach)
    ssh -t "$REMOTE" "tmux attach -t $SESSION"
    ;;

  stop)
    ssh "$REMOTE" "tmux send-keys -t $SESSION C-c"
    echo "[remote.sh] sent Ctrl-C — training will save final checkpoint and exit"
    ;;

  kill)
    ssh "$REMOTE" "tmux kill-session -t $SESSION 2>/dev/null; tmux kill-session -t monitor 2>/dev/null; tmux kill-session -t train_stand 2>/dev/null; true"
    echo "[remote.sh] tmux sessions '$SESSION', 'monitor', 'train_stand' killed"
    ;;

  gpu)
    ssh "$REMOTE" "nvidia-smi"
    ;;

  sync)
    "$(dirname "$0")/pull_checkpoints.sh" &
    pid=$!
    sleep 35
    kill "$pid" 2>/dev/null || true
    echo "[remote.sh] one sync tick done"
    ;;

  *)
    echo "usage: $0 {start|start_stand|log|status|health|attach|stop|stop_stand|kill|gpu|sync}"
    echo "  ENVS=N RESUME=1 ./remote.sh start          # walk task"
    echo "  STAND_ENVS=N    ./remote.sh start_stand    # parallel stand task"
    exit 1
    ;;
esac
