#!/bin/bash
# Drive the remote H100 from your laptop. One command per action.
#   ./remote.sh start    # kick training in detached tmux at H100-max settings
#   ./remote.sh log      # live-tail the training log (Ctrl-C only stops the tail)
#   ./remote.sh status   # last log lines + nvidia-smi snapshot
#   ./remote.sh attach   # ssh in and attach to the tmux session
#   ./remote.sh stop     # send Ctrl-C to training (graceful save), tmux survives
#   ./remote.sh kill     # nuke the tmux session entirely
#   ./remote.sh gpu      # one-shot nvidia-smi
#   ./remote.sh sync     # one-shot pull of checkpoints + mlruns + rollouts
set -euo pipefail

REMOTE=root@51.159.167.125
REPO=gibbonBot-RL
SESSION=train
RUN_DIR="$REPO/runs/ape_walk_mjx"

# H100-max defaults — override with env vars if needed
ENVS=${ENVS:-16384}
ROLLOUT_EVERY=${ROLLOUT_EVERY:-50}
ROLLOUT_ENVS=${ROLLOUT_ENVS:-256}
RESUME=${RESUME:-}     # set RESUME=1 to add --resume

cmd=${1:-help}

case "$cmd" in
  start)
    resume_flag=""
    [[ -n "$RESUME" ]] && resume_flag="--resume"
    # python3 -u so crashes are flushed; keep stdbuf in case tee buffers; trap the
    # exit code so a crashed run leaves a visible footer in the log instead of
    # vanishing.
    # In-process tee writes train.log line-buffered (shell tee buffered at 4KB).
    # We just truncate the file at start; python appends from there.
    train_cmd="cd $REPO && export JAX_COMPILATION_CACHE_DIR=~/.jax_cache PYTHONUNBUFFERED=1 \
&& mkdir -p $RUN_DIR && : > $RUN_DIR/train.log \
&& python3 -u train_walk_mjx.py --no-watch --envs $ENVS \
     --save-rollouts-every $ROLLOUT_EVERY --rollout-envs $ROLLOUT_ENVS $resume_flag; \
   ec=\$?; echo \"[remote.sh] python exited with code \$ec\" >> $RUN_DIR/train.log; \
   echo \"[remote.sh] tmux command finished — pane will close\""
    # auto-replace dead/stale sessions instead of refusing
    ssh "$REMOTE" "tmux kill-session -t $SESSION 2>/dev/null; \
                   tmux kill-session -t monitor 2>/dev/null; \
                   tmux new-session -d -s $SESSION '$train_cmd'; \
                   tmux new-session -d -s monitor 'cd $REPO && ENVS=$ENVS stdbuf -oL -eL python3 -u monitor.py'"
    echo "[remote.sh] training started in tmux session '$SESSION' on $REMOTE"
    echo "[remote.sh] watchdog started in tmux session 'monitor' (auto-restarts on stall)"
    echo "[remote.sh] envs=$ENVS  rollout_every=$ROLLOUT_EVERY  rollout_envs=$ROLLOUT_ENVS  resume=${RESUME:-auto}"
    echo "[remote.sh] watch with: ./remote.sh log    OR    ./remote.sh status    OR    ./remote.sh health"
    ;;

  log)
    ssh "$REMOTE" "tail -f $RUN_DIR/train.log"
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
    ssh "$REMOTE" "tmux kill-session -t $SESSION 2>/dev/null; tmux kill-session -t monitor 2>/dev/null; true"
    echo "[remote.sh] tmux sessions '$SESSION' and 'monitor' killed"
    ;;

  gpu)
    ssh "$REMOTE" "nvidia-smi"
    ;;

  sync)
    ./pull_checkpoints.sh &
    pid=$!
    sleep 35
    kill "$pid" 2>/dev/null || true
    echo "[remote.sh] one sync tick done"
    ;;

  *)
    echo "usage: $0 {start|log|status|health|attach|stop|kill|gpu|sync}"
    echo "  ENVS=N RESUME=1 ./remote.sh start    # override defaults"
    exit 1
    ;;
esac
