# Quickstart — train on a remote GPU

The whole workflow, in commands you can paste. Everything below assumes you
have a remote H100 (or similar CUDA box) at the IP set in `remote.sh`. For
the deep-dive (cost notes, teardown, troubleshooting), see
[docs/REMOTE_GPU.md](docs/REMOTE_GPU.md).

## TL;DR

```bash
# 0. (one-time) prep the remote box
ssh root@<IP> 'apt update && apt install -y python3-pip python3-venv git tmux rsync'
ssh root@<IP> 'git clone ssh://git@codeberg.org/tony_olivabot/gibbonBot-RL.git'
ssh root@<IP> 'cd gibbonBot-RL && python3 -m venv .venv && source .venv/bin/activate \
              && pip install --upgrade pip "jax[cuda12]" mujoco mujoco-mjx flax optax numpy mlflow'

# 1. push your local changes
git push

# 2. start training in the background on the remote
./remote.sh start

# 3. confirm it's actually running
./remote.sh status

# 4. watch progress (any of these — separate terminals)
./remote.sh log                    # raw log tail
./pull_checkpoints.sh              # syncs checkpoints + mlruns + rollouts
python view.py --env walk          # visual policy hot-swap
mlflow ui                          # http://localhost:5000 — reward curves

# 5. when you're done
./remote.sh stop                   # graceful save
./remote.sh sync                   # final pull
# then destroy the instance in your cloud provider's console
```

## What `./remote.sh` does

Single helper, one sub-command per action — no need to remember ssh+tmux flags:

| Command | What |
|---|---|
| `./remote.sh start` | Kick training in detached tmux at H100-max defaults (16384 envs, rollout dumps every 50 updates) |
| `./remote.sh status` | Last 30 log lines + `nvidia-smi` summary + tmux session list |
| `./remote.sh log` | Live `tail -f` of `train.log` (Ctrl-C only stops the tail, training keeps running) |
| `./remote.sh attach` | SSH in and join the tmux session interactively |
| `./remote.sh stop` | Send Ctrl-C to training — fires the graceful save handler |
| `./remote.sh kill` | Nuke the tmux session entirely (use after `stop` saved) |
| `./remote.sh gpu` | One-shot `nvidia-smi` |
| `./remote.sh sync` | One tick of `pull_checkpoints.sh` |

Override defaults per invocation via env vars:

```bash
ENVS=8192 ./remote.sh start          # smaller batch
RESUME=1 ./remote.sh start           # continue from latest checkpoint
ROLLOUT_EVERY=100 ./remote.sh start  # less frequent rollout dumps
```

## Three lenses on training progress

Pick whichever fits the moment. They're independent.

**Visual** — does the policy actually look like it's walking?
```bash
./pull_checkpoints.sh        # leave running in one terminal
python view.py --env walk    # hot-swaps to each new checkpoint
```

**Metric curves** — is reward climbing? Is sps stable?
```bash
./pull_checkpoints.sh   # also syncs mlruns/
mlflow ui               # http://localhost:5000
```

**Raw log** — lowest overhead, just see the printed updates:
```bash
./remote.sh log
```

If `mean_rew` is climbing in the log but `view.py` looks stuck, your
`pull_checkpoints.sh` isn't running or hasn't synced yet — check the local
file mtimes in `runs/ape_walk_mjx/`.

## What gets saved where

The H100's compute output, after a session:

| What | Where | Auto-pulled? |
|---|---|---|
| Policy weights | `runs/ape_walk_mjx/mjx_*.npz` | yes (by `pull_checkpoints.sh`) |
| Per-update metrics + hyperparams | `mlruns/` | yes |
| Rollout transitions (opt-in) | `runs/ape_walk_mjx/rollouts/*.npz` | yes |
| Run lineage | `runs/ape_walk_mjx/lineage.json` | tracked in git directly |
| Stdout log | `runs/ape_walk_mjx/train.log` | yes |

To keep a checkpoint in git permanently (e.g. "first run that walked"):

```bash
mkdir -p runs/ape_walk_mjx/keepers
cp runs/ape_walk_mjx/mjx_<good-step>.npz runs/ape_walk_mjx/keepers/v1_first_walk.npz
git add runs/ape_walk_mjx/keepers/ && git commit -m "keep: v1 first walk"
```

The `keepers/` subdir is the only place under `runs/` that **isn't** gitignored.

## Local-only iteration

If you don't have a remote GPU and just want to develop against `view.py`:

```bash
python view.py --env walk     # POLICY="latest" loads newest checkpoint
```

Edit `POLICY` at the top of `view.py` to switch between `"zero"` / `"random"` /
`"latest"` for sanity checks before committing to a long training run.

## Troubleshooting

| Symptom | First check |
|---|---|
| `./remote.sh start` says "session already exists" | `./remote.sh kill` then retry |
| `attach` shows `[exited]` immediately | Training crashed — read the log: `ssh root@<IP> 'tail -80 gibbonBot-RL/runs/ape_walk_mjx/train.log'` |
| `view.py` policy looks unchanged after hours | `pull_checkpoints.sh` not running, or remote `mean_rew` is flat (different problem — see mlflow) |
| `GLFWError: DISPLAY missing` | You ran `view.py` over plain SSH. Run it locally instead. |
| GPU util < 50% in `./remote.sh status` | Try `ENVS=24576 ./remote.sh start` |
