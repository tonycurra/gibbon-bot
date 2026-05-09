# Remote GPU training (Scaleway H100)

Step-by-step for renting a remote H100, training fast, and not wasting money. Tested on Scaleway H100 PCIe 80GB (€2.73/hour, billed per minute) but works on any rented Linux box with an NVIDIA GPU.

---

## Driving it from your laptop (recommended)

`remote.sh` wraps the common ssh+tmux dance in single commands:

```bash
./remote.sh start     # kicks training in detached tmux on the H100
                      # defaults: --envs 16384, --save-rollouts-every 50

./remote.sh status    # last 30 log lines + nvidia-smi + tmux session list
./remote.sh log       # live `tail -f` of train.log (Ctrl-C just stops the tail)
./remote.sh attach    # ssh in and join the tmux session interactively
./remote.sh stop      # Ctrl-C the training — fires graceful save handler
./remote.sh kill      # nukes the tmux session (use after stop has saved)
./remote.sh gpu       # one-shot nvidia-smi
./remote.sh sync      # one tick of pull_checkpoints.sh
```

Override defaults via env vars: `ENVS=8192 ./remote.sh start`,
`RESUME=1 ./remote.sh start`, etc. Edit the `REMOTE=` line at the top of
`remote.sh` if the IP changes.

Typical session: `./remote.sh start` → walk away → `./remote.sh status`
periodically to confirm GPU util is pinned and reward is climbing →
`./remote.sh stop` when done → destroy the instance in the Scaleway console.

## TL;DR

You'll be training within 10 minutes. Two terminals: one ssh'd into the remote, one local for the viewer.

```bash
# === Local: push your latest code ===
git push origin mjx-gpu

# === SSH in (replace IP with yours) ===
ssh root@51.159.187.87

# === On remote, one-time setup (~5 min) ===
apt update && apt install -y python3-pip python3-venv git tmux rsync
git clone ssh://git@codeberg.org/tony_olivabot/gibbonBot-RL.git ape && cd ape
git checkout mjx-gpu
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip "jax[cuda12]" mujoco mujoco-mjx flax optax numpy mlflow

# === Start training in tmux ===
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
tmux new -s train
python train_walk_mjx.py --no-watch --envs 8192
# Ctrl+B then D to detach. Reattach later: tmux a -t train
```

```bash
# === Locally: stream checkpoints + mlflow runs down, watch policy ===
# Terminal 1 (sync loop — pulls .npz + mlruns every 30s, leave running):
./pull_checkpoints.sh

# Terminal 2 (viewer hot-swaps to each new checkpoint):
python view.py --env walk

# Terminal 3 (live metric curves: sps, mean_rew, mean_dist):
mlflow ui
# → http://localhost:5000
```

That's it. Three lenses on the same run: visual policy, metric curves, and (optionally) the raw log via `tail -f runs/ape_walk_mjx/train.log`.

---

## One-time setup on the remote

After SSH'ing in:

```bash
apt update && apt install -y python3-pip python3-venv git tmux rsync
git clone ssh://git@codeberg.org/tony_olivabot/gibbonBot-RL.git ape
cd ape
git checkout mjx-gpu
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade "jax[cuda12]"          # CRITICAL: cuda12 wheel for H100
pip install mujoco mujoco-mjx flax optax numpy
```

> **The `jax[cuda12]` part is non-negotiable on H100.** The default `pip install jax` gives you a CPU-only wheel and your H100 will sit idle while training crawls.

Set the JIT cache once per session so re-runs skip the ~3-5 min compile:

```bash
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
# (Add this to ~/.bashrc on the remote so it persists across SSH sessions.)
```

---

## Training in tmux

`tmux` keeps the training alive when SSH disconnects (which it will).

```bash
tmux new -s train             # create a new session called "train"
python train_mjx.py --no-watch --envs 8192
# Detach: Ctrl+B then D
# Reattach later: tmux a -t train
# List sessions: tmux ls
# Kill session: tmux kill-session -t train
```

Recommended starting flags for H100 80GB:

```bash
python train_mjx.py --no-watch --envs 8192
```

Push to `--envs 16384` if you want to use more VRAM (~50-60GB usage at that level). Above 16384 the gains diminish.

`--no-watch` is required — the remote box has no display.

---

## Watching from your laptop

Three independent lenses — pick whichever you need.

**1. Visual policy + metric history** (the default):
```bash
# Terminal 1 — pulls .npz checkpoints AND mlruns/ every 30s:
./pull_checkpoints.sh

# Terminal 2 — viewer hot-swaps to each new checkpoint:
python view.py --env walk

# Terminal 3 — local mlflow UI on the synced runs:
mlflow ui     # → http://localhost:5000
```

The `mlflow ui` reads `./mlruns/` (synced down by the pull script), so reward / sps / mean_dist curves stay live without ssh tunneling.

**2. Just the log** (lowest overhead):
```bash
ssh root@<IP> 'tail -f gibbonBot-RL/runs/ape_walk_mjx/train.log'
```

**3. Live mlflow over ssh tunnel** (skip the rsync, view the remote store directly):
```bash
ssh -L 5000:localhost:5000 root@<IP>
# on remote:
cd gibbonBot-RL && source .venv/bin/activate && mlflow ui --host 0.0.0.0
```

## Preserving what the GPU produced

The compute artifacts you don't want to lose:

| What | Where | How to preserve |
|---|---|---|
| Policy weights | `runs/<task>/mjx_*.npz` | gitignored by default — pulled by `pull_checkpoints.sh` |
| Metric history + hyperparams | `mlruns/` | gitignored — pulled by `pull_checkpoints.sh` |
| Run lineage | `runs/<task>/lineage.json` | tracked in git |
| Rollout transitions (opt-in) | `runs/<task>/rollouts/*.npz` | gitignored — pulled by `pull_checkpoints.sh` |

To version a checkpoint that's worth keeping (e.g. "first run that walked"), copy it into `runs/<task>/keepers/` — that subdir is **not** gitignored, so you can `git add` curated weights without polluting history with every intermediate save.

```bash
cp runs/ape_walk_mjx/mjx_000001523712.npz \
   runs/ape_walk_mjx/keepers/v1_first_walk_2M_steps.npz
git add runs/ape_walk_mjx/keepers/ && git commit -m "keep: v1 first walk"
```

---

## Saving rollout transitions

PPO normally consumes its rollout buffer in-place and discards it. To preserve
the actual `(obs, action, reward, done, value, logp, advantage, return)`
tuples the GPU produced — useful for offline RL, behavior cloning, or replay
analysis later — opt in with:

```bash
python train_walk_mjx.py --no-watch --envs 8192 \
    --save-rollouts-every 50 \
    --rollout-envs 256 \
    --rollout-keep 100
```

Each dump is one PPO update's experience, sub-sampled to `--rollout-envs`
trajectories (full `N_STEPS` length). Default `--rollout-envs 256` keeps each
file around ~10MB compressed at OBS_DIM=86. `--rollout-keep` is a ring buffer
size — older files are deleted to bound disk use.

Files land in `runs/ape_walk_mjx/rollouts/rollout_<update>_<steps>.npz` and
are pulled by `pull_checkpoints.sh`. Load them with `utils.rollout_io`:

```python
from utils.rollout_io import list_rollouts, load_rollout
files = list_rollouts("runs/ape_walk_mjx")
r = load_rollout(files[-1])  # dict: obs/act/rew/done/val/logp/adv/ret
```

Tradeoff: dumping forces a `device_get` sync, so don't set
`--save-rollouts-every` too aggressively (1 = every update would noticeably
slow training). 50–100 is a good default at 8192 envs.

## Cost-saving rules

You're paying €2.73/hour = €0.045/minute. Every minute counts.

- **Don't touch `params_mjx.py` mid-run.** Each change triggers a JIT recompile (~30s at 8192 envs ≈ €0.025 wasted per recompile). Set params before starting; if you must tweak, batch all your changes and accept one recompile.
- **First JIT compile takes 3-5 minutes.** That's normal. The terminal will look hung. Wait. Once the first `[save]` line appears you're flying.
- **Use `--resume`** if you stop and restart on the same instance. Picks up from latest checkpoint.
- **`JAX_COMPILATION_CACHE_DIR=~/.jax_cache`** survives across runs on the same instance (but NOT across new instances — the cache lives on remote disk).

Approximate throughput on H100 at 8192 envs: ~1-2M training steps/sec.

| Steps | Time | Cost |
|-------|------|------|
| 100M  | ~1-2 min | €0.05-0.10 |
| 1B    | ~10-15 min | €0.50-0.70 |
| 10B   | ~2-3 hours | €5-8 |

For a real "train it from scratch to standing" run, plan for ~5-10B steps.

---

## Recommended H100 settings

| Setting | Value | Why |
|---------|-------|-----|
| `--envs` | **8192** | Sweet spot for H100. Try 16384 to push it. |
| `--no-watch` | required | No display on remote |
| `LEARNING_RATE` | **3e-4** | Lower than local default — bigger batch is more stable, can use smaller LR |
| `PHYSICS_STEPS_PER_ACTION` | your choice | Doesn't affect VRAM. Use 200 for smooth motion, 16 for raw throughput. |
| Other params | "Smooth" preset values | Tune locally first in `--tune` mode, commit, then deploy |

> **Don't run `--tune` mode on a paid GPU.** The tuning UI is for local iteration. Find your params locally with 32 envs, commit them to `params_mjx.py`, then deploy to the H100 for the long run.

---

## Stopping cleanly

In the tmux session running training:

```bash
# Ctrl+C — fires the graceful shutdown handler
^C
[stop] Ctrl+C received — finishing current rollout, then saving...
[stop] Ctrl+C — flushing final checkpoint at 8,716,288 steps
[save] mjx_000008716288.npz  (+ mjx_000008716288.params.json)
```

The checkpoint is now safe. Exit tmux:

```bash
exit            # closes the tmux pane
exit            # closes the SSH session
```

Then **shut the instance down from the Scaleway web console**. The meter does not stop until the instance is destroyed (or you specifically stop+detach storage).

---

## Teardown checklist

1. Ctrl+C in the tmux training session, wait for the final save line.
2. One last sync from your laptop (run `./pull_checkpoints.sh` for one tick, or):
   ```bash
   rsync -avz root@<IP>:gibbonBot-RL/runs/ape_walk_mjx/ runs/ape_walk_mjx/
   rsync -avz root@<IP>:gibbonBot-RL/mlruns/             mlruns/
   ```
3. Sanity check the count matches:
   ```bash
   # On remote:
   ls ~/ape/runs/ape_stand_mjx/mjx_*.npz | wc -l
   # Locally:
   ls runs/ape_stand_mjx/mjx_*.npz | wc -l
   ```
4. **Shut down the instance** in the Scaleway console. Verify billing has stopped.

---

## Resuming a future session

A new instance = new disk = no JIT cache, no clone. So one-time setup repeats. But the workflow is faster because you've done it once:

```bash
# On a fresh remote box:
apt update && apt install -y python3-pip python3-venv git tmux rsync
git clone ssh://git@codeberg.org/tony_olivabot/gibbonBot-RL.git ape && cd ape
git checkout mjx-gpu
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip "jax[cuda12]" mujoco mujoco-mjx flax optax numpy

# If you want to continue from a previous run, push the latest checkpoint up first:
# (locally)
rsync -avz runs/ape_stand_mjx/mjx_*.npz* root@<IP>:ape/runs/ape_stand_mjx/

# Then on remote:
export JAX_COMPILATION_CACHE_DIR=~/.jax_cache
tmux new -s train
python train_mjx.py --no-watch --envs 8192 --resume
```
