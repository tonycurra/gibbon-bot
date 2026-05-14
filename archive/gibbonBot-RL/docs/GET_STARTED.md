# Get started — train stand locally in Docker

This is the **canonical workflow** for running training on your own machine.
We use Docker so the CUDA + JAX + MuJoCo versions are pinned and identical
to what runs on remote GPU hosts. You don't need to know Docker beyond the
six commands in this doc.

> **Inspection vs training.** The `look.py` viewer (and any one-off Python
> snippet you run locally) uses your **venv**. The trainer (`train_mjx.py`)
> runs inside **Docker**. They share the same `data/`, `runs/`, and
> `mlruns/` because Docker mounts those folders from your repo. So edits to
> `data/scene.xml` or `envs/stand_env_mjx.py` take effect immediately for
> the next training run — no rebuild needed.

---

## What we're training right now

- **Task**: `stand` — keep the torso upright on the floor with all 4 limbs.
- **Scene**: `data/scene.xml` (the new 4-limb, two-slab-torso, radians-compiler scene).
- **Env**: `envs/stand_env_mjx.py`.
- **Backend**: MJX + MuJoCo-Warp. The Docker container has
  `MUJOCO_BACKEND=mjwarp` set by default, so you don't pass anything.
  To force pure MJX (no Warp) for a single run:
  `MUJOCO_BACKEND=mjx scripts/train_bg.sh docker stand`.
- **Memory**: ApeBrain episodic store is **off** unless you pass `--brain-dir`.
- **Resume**: the trainer auto-resumes from the newest compatible checkpoint
  in `runs/ape_stand_mjx/`. If it can't, it starts fresh and logs a warning.

---

## One-time setup (5 min)

```bash
# 1. Build the docker image (~5 GB; only needs to be done once).
docker build -t gibbonbot:latest .

# 2. Sanity-check that the GPU shows up inside the container.
docker run --rm --gpus all gibbonbot:latest nvidia-smi
```

You should see your GPU listed. If `--gpus all` fails, install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Daily commands

```bash
# Start stand training in the background (detached). 
# Uses Docker, --warp backend, auto-resumes if checkpoints exist.
scripts/train_bg.sh docker stand

# Watch progress (separate terminal — Ctrl-C just stops the tail, not the trainer).
scripts/train_bg.sh tail            # last 40 lines
scripts/train_bg.sh tail 200        # last 200 lines
tail -f train.log                   # live follow

# Status: PID, uptime, last 20 log lines.
scripts/train_bg.sh status

# Stop. Kills every train_mjx.py process on the box — no exceptions.
scripts/train_bg.sh stop
```

> **Why `docker` and not `start`?** `start` runs the trainer in your local
> venv (faster, no container overhead, but versions can drift). `docker`
> runs it in the container (slower to launch by ~30s, but guaranteed
> reproducible). We use `docker` as the canonical path; reach for `start`
> only if you've made a Python change you want to test instantly.

---

## Watch the training

Three options, pick whatever you prefer:

```bash
# 1. Raw log tail.
tail -f train.log

# 2. MLflow web UI — reward curves, losses, system metrics.
docker compose up -d mlflow_ui
# → open http://localhost:5000 in your browser

# 3. The custom dashboard (optional; reads MLflow + log + runs/).
scripts/dashboard.sh
# → open http://localhost:8765
```

## Inspect the current policy

The trainer saves checkpoints to `runs/ape_stand_mjx/mjx_*.npz` every
~25,000 steps. To watch the current best policy in MuJoCo's GUI viewer:

```bash
# Newest compatible checkpoint:
python view.py --task stand

# Or the best by reward-per-MLflow:
python view.py --task stand --best
```

Both use your local **venv**, not Docker, because the GUI viewer needs a
display. They hot-reload whenever a new checkpoint appears.

To just look at the rest pose (no policy, no checkpoint, just gravity):

```bash
python look.py
```

---

## What's actually happening when you start a run

1. `scripts/train_bg.sh docker stand` runs:
   ```
   docker compose run trainer python train_mjx.py --task stand --warp --no-watch
   ```
2. Docker spins up `gibbonbot:latest` with your GPU attached and these mounts:
   - `./data` → `/app/data` (so scene edits land without rebuild)
   - `./envs` → `/app/envs` (same)
   - `./scripts` → `/app/scripts`
   - `./runs` → `/app/runs` (so checkpoints persist on host)
   - `./mlruns` → `/app/mlruns`
   - `./mlflow.db` → `/app/mlflow.db`
3. `train_mjx.py` finds `runs/ape_stand_mjx/`, loads the newest compatible
   `mjx_*.npz` checkpoint if the shapes match the current network, and
   continues training. Otherwise it starts at step 0.
4. Every ~25k env-steps it writes `mjx_<NNN>.npz` + a `.safetensors` sidecar
   and logs reward to MLflow.

---

## Troubleshooting

**`scripts/train_bg.sh stand` → "usage:..."**
The first arg has to be one of `start|docker|status|tail|stop`. The
thing you want is `scripts/train_bg.sh docker stand`. (No `--warp`
needed — Warp is already on by default inside the container via
`MUJOCO_BACKEND=mjwarp` in `docker-compose.yml`.)

**"another process is already using the GPU — refusing to start"**
The wrapper refuses to launch a second trainer on the same GPU because JAX
will silently fight for memory. Run `scripts/train_bg.sh stop` first, or
override with `ALLOW_GPU_SHARE=1 scripts/train_bg.sh docker stand`
(only if you really know there's nothing important running).

**Checkpoint shape mismatch on resume**
Means the obs-dim or action-dim of `envs/stand_env_mjx.py` changed since
the last run. The trainer logs the mismatch and starts fresh. Old
checkpoints stay in `runs/` — they don't get deleted.

**Image build fails on `apt-get`**
Usually a transient network issue. Re-run `docker build -t gibbonbot:latest .`
once or twice. If it persists, check your proxy / DNS.

---

## Files you'll actually touch

| File | What it controls |
|---|---|
| `data/scene.xml` | The robot + floor + lights. See `data/SCENE_GEOMETRY.md` before edits. |
| `envs/stand_env_mjx.py` | Reward function, episode termination, obs construction. |
| `envs/params_stand_mjx.py` | Hyperparams (n_envs, lr, ppo epochs, etc.). |
| `train_mjx.py` | The trainer itself; rarely edit. |
| `docs/LESSONS.md` | Append a note when a run teaches you something. |
| `docs/BACKLOG.md` | Ideas not yet tried. |

All four can be edited mid-training — Python files only land on the **next**
trainer restart, but `data/scene.xml` edits affect the **next env reset**
(every few seconds during training).
