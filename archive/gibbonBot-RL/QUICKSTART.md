# Quickstart

Two modes: **GO** (default) when you're done tweaking and want to train, and **TUNE** when you want to fiddle with sliders.

---

## GO mode — just train it

```bash
python train_mjx.py
```

Auto-picks **2048 envs on GPU** (or 64 on CPU). Spawns the viewer. **Does NOT spawn the params UI** — no temptation to tweak.

Terminal shows per-episode trend so you can tell if it's learning:

```
Ep  12 TIMEOUT torso=0.282m  ep=2000  reward≈0.412 ↑  (best 0.418)  [mjx_000262144.npz]
Ep  13 FLIPPED torso=0.094m  ep= 643  reward≈0.218 ↓  (best 0.418)  [mjx_000262144.npz]
Ep  14 TIMEOUT torso=0.301m  ep=2000  reward≈0.451 ↑  (best 0.451)  ← new best!  [mjx_000393216.npz]
```

`↑` / `↓` is the trend vs the rolling 20-episode mean. When `↑` outpaces `↓` and the best keeps climbing, training is working. Walk away. Come back in 30 minutes.

Other flags:

```bash
python train_mjx.py --envs 4096      # override env count
python train_mjx.py --resume         # continue from latest checkpoint
python train_mjx.py --no-watch       # headless, no viewer
```

Checkpoints save to `runs/ape_stand_mjx/mjx_<steps>.npz` every 1M steps. Run metadata is appended to `runs/ape_stand_mjx/lineage.json`.

---

## TUNE mode — slider UI

```bash
python train_mjx.py --tune
```

Same as GO mode but with **32 envs**, saves every 5k steps, and the categorised slider UI (`envs/params_mjx.py`) auto-launches.

Reward, motor, smoothness and learning params hot-reload between rollouts — no restart needed. Only `PHYSICS_STEPS_PER_ACTION` (marked ⚙ in the UI) requires a restart, since it's baked into the JIT compile.

Categories in the slider UI (top → bottom by visible impact):

| Category | What's in it |
|----------|--------------|
| ⚡ **Motion** | `PHYSICS_STEPS_PER_ACTION` ⚙, `MOTOR_MAX_SPEED`, `ACTION_SMOOTH_ALPHA`, `VIEWER_SPEED` |
| 🧊 **Smoothness Penalties** | `W_SMOOTH`, `W_JVEL`, `W_ENERGY` |
| 🎯 **Goal Shaping** | `TARGET_Z`, `TARGET_TOL`, `W_HEIGHT`, `W_UPRIGHT` |
| 💀 **Survival** | `W_ALIVE`, `FLIP_PENALTY` |
| 🤝 **Arm Safety** | `ARM_SAFETY_MARGIN`, `W_PROXIMITY` |
| 🧠 **Learning** | `LEARNING_RATE`, `ENT_COEF`, `CLIP_EPS` |

**Live feedback bars** sit next to motor / smoothness / goal params, fed by `runs/ape_stand_mjx/live_stats.json` (written by the viewer). Bars colour-shift green → amber → red as the measured value approaches the slider's max.

**Bottom stats panel** shows current checkpoint, torso z, upright, ep length, rolling flip rate, and avg episode length.

**Preset buttons** at the top:
- **Smooth** — historical "good" smoothness values
- **Fast train** — minimal penalties, fast iteration
- **Reset defaults** — restore starter values

---

## Walk-to-tree training

Separate environment + params + train script for "walk toward a tree without
dragging torso on the ground":

```bash
python gen_trees.py --inject data/scene.xml --at 3.0 0.0   # one tree at (3, 0)
python train_walk_mjx.py                                    # train with viewer
```

Files:

- `envs/walk_env_mjx.py` — env (OBS_DIM=86, adds tree-relative xy + dist)
- `envs/params_walk_mjx.py` — knobs: `W_PROGRESS`, `GROUND_CLEARANCE`, `W_GROUND`, `SUCCESS_DIST`
- `train_walk_mjx.py` — PPO loop, mirrors `train_mjx.py`, saves to `runs/ape_walk_mjx/`

To watch a trained walk policy without training:
```bash
python view.py --env walk
```

The terminal log shows `mean_dist` per update — should trend down from ~3.0 toward
`SUCCESS_DIST` (default 0.6).

---

## Forest scene generator

```bash
python gen_trees.py --n 8 --seed 42                  # write data/trees_forest.xml
python gen_trees.py --inject data/scene.xml --n 12   # inject into existing scene
python gen_trees.py --inject data/scene.xml --at 3 0 # one tree at fixed pos
python view_scene.py data/trees_forest.xml           # view standalone forest
```

Trees are capsule trunks + random branch capsules, friction tuned to ~0.7
(realistic for dry bark). Heights sampled from terrain.png so trees plant
correctly on uneven ground.

---

## Motor specs

`data/motors.md` documents the Waveshare ST32xx servos. Per-class joint damping
is set in `<default>` blocks at the top of `data/scene.xml` — derived from the
linear DC motor model `damping ≈ τ_stall / ω_no_load`:

- ST3235 / ST3215 main joints → `damping=0.62` (2.94 N·m / 4.71 rad/s)
- ST3215-HS gripper/foot      → `damping=0.18` (1.96 N·m / 11.10 rad/s)

If you tune reward shaping, note that `qvel²` penalties scale with motor speed —
the original stand env's `W_JVEL=0.5` was tuned for over-damped motors and will
over-penalise now. Walk env has been re-tuned to `W_JVEL=0.005`.

---

## Standalone viewer

To watch any saved checkpoint without training:

```bash
python utils/watch_mjx.py            # loads latest mjx_*.npz
python utils/watch_mjx.py --since 0  # explicitly load any checkpoint
```

The viewer hot-reloads `envs/params_mjx.py` — `VIEWER_SPEED` and `MOTOR_MAX_SPEED` change live.

---

## Track what worked

Every saved checkpoint automatically writes a `.params.json` sidecar capturing the params active at that moment. So you can always go back and see what produced a good policy.

```bash
python utils/tag.py "first stable stand, smooth motion"   # tag latest checkpoint
python utils/tag.py 262144 "before W_SMOOTH bump"         # tag by partial match
python utils/tag.py --list                                # list all tagged ckpts
python utils/lineage.py --checkpoints                     # tabular params per ckpt
```

Tags go to `runs/ape_stand_mjx/notes.md` (git-tracked). Workflow: train → spot a great episode in the viewer → grab the checkpoint name from the terminal → tag it. Future-you will thank present-you.
