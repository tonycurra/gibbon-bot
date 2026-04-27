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

## Standalone viewer

To watch any saved checkpoint without training:

```bash
python utils/watch_mjx.py            # loads latest mjx_*.npz
python utils/watch_mjx.py --since 0  # explicitly load any checkpoint
```

The viewer hot-reloads `envs/params_mjx.py` — `VIEWER_SPEED` and `MOTOR_MAX_SPEED` change live.
