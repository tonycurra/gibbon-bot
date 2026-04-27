# Quickstart

Two workflows: **`--test`** for tuning (32 envs, frequent saves, viewer + params UI auto-launch), then scale up to a full run.

---

## Step 1 — Test mode (tune live)

```bash
python train_mjx.py --test
```

This starts:

1. MJX training with **32 parallel envs**, saving every 25k steps
2. A MuJoCo viewer (`utils/watch_mjx.py`) that hot-swaps to each new checkpoint
3. The categorised slider UI (`envs/params_mjx.py`) for live param tuning

Reward, motor, smoothness and learning params (`LEARNING_RATE`, `ENT_COEF`, `CLIP_EPS`, etc.) hot-reload between rollouts — no restart needed.

The watcher only loads checkpoints saved *after* training started (via `--since <timestamp>`), so old runs don't pollute your view.

---

## Step 2 — Tune in the slider UI

The UI groups params by category:

| Category | What to watch for |
|----------|------------------|
| **Viewer** | `VIEWER_SPEED` — display only, doesn't affect physics. Lower to study motion, raise to see more episodes. |
| **Motor** | `MOTOR_MAX_SPEED` — too low: frozen. Too high: jerky. |
| **Height Goal** | Start `TARGET_Z` low (~0.2m), wide `TARGET_TOL` (~0.1). Tighten as it learns. |
| **Arm Spacing** | Increase `W_PROXIMITY` if arms tangle. |
| **Survival** | `W_ALIVE` discourages early give-up; `FLIP_PENALTY` discourages flipping. |
| **Smoothness** | `W_SMOOTH` reduces twitching, `W_JVEL` reduces flailing, `W_ENERGY` rewards efficiency. |
| **Learning** | Lower `LEARNING_RATE` if reward oscillates; raise `ENT_COEF` if policy gets stuck. |

You can either drag the slider or type an exact value in the entry box (Enter to commit).

Terminal output during training:

```
   iter   total_steps   ep_len   mean_reward
      1         4,096       76         0.398
      2         8,192       84         0.868
[save] mjx_000000008192.npz
```

---

## Step 3 — Scale up

Once behaviour looks reasonable:

```bash
python train_mjx.py                  # 1024 envs, save every 1M steps
python train_mjx.py --envs 2048      # more envs (needs more VRAM)
python train_mjx.py --resume         # continue from latest checkpoint
python train_mjx.py --no-watch       # headless, no viewer/UI
```

Checkpoints save to `runs/ape_stand_mjx/mjx_<steps>.npz`. Run metadata is appended to `runs/ape_stand_mjx/lineage.json`.

---

## Standalone viewer

To watch any saved checkpoint without training:

```bash
python utils/watch_mjx.py            # loads latest mjx_*.npz
python utils/watch_mjx.py --since 0  # explicitly load any checkpoint
```

The viewer hot-reloads `envs/params_mjx.py` — adjust `VIEWER_SPEED` and `MOTOR_MAX_SPEED` live.
