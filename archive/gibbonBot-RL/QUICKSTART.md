# Quickstart

The recommended workflow is: tune with `test.py`, then scale up with `train.py`.

---

## Step 1 — Test and tune (CPU, live viewer)

```bash
python test.py
```

This trains on CPU in a single environment with the MuJoCo viewer embedded — you're watching the actual robot being trained, not a replay. Episodes print to the terminal:

```
Ep    1 | FLIPPED    | steps=  23 | height=0.312m | h_rew=4.87 | prox=0.14 | reward=  -89.4
Ep    2 | TIMEOUT    | steps=2000 | height=0.418m | h_rew=3.21 | prox=0.02 | reward= 6203.1
```

**Options:**

```bash
python test.py               # fresh run
python test.py --resume      # continue from last test checkpoint
python test.py --inherit     # warm-start weights from runs/ape_stand/ (production run)
python test.py --no-watch    # headless (no viewer)
```

Checkpoints save to `runs/ape_stand_test/` every 10k steps.

---

## Step 2 — Tune parameters live

In a second terminal, open the slider UI:

```bash
python envs/params.py
```

Drag the sliders — changes take effect within one episode, no restart needed.

**Key things to tune:**

| Slider | What to watch for |
|--------|------------------|
| `REALTIME_FACTOR` | Speed up the viewer so you can see more episodes faster |
| `TARGET_Z` | Start low (0.2m) and increase once the robot can hold a height |
| `TARGET_TOL` | Wider sigma = easier early on; tighten once it's consistent |
| `W_HEIGHT` / `W_UPRIGHT` | Balance height vs staying level |
| `W_PROXIMITY` | Increase if arms keep tangling; 0 to disable |
| `ARM_SAFETY_MARGIN` | How far apart arms need to stay before penalty kicks in |

**Terminal output guide:**
- `FLIPPED` — robot went upside-down (flip penalty applied)
- `COLLISION` — arms touched (info only, no termination)
- `TIMEOUT` — episode ran the full 2000 steps (good)

---

## Step 3 — Scale up with GPU training

Once the reward is trending up and behaviour looks reasonable in the viewer:

```bash
python train.py
```

This auto-detects your GPU, trains headless, and opens a separate replay window showing the best checkpoint so far.

**Options:**

```bash
python train.py                # GPU auto-detect + replay viewer
python train.py --cpu          # force CPU
python train.py --no-watch     # fully headless, no window
python train.py --envs 4       # parallel environments (faster on GPU)
python train.py --inherit      # warm-start from best checkpoint in runs/ape_stand/
python train.py --resume       # continue exact run from last checkpoint
```

Checkpoints save to `runs/ape_stand/` every 50k steps. The best checkpoint is always at `runs/ape_stand/best/best_model.zip`.

---

## Typical flow

```
python test.py                   # terminal 1: training + viewer
python envs/params.py            # terminal 2: slider UI

# ... tune until reward climbs and behaviour looks good ...

python train.py --envs 4         # terminal 1: full GPU run
```

To carry test weights into the full run:

```bash
# test.py saves to runs/ape_stand_test/
# train.py reads from runs/ape_stand/
# so copy the best test checkpoint first:
cp runs/ape_stand_test/best/best_model.zip runs/ape_stand/best/best_model.zip
python train.py --inherit
```
