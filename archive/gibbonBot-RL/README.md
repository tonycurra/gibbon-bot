# 🗂️ Robotic Ape Project - Organized

**📁 See [README_ORGANIZED.md](README_ORGANIZED.md) for clean structure and quick start**

A 4-limb robotic ape built from **4x RoArm M3 Pro** arms, trained with reinforcement learning to walk, climb, and eventually brachiate — using real ape video as motion reference.

**🚀 Quick Start**: See [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## Status

| Layer | Tool | Status |
|-------|------|--------|
| Physics sim | MuJoCo 3.x | ✅ Working |
| Robot description | URDF → MJCF | ✅ Working |
| RL — reach tree | `ape_env.py` + PPO | ✅ Working |
| RL — stand up (CPU) | `stand_env.py` + SB3 PPO | ✅ Training |
| RL — stand up (GPU) | `stand_env_mjx.py` + JAX PPO | ✅ Working — 46M steps/hr |
| RL — tree navigation (GPU) | `stand_env_mjx.py` + tree-seeking | ✅ Working |
| Live viewer | `watch.py` | ✅ Working |
| Generational transfer | `--inherit` flag | ✅ Working |
| Motion reference | AMP / video tracking | ⬜ Next (see FUTURE.md) |
| Real hardware | ROS2 / MoveIt | ⬜ Later |

---

## The robot

- **Torso**: ~3 kg floating base (6 DOF free joint), 30×20×10 cm
- **4 limbs**: RoArm M3 Pro, each with 5 DOF + gripper = **24 actuated joints**
- **Actuators**: position-controlled, kp=8, torque limits from STS3215 servo datasheet
- **Ground**: flat plane (z=0), grey — simple and MJX-compatible
- **Collision**: arm meshes are visual only; capsule proxies on outer links (link3–gripper) handle inter-arm contacts at low GPU cost

```
        [fl_]  [fr_]
          |      |
        +---------+
        |  torso  |
        +---------+
          |      |
        [bl_]  [br_]
```

---

## File map

```
gb2.1/
│
├── README.md               ← you are here
├── DEVLOG.md               ← full change history with rationale
├── FUTURE.md               ← AMP, video tracking, MJX pipeline plans
├── REMOTE.md               ← remote GPU/CPU SSH workflow + sync setup
├── remote_sync.sh          ← bidirectional sync with H100 server
├── scene.xml               ← MJCF scene (flat ground, capsule tree for MJX compat)
├── lineage.py              ← records hardware params + inheritance chain per run
│
├── envs/                   ← Gymnasium environments + reward shaping
│   ├── ape_env.py          ← reach task: walk to tree 5m away
│   ├── stand_env.py        ← stand task: lift torso to 0.30m, hold stable
│   ├── stand_env_mjx.py    ← MJX version: pure JAX, GPU-parallel
│   └── hints.py            ← live reward shaping (hot-reloaded each episode)
│
├── train.py                ← CPU PPO via SB3 (--task, --resume, --inherit, --watch)
├── train_mjx.py            ← GPU PPO via JAX/MJX, 1024 envs, ~46M steps/hr
│
├── tools/
│   ├── sim.py              ← interactive viewer, physics only (no policy)
│   ├── watch.py            ← live viewer, hot-swaps checkpoints (--task reach|stand)
│   ├── watch_multi.py      ← split-screen viewer for reach task
│   ├── eval.py             ← evaluate a saved checkpoint
│   ├── test_joints.py      ← manual joint tester (gravity off, Ctrl+A)
│   ├── build_scene.py      ← converts ape.urdf → scene.xml
│   └── add_arm_capsules.py ← adds capsule collision proxies to scene.xml (run after build_scene.py)
│
└── roarm_ws/
    └── roarm_description/
        ├── ape.urdf            ← EDIT THIS to change the robot
        ├── urdf/roarm_m3/
        │   └── roarm_m3.xacro  ← original single-arm URDF (DO NOT EDIT)
        └── meshes/roarm_m3/    ← STL mesh files (DO NOT EDIT)
```

---

## Quick start

```bash
source .venv/bin/activate
```

### Watch the robot (no training)
```bash
python tools/sim.py        # physics only, interactive viewer
```

### Train locally — two paths

**GPU (MJX, fastest, ~46M steps/hr on RTX 4070)**
```bash
python train_mjx.py                  # 1024 parallel envs
python train_mjx.py --envs 2048      # more envs if VRAM allows
```
GPU has no built-in viewer — open one in a second terminal:
```bash
python tools/watch.py --task stand   # polls runs/ape_stand_mjx/best/
```

**Hints** (reward shaping in `envs/hints.py`) work on both paths:
- **CPU**: hot-reloaded every episode — edit while training runs.
- **MJX**: baked in at startup (JAX JIT can't hot-swap) — restart training to apply changes.

**CPU (SB3, slower but with live viewer + hints)**
```bash
python train.py --task stand --watch        # train + viewer in one command
python train.py --task stand --envs 8       # 8 parallel envs
python train.py --task stand --resume       # continue same run
python train.py --task stand --inherit      # new run, inherit weights from best
```
While running, edit `envs/hints.py` to steer rewards live — picked up at next episode.

### Generational transfer — use after changing hardware params
```bash
# 1. change kp, torque, etc. in scene.xml or stand_env.py
# 2. then:
python train.py --task stand --inherit --watch
# agent starts with all prior knowledge, adapts to new physics
```

### View training lineage
```bash
python train.py --lineage --task stand
```

### Evaluate a checkpoint
```bash
python tools/eval.py --task stand
python tools/eval.py --task stand --model runs/ape_stand/best/best_model.zip
```

---

## Training outputs

| Path | Contents |
|------|----------|
| `runs/ape_stand/` | SB3 checkpoints every 50k steps + TensorBoard logs |
| `runs/ape_stand/best/best_model.zip` | Best policy by eval reward |
| `runs/ape_stand/lineage.json` | Per-run hardware params + inheritance chain |
| `runs/ape_stand_mjx/` | MJX checkpoints (`.npz`) + exported `best_model.zip` |

```bash
tensorboard --logdir runs/    # visualise reward curves
```

## Recent Improvements

### Adaptive Physics
- **Adaptive stiffness**: Joint stiffness (kp) now varies based on position
- Higher stiffness near center positions for stability
- Lower stiffness at extremes for smoother motion
- Range: 4.8-9.6 (compared to fixed 8.0 previously)

### Training Optimizations
- **Learning rate scheduling**: Cosine decay from 3e-4 to 1e-5 over 50M steps
- **Layer normalization**: Added to all policy network layers for better stability
- **AdamW optimizer**: Weight decay for better regularization

### Curriculum Learning
- **Progressive difficulty**: Target height gradually increases from 0.25m to 0.40m
- **Duration**: 10M steps curriculum phase
- **Smoother learning**: Easier initial phase helps policy converge faster

### Tree Navigation (New!)
- **Tree-seeking rewards**: Distance-based rewards encourage approaching trees
- **Relative positioning**: Robot observes tree position relative to itself
- **Randomized targets**: Trees spawn at random positions (3-7m range)
- **Configurable weight**: Adjust tree-seeking importance via `hints.py`
- **Observation expansion**: Added 3D tree position to observation space (OBS_DIM: 59→62)

### Expected Benefits
- **Faster convergence**: Curriculum + adaptive physics should reduce training time
- **Better final performance**: Layer norm + LR scheduling improve policy quality
- **More natural motion**: Adaptive stiffness creates more biologically plausible movement

---

## Live reward shaping (hints)

While training runs (CPU path), edit `hints.py` to encourage specific behaviours.
Reloaded automatically at the start of each episode — no restart needed.

```python
# hints.py — set weight > 0 to activate
"spread_legs":      0.3   # wider base
"front_arms_down":  0.2   # keep front grippers planted
"symmetry":         0.1   # discourage leaning
"stillness":        0.0   # (off) penalise joint velocity
```

---

## Environment details

### Stand task

| Item | Value |
|------|-------|
| Observation | 59-dim: torso z/quat/lin-vel/ang-vel + 24 joint pos/vel |
| Action | 24-dim joint position targets, normalised to [-1, 1] |
| Reward | `height_reward + upright_reward + 0.05 alive − 5e-4 ctrl² − 1e-3 jvel² − 2e-2 Δaction²` |
| height_reward | Gaussian peak 1.0 at 0.30m (σ=0.08m) |
| upright_reward | 0.3 × quat_w² |
| Termination | torso < 3cm (fallen) |
| Timeout | 2000 steps |
| Steps/action | 12 physics steps — slower, more deliberate motion |

### Reach task

| Item | Value |
|------|-------|
| Observation | 63-dim: as above + tree direction + distance |
| Action | 24-dim joint targets |
| Reward | `progress×2 + 0.1 alive − 5e-4 ctrl² − 1e-3 jvel²` |
| Termination | Fallen or success (dist < 1.5m to tree) |
| Goal | Reach tree at (5, 0, 0) |

---

## Remote GPU training

For massive speedup, train on a remote H100 while watching live on the laptop.
Full setup in [REMOTE.md](REMOTE.md). Quick version:

```bash
./remote_sync.sh                      # local: bidirectional sync loop
python tools/watch.py --task stand    # local: watches synced checkpoints
# then edit envs/hints.py to steer the remote training in real time
```

---

## Hardware notes (Acer Predator Helios P16)

- **GPU**: RTX 4070 Laptop (8 GB VRAM) — used by MJX training
- **CPU**: Intel i9 (16 cores) — used by SB3 training
- MJX with 1024 envs: ~46M steps/hr, GPU-Util visible in `nvidia-smi`
- SB3 with 1 env: ~1M steps/hr, CPU-only, GPU idle

---

## Future plans

See [FUTURE.md](FUTURE.md) for full details:
- **Pipeline A**: manual keyframe reference motion (1–2 days)
- **Pipeline B**: MediaPipe video landmark tracking (3–5 days)
- **Pipeline D**: already done ✅ (MJX GPU physics)
- **Full AMP**: adversarial motion priors from ape video

---

## Git branches

| Branch | Purpose |
|--------|---------|
| `master` | baseline snapshot of original ROS2 package |
| `sim/mujoco-setup` | all simulation and RL work |
