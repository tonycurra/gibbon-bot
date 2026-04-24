# Robotic Ape Project

A 4-limb robotic ape built from **4x RoArm M3 Pro** arms, trained with reinforcement learning to walk, climb, and eventually brachiate — using real ape video as motion reference.

---

## Status

| Layer | Tool | Status |
|-------|------|--------|
| Physics sim | MuJoCo 3.x | ✅ Working |
| Robot description | URDF → MJCF | ✅ Working |
| RL — reach tree | `ape_env.py` + PPO | ✅ Working |
| RL — stand up (CPU) | `stand_env.py` + SB3 PPO | ✅ Training |
| RL — stand up (GPU) | `stand_env_mjx.py` + JAX PPO | ✅ Working — 46M steps/hr |
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
│   └── build_scene.py      ← converts ape.urdf → scene.xml
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

### Train on GPU — recommended
```bash
python train_mjx.py                  # 1024 parallel envs, ~46M steps/hr
python train_mjx.py --envs 2048      # more envs if VRAM allows
```

### Train on CPU (SB3)
```bash
python train.py --task stand --watch        # train + live viewer, 1 env
python train.py --task stand --envs 8       # 8 parallel envs
python train.py --task stand --resume       # continue same run
python train.py --task stand --inherit      # new run, inherit weights from best checkpoint
python train.py --task stand --inherit --watch --envs 8
```

### Watch training live (CPU path)
```bash
python tools/watch.py --task stand   # loads best checkpoint, hot-swaps on new saves
```

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
