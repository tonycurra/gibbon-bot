# Robotic Ape Project

A 4-limb robotic ape built from **4x RoArm M3 Pro** arms, trained with reinforcement learning to walk, climb, and eventually brachiate — using real ape video as motion reference.

---

## Status

| Layer | Tool | Status |
|-------|------|--------|
| Physics sim | MuJoCo 3.x | ✅ Working |
| Robot description | URDF → MJCF | ✅ Working |
| RL — reach tree | `ape_env.py` + PPO | ✅ Working |
| RL — stand up | `stand_env.py` + PPO | ✅ Training |
| Live viewer | `watch.py` | ✅ Working |
| Motion reference | AMP / ViTPose | ⬜ Next |
| Real hardware | ROS2 / MoveIt | ⬜ Later |

---

## The robot

- **Torso**: 2 kg floating base (6 DOF free joint), 30×20×10 cm
- **4 limbs**: RoArm M3 Pro, each with 5 DOF + gripper = **24 actuated joints**
- **Total mass**: ~3.8 kg (2 kg torso + ~0.45 kg × 4 arms)
- **Actuators**: position-controlled, kp=20, torque limits from STS3215 servo datasheet

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
├── PLAN.md                 ← next steps and ideas
├── QUICKSTART.md           ← original project roadmap
│
│── Training & eval
├── train.py                ← PPO training (--task reach|stand), saves to runs/<task>/
├── eval.py                 ← evaluate a checkpoint (--task reach|stand)
├── watch.py                ← live viewer, hot-swaps checkpoints (--task reach|stand)
├── watch_multi.py          ← split-screen viewer (N envs in a grid) — reach task only
│
│── Environments
├── ape_env.py              ← reach task: walk to tree 5m away
├── stand_env.py            ← stand task: lift torso to 0.30m, hold stable
├── terrain.py              ← generates 64×64 agricultural heightfield at runtime
│
│── Simulation core
├── build_scene.py          ← converts ape.urdf → scene.xml (run after any URDF edit)
├── scene.xml               ← generated MJCF — do NOT edit by hand
│
│── Dev tools
├── view_robot.py           ← MuJoCo viewer (full physics, grass terrain)
├── test_joints.py          ← manual joint tester (gravity off, torso pinned, Ctrl+A)
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

# View the robot on terrain
python view_robot.py

# Train — reach task (default) or stand task
python train.py                   # reach: walk to tree
python train.py --task stand      # stand: lift and hold height
python train.py --task stand --resume   # continue from last checkpoint

# Watch training live
python watch.py                   # reach task
python watch.py --task stand      # stand task

# Split-screen multi-env viewer (reach task)
python watch_multi.py             # 2×2 grid
python watch_multi.py --n 8       # 2×4 grid

# Evaluate a checkpoint
python eval.py
python eval.py --task stand
python eval.py --task stand --model runs/ape_stand/ape_stand_final.zip
```

### Two-terminal workflow

```
Terminal 1:  python train.py --task stand
Terminal 2:  python watch.py --task stand
```

The viewer auto-loads new checkpoints as training saves them.

### After editing ape.urdf

```bash
python build_scene.py   # regenerates scene.xml
python view_robot.py    # verify visually
```

---

## Environment details

### Reach task (`ape_env.py`)

| Item | Value |
|------|-------|
| Observation | 63-dim: torso z/quat/vel + 24 joint pos/vel + tree direction + distance |
| Action | 24-dim joint position targets, normalised to [-1, 1] |
| Reward | `progress×2 + 0.1 alive − 5e-4×ctrl² − 1e-3×jvel²` |
| Termination | Fallen (torso < 0.15m above terrain) or success (dist < 1.5m to tree) |
| Timeout | 2000 steps |
| Goal | Reach the tree at (5, 0, 0) |

### Stand task (`stand_env.py`)

| Item | Value |
|------|-------|
| Observation | 59-dim: torso z/quat/vel + 24 joint pos/vel |
| Action | 24-dim joint position targets, normalised to [-1, 1] |
| Reward | `height_reward + upright_reward + 0.05 alive − 5e-4×ctrl² − 1e-3×jvel²` |
| height_reward | Gaussian peak 1.0 at 0.30m above terrain (σ=0.08m) |
| upright_reward | 0.3 × quat_w² — penalises torso tilt |
| Termination | Fallen (torso < 3cm above terrain surface) |
| Timeout | 2000 steps |
| Goal | Lift torso to 30cm and hold it stably |

---

## Git branches

| Branch | Purpose |
|--------|---------|
| `master` | baseline snapshot of original ROS2 package — nothing changed |
| `sim/mujoco-setup` | all simulation and RL work |
