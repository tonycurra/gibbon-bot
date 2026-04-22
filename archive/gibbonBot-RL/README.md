# Robotic Ape Project

A 4-limb robotic ape built from **4x RoArm M3 Pro** arms (used as both arms and legs), trained with reinforcement learning to walk, climb, and eventually brachiate — using real ape video as motion reference.

---

## What this repo is

| Layer | Tool | Status |
|-------|------|--------|
| Physics sim | MuJoCo 3.x | ✅ Working |
| Robot description | URDF → MJCF | ✅ Working |
| Viewer | `view_robot.py` | ✅ Working |
| RL training | MJX + JAX + PPO | ⬜ Next |
| Motion reference | MediaPipe / ViTPose | ⬜ Later |
| Real hardware | ROS2 / MoveIt | ⬜ Later |

---

## The robot

- **Torso**: floating base (6 DOF free joint)
- **4 limbs**: RoArm M3 Pro, each with 5 DOF + gripper = **24 actuated joints**
- **Total DOF**: 30 (6 base + 24 joints)
- **Foot contact**: gripper tip of each arm

### Arm layout

```
        [fl_]  [fr_]
          |      |
        +---------+
        |  torso  |
        +---------+
          |      |
        [bl_]  [br_]
```

Prefixes: `fl_` front-left · `fr_` front-right · `bl_` back-left · `br_` back-right

### Joint limits (per arm)

| Joint | Range | Effort | Velocity |
|-------|-------|--------|----------|
| Base yaw (`base_link_to_link1`) | ±180° | 2.94 Nm | 6.28 rad/s |
| Shoulder (`link1_to_link2`) | ±90° | 2.94 Nm | 6.28 rad/s |
| Elbow (`link2_to_link3`) | -57° to +169° | 2.94 Nm | 6.28 rad/s |
| Wrist pitch (`link3_to_link4`) | ±90° | 2.94 Nm | 6.28 rad/s |
| Wrist roll (`link4_to_link5`) | ±180° | 2.94 Nm | 6.28 rad/s |
| Gripper (`link5_to_gripper_link`) | 0° to 86° | 1.47 Nm | 6.28 rad/s |

Motor specs from STS3215 servo datasheet (12V). Joint damping: 2.0 Nm·s/rad.

---

## File map

```
gb2.1/
│
├── README.md               ← you are here
├── DEVLOG.md               ← full change history with rationale
├── QUICKSTART.md           ← original project roadmap (read this)
│
├── ape.urdf  (→ in roarm_ws/roarm_description/)
├── build_scene.py          ← converts ape.urdf → scene.xml (run after editing URDF)
├── scene.xml               ← generated MJCF: floor + freejoint + keyframe (do not edit by hand)
├── view_robot.py           ← launches MuJoCo viewer
│
└── roarm_ws/
    └── roarm_description/
        ├── ape.urdf            ← EDIT THIS to change the robot
        ├── urdf/roarm_m3/
        │   └── roarm_m3.xacro  ← original single-arm description (DO NOT EDIT)
        └── meshes/roarm_m3/    ← STL files for all links (DO NOT EDIT)
            base_link.stl · link1-5.stl · gripper_link.stl · ...
```

---

## Quick start

```bash
# 1. activate the venv
source .venv/bin/activate

# 2. (first time only) install mujoco
pip install mujoco

# 3. view the robot
python view_robot.py
```

### After editing ape.urdf

```bash
python build_scene.py   # regenerates scene.xml
python view_robot.py
```

### Viewer controls

| Key / action | Effect |
|---|---|
| Left-drag | rotate camera |
| Right-drag | pan |
| Scroll | zoom |
| Space | pause / resume |
| Backspace | reset to home pose |

---

## How the pipeline works

```
ape.urdf  ──(build_scene.py)──►  scene.xml  ──(view_robot.py)──►  MuJoCo viewer
```

1. **`ape.urdf`** — standalone URDF, no ROS/xacro. Defines the torso + 4 arms with a `floating` joint at the root so MuJoCo gives the torso 6 DOF.
2. **`build_scene.py`** — loads the URDF, dumps it as MJCF via `mj_saveLastXML`, then post-processes the XML to add: floor plane, `meshdir` compiler path, and a `home` keyframe (torso at z=0.5 m, all joints at 0).
3. **`scene.xml`** — the generated MJCF ready for simulation. Load this in MuJoCo, not the URDF directly.
4. **`view_robot.py`** — passive viewer: loads `scene.xml`, resets to the `home` keyframe, steps physics in a loop.

---

## Git branches

| Branch | Purpose |
|--------|---------|
| `master` | baseline snapshot of the original ROS2 package — nothing changed here |
| `sim/mujoco-setup` | all simulation work so far |

---

## Known gaps / roadmap

- [ ] **Tune mount angles** — arms currently point straight down (90°). Adjust `rpy` in the 4 `torso_to_*_base_link` joints in `ape.urdf` until stance looks right.
- [ ] **Actuators** — no actuators in `scene.xml` yet. Add `<actuator>` block to MJCF for RL.
- [ ] **RL environment** — Gymnasium wrapper + PPO (see QUICKSTART.md step 5).
- [ ] **Real motor specs** — STS3215 values are estimates; get exact torque curve from hardware docs.
- [ ] **AMP** — motion priors from ape video (QUICKSTART step 7).
