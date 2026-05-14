# `data/scene.xml` — the canonical robot scene

This is the **only** active MuJoCo scene for the gibbon robot. Loaded by
`look.py`, `view.py`, `train_mjx.py`, and every env under `envs/`.
Older variants live in `data/scenes_archive/` for reference.

Two sibling docs cover the dimensions and the real-robot motor specs:

- [`SCENE_GEOMETRY.md`](SCENE_GEOMETRY.md) — full dimensional reference (mesh sizes, inner-face-gap formula, mount positions).
- [`motors.md`](motors.md) — ST32xx servo datasheets, per-joint assignment.
- [`controls.md`](controls.md) — real-robot web UI / sim-to-real differences.

---

## Units and conventions

| | |
|---|---|
| Length | metres (SI) |
| Angles | **radians** (`<compiler angle="radian"/>`) |
| Mass | kilograms |
| Z up | floor at z = 0, torso spawns at z = 0.5310 |
| Frame convention | `+x` = robot front, `+y` = robot left, `+z` = up |
| Mesh scale | STLs are in mm, scaled `0.001` |

**No `ref` on joints.** Pose offsets are baked into body `quat`s. This keeps
qpos = 0 as the canonical rest pose — important for training reward shaping
and for matching the real robot's zero-encoder pose.

---

## Body layout

```
                  +x (front)
                       |
       FL arm  ●---torso_arms (blue)---●  FR arm
               |                       |
               |    torso_legs (brown) |
               |                       |
       BL leg  ●-----------------------●  BR leg
                       |
                  -x (back)
```

### Torso (two-slab box, single body)

| Geom | Pos (x y z) | Half-size (sx sy sz) | Colour | Z extent |
|---|---|---|---|---|
| `torso_legs` | `-0.02485 0 -0.04325` | `0.10405 0.1408 0.04925` | brown `rgba 0.40 0.25 0.15 1` | flush with full base_link Z |
| `torso_arms` | `+0.12   0 -0.04325` | `0.0408 0.10 0.04925`   | blue  `rgba 0.20 0.45 0.65 1` | flush with full base_link Z |

Total torso slab height = **98.5 mm** (matches the base_link mounting-face Z
extent exactly — every face is flush to < 0.0001 m). See
`scenes_archive/scene_torso_30mm.xml` for the pre-height-bump version.

### Limb mount positions (base_link mesh centres)

| Limb | x | y | z | Mount quat |
|---|---:|---:|---:|---|
| FL (front-left arm)  | +0.12 | +0.1711 | -0.05 | `0.5 -0.5 0.5 0.5` |
| FR (front-right arm) | +0.12 | -0.1711 | -0.05 | `0.5  0.5 0.5 -0.5` |
| BL (back-left leg)   | -0.20 | +0.10   | -0.05 | `0 -0.7071068 0 0.7071068` |
| BR (back-right leg)  | -0.20 | -0.10   | -0.05 | `0 -0.7071068 0 0.7071068` |

The back-leg `link1` body quats include a 180° hip-yaw pre-rotation so the
brackets face the same way as the arms when qpos = 0. The leg mount-mesh
geoms in the torso body also carry an inverted quat
(`0 0.7071068 0 -0.7071068`) so the visible bracket orientation matches.

### Arm/leg orientation (rest pose, qpos = 0)

- **Arms** extend forward and to the side from the FL/FR mounts.
  link1 (yaw shaft) is vertical out of the torso top; link2 (upper arm)
  reaches forward-out; link3 (forearm) hangs down; link5 (wrist) + gripper
  point downward.
- **Legs** are mirrored copies of the arm chain rotated 180° around the
  hip-yaw axis, so qpos = 0 puts the leg in the same downward-pointing
  rest pose as the arm — only the mount position and yaw bake differ.

This symmetry is intentional: all four limbs share the same URDF chain, the
same `link1…link5+gripper` meshes, and the same motor assignment. The robot
is a 4-arm gibbon, not an arm+leg quadruped.

---

## Joints and motors

24 actuated DoF total = 6 per limb × 4 limbs. Each limb has the same chain:

| # | Joint name (per limb) | Axis | Range (rad) | Motor class | Stall torque | Damping |
|---|---|---|---|---|---|---|
| 1 | `*_shoulder_yaw` / `*_hip_yaw`     | Z | `[-π, +π]`           | ST3235     | 2.94 N·m | 0.62 |
| 2 | `*_shoulder_pitch` / `*_hip_pitch` | Z | `[-π/2, +π/2]`       | ST3235     | 2.94 N·m | 0.62 |
| 3 | `*_elbow` / `*_knee`               | Z | `[-1.00, +2.95]`     | ST3235     | 2.94 N·m | 0.62 |
| 4 | `*_forearm_roll` / `*_shin_roll`   | Z | `[-π/2, +π/2]`       | ST3235     | 2.94 N·m | 0.62 |
| 5 | `*_wrist` / `*_ankle`              | Z | `[-π, +π]`           | ST3215     | 2.94 N·m | 0.62 |
| 6 | `*_gripper` / `*_foot`             | Z | `[0, 1.5]`           | ST3215-HS  | 1.96 N·m | 0.18 |

Each `<default class="...">` block fixes the damping; per-joint
`actuatorfrcrange` clamps the applied torque. The matching `<position>`
actuators all use `kp=8` and `forcelimited="true"` with `forcerange` set
to the motor's stall torque. Naming uses arm-style for the front limbs
(`shoulder_*`, `elbow`, `wrist`, `gripper`) and leg-style for the back
limbs (`hip_*`, `knee`, `ankle`, `foot`) but the kinematics are identical.

Prefix convention: `fl_`, `fr_`, `bl_`, `br_` for front-left/right and
back-left/right respectively.

### Why `kp=8` and these damping values

`damping = τ_stall / ω_no_load` (motor characteristic). With `kp=8` and
`forcelimited=true`, position commands clamp to the real servo's stall
torque, giving a sim that respects the same saturation curve as the
hardware. See `motors.md` for the derivation.

---

## Free joint and spawn

The torso carries a single `world_to_torso` free joint at the top of the
kinematic tree. Default spawn:

- `body name="torso" pos="0 0 0.5310"` → torso slab at z ≈ 0.481 to 0.580.
- Floor is a plane at z = 0, `condim=3`, size 5 × 5 m.
- One directional light at `(0, 0, 3)` pointing down, `castshadow=false`.

No skybox, fog, haze, headlight, or tree assets — keep the scene minimal so
MJX cold-compiles fast and view.py launches without OpenGL surprises.

---

## What is **not** in this file

- No trees / grove geometry → see the `grove_scene` branch and
  `docs/GROVE_MASTERPLAN.md`.
- No tendons, sites, or sensors.
- No contact-pair exclusions (default MuJoCo self-collision).
- No keyframes — every env builds its initial qpos in code.

If you need any of the above, add it inside the env XML loader rather than
mutating `scene.xml` directly.
