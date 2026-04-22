# Dev Log — Robotic Ape Simulation

## Rules
- Never edit files under `roarm_ws/roarm_description/urdf/` or `meshes/` directly.
- All simulation-specific files live at the repo root or in `sim/`.
- Every meaningful change gets a git commit on a feature branch.

---

## 2026-04-22 — Initial MuJoCo setup (branch: `sim/mujoco-setup`)

### What was done
1. **Read** the existing ROS2 package (`roarm_description`) and `QUICKSTART.md`.
2. **Created `roarm_ws/roarm_description/ape.urdf`** — a standalone URDF (no ROS/xacro
   dependencies) assembling the full 4-limb ape robot:
   - Root link: `torso` (free-floating box, 30×20×10 cm, 1.0 kg)
   - 4 arms mounted with fixed joints at torso corners, prefixes `fl_` `fr_` `bl_` `br_`
   - All joint/link names from the original `roarm_m3.xacro` preserved verbatim, just prefixed
   - Mesh paths changed from `$(find roarm_description)/...` to relative `meshes/roarm_m3/...`
   - Motor limits filled in from STS3215 servo datasheet: **effort 2.94 Nm, velocity 6.28 rad/s**
     (was `effort="0" velocity="0"` in the original, which means no force simulation)
   - Gripper joints get half torque (1.47 Nm) since they're lighter-duty

3. **Created `view_robot.py`** — drop-in MuJoCo viewer. Run with `python view_robot.py`.

4. **Original files untouched**: `roarm_m3.xacro`, all STL meshes, launch files.

### Mount orientations (initial)
| Arm | xyz on torso | rpy |
|-----|-------------|-----|
| fl_ | 0.12  0.10 -0.05 | 0  0.785 0 |
| fr_ | 0.12 -0.10 -0.05 | 0  0.785 0 |
| bl_ |-0.12  0.10 -0.05 | 0 -0.785 0 |
| br_ |-0.12 -0.10 -0.05 | 0 -0.785 0 |

---

## 2026-04-22 — Fix spawn angle (commit `56cb352`)

**Problem**: torso spawned at z=0 (ground level); arms immediately collided with the floor,
producing a violent sideways shove — robot appeared at an angle on launch.

**Fix**: added `data.qpos[2] = 0.5` + `mj_forward()` in `view_robot.py` before the viewer
loop to lift the torso 0.5 m above ground at t=0.

---

## 2026-04-22 — Arm mount angle 45° → 90° (commit `ac41507`)

Changed `rpy` pitch on all 4 mount joints from `±0.785` rad (45°) to `±1.5708` rad (90°).
Arms now point straight down from the torso instead of angled forward/backward.

Joints changed: `torso_to_fl_base_link`, `torso_to_fr_base_link`, `torso_to_bl_base_link`,
`torso_to_br_base_link` in `ape.urdf`.

---

## 2026-04-22 — Joint damping added (commit `62e6b88`)

**Problem**: all joints had no `<dynamics>` element → zero damping in MuJoCo → frictionless
hinges that oscillate forever (energy never dissipates).

**Fix**: added `<dynamics damping="0.5"/>` to all 24 revolute joints via `replace_all` on the
`velocity="6.28"/>` pattern.

---

## 2026-04-22 — Floor, gravity, freejoint, build pipeline (commit `1e388fd`)

Three separate problems fixed together:

**1. No floor / robot falling forever**
MuJoCo has gravity by default but no ground plane unless you define one. Added a floor geom
(`type="plane"`) in `build_scene.py` post-processing.

**2. Torso fixed to world (not free-floating)**
MuJoCo's URDF loader welds the root link to the world by default. Confirmed by checking
`model.nq=24` (should be 31 for 6DOF base + 24 arm joints).
Fix: added `<link name="world"/>` + `<joint type="floating">` from world → torso in `ape.urdf`.
Result: `nq=31, nv=30` — torso is now free-floating.

**3. Damping raised 0.5 → 2.0**
0.5 was still too low; joints continued to oscillate visibly. Bumped to 2.0 Nm·s/rad.

**New file: `build_scene.py`**
One-shot converter: loads `ape.urdf` → saves MJCF → patches XML to add:
- floor geom with friction
- `meshdir` compiler attribute (fixes mesh path resolution when scene.xml is at repo root)
- `home` keyframe (torso z=0.5, qw=1, all arm joints=0)

`scene.xml` is the generated output. `view_robot.py` now loads `scene.xml` and resets to the
`home` keyframe instead of manually setting `data.qpos`.

**Workflow going forward**:
```
edit ape.urdf  →  python build_scene.py  →  python view_robot.py
```

---

## 2026-04-22 — Position actuators + checkerboard floor (commit `a404e0b`)

**Problem**: `model.nu=0` — no actuators, so the viewer's Ctrl+A panel was empty and joints
couldn't be driven manually.

**Fix**: `build_scene.py` now iterates all revolute joints and emits a `<position>` actuator
for each one:
- `kp=50` (proportional gain)
- `ctrlrange` from `jnt_range` (position limits)
- `forcerange` from `jnt_actfrcrange` (effort limits already stored from URDF `<limit effort>`)
- Result: `nu=24`, one actuator per joint, correct torque limits wired in

**Also**: replaced the grey plane with a black/white checkerboard texture for easier visual
reference. `scene.xml` gets a `<texture builtin="checker">` + `<material>` + floor geom with
`material="floor_mat"`.

---

## 2026-04-22 — Agricultural terrain + test_joints.py (commit `0dcff40`)

**New file: `terrain.py`**
Generates a 64×64 heightfield that looks like plowed farmland:
- `hills`: two overlapping low-frequency sine waves
- `furrows`: medium-frequency ridges along one axis (`0.5 * sin(X * 2.5)`)
- `noise`: small gaussian noise for surface texture
- Values normalized to [0, 1]; MuJoCo scales by `max_height=0.5 m`

`terrain.apply(model)` writes the array into `model.hfield_data` at runtime.
Must be called after `from_xml_path` and before `MjData` is created.

**`build_scene.py` changes**:
- Removed flat plane, added `<hfield name="terrain">` asset (64×64, 10×10 m, max 0.5 m)
- Added grass checker texture: two shades of green (`rgb1=0.18 0.42 0.09`, `rgb2=0.28 0.55 0.14`)
- Home keyframe z raised `0.5 → 1.0 m` (must clear 0.5 m max terrain height)

**New file: `test_joints.py`**
Dedicated manual joint tester:
- Gravity off (`model.opt.gravity[:] = 0`)
- `kp` overridden to 200 for crisp slider response
- Torso velocity zeroed each step (pins base in place)
- Prints all 24 actuator names + ranges on launch
- Usage: `python test_joints.py` → Ctrl+A in viewer → drag sliders

---

## 2026-04-22 — Friction fix: robot sliding on terrain (commit `ba8d127`)

**Problem**: robot was visibly sliding on the grass terrain even after landing.

**Root cause**: MuJoCo computes effective contact friction as `min(geom_A, geom_B)`.
The terrain had `friction=1.2` but robot geoms inherited MuJoCo's default of `1.0`,
so effective friction was `1.0`. At oblique contact angles (arms landing at ~90° from
torso, gripper tips hitting sloped terrain), lateral forces exceed what 1.0 can resist.

**Fix in `build_scene.py`**:
- Added `<default class="robot"><geom friction="2.0 0.01 0.001" condim="4"/></default>`
- Tagged all non-terrain geoms with `class="robot"` in post-processing loop
- Terrain friction also raised to `2.0 0.01 0.001`
- `condim` raised `3 → 4` on both sides: condim=4 adds spinning friction (resists torque
  at contact point), condim=3 only handles sliding

**Note**: some residual sliding remains — this is expected without an active stance
controller. The robot has no policy telling feet to stay planted. Friction fix reduces
passive drift; active control is needed to fully eliminate it (RL phase).

---

## 2026-04-22 — Static tree obstacle (commit after ba8d127)

Added a simple tree 5 m from the robot spawn at world pos (5, 0, 0).
Implemented as a static body (no joint = welded to world) with two geoms:
- `trunk`: brown cylinder, radius 0.12 m, 2 m tall (half-height 1.0), centre at z=1.0
- `canopy`: dark green sphere, radius 1.3 m, centre at z=3.0

Added in `build_scene.py` via `ET.SubElement(worldbody, "body", ...)` after the terrain geom.
The tree is a solid collision object — the robot will bump into it.
