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
