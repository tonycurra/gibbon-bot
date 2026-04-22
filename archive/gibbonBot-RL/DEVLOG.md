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

### Mount orientations
| Arm | xyz on torso | rpy |
|-----|-------------|-----|
| fl_ | 0.12  0.10 -0.05 | 0  0.785 0 |
| fr_ | 0.12 -0.10 -0.05 | 0  0.785 0 |
| bl_ |-0.12  0.10 -0.05 | 0 -0.785 0 |
| br_ |-0.12 -0.10 -0.05 | 0 -0.785 0 |

These are from QUICKSTART.md step 3. Adjust after viewing in step 4.

### Known gaps / next steps
- [ ] Run `python view_robot.py` and check that the stance looks right. Tweak mount angles if
      arms clip through the torso or the robot falls badly.
- [ ] Convert `ape.urdf` → `ape.xml` (native MJCF) for actuator/contact/sensor control.
      Use: `mujoco.mj_saveLastXML("ape.xml", model)` after loading the URDF.
- [ ] Add actuators to MJCF (position or torque) for RL training.
- [ ] Set up Gymnasium environment + PPO (Step 5 in QUICKSTART).

### How to check if mujoco is installed
```bash
python -c "import mujoco; print(mujoco.__version__)"
```
If not: `pip install mujoco`
