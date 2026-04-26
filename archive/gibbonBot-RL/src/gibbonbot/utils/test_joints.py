"""
Manual joint tester.

Loads scene.xml with:
  - gravity OFF  (arms don't sag)
  - torso pinned (base stays at home position)
  - kp boosted   (sliders feel responsive)

How to use:
  1. python test_joints.py
  2. Press Ctrl+A in the viewer to open the actuator panel
  3. Drag any slider to move that joint
  4. Backspace resets all joints to zero

Joint naming:
  act_fl_*  front-left arm
  act_fr_*  front-right arm
  act_bl_*  back-left arm
  act_br_*  back-right arm

  *_base_link_to_link1   = shoulder yaw    (±180°)
  *_link1_to_link2       = shoulder pitch  (±90°)
  *_link2_to_link3       = elbow           (-57° to +169°)
  *_link3_to_link4       = wrist pitch     (±90°)
  *_link4_to_link5       = wrist roll      (±180°)
  *_link5_to_gripper_link = gripper        (0° to 86°)
"""

import pathlib
import mujoco
import mujoco.viewer

SCENE = pathlib.Path(__file__).parent / "scene.xml"
KP_TEST = 200   # stiffer than default 50 — sliders feel immediate

model = mujoco.MjModel.from_xml_path(str(SCENE))
data  = mujoco.MjData(model)

# gravity off so arms hold position without fighting
model.opt.gravity[:] = 0

# boost kp so slider movements are crisp
model.actuator_gainprm[:, 0] = KP_TEST

mujoco.mj_resetDataKeyframe(model, data, 0)

print(f"Loaded {model.nu} actuators — Ctrl+A for slider panel\n")
for i in range(model.nu):
    lo, hi = model.actuator_ctrlrange[i]
    print(f"  {i:2d}  {model.actuator(i).name:<40s}  [{lo:.2f}, {hi:.2f}] rad")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # pin torso: zero its velocity each step so it stays put
        data.qvel[:6] = 0
        mujoco.mj_step(model, data)
        viewer.sync()
