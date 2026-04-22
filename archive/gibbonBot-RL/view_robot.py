"""
Launch the ape robot in the MuJoCo passive viewer.

Run from the project root:
    python view_robot.py

Controls (MuJoCo viewer):
    Left-drag    — rotate camera
    Right-drag   — pan
    Scroll       — zoom
    Ctrl+A       — toggle actuator panel
    Space        — pause/resume
    Backspace    — reset simulation
"""

import pathlib
import mujoco
import mujoco.viewer

SCENE = pathlib.Path(__file__).parent / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # "home": torso at z=0.5, arms at 0

print(f"Loaded: {SCENE.name}")
print(f"  bodies:  {model.nbody}")
print(f"  joints:  {model.njnt}  (incl. freejoint)")
print(f"  DOF:     {model.nv}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
