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

URDF = pathlib.Path(__file__).parent / "roarm_ws/roarm_description/ape.urdf"

model = mujoco.MjModel.from_xml_path(str(URDF))
data = mujoco.MjData(model)

print(f"Loaded: {URDF.name}")
print(f"  bodies:  {model.nbody}")
print(f"  joints:  {model.njnt}")
print(f"  DOF:     {model.nv}")
print(f"  actuators: {model.nu}  (0 = none yet, add in MJCF for RL)")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
