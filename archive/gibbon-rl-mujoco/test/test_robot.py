import mujoco
import mujoco.viewer
import time

# Load the robot model
model = mujoco.MjModel.from_xml_path("test/robot.xml")
data = mujoco.MjData(model)

# Open a visualizer window
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
