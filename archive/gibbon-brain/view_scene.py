#!/usr/bin/env python3
"""Simple viewer for gibbonBot scene."""

import mujoco
import mujoco.viewer

# Load the scene
model = mujoco.MjModel.from_xml_path(
    "gibbonBot-RL-brain-modeling_data/gibbonbot-rl/scene.xml"
)
data = mujoco.MjData(model)

print("MuJoCo Viewer - GibbonBot Scene")
print("Controls:")
print("  Left-drag: Rotate camera")
print("  Right-drag: Pan")
print("  Scroll: Zoom")
print("  ESC: Exit")

mujoco.viewer.launch(model, data)
