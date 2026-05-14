#!/usr/bin/env python3
"""Inspect the robot in the canonical scene. Loads data/scene.xml
(4-limb architecture, floor + one no-shadow light, no fog/haze/skybox/tree)
and settles under gravity with zero actuator command. No policy, no
checkpoint, no training — eyeball-only.

Usage:
    python look.py
"""
import pathlib
import time

import mujoco
import mujoco.viewer

SCENE = pathlib.Path("data/scene.xml")
RT_HZ = 60

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

dt = model.opt.timestep
steps_per_frame = max(1, int((1.0 / RT_HZ) / dt))
print(
    f"[look] {SCENE}  bodies={model.nbody}  joints={model.njnt}  "
    f"dt={dt*1000:.2f}ms  steps/frame={steps_per_frame}"
)

viewer = mujoco.viewer.launch_passive(model, data)
try:
    while viewer.is_running():
        t0 = time.time()
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, 1.0 / RT_HZ - (time.time() - t0)))
finally:
    viewer.close()
