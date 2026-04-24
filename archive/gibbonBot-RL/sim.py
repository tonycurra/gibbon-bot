"""
Interactive simulation viewer.

Opens the MuJoCo scene and runs physics in real time.
Use the built-in viewer controls to orbit, zoom, pause, and reset.

Usage:
    python sim.py
"""

import mujoco
import mujoco.viewer
import pathlib

SCENE = pathlib.Path(__file__).parent / "scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # reset to keyframe 0 (default standing pose)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    print("Simulation running — close the window to exit.")
    print("  Space  : pause / resume")
    print("  Ctrl+R : reset")
    print("  Mouse  : orbit / pan / zoom")

    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
