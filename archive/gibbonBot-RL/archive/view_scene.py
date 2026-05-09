"""Interactive MuJoCo viewer.

Run:
    python view_scene.py                          # opens data/scene.xml (robot)
    python view_scene.py data/physics_test.xml    # opens any other MJCF
    python view_scene.py path/to/your.xml         # ditto, any path

Controls:
    Space             pause / resume physics
    Backspace         reset to keyframe (if any)
    Tab               toggle side panels (joint sliders, sim settings)
    H                 help overlay with all shortcuts
    Ctrl+Right-drag   drag a body
"""
import sys
from pathlib import Path
import mujoco.viewer

DEFAULT = Path(__file__).parent / "data" / "scene.xml"

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT)
    mujoco.viewer.launch_from_path(path)
