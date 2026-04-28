"""Viewer for data/scene.xml.

Run:  python view_scene.py

Controls:
  Space           pause / resume physics
  Backspace       reset to home keyframe
  Tab             toggle side panels (joint sliders)
  H               help overlay (all shortcuts)
  Ctrl+Right-drag drag a body

Slider ranges (viewer shows radians):
  shoulder_yaw / hip_yaw    -3.14  to  3.14  =  -180° to  180°
  shoulder_pitch / hip_pitch -1.57  to  1.57  =   -90° to   90°
  elbow / knee               -1.00  to  2.95  =   -57° to  169°
  forearm_roll / shin_roll   -1.57  to  1.57  =   -90° to   90°
  wrist / ankle              -3.14  to  3.14  =  -180° to  180°
  gripper / foot              0.00  to  1.50  =     0° to   86°
"""
from pathlib import Path
import mujoco.viewer

SCENE = Path(__file__).parent / "data" / "scene.xml"

if __name__ == "__main__":
    mujoco.viewer.launch_from_path(str(SCENE))
