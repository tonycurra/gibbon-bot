"""
Convert ape.urdf → scene.xml (native MJCF).

Adds:
  - floor plane with friction
  - "home" keyframe: torso at z=0.5, all joints at 0

Run once whenever ape.urdf changes:
    python build_scene.py
"""

import pathlib
import xml.etree.ElementTree as ET
import mujoco

ROOT = pathlib.Path(__file__).parent
URDF = ROOT / "roarm_ws/roarm_description/ape.urdf"
OUT  = ROOT / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(URDF))
mujoco.mj_saveLastXML(str(OUT), model)

tree = ET.parse(OUT)
mjcf = tree.getroot()

# fix mesh paths: scene.xml lives at repo root, STLs are in roarm_ws/roarm_description/
compiler = mjcf.find("compiler")
if compiler is None:
    compiler = ET.SubElement(mjcf, "compiler")
compiler.set("meshdir", str(ROOT / "roarm_ws/roarm_description"))

# floor plane
worldbody = mjcf.find("worldbody")
floor = ET.Element("geom", {
    "name":     "floor",
    "type":     "plane",
    "size":     "5 5 0.05",
    "rgba":     ".8 .8 .8 1",
    "condim":   "3",
    "friction": "1 0.005 0.0001",
})
worldbody.insert(0, floor)

# home keyframe: model.nq=31 total (7 freejoint + 24 arm)
# freejoint layout: x y z qw qx qy qz  (indices 0-6)
qpos = [0.0] * model.nq
qpos[2] = 0.5   # z
qpos[3] = 1.0   # qw (unit quaternion)
key_str = " ".join(f"{v:.6g}" for v in qpos)
keyframe = mjcf.find("keyframe")
if keyframe is None:
    keyframe = ET.SubElement(mjcf, "keyframe")
ET.SubElement(keyframe, "key", {"name": "home", "qpos": key_str})

tree.write(str(OUT), xml_declaration=True, encoding="unicode")
print(f"Written: {OUT}")
print(f"  nq={model.nq}  nv={model.nv}  nbody={model.nbody}")
