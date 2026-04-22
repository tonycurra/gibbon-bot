"""
Convert ape.urdf → scene.xml (native MJCF).

Adds:
  - checkerboard floor with friction
  - position actuators for all 24 arm joints (torque limits from URDF)
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

# ── mesh path fix ────────────────────────────────────────────────────────────
# scene.xml lives at repo root; STLs are in roarm_ws/roarm_description/
compiler = mjcf.find("compiler")
if compiler is None:
    compiler = ET.SubElement(mjcf, "compiler")
compiler.set("meshdir", str(ROOT / "roarm_ws/roarm_description"))

# ── checkerboard floor ───────────────────────────────────────────────────────
asset = mjcf.find("asset")
if asset is None:
    asset = ET.SubElement(mjcf, "asset")
ET.SubElement(asset, "texture", {
    "name":    "checker",
    "type":    "2d",
    "builtin": "checker",
    "rgb1":    "0.15 0.15 0.15",
    "rgb2":    "0.9 0.9 0.9",
    "width":   "512",
    "height":  "512",
    "mark":    "edge",
    "markrgb": "0.1 0.1 0.1",
})
ET.SubElement(asset, "material", {
    "name":        "floor_mat",
    "texture":     "checker",
    "texrepeat":   "4 4",
    "reflectance": "0.1",
})

worldbody = mjcf.find("worldbody")
floor = ET.Element("geom", {
    "name":     "floor",
    "type":     "plane",
    "size":     "5 5 0.05",
    "material": "floor_mat",
    "condim":   "3",
    "friction": "1 0.005 0.0001",
})
worldbody.insert(0, floor)

# ── position actuators (one per revolute joint) ──────────────────────────────
# Torque limits come straight from the URDF <limit effort="..."> via jnt_actfrcrange.
# kp=50: stiff enough to hold a pose, soft enough not to explode.
# In the viewer: Ctrl+A opens the actuator slider panel to drive joints manually.
actuator = mjcf.find("actuator")
if actuator is None:
    actuator = ET.SubElement(mjcf, "actuator")

for i in range(model.njnt):
    jtype = int(model.jnt_type[i])
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
        continue
    name  = model.joint(i).name
    lo, hi   = model.jnt_range[i]
    flo, fhi = model.jnt_actfrcrange[i]
    ET.SubElement(actuator, "position", {
        "name":          f"act_{name}",
        "joint":         name,
        "kp":            "50",
        "ctrlrange":     f"{lo:.6g} {hi:.6g}",
        "forcelimited":  "true",
        "forcerange":    f"{flo:.6g} {fhi:.6g}",
    })

# ── home keyframe ────────────────────────────────────────────────────────────
# model.nq = 31 total: 7 (freejoint: x y z qw qx qy qz) + 24 arm joints
qpos = [0.0] * model.nq
qpos[2] = 0.5   # z
qpos[3] = 1.0   # qw
key_str = " ".join(f"{v:.6g}" for v in qpos)
keyframe = mjcf.find("keyframe")
if keyframe is None:
    keyframe = ET.SubElement(mjcf, "keyframe")
ET.SubElement(keyframe, "key", {"name": "home", "qpos": key_str})

tree.write(str(OUT), xml_declaration=True, encoding="unicode")
print(f"Written: {OUT}")
print(f"  nq={model.nq}  nv={model.nv}  nbody={model.nbody}  nu={model.nu}")
