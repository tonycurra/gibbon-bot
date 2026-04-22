"""
Convert ape.urdf → scene.xml (native MJCF).

Adds:
  - agricultural grass terrain (heightfield, filled at runtime by terrain.py)
  - position actuators for all 24 arm joints (torque limits from URDF)
  - "home" keyframe: torso at z=1.0 (above max terrain height), all joints at 0

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

# ── contact defaults ─────────────────────────────────────────────────────────
# MuJoCo default friction is 1.0 sliding / 0.005 torsional / 0.0001 rolling.
# Effective friction at contact = min(geom_a, geom_b), so robot geoms need to
# be at least as high as terrain to avoid the terrain value being dragged down.
# condim=4 enables the full friction pyramid (sliding + spinning) vs basic condim=3.
default = mjcf.find("default")
if default is None:
    default = ET.SubElement(mjcf, "default")
robot_default = ET.SubElement(default, "default", {"class": "robot"})
ET.SubElement(robot_default, "geom", {
    "friction": "2.0 0.01 0.001",
    "condim":   "4",
})

# tag every robot geom with class="robot" so the default applies
worldbody = mjcf.find("worldbody")
for geom in worldbody.iter("geom"):
    if geom.get("name") != "terrain":
        geom.set("class", "robot")

# ── grass terrain (heightfield) ──────────────────────────────────────────────
# Heightfield dimensions: 64×64 grid, 10×10 m, max height 0.5 m, base 0.05 m.
# The actual height data is written at runtime by terrain.py (model.hfield_data).
from terrain import NROW, NCOL
asset = mjcf.find("asset")
if asset is None:
    asset = ET.SubElement(mjcf, "asset")

ET.SubElement(asset, "hfield", {
    "name":  "terrain",
    "nrow":  str(NROW),
    "ncol":  str(NCOL),
    "size":  "5 5 0.5 0.05",      # half-x, half-y, max-height, base-thickness
})
ET.SubElement(asset, "texture", {
    "name":    "grass",
    "type":    "2d",
    "builtin": "checker",
    "rgb1":    "0.18 0.42 0.09",   # dark grass
    "rgb2":    "0.28 0.55 0.14",   # light grass
    "width":   "512",
    "height":  "512",
})
ET.SubElement(asset, "material", {
    "name":        "grass_mat",
    "texture":     "grass",
    "texrepeat":   "6 6",
    "reflectance": "0.05",
})

worldbody = mjcf.find("worldbody")
terrain_geom = ET.Element("geom", {
    "name":     "terrain",
    "type":     "hfield",
    "hfield":   "terrain",
    "material": "grass_mat",
    "condim":   "4",
    "friction": "2.0 0.01 0.001",
})
worldbody.insert(0, terrain_geom)

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
qpos[2] = 1.0   # z — above max terrain height (0.5 m)
qpos[3] = 1.0   # qw
key_str = " ".join(f"{v:.6g}" for v in qpos)
keyframe = mjcf.find("keyframe")
if keyframe is None:
    keyframe = ET.SubElement(mjcf, "keyframe")
ET.SubElement(keyframe, "key", {"name": "home", "qpos": key_str})

tree.write(str(OUT), xml_declaration=True, encoding="unicode")
print(f"Written: {OUT}")
print(f"  nq={model.nq}  nv={model.nv}  nbody={model.nbody}  nu={model.nu}")
