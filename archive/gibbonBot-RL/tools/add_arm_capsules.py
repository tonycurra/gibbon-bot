"""
One-shot scene.xml surgery: add capsule collision proxies to each arm link.

The arm meshes are kept for visuals (contype=0, conaffinity=0) and a thin
capsule is added per link spanning origin → child-link's origin.

Per-arm collision bitmasks ensure:
  - arms collide with floor and with OTHER arms
  - same-arm adjacent links do NOT collide (they're physically attached)

Run once: python tools/add_arm_capsules.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import xml.etree.ElementTree as ET

SCENE = pathlib.Path(__file__).parent.parent / "scene.xml"

# child offsets (same for every arm). Capsule on parent runs origin→child_pos.
CHILD_OFFSETS = {
    "link1":        (0.0,      0.0,      0.051959),
    "link2":        (0.236815, 0.030002, 0.0),
    "link3":        (0.0,     -0.144586, 0.0),
    "link4":        (0.015147,-0.053653, 0.0),
    "link5":        (0.0,      0.018821, 0.052035),
    "gripper_link": (0.0,      0.0,      0.04),  # terminal: short stub
}

# only outer arm segments collide — the inner ones (link1, link2) rarely clash
# and add lots of pairwise checks that crater MJX SPS.
COLLIDING_LINKS = {"link3", "link4", "link5", "gripper_link"}

CAPSULE_RADIUS = {
    "link1":        0.025,
    "link2":        0.022,
    "link3":        0.020,
    "link4":        0.018,
    "link5":        0.018,
    "gripper_link": 0.015,
}

# bitmask per arm: each arm gets unique contype, conaffinity excludes self
ARM_BITS = {"fl": 2, "fr": 4, "bl": 8, "br": 16}
FLOOR_BIT = 1


def main():
    tree = ET.parse(SCENE)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # ensure floor sees all arm bits
    for g in worldbody.iter("geom"):
        if g.get("name") == "floor":
            g.set("contype", str(FLOOR_BIT))
            affinity = FLOOR_BIT | sum(ARM_BITS.values())
            g.set("conaffinity", str(affinity))

    n_disabled = 0
    n_added = 0
    # walk every arm body
    for body in worldbody.iter("body"):
        name = body.get("name", "")
        if not name.startswith(("fl_", "fr_", "bl_", "br_")):
            continue
        prefix, link = name.split("_", 1)
        if link not in CHILD_OFFSETS:
            continue

        # disable the existing mesh geom's collisions
        for g in body.findall("geom"):
            if g.get("type") == "mesh":
                g.set("contype", "0")
                g.set("conaffinity", "0")
                n_disabled += 1

        # add capsule
        x, y, z = CHILD_OFFSETS[link]
        r = CAPSULE_RADIUS[link]
        if link in COLLIDING_LINKS:
            my_bit = ARM_BITS[prefix]
            affinity = FLOOR_BIT | (sum(ARM_BITS.values()) ^ my_bit)
            contype, conaff = str(my_bit), str(affinity)
        else:
            contype, conaff = "0", "0"
        cap = ET.SubElement(body, "geom", {
            "name":        f"{name}_cap",
            "type":        "capsule",
            "fromto":      f"0 0 0  {x:.6g} {y:.6g} {z:.6g}",
            "size":        f"{r}",
            "rgba":        "0.9 0.4 0.4 0.4",
            "contype":     contype,
            "conaffinity": conaff,
            "condim":      "3",
            "friction":    "1.0 0.005 0.0001",
        })
        n_added += 1

    tree.write(SCENE, xml_declaration=True, encoding="unicode")
    print(f"Disabled {n_disabled} mesh geom collisions.")
    print(f"Added {n_added} capsule collision proxies.")
    print(f"Wrote: {SCENE}")


if __name__ == "__main__":
    main()
