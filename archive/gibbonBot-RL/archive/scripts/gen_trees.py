"""Generate a forest of simple capsule trees for climbing tests.

Each tree: vertical trunk (capsule) + N branches sticking out at random heights
and angles. All collidable. No realism — just structure for the robot to grip.

Usage:
    python gen_trees.py                    # writes data/trees_forest.xml
    python gen_trees.py --n 12 --seed 7    # 12 trees, seed 7

Then view:
    python view_scene.py data/trees_forest.xml
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image

# hfield params must match the scene XML
HFIELD_HALF_X = 5.0
HFIELD_HALF_Y = 5.0
HFIELD_MAX_HEIGHT = 0.10
TERRAIN_PNG = Path(__file__).parent / "data" / "terrain.png"


def load_terrain_sampler():
    """Return a function (x, y) -> z that samples the heightmap."""
    img = Image.open(TERRAIN_PNG).convert("L")
    w, h = img.size
    pixels = img.load()

    def sample(x, y):
        # Map world (x, y) ∈ [-half, +half] → pixel (u, v) ∈ [0, w-1] × [0, h-1].
        # MuJoCo hfield row 0 is +Y, so flip v.
        u = (x + HFIELD_HALF_X) / (2 * HFIELD_HALF_X) * (w - 1)
        v = (1.0 - (y + HFIELD_HALF_Y) / (2 * HFIELD_HALF_Y)) * (h - 1)
        ui = max(0, min(w - 1, int(round(u))))
        vi = max(0, min(h - 1, int(round(v))))
        return pixels[ui, vi] / 255.0 * HFIELD_MAX_HEIGHT

    return sample

HEADER = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="trees_forest">
  <compiler angle="degree"/>

  <asset>
    <hfield name="terrain" file="terrain.png" size="5.0 5.0 0.1000 0.05" />
    <mesh name="terrain_surface" file="terrain_surface.obj"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.350 0.550 0.850" rgb2="0.700 0.833 0.960" width="256" height="256" />
  </asset>

  <visual>
    <headlight ambient="0.400 0.420 0.450" diffuse="0.6 0.6 0.6" specular="0 0 0" />
    <quality shadowsize="4096" offsamples="8" />
    <map fogstart="20" fogend="60" znear="0.05" zfar="200" />
    <rgba haze="0.65 0.74 0.82 1" />
    <global offwidth="1920" offheight="1920" />
  </visual>

  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="-3.536 3.536 8.660" dir="0.354 -0.354 -0.866" diffuse="0.950 0.920 0.850" specular="0.2 0.2 0.2" />
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 0" rgba="0 0 0 0" condim="4"/>
    <geom name="terrain_visual" type="mesh" mesh="terrain_surface" rgba="0.45 0.32 0.18 1" contype="0" conaffinity="0"/>
"""

FOOTER = """  </worldbody>
</mujoco>
"""

BARK_RGBA = "0.50 0.47 0.43 1"


def random_branches(rng, trunk_radius, trunk_height, count):
    """Yield (start_xyz, end_xyz, radius) for `count` random branches."""
    for _ in range(count):
        # Branch attaches somewhere in the upper 70% of the trunk
        h = rng.uniform(0.35 * trunk_height, 0.95 * trunk_height)
        # Random horizontal direction, slight upward tilt
        theta = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(-0.1, 0.5)  # mostly horizontal-to-up
        length = rng.uniform(0.4, 1.2) * trunk_height * 0.4
        r = trunk_radius * rng.uniform(0.25, 0.55)
        dx = math.cos(theta) * length * math.cos(tilt)
        dy = math.sin(theta) * length * math.cos(tilt)
        dz = length * math.sin(tilt)
        # start just inside the trunk so the branch visually merges
        start = (math.cos(theta) * trunk_radius * 0.5,
                 math.sin(theta) * trunk_radius * 0.5,
                 h)
        end = (start[0] + dx, start[1] + dy, start[2] + dz)
        yield start, end, r


def emit_tree(idx, x, y, ground_z, rng):
    """Return XML fragment for one tree at (x, y), planted at ground_z."""
    trunk_radius = rng.uniform(0.06, 0.18)        # 6cm thin sapling → 18cm trunk
    trunk_height = rng.uniform(2.0, 5.0)
    n_branches = rng.randint(3, 8)

    # Sink the trunk base slightly below the surface so it visibly plants in.
    z_base = ground_z - trunk_radius * 0.5
    z_top = ground_z + trunk_height

    lines = [f'    <!-- tree {idx}: r={trunk_radius:.3f} h={trunk_height:.2f} ground_z={ground_z:.3f} -->']
    lines.append(
        f'    <geom name="t{idx}_trunk" type="capsule" '
        f'fromto="{x:.3f} {y:.3f} {z_base:.3f} {x:.3f} {y:.3f} {z_top:.3f}" '
        f'size="{trunk_radius:.3f}" rgba="{BARK_RGBA}" '
        f'contype="1" conaffinity="31" friction="0.7 0.01 0.001"/>'
    )
    for bi, (s, e, r) in enumerate(random_branches(rng, trunk_radius, trunk_height, n_branches)):
        sx, sy, sz = x + s[0], y + s[1], ground_z + s[2]
        ex, ey, ez = x + e[0], y + e[1], ground_z + e[2]
        lines.append(
            f'    <geom name="t{idx}_b{bi}" type="capsule" '
            f'fromto="{sx:.3f} {sy:.3f} {sz:.3f} {ex:.3f} {ey:.3f} {ez:.3f}" '
            f'size="{r:.3f}" rgba="{BARK_RGBA}" '
            f'contype="1" conaffinity="31" friction="0.7 0.01 0.001"/>'
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="number of trees")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spread", type=float, default=4.0,
                   help="half-extent of XY area trees are placed in")
    p.add_argument("--out", default="data/trees_forest.xml",
                   help="standalone scene path (ignored if --inject is set)")
    p.add_argument("--inject", default=None,
                   help="path to a scene XML; trees are written between "
                        "TREE-WORLDBODY-BEGIN/END markers in-place")
    p.add_argument("--min-robot-dist", type=float, default=1.5,
                   help="when injecting, keep trees this far from origin (robot spawn)")
    p.add_argument("--at", type=float, nargs=2, default=None,
                   help="force a single tree at (x, y), overrides --n and random placement")
    args = p.parse_args()

    rng = random.Random(args.seed)
    sample_z = load_terrain_sampler()

    # Place trees on a jittered grid so they don't overlap
    if args.at is not None:
        placed = [tuple(args.at)]
    else:
        placed = []
        min_sep = 1.5
        min_robot = args.min_robot_dist if args.inject else 0.0
        attempts = 0
        while len(placed) < args.n and attempts < 2000:
            x = rng.uniform(-args.spread, args.spread)
            y = rng.uniform(-args.spread, args.spread)
            if math.hypot(x, y) < min_robot:
                attempts += 1
                continue
            if all(math.hypot(x - px, y - py) > min_sep for px, py in placed):
                placed.append((x, y))
            attempts += 1

    tree_blocks = [emit_tree(i, x, y, sample_z(x, y), rng)
                   for i, (x, y) in enumerate(placed)]

    if args.inject:
        target = Path(args.inject)
        text = target.read_text()
        begin = "<!-- TREE-WORLDBODY-BEGIN -->"
        end = "<!-- TREE-WORLDBODY-END -->"
        if begin not in text or end not in text:
            raise SystemExit(f"markers {begin}/{end} not found in {target}")
        b = text.index(begin) + len(begin)
        e = text.index(end)
        new_text = text[:b] + "\n" + "".join(tree_blocks) + "    " + text[e:]
        target.write_text(new_text)
        print(f"injected {len(placed)} trees into {target}")
    else:
        out = Path(args.out)
        out.write_text(HEADER + "".join(tree_blocks) + FOOTER)
        print(f"wrote {out} ({len(placed)} trees)")


if __name__ == "__main__":
    main()
