"""Tree generation orchestrator.

Builds one or more procedural olive trees and writes data/grove.xml.

Currently uses the primitive path (trees.primitives). Track B will swap
this for Blender + Sapling Tree Gen when --use-blender is on; primitive
remains the no-dep fallback.

Run via:
    python -m trees.generate --variety mature
    python -m trees.generate --variety ancient --size 1.3 --detachment 0.6
    python -m trees.generate --variety young --leaves 0.8 --branches 3
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .primitives import build_tree_primitives, emit_grove_xml
from .varieties import VARIETIES

REPO_ROOT = Path(__file__).resolve().parent.parent
TERRAIN_XML = REPO_ROOT / "data" / "terrain.xml"
TERRAIN_PNG = REPO_ROOT / "data" / "terrain.png"
GROVE_XML   = REPO_ROOT / "data" / "grove.xml"
HALF_EXTENT = 5.0


def terrain_height_at(x: float, y: float) -> float:
    """Sample the live heightfield at world (x, y)."""
    img = np.asarray(Image.open(TERRAIN_PNG).convert("L")) / 255.0
    h, w = img.shape
    m = re.search(
        r'<hfield\b[^>]*\bsize="\S+\s+\S+\s+([\d.]+)',
        TERRAIN_XML.read_text(),
    )
    zmax = float(m.group(1)) if m else 1.0
    u = float(np.clip((x + HALF_EXTENT) / (2 * HALF_EXTENT), 0, 1))
    v = float(np.clip((HALF_EXTENT - y) / (2 * HALF_EXTENT), 0, 1))
    return float(img[int(v * (h - 1)), int(u * (w - 1))]) * zmax


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variety", choices=list(VARIETIES.keys()), default="mature")
    p.add_argument("--size", type=float, default=1.0)
    p.add_argument("--branches", type=int, default=None)
    p.add_argument("--leaves", type=float, default=0.6)
    p.add_argument("--detachment", type=float, default=0.3)
    p.add_argument("--pos", type=float, nargs=2, default=[2.0, 0.0])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    V = VARIETIES[args.variety]
    size = max(0.3, args.size)
    leaves = max(0.1, min(1.0, args.leaves))
    detach = max(0.0, min(1.0, args.detachment))
    n_branches = max(2, min(8, args.branches if args.branches is not None else V["n_branches"]))

    rng = np.random.default_rng(args.seed)
    bx, by = args.pos
    bz = terrain_height_at(bx, by) - 0.05  # bury base 5cm

    trees = [
        build_tree_primitives(
            variety_params=V,
            size=size,
            n_branches=n_branches,
            leaves=leaves,
            detach=detach,
            base_pos=(bx, by, bz),
            rng=rng,
        )
    ]
    emit_grove_xml(trees, GROVE_XML)

    height = V["height"] * size
    base_r = V["base_r"] * size
    print(f"Wrote {GROVE_XML}")
    print(f"  variety={args.variety} size={size:.2f} → height={height:.2f}m, trunk_r={base_r:.2f}m")
    print(f"  branches={n_branches}  leaves={leaves:.2f}  detachment={detach:.2f}")
    print(f"  position=({bx:.2f}, {by:.2f}, {bz:.2f})  seed={args.seed}")


if __name__ == "__main__":
    main()
