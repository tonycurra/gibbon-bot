"""Tree generation orchestrator.

Builds one olive tree and writes data/grove.xml.

Two backends:
  * **blender** — invokes Blender headlessly with the Sapling Tree Gen
    addon (`trees/blender_sapling.py`). Emits real OBJ meshes.
  * **primitives** — pure-Python fallback (capsules + ellipsoids) used
    when Blender is missing or `--primitives` is forced.

Run via:
    python -m trees.generate --variety mature
    python -m trees.generate --variety ancient --size 1.3 --detachment 0.6
    python -m trees.generate --variety young --leaves 0.8 --branches 3
    python -m trees.generate --primitives          # force fallback
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .articulate import emit_grove_xml_articulated
from .primitives import build_tree_primitives, emit_grove_xml
from .varieties import VARIETIES

REPO_ROOT     = Path(__file__).resolve().parent.parent
TERRAIN_XML   = REPO_ROOT / "data" / "terrain.xml"
TERRAIN_PNG   = REPO_ROOT / "data" / "terrain.png"
GROVE_XML     = REPO_ROOT / "data" / "grove.xml"
TREES_DIR     = REPO_ROOT / "data" / "trees"
BLENDER_SCRIPT = REPO_ROOT / "trees" / "blender_sapling.py"
HALF_EXTENT   = 5.0

BARK   = "0.48 0.42 0.34 1"   # silvery-warm gray, characteristic of olive bark
LEAVES = "0.50 0.58 0.40 1"   # silvery-green, leaning a touch dusty


# ─────────────────────────── terrain height sampling ─────────────────────────
def terrain_height_at(x: float, y: float) -> float:
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


# ─────────────────────────────── arg parsing ─────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variety", choices=list(VARIETIES.keys()), default="mature")
    p.add_argument("--size", type=float, default=1.0)
    p.add_argument("--branches", type=int, default=None)
    p.add_argument("--leaves", type=float, default=0.6)
    p.add_argument("--detachment", type=float, default=0.3)
    p.add_argument("--pos", type=float, nargs=2, default=[2.0, 0.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--primitives", action="store_true",
                   help="force the primitive backend (skip Blender even if present)")
    p.add_argument("--rigid-branches", action="store_true",
                   help="emit static branch mesh instead of articulated bodies (faster training; no flex)")
    p.add_argument("--articulate-max", type=int, default=8,
                   help="cap number of articulated branches per tree (default 8)")
    return p.parse_args(argv)


# ─────────────────────────────── Blender path ────────────────────────────────
def find_blender() -> str | None:
    """Return path to `blender` CLI, or None if not on PATH."""
    return shutil.which("blender")


def run_blender(blender_exe: str, output_id: str, args, n_branches: int,
                size: float, leaves: float, detach: float,
                base_pos: tuple[float, float]) -> dict | None:
    """Invoke Blender headlessly. Returns parsed JSON sidecar on success."""
    TREES_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_exe, "--background", "--python", str(BLENDER_SCRIPT), "--",
        "--output-id", output_id,
        "--output-dir", str(TREES_DIR),
        "--variety", args.variety,
        "--size", str(size),
        "--branches", str(n_branches),
        "--leaves", str(leaves),
        "--detachment", str(detach),
        "--seed", str(args.seed),
        "--pos", str(base_pos[0]), str(base_pos[1]),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("[trees.generate] Blender timed out (60s)", file=sys.stderr)
        return None
    if proc.returncode != 0:
        # Blender prints all output to stdout; show last few lines for debugging
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
        print(f"[trees.generate] Blender failed (rc={proc.returncode}):\n{tail}",
              file=sys.stderr)
        return None
    sidecar_path = TREES_DIR / f"{output_id}.json"
    if not sidecar_path.exists():
        print(f"[trees.generate] sidecar missing: {sidecar_path}", file=sys.stderr)
        return None
    return json.loads(sidecar_path.read_text())


def emit_grove_xml_from_blender(sidecars: list[dict], output_path: Path) -> None:
    """Build grove.xml referencing OBJ meshes + a trunk collision capsule.

    sidecars are JSON dicts as written by trees/blender_sapling.py. Each
    one's `pos` field is interpreted as the world-space anchor where the
    tree's local origin (Blender (0,0,0)) lands.
    """
    asset_lines: list[str] = []
    body_lines: list[str] = []

    for tid, side in enumerate(sidecars):
        bx, by, bz = side["pos"]
        wood_obj   = side["wood_obj"]
        leaves_obj = side.get("leaves_obj")

        asset_lines.append(
            f'    <mesh name="tree_{tid}_wood" file="trees/{wood_obj}"/>'
        )
        if leaves_obj:
            asset_lines.append(
                f'    <mesh name="tree_{tid}_leaves" file="trees/{leaves_obj}"/>'
            )

        # Collision capsule for the trunk: take the first spline (trunk)
        # and use its first→last point as the capsule fromto. Use the
        # max radius along that spline.
        trunk = side["branches"][0]
        p_base = trunk["points"][0]["pos"]
        p_top  = trunk["points"][-1]["pos"]
        r_max  = max(p["radius"] for p in trunk["points"])
        # max(0.05, ...) keeps the capsule visible even if Sapling's curve
        # radii are unitless — we treat them as "tens of cm" approx.
        r = max(0.05, r_max * 0.10)

        pb = (bx + p_base[0], by + p_base[1], bz + p_base[2])
        pt = (bx + p_top[0],  by + p_top[1],  bz + p_top[2])
        body_lines.append(
            f'    <geom name="tree_{tid}_trunk_col" type="capsule" '
            f'fromto="{pb[0]:.3f} {pb[1]:.3f} {pb[2]:.3f} '
            f'{pt[0]:.3f} {pt[1]:.3f} {pt[2]:.3f}" '
            f'size="{r:.3f}" rgba="0 0 0 0" '
            f'contype="1" conaffinity="31"/>'
        )
        body_lines.append(
            f'    <geom name="tree_{tid}_wood_visual" type="mesh" '
            f'mesh="tree_{tid}_wood" '
            f'pos="{bx:.3f} {by:.3f} {bz:.3f}" rgba="{BARK}" '
            f'contype="0" conaffinity="0"/>'
        )
        if leaves_obj:
            body_lines.append(
                f'    <geom name="tree_{tid}_leaves_visual" type="mesh" '
                f'mesh="tree_{tid}_leaves" '
                f'pos="{bx:.3f} {by:.3f} {bz:.3f}" rgba="{LEAVES}" '
                f'contype="0" conaffinity="0"/>'
            )

    out = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<!-- generated by trees.generate (Blender path) -->',
        '<mujocoinclude>',
        '  <asset>',
        *asset_lines,
        '  </asset>',
        '  <worldbody>',
        *body_lines,
        '  </worldbody>',
        '</mujocoinclude>',
        '',
    ]
    output_path.write_text('\n'.join(out))


# ─────────────────────────────────── main ────────────────────────────────────
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

    blender_exe = None if args.primitives else find_blender()

    if blender_exe:
        output_id = "tree_0"   # single tree for now; Phase 3 will index by position
        sidecar = run_blender(
            blender_exe, output_id, args,
            n_branches=n_branches, size=size, leaves=leaves, detach=detach,
            base_pos=(bx, by),
        )
        if sidecar is not None:
            sidecar["pos"] = [bx, by, bz]
            if args.rigid_branches:
                emit_grove_xml_from_blender([sidecar], GROVE_XML)
                mode = "static mesh"
            else:
                # articulated: trunk static, branches as ball-jointed body chains
                fol_size = 0.20 * size * leaves * (1.0 - 0.5 * detach)
                emit_grove_xml_articulated(
                    [sidecar], GROVE_XML,
                    bark=BARK, leaves=LEAVES, fol_size=fol_size,
                    max_articulated=args.articulate_max,
                )
                mode = "articulated"
            print(f"Wrote {GROVE_XML} (Blender + {mode}, {output_id})")
            print(f"  variety={args.variety} size={size:.2f}  branches={n_branches}")
            print(f"  leaves={leaves:.2f}  detachment={detach:.2f}  seed={args.seed}")
            print(f"  position=({bx:.2f}, {by:.2f}, {bz:.2f})")
            return
        print("[trees.generate] Blender path failed — falling back to primitives.",
              file=sys.stderr)

    # ── primitive fallback ──
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
    print(f"Wrote {GROVE_XML} (primitives)")
    print(f"  variety={args.variety} size={size:.2f} → height={height:.2f}m, trunk_r={base_r:.2f}m")
    print(f"  branches={n_branches}  leaves={leaves:.2f}  detachment={detach:.2f}")
    print(f"  position=({bx:.2f}, {by:.2f}, {bz:.2f})  seed={args.seed}")


if __name__ == "__main__":
    main()
