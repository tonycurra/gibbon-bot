"""Primitive-based tree builder — the Phase 1 fallback path.

Emits MuJoCo capsule + ellipsoid geoms directly into grove.xml. No mesh
files generated. Used when Blender is not on PATH, or when --primitives
is forced. Visual quality is "ugly but real" — kept as a working baseline
the rest of the pipeline can fall back to.
"""
from pathlib import Path

import numpy as np

BARK   = "0.36 0.27 0.18 1"     # warm brown
LEAVES = "0.42 0.55 0.32 1"     # olive green


def build_tree_primitives(variety_params, size, n_branches, leaves, detach, base_pos, rng, n_trunk=4):
    """Build a single tree's primitive description.

    Returns three lists:
        trunk_caps  : [(p0, p1, radius), ...]   collision-enabled
        branches    : [(p0, p1, radius), ...]   visual-only (Phase 1)
        foliage     : [(center, [rx, ry, rz]), ...]
    """
    V = variety_params
    height = V["height"] * size
    base_r = V["base_r"] * size
    top_r  = V["top_r"]  * size
    gnarl  = V["gnarl"]
    canopy = V["canopy"] * size
    fol_r  = V["fol_r"]  * size

    bx, by, bz = base_pos

    # ---------- trunk: 4 capsules along a slightly bent gnarled curve ----------
    trunk_pts = []
    for i in range(n_trunk + 1):
        t = i / n_trunk
        dx = gnarl * np.sin(t * 1.7 * np.pi) * 0.35 * rng.uniform(0.7, 1.3)
        dy = gnarl * np.sin(t * 2.3 * np.pi + 0.7) * 0.35 * rng.uniform(0.7, 1.3)
        trunk_pts.append((bx + dx, by + dy, bz + t * height))

    trunk_caps = []
    for i in range(n_trunk):
        t = (i + 0.5) / n_trunk
        r = base_r + (top_r - base_r) * t
        trunk_caps.append((trunk_pts[i], trunk_pts[i + 1], r))

    # ---------- branches: capsules off upper trunk ----------
    branches = []
    for b in range(n_branches):
        t = rng.uniform(0.4, 0.9)
        seg = min(int(t * n_trunk), n_trunk - 1)
        p0 = np.array(trunk_pts[seg])
        az = 2 * np.pi * b / n_branches + rng.uniform(-0.4, 0.4)
        el = rng.uniform(np.pi / 7, np.pi / 2.3)
        blen = canopy * rng.uniform(0.7, 1.2)
        direction = np.array(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
        )
        p1 = p0 + direction * blen
        r0 = top_r * rng.uniform(0.45, 0.7)
        branches.append((tuple(p0), tuple(p1), r0))

    # ---------- foliage: ellipsoids at trunk crown + each branch tip ----------
    foliage = []
    crown = np.array(trunk_pts[-1]) + np.array([0, 0, fol_r * 0.4])
    foliage.append((crown, [fol_r * 1.0, fol_r * 1.0, fol_r * 0.6]))
    for (_, p1, _) in branches:
        spread = 1.0 - 0.55 * detach
        fr = fol_r * leaves * rng.uniform(0.6, 1.0) * spread
        foliage.append((np.array(p1), [fr * 1.1, fr * 1.1, fr * 0.7]))

    return trunk_caps, branches, foliage


def _f3(v):
    return f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}"


def _f6(a, b):
    return f"{a[0]:.3f} {a[1]:.3f} {a[2]:.3f} {b[0]:.3f} {b[1]:.3f} {b[2]:.3f}"


def build_grove_blocks(trees) -> tuple[str, str]:
    """Return (assets_xml, worldbody_xml) for a list of primitive trees.

    Primitives have no asset declarations — only worldbody geoms.
    """
    body_lines: list[str] = []
    for tid, (trunk_caps, branches, foliage) in enumerate(trees):
        for i, (p0, p1, r) in enumerate(trunk_caps):
            body_lines.append(
                f'    <geom name="tree_{tid}_trunk_{i}" type="capsule" '
                f'fromto="{_f6(p0, p1)}" size="{r:.3f}" rgba="{BARK}" '
                f'contype="1" conaffinity="31"/>'
            )
        for i, (p0, p1, r) in enumerate(branches):
            body_lines.append(
                f'    <geom name="tree_{tid}_branch_{i}" type="capsule" '
                f'fromto="{_f6(p0, p1)}" size="{r:.3f}" rgba="{BARK}" '
                f'contype="0" conaffinity="0"/>'
            )
        for i, (c, sz) in enumerate(foliage):
            body_lines.append(
                f'    <geom name="tree_{tid}_foliage_{i}" type="ellipsoid" '
                f'pos="{_f3(c)}" size="{sz[0]:.3f} {sz[1]:.3f} {sz[2]:.3f}" '
                f'rgba="{LEAVES}" contype="0" conaffinity="0"/>'
            )
    return ("", "\n".join(body_lines))
