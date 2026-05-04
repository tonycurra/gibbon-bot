"""Build articulated branch chains from a Blender Sapling skeleton JSON.

Each non-trunk branch becomes a nested `<body>` chain of N segments, joined
by ball joints with stiffness + damping. Robot can push branches and they
spring back. The trunk stays static (one rigid mesh + collision capsule);
only branches articulate.

Track 2c first pass:
- skeleton splines from `data/trees/<id>.json` are decimated to 2-4
  segments per branch (keeps DOF count manageable)
- each segment is a capsule with collision enabled (climbable)
- leaf cluster ellipsoid attaches to the LAST segment so foliage moves
  with the branch
- per-level stiffness presets approximate woody → twiggy flex
"""
from __future__ import annotations

import math
from pathlib import Path

# Default rgba constants — caller can override
BARK_DEFAULT   = "0.48 0.42 0.34 1"
LEAVES_DEFAULT = "0.50 0.58 0.40 1"

# Per-segment ball joint defaults. Higher level (twigs) = lower stiffness.
# stiffness in N·m/rad, damping in N·m·s/rad.
LEVEL_PRESETS = [
    # placeholder for level 0 (trunk); we don't articulate trunk
    dict(stiffness=80.0, damping=8.0, range=0.30, density=900.0),
    # level 1 — primary branches: woody, stiff
    dict(stiffness=18.0, damping=2.5, range=0.50, density=750.0),
    # level 2 — secondary branches: more flex
    dict(stiffness=8.0,  damping=1.2, range=0.65, density=650.0),
    # level 3 — twigs: limp
    dict(stiffness=4.0,  damping=0.8, range=0.85, density=550.0),
]


def _decimate(points: list[dict], n_segments: int) -> list[dict]:
    """Sample n_segments+1 points evenly from a denser spline."""
    n_segments = max(2, min(4, n_segments))
    if len(points) <= n_segments + 1:
        return list(points)
    out = []
    for i in range(n_segments + 1):
        t = i / n_segments
        idx = int(round(t * (len(points) - 1)))
        out.append(points[idx])
    return out


def _classify_level(branch_idx: int, total_branches: int) -> int:
    """Heuristic: branch 0 = trunk (level 0); first quarter = level 1;
    next half = level 2; last quarter = level 3 (twigs).

    Sapling emits splines roughly in level order, so this is approximately
    right without the JSON carrying explicit level info.
    """
    if branch_idx == 0:
        return 0
    if total_branches <= 4:
        return 1
    frac = branch_idx / max(1, total_branches - 1)
    if frac < 0.30:
        return 1
    if frac < 0.70:
        return 2
    return 3


def _seg_count(level: int, length: float) -> int:
    """Pick number of segments for a branch by its level + length."""
    if level == 1:
        return 3 if length > 0.4 else 2
    if level == 2:
        return 3 if length > 0.5 else 2
    return 2  # twigs


def _length(p0: list[float], p1: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, p1)))


def _f3(v) -> str:
    return f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"


def emit_branch_chain(
    branch: dict,
    parent_pos: tuple[float, float, float],
    body_name: str,
    level: int,
    bark: str,
    leaves: str,
    fol_size: float,
    indent: int = 4,
) -> list[str]:
    """Recursively build nested <body> chain. Returns list of XML lines.

    `parent_pos` is the worldbody offset to the tree base (where Blender's
    (0,0,0) lands). The branch's points are in Blender-local coords, so
    we translate by parent_pos when computing world positions.
    """
    pts = branch["points"]
    total_len = _length(pts[0]["pos"], pts[-1]["pos"])
    n_seg = _seg_count(level, total_len)
    pts = _decimate(pts, n_seg)
    if len(pts) < 2:
        return []

    preset = LEVEL_PRESETS[min(level, 3)]
    stiffness = preset["stiffness"]
    damping = preset["damping"]
    rng = preset["range"]
    density = preset["density"]

    px, py, pz = parent_pos
    lines: list[str] = []
    pad = " " * indent

    # First body: anchored at branch's first point in world space.
    p0 = pts[0]["pos"]
    base_world = (px + p0[0], py + p0[1], pz + p0[2])

    # We carry the "current world position" so each child body can express
    # its pos as an offset from its parent body's frame.
    prev_world = base_world

    # Build chain bodies inside-out via a stack of close tags
    close_tags: list[str] = []
    for i in range(len(pts) - 1):
        p_curr = pts[i]["pos"]
        p_next = pts[i + 1]["pos"]
        # offset from prev body frame to this body
        if i == 0:
            local_pos = base_world  # first body is in worldbody frame
        else:
            local_pos = (p_curr[0] - pts[i - 1]["pos"][0],
                         p_curr[1] - pts[i - 1]["pos"][1],
                         p_curr[2] - pts[i - 1]["pos"][2])
        # capsule along this segment in body-local frame: from origin to delta
        delta = (p_next[0] - p_curr[0],
                 p_next[1] - p_curr[1],
                 p_next[2] - p_curr[2])
        seg_len = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
        # radius from spline radius, with a minimum so capsules render
        r_curr = max(0.012, pts[i]["radius"] * 0.10)
        # taper between adjacent radii
        r_next = max(0.010, pts[i + 1]["radius"] * 0.10)
        r = (r_curr + r_next) / 2

        seg_name = f"{body_name}_s{i}"
        lines.append(
            f'{pad}<body name="{seg_name}" pos="{_f3(local_pos)}">'
        )
        lines.append(
            f'{pad}  <joint type="ball" stiffness="{stiffness:.2f}" '
            f'damping="{damping:.2f}" range="0 {rng:.2f}"/>'
        )
        lines.append(
            f'{pad}  <geom type="capsule" '
            f'fromto="0 0 0 {_f3(delta)}" size="{r:.4f}" '
            f'rgba="{bark}" density="{density:.0f}" '
            f'contype="1" conaffinity="31"/>'
        )
        close_tags.append(f"{pad}</body>")
        pad += "  "
        prev_world = (prev_world[0] + delta[0],
                      prev_world[1] + delta[1],
                      prev_world[2] + delta[2])

    # leaf cluster: ellipsoid in the LAST body's frame, at end of last segment
    if fol_size > 0 and level >= 1:
        last_p = pts[-1]["pos"]
        prev_p = pts[-2]["pos"]
        tip_local = (last_p[0] - prev_p[0],
                     last_p[1] - prev_p[1],
                     last_p[2] - prev_p[2])
        sx = sy = fol_size
        sz = fol_size * 0.55
        lines.append(
            f'{pad}<geom type="ellipsoid" pos="{_f3(tip_local)}" '
            f'size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="{leaves}" '
            f'contype="0" conaffinity="0"/>'
        )

    # close all bodies inside-out
    for tag in reversed(close_tags):
        lines.append(tag)

    return lines


def build_grove_blocks_articulated(
    sidecars: list[dict],
    bark: str = BARK_DEFAULT,
    leaves: str = LEAVES_DEFAULT,
    fol_size: float = 0.18,
    max_articulated: int = 8,
) -> tuple[str, str]:
    """Write grove.xml with articulated primary branches.

    Trunk: static mesh visual + invisible collision capsule.
    Wood mesh: shows all branches (Sapling produces hundreds — they stay
    rigid as part of the visual).
    Articulated branches: only the longest `max_articulated` non-trunk
    splines become flexible chains. Their collision capsules are
    invisible (alpha=0) — the mesh provides the visual. Leaf-cluster
    ellipsoids are attached to articulated branch tips so foliage moves
    with the bend.

    This keeps DOF count tractable: 8 branches × 2-3 segments × 4 qpos
    per ball joint ≈ 80 added qpos per tree, instead of thousands.
    """
    asset_lines: list[str] = []
    body_lines: list[str] = []

    for tid, side in enumerate(sidecars):
        bx, by, bz = side["pos"]
        wood_obj = side["wood_obj"]

        # asset: wood mesh only (leaves on articulated tips replace any leaf OBJ)
        asset_lines.append(
            f'    <mesh name="tree_{tid}_wood" file="trees/{wood_obj}"/>'
        )

        branches_data = side["branches"]
        if not branches_data:
            continue

        trunk = branches_data[0]
        p_base = trunk["points"][0]["pos"]
        p_top = trunk["points"][-1]["pos"]
        r_max = max(p["radius"] for p in trunk["points"])
        r = max(0.05, r_max * 0.10)
        pb = (bx + p_base[0], by + p_base[1], bz + p_base[2])
        pt = (bx + p_top[0], by + p_top[1], bz + p_top[2])
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
            f'pos="{bx:.3f} {by:.3f} {bz:.3f}" rgba="{bark}" '
            f'contype="0" conaffinity="0"/>'
        )

        # Pick the LONGEST non-trunk branches to articulate. Sapling emits
        # hundreds of splines but most are short twigs; the structural
        # branches the robot can grab are the long ones.
        non_trunk = branches_data[1:]
        ranked = sorted(
            non_trunk,
            key=lambda b: _length(b["points"][0]["pos"], b["points"][-1]["pos"]),
            reverse=True,
        )
        articulated = ranked[:max_articulated]

        for bid, branch in enumerate(articulated):
            length = _length(branch["points"][0]["pos"], branch["points"][-1]["pos"])
            level = 1 if length > 0.5 else 2
            body_name = f"tree_{tid}_b{bid}"
            chain = emit_branch_chain(
                branch=branch,
                parent_pos=(bx, by, bz),
                body_name=body_name,
                level=level,
                bark="0 0 0 0",  # invisible — wood mesh provides the visual
                leaves=leaves,
                fol_size=fol_size,
                indent=4,
            )
            body_lines.extend(chain)

    return ("\n".join(asset_lines), "\n".join(body_lines))
