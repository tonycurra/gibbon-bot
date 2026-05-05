"""Generate one olive tree using Blender's Sapling Tree Gen addon.

This script runs *inside* Blender, not in our normal venv:

    blender --background --python trees/blender_sapling.py -- \\
        --output-id tree_42_mature \\
        --variety mature --size 1.0 --branches 5 \\
        --leaves 0.6 --detachment 0.3 --seed 42 \\
        --output-dir data/trees

Outputs into <output-dir>:
    <output-id>_wood.obj      trunk + branches (visual mesh, smooth-shaded)
    <output-id>_leaves.obj    leaf cards (only if Sapling produced them)
    <output-id>.json          sidecar — branch midlines + olive tip positions

Track C (articulated branches) reads the JSON to place collision capsules
and hinge chains along the same skeleton. Track D (dynamic olives) reads
`olives` for body anchor positions.

Tested against Blender 3.6 + 4.x. Sapling addon is shipped with Blender;
we only need to enable it.
"""
import argparse
import json
import sys
from pathlib import Path

# bpy is only available inside Blender. Fail loudly if invoked otherwise.
try:
    import bpy
except ImportError:  # pragma: no cover
    print(
        "ERROR: this script must run inside Blender:\n"
        "  blender --background --python trees/blender_sapling.py -- ...",
        file=sys.stderr,
    )
    sys.exit(2)


# ───────────────────────── variety → sapling params ──────────────────────────
# Sapling Tree Gen exposes ~80 properties. We tune a curated subset per
# variety to get olive-ish silhouettes. Numbers came from trial-and-error
# starting from the addon's "Quaking Aspen" preset and pushing toward
# olive characteristics: short, wide, gnarled trunks; many low branches;
# silvery leaf clusters.
#
# Keys map to operator kwargs of `bpy.ops.curve.tree_add`. For Blender 4
# the operator name is unchanged; properties only differ in a few corners.
# Olive-tuned Sapling parameters. Real olives are SHORT, THICK, low-branching,
# with small silvery-green leaves and gnarled trunks. Generic tree presets
# look more like aspens — these push toward Mediterranean orchard shapes:
#   - high `ratio` (thick trunk) and strong `taper`
#   - `base_size` low so branches start near the ground
#   - many short upper-level branches (`branches[2]` high)
#   - small leaves (real olive leaf ≈ 3cm)
#   - `prune_*` shapes the canopy into a compact dense crown
#   - `attract_up` and `curve_v` add upright gnarliness
VARIETIES = {
    "young": dict(
        scale=2.0,
        scale_v=0.2,
        length=(0.5, 0.45, 0.30, 0.20),
        length_v=(0.0, 0.10, 0.10, 0.10),
        ratio=0.020,
        ratio_power=1.0,
        taper=(1.3, 1.5, 1.5, 1.5),
        branches=(0, 8, 18, 0),
        levels=3,
        base_size=0.20,
        base_splits=0,
        split_angle=(0, 32, 35, 30),
        split_angle_v=(0, 8, 8, 8),
        attract_up=(0, 0.4, 0.3, 0.0),
        curve=(0, 30, 40, 0),
        curve_v=(20, 60, 80, 0),
        prune_ratio=0.4,
        prune_width=0.6,
        prune_width_peak=0.5,
        prune_height=0.55,
        leaf_count=140,
        leaf_scale=0.030,
        leaf_scale_x=0.4,
    ),
    "mature": dict(
        # Real olive: ~4-6m tall, often wider than tall, multi-trunk from base,
        # gnarled, with horizontal-spreading rounded canopy.
        scale=3.0,
        scale_v=0.3,
        length=(0.45, 0.55, 0.35, 0.20),
        length_v=(0.0, 0.20, 0.20, 0.10),
        ratio=0.055,                 # thick trunk relative to height
        ratio_power=1.2,
        taper=(1.8, 1.7, 1.5, 1.5),  # strong taper toward branch tips
        branches=(0, 16, 32, 0),     # dense canopy twigs
        levels=3,
        base_size=0.05,              # branches start very low (olive habit)
        base_splits=2,               # multi-trunk from the base
        split_angle=(0, 50, 42, 30), # wide branching angles → spreading
        split_angle_v=(0, 12, 12, 8),
        attract_up=(0, 0.15, 0.05, 0.0),  # branches go OUT, not up
        curve=(0, 45, 55, 0),
        curve_v=(80, 110, 130, 0),   # heavy gnarl on trunk + branches
        prune_ratio=0.7,
        prune_width=1.20,            # canopy wider than tall
        prune_width_peak=0.40,
        prune_height=0.45,           # squat rounded crown
        leaf_count=280,
        leaf_scale=0.030,
        leaf_scale_x=0.3,            # narrow lance-shape leaves
    ),
    "ancient": dict(
        scale=3.0,                   # not very tall — squat and wide
        scale_v=0.4,
        length=(0.40, 0.60, 0.35, 0.20),
        length_v=(0.0, 0.20, 0.20, 0.10),
        ratio=0.085,                 # massively thick trunk
        ratio_power=1.2,
        taper=(2.0, 1.8, 1.5, 1.5),  # dramatic taper
        branches=(0, 18, 30, 0),
        levels=3,
        base_size=0.06,              # branches start very low (knotty base)
        base_splits=2,               # multi-trunk character
        split_angle=(0, 45, 40, 30),
        split_angle_v=(0, 15, 12, 8),
        attract_up=(0, 0.35, 0.25, 0.0),
        curve=(0, 50, 60, 0),
        curve_v=(70, 120, 140, 0),   # heavy gnarl
        prune_ratio=0.7,
        prune_width=1.10,            # wider than tall
        prune_width_peak=0.4,
        prune_height=0.50,
        leaf_count=320,
        leaf_scale=0.035,
        leaf_scale_x=0.4,
    ),
    "olive": dict(
        # Built to spec: squat & thick (1m+ trunk on 5m tree), multi-trunk burly
        # base, primaries break at ~1-2m, asymmetric billowing canopy as wide as
        # tall or wider, heavy gnarl, narrow lanceolate leaves. The character
        # lives in the trunk — push gnarl + ratio hard, keep canopy a cloud.
        scale=4.5,                    # ~4-5m tall
        scale_v=0.5,                  # asymmetric — symmetry kills it
        length=(0.40, 0.55, 0.40, 0.22),
        length_v=(0.0, 0.30, 0.25, 0.15),  # uneven branch lengths
        ratio=0.110,                  # ~0.5m radius trunk (1m diameter)
        ratio_power=1.3,              # branches stay thick a long way out
        taper=(1.5, 1.6, 1.5, 1.5),   # less taper → trunk holds thickness
        branches=(0, 14, 36, 0),      # few primaries, dense secondaries
        levels=3,
        base_size=0.25,               # primaries break at ~1-2m up
        base_splits=2,                # multi-trunk burly base
        split_angle=(0, 60, 45, 30),  # wide spread → outward then up
        split_angle_v=(0, 20, 15, 10),
        attract_up=(0, 0.20, 0.45, 0.0),  # spread out, then secondaries lift
        curve=(0, 50, 60, 0),
        curve_v=(140, 160, 150, 0),   # extreme gnarl (esp. trunk: 140)
        prune_ratio=0.85,
        prune_width=1.30,             # canopy WIDER than tall
        prune_width_peak=0.35,        # widest below middle = billowing
        prune_height=0.40,            # squat rounded crown
        leaf_count=380,               # dense but airy
        leaf_scale=0.045,             # 4-5cm leaves
        leaf_scale_x=0.22,            # narrow lanceolate
    ),
}


# ─────────────────────────────── arg parsing ─────────────────────────────────
def parse_args(argv):
    # Blender convention: user args come after `--`.
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--output-id", required=True)
    p.add_argument("--output-dir", default="data/trees")
    p.add_argument("--variety", choices=list(VARIETIES.keys()), default="mature")
    p.add_argument("--size", type=float, default=1.0)
    p.add_argument("--branches", type=int, default=5)
    p.add_argument("--leaves", type=float, default=0.6)
    p.add_argument("--detachment", type=float, default=0.3)  # currently unused inside Blender
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos", type=float, nargs=2, default=[0.0, 0.0])  # passed through to sidecar
    return p.parse_args(argv)


# ─────────────────────────── blender-side helpers ────────────────────────────
def reset_scene():
    """Clear the default cube/lamp/camera so we have an empty scene."""
    if bpy.data.objects:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)


def enable_sapling():
    """Make sure the Sapling Tree Gen addon is loaded.

    Tries `bpy.ops.preferences.addon_enable` first; if that's a no-op
    in headless mode (some Blender builds), falls back to addon_utils.
    """
    module = "add_curve_sapling"
    addons = bpy.context.preferences.addons
    if module in addons:
        return
    try:
        bpy.ops.preferences.addon_enable(module=module)
    except Exception:
        pass
    if module not in bpy.context.preferences.addons:
        # last-ditch: addon_utils direct enable
        import addon_utils
        addon_utils.enable(module, default_set=False, persistent=True)
    if not hasattr(bpy.ops.curve, "tree_add"):
        print("ERROR: Sapling tree_add operator missing after enable attempt",
              file=sys.stderr)
        sys.exit(3)


def build_tree(args) -> tuple:
    """Invoke Sapling's tree_add operator. Returns (wood_obj, leaf_obj).

    Detects new objects by diffing the object set across the operator
    call — robust to operator-name renames or active-object quirks in
    headless mode.
    """
    V = VARIETIES[args.variety]
    branch_count = max(2, min(40, args.branches * 4))
    leaf_count = int(V["leaf_count"] * args.leaves)

    before = {o.name for o in bpy.data.objects}

    # Sapling's tree_add accepts many props as kwargs. Order/quantity of
    # `branches` and `length` tuples must match `levels`. Levels=2 keeps
    # things light; bumping to 3 gives bushier subdivision.
    # branch_count from CLI replaces level-1 (primary branches); level-2
    # stays at the variety's preset for a denser canopy.
    branches_tuple = (0, branch_count, V["branches"][2], V["branches"][3])

    op_kwargs = dict(
        do_update=True,
        bevel=True,
        bevelRes=4,
        resU=4,
        seed=args.seed,
        scale=V["scale"] * args.size,
        scaleV=V["scale_v"],
        length=V["length"],
        lengthV=V.get("length_v", (0, 0, 0, 0)),
        ratio=V["ratio"],
        ratioPower=V.get("ratio_power", 1.2),
        taper=V.get("taper", (1, 1, 1, 1)),
        branches=branches_tuple,
        levels=V["levels"],
        baseSize=V.get("base_size", 0.25),
        baseSplits=V.get("base_splits", 0),
        splitAngle=V.get("split_angle", (0, 30, 30, 30)),
        splitAngleV=V.get("split_angle_v", (0, 0, 0, 0)),
        attractUp=V.get("attract_up", (0, 0, 0, 0)),
        curve=V.get("curve", (0, 0, 0, 0)),
        curveV=V.get("curve_v", (0, 0, 0, 0)),
        prune=V.get("prune_ratio", 0) > 0,
        pruneRatio=V.get("prune_ratio", 0),
        pruneWidth=V.get("prune_width", 0.5),
        pruneWidthPeak=V.get("prune_width_peak", 0.5),
        showLeaves=leaf_count > 0,
        leaves=leaf_count,
        leafShape="hex",
        leafScale=V["leaf_scale"] * args.size,
        leafScaleX=V.get("leaf_scale_x", 1.0),
    )
    try:
        bpy.ops.curve.tree_add(**op_kwargs)
    except TypeError as e:
        # An unsupported kwarg can fail this whole call. Try minimal args.
        print(f"[blender] tree_add full kwargs failed ({e}); retrying minimal",
              file=sys.stderr)
        bpy.ops.curve.tree_add(
            do_update=True,
            seed=args.seed,
            scale=V["scale"] * args.size,
            showLeaves=leaf_count > 0,
        )

    after = {o.name for o in bpy.data.objects}
    new_names = after - before

    if not new_names:
        # diagnostic: list everything we have
        print("[blender] no new objects after tree_add. existing:", file=sys.stderr)
        for o in bpy.data.objects:
            print(f"  - {o.name} type={o.type}", file=sys.stderr)
        return None, None

    wood = None
    leaves = None
    # Sapling emits an "envelope" curve when prune=True (the prune volume) —
    # skip it; the actual tree curve is usually named "tree".
    for name in new_names:
        obj = bpy.data.objects[name]
        if obj.type == "CURVE" and "envelope" not in obj.name.lower():
            if wood is None:
                wood = obj
        elif obj.type == "MESH" and "leaves" in obj.name.lower():
            leaves = obj

    # Clean up the envelope so it doesn't clutter the saved .blend.
    for name in new_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and "envelope" in obj.name.lower():
            bpy.data.objects.remove(obj, do_unlink=True)

    print(f"[blender] tree_add produced: wood={wood.name if wood else None}"
          f"  leaves={leaves.name if leaves else None}",
          file=sys.stderr)
    return wood, leaves


def extract_skeleton(curve_obj) -> list:
    """Walk Sapling's curve splines → list of branches.

    Each branch dict:
        {"id": int, "points": [{"pos": [x,y,z], "radius": float}, ...]}

    We do this BEFORE converting to mesh, while the curve data still has
    per-point radii.
    """
    out = []
    for spline_idx, spline in enumerate(curve_obj.data.splines):
        pts = []
        if spline.type == "BEZIER":
            iter_pts = spline.bezier_points
        else:
            iter_pts = spline.points
        for p in iter_pts:
            co = list(p.co)[:3]
            pts.append({"pos": co, "radius": float(p.radius)})
        if len(pts) >= 2:
            out.append({"id": spline_idx, "points": pts})
    return out


def sample_olives(branches: list, n_olives: int, seed: int) -> list:
    """Pick olive attach points: tips of upper-level branches.

    We use the *last* point of branches with at least two points, weighted
    toward the leafiest splines (last in the spline array — Sapling tends
    to emit them last).
    """
    import random
    rng = random.Random(seed)
    candidates = []
    for b in branches:
        if len(b["points"]) >= 2:
            tip = b["points"][-1]
            candidates.append({"pos": tip["pos"], "branch_id": b["id"]})
    # Bias toward later branches (tips of upper levels).
    candidates = candidates[-min(len(candidates), 4 * n_olives):]
    rng.shuffle(candidates)
    return candidates[:n_olives]


def export_obj(obj, filepath: Path) -> None:
    """Convert curve→mesh if needed, smooth-shade, export selected as OBJ."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if obj.type == "CURVE":
        # Cap the bevelled tube ends so the trunk/branches aren't hollow
        # open cylinders — without this you can see through the trunk.
        try:
            obj.data.use_fill_caps = True
        except AttributeError:
            pass
        bpy.ops.object.convert(target="MESH")

    # smooth shading for nicer renderer output
    if bpy.ops.object.shade_smooth.poll():
        bpy.ops.object.shade_smooth()

    # Blender 4.x: bpy.ops.wm.obj_export. Blender 3.x: bpy.ops.export_scene.obj.
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(filepath),
            export_selected_objects=True,
            export_normals=True,
            export_uv=True,
            export_materials=False,
            forward_axis="Y",
            up_axis="Z",
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=str(filepath),
            use_selection=True,
            use_normals=True,
            use_uvs=True,
            use_materials=False,
            axis_forward="Y",
            axis_up="Z",
        )


# ─────────────────────────────────── main ────────────────────────────────────
def main() -> None:
    args = parse_args(sys.argv)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    enable_sapling()

    wood, leaves = build_tree(args)
    if wood is None:
        print("ERROR: Sapling produced no tree curve", file=sys.stderr)
        sys.exit(1)

    # Skeleton extraction must precede mesh conversion.
    branches = extract_skeleton(wood)
    n_olives = max(0, int(8 * args.leaves))
    olives = sample_olives(branches, n_olives=n_olives, seed=args.seed)

    wood_path = out_dir / f"{args.output_id}_wood.obj"
    leaves_path = out_dir / f"{args.output_id}_leaves.obj"
    sidecar_path = out_dir / f"{args.output_id}.json"

    sidecar = {
        "id": args.output_id,
        "variety": args.variety,
        "size": args.size,
        "branches_request": args.branches,
        "leaves": args.leaves,
        "detachment": args.detachment,
        "seed": args.seed,
        "pos": list(args.pos),
        "wood_obj": wood_path.name,
        "leaves_obj": leaves_path.name if leaves is not None else None,
        "branches": branches,
        "olives": olives,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    export_obj(wood, wood_path)
    if leaves is not None:
        export_obj(leaves, leaves_path)

    blend_path = out_dir / f"{args.output_id}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    n_pts = sum(len(b["points"]) for b in branches)
    print(
        f"[blender_sapling] wrote {wood_path.name} "
        f"({len(branches)} splines, {n_pts} pts) "
        f"+ {leaves_path.name if leaves else '(no leaves)'} "
        f"+ {sidecar_path.name} ({len(olives)} olives)"
    )


if __name__ == "__main__":
    main()
