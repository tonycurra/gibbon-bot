# Scene geometry — facts, formulas, gotchas

Source of truth for the robot's physical dimensions and how they map into
`scene.xml` / `scene_minimal.xml`. Read this **before** editing any limb
position, quat, or joint range.

## Units

**SI everywhere.** All `pos`, `size`, mesh `scale`, and lengths in the XML are
**meters**. STL files are authored in **millimeters**; the `<mesh scale="0.001 0.001 0.001">` on each mesh asset converts them. **Never** edit a position
without remembering the world is in meters.

The compiler is `angle="radian"` in `scene_minimal.xml`. All `range` and
`ctrlrange` values on joints/actuators are in radians. Conversion table:

| Joint | Range (rad) | Range (deg) |
|---|---|---|
| shoulder_yaw / hip_yaw | ±π (`±3.1416`) | ±180° |
| shoulder_pitch / hip_pitch | ±π/2 (`±1.5708`) | ±90° |
| elbow / knee | `-1.0` to `2.95` | ≈ -57° to 169° |
| forearm_roll / shin_roll | ±π/2 (`±1.5708`) | ±90° |
| wrist / ankle | ±π (`±3.1416`) | ±180° |
| gripper / foot | `0` to `1.5` | 0 to ≈86° |

## Real-robot dimensions (roarm M3 — measured from the STLs)

```
base_link bracket : 99 × 82 × 73 mm   (mesh x × y × z, scaled to m for the sim)
link1             : 55 × 72 × 60 mm
link2 (humerus)   : 257 × 56 × 79 mm   ← the long bone
link3             : 28 × 165 × 36 mm
link4             : 54 × 62 × 40 mm
link5             : 47 × 36 × 121 mm
gripper_link      : 78 × 25 × 39 mm
approx full reach : ≈ 546 mm (sum of joint offsets, fully straight)
```

To re-measure at any time:

```python
import trimesh
m = trimesh.load('data/meshes/base_link.stl', force='mesh')
print(m.bounds, m.bounds[1] - m.bounds[0])
```

## Source of truth for the kinematic chain

`data/scenes_archive/ape_early.urdf` is the **original URDF** from the design.
The MJCF in `data/scene_minimal.xml` was derived from it. Whenever
joint-origin geometry is ambiguous, the URDF wins. Mount positions per URDF:

| Limb | xyz (m) | rpy around Y (rad) |
|---|---|---|
| FL | `0.12  0.10 -0.05` | `+0.785` (45° forward) |
| FR | `0.12 -0.10 -0.05` | `+0.785` (same as FL) |
| BL | `-0.12  0.10 -0.05` | `-0.785` (45° backward) |
| BR | `-0.12 -0.10 -0.05` | `-0.785` (same as BL) |

In the URDF, **left and right are NOT mirrored** — every limb is the same
real roarm M3 arm. The MJCF version in scene_minimal.xml uses different,
hand-edited mount quats (90° axis-swaps + L/R mirroring on the arms), but
the underlying physical limb is identical across all 4 mounts.

## "Base-to-base" distance — the most common foot-gun

When the user says "200 mm base-to-base" they almost always mean:

> **The gap between the two inner faces of the base_link meshes (the face
> furthest from where the arm extends).**

NOT center-to-center body positions. Compute it like this:

- `base_link.stl` bounds: mesh-`z` ∈ `[-71.1, 1.5]` mm. The bottom face is
  at `mesh_z = −71.1 mm = −0.0711 m`.
- The FL `link1` body quat `(0.5, -0.5, 0.5, 0.5)` rotates mesh-`z` →
  world-`+y`. So the inner face is at `world_y = body_y − 0.0711`.
- FR's mirror quat `(0.5, 0.5, 0.5, -0.5)` puts the inner face at
  `world_y = body_y + 0.0711`.

**Formula** (for the front arms, by symmetry):

```
gap_between_inner_faces = 2 × (body_y − 0.0711)
⇒ body_y = gap / 2 + 0.0711
```

| Desired gap (mm) | `body_y` (m) | Center-to-center (mm) |
|---|---|---|
| 0   (touching)   | 0.0711 | 142  |
| 100              | 0.1211 | 242  |
| **200 (current)** | **0.1711** | **342** |
| 300              | 0.2211 | 442  |
| 400              | 0.2711 | 542  |

The **URDF default** (`body_y = ±0.10`) gives a 58 mm gap — the bases are
almost touching. The current scene_minimal.xml uses **`body_y = ±0.1711`**
for a clean **200 mm gap**. If you change one, recompute via the formula
and update both this table and the XML comment.

## Joint rest pose — never use `ref` for orientation offsets

If you want a limb to start at a pose that's rotated relative to the URDF
default (e.g. back legs facing backward), **bake the rotation into the body
`quat`**, not into the joint's `ref` attribute.

`ref` sets the joint angle at `qpos = 0` but does NOT change the actuator's
default command. With a position actuator at `ctrl = 0` and `ref = π`, the
scene loads at `qpos = ref = π`, but the controller immediately drives it
to `qpos = 0` — you get a 180° yaw kick the moment the sim starts.

The current scene_minimal.xml encodes the 180° back-leg rotation in the
`bl_link1` and `br_link1` body quats:

- Original (URDF-derived) quat: `(0.7071068, 0, -0.7071068, 0)`
- After post-multiplying by 180° around local Z (joint axis):
  `(0, -0.7071068, 0, 0.7071068)`

`qpos = 0` is now the rest pose; no startup motion.

## Torso construction

The torso is **two box geoms** in a single `torso` body — one slab per
limb pair — sized so every face is flush with the corresponding base_link
mounting face. All values in body-local meters; the body sits at world
`pos="0 0 0.5310"`.

| Geom | `pos` (body-local) | `size` (half-extents) | World extent |
|---|---|---|---|
| `torso_legs` (brown) | `-0.02485 0 -0.04325` | `0.10405 0.1408 0.04925` | X∈[-0.1289, +0.0792], Y∈[±0.1408], Z∈[+0.4385, +0.5370] |
| `torso_arms` (blue)  | `+0.12 0 -0.04325`    | `0.0408 0.10 0.04925`   | X∈[+0.0792, +0.1608], Y∈[±0.10], Z∈[+0.4385, +0.5370] |

Verified contacts (all within 0.0001 m):
- `torso_legs.−X` face (`x=−0.1289`) ↔ BL/BR inner X face — leg `base_link`
  mounting planes (after the 180° rotation around link2 axis baked into the
  mount-geom quat).
- `torso_arms.+Y` and `−Y` faces (`y=±0.10`) ↔ FL/FR inner Y face — arm
  `base_link` mounting planes.
- `torso_legs.+X` ↔ `torso_arms.−X` at `x=+0.0792` — the two slabs share a
  face, forming a single visually-continuous structure.
- Z extent of both slabs (`+0.4385 → +0.5370`, 98.5 mm tall) matches the
  full Z extent of every base_link mounting face.

The two slabs were verified by loading the model and computing the world
AABB of each `base_link` mesh from `model.mesh_vert + data.geom_xmat +
data.geom_xpos`. If you change any limb position or quat, re-run that
check (script lives inline in the commit that introduced this section).

## Files

- `data/scene_minimal.xml` — inspect-only scene used by `look.py`. 4 limbs,
  floor, one no-shadow light, no fog/haze/skybox/tree. Compiler in radians.
- `data/scene.xml` — full training scene. Currently the **pre-mk2 (old
  4-limb)** architecture. The mk2 torso-mesh variant is backed up at
  `data/scene.xml.mk2_backup`.
- `data/scenes_archive/` — historical scene variants. `ape_early.urdf` is
  the URDF source of truth.
- `data/meshes/` — STL meshes, byte-identical to
  `data/roarm_ws/src/roarm_main/roarm_description/meshes/roarm_m3/`
  (verified by md5; do not modify).
