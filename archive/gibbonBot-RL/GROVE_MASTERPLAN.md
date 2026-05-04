# Olive Grove Masterplan

> Virtual olive grove for an ape robot to learn locomotion and climbing in,
> rendered as realistically as MuJoCo's renderer allows, GPU-trained via MJX.

This file is the cross-session plan. Update **Status** lines as work lands so
the next session can pick up cold.

---

## North Star

A procedural Mediterranean olive grove (terrain + many trees) where the robot
can train. Trees vary by **variety** (young / mature / ancient), **size**,
**detachment** (canopy openness), **spawning** (count + distribution),
**leaves** (density + colour), **branches** (count + taper + angle). Robot
should be able to climb / brachiate / push off branches.

## Decisions (do not relitigate)

- **Renderer**: MJX for training, browser MJPEG (`html_viewer` branch) for
  inspection only. Anything that ships must work in MJX first.
- **Trees are static + visual-only-mesh**:
  - No joints on trees (would shift `qpos[7:]` slicing in `stand_env_mjx.py`).
  - Mesh geoms get `contype=0, conaffinity=0` automatically at load
    (`envs/stand_env_mjx.py:85-88`). Trees inherit this — visual mesh only.
  - For *physical* tree interaction (climbing): use **capsule/box collision
    proxies** for trunk + main branches alongside the visual mesh.
- **Body indices are sacred**: `_ARM_GROUPS_NP = [2..25]` in
  `stand_env_mjx.py`. Anything that adds a `<body>` element to scene.xml
  *before* the arm chains breaks training silently. Tree geoms must live
  directly under `<worldbody>` (no `<body>` wrapper) — same pattern as `floor`
  and `terrain_visual`.
- **Scene split via `<include>`**: MuJoCo 3.8 supports `<include file="..."/>`.
  Top-level `data/scene.xml` becomes a thin manifest; robot, terrain, and
  grove live in their own files. Generated content (terrain, grove) is
  rewritten in place by their respective scripts.
- **Procedural assets are committed to disk, not git**: `data/*.obj`,
  `data/*.png` are gitignored. The *generators* (`gen_terrain.py`,
  `gen_grove.py`) are the source of truth.

## Architecture

```
data/
  scene.xml         ← thin manifest: <include>s of the three below + globals
  robot.xml         ← <body name="torso">…</body>: hand-edited, stable
  terrain.xml       ← <hfield>, floor geom, terrain_visual mesh, sun, sky
  grove.xml         ← <mesh> assets + tree geoms; rewritten by gen_grove.py
  terrain.png        (gitignored — heightfield)
  terrain_surface.obj (gitignored — top-surface mesh)
  trees/<id>.obj     (gitignored — per-tree procedural meshes)

gen_terrain.py      ← already exists, patches terrain.xml in place
gen_grove.py        ← NEW: procedural olive trees → grove.xml + .obj files
```

## Phases

### Phase 0 — Scaffolding   `Status: DONE (commit pending)`
- [x] Split `data/scene.xml` → `scene.xml` (manifest) + `robot.xml` + `terrain.xml`.
- [x] Verify MJX still loads — `nbody=26, ngeom=39, nu=24, nq=31` matches.
- [x] `gen_terrain.py` now patches `terrain.xml` + `robot.xml` (split spawn z).
- [ ] Verify browser viewer still loads (deferred to next session — branch
      `html_viewer` is on a previous commit; will rebase or cherry-pick the
      split when grove work needs it).

### Phase 1 — One olive tree, ugly but real   `Status: DONE`
- [x] `gen_grove.py --variety {young|mature|ancient}` writes `data/grove.xml`.
- [x] Trunk: 4 stacked capsules along a slightly bent gnarled curve, all
      collision-enabled (climbable). Tapered base→top.
- [x] Branches: 3–6 outward+up capsules (visual-only for now; collision in
      Phase 5 climbing pass).
- [x] Foliage: ellipsoid clusters at trunk crown + each branch tip, oblate.
- [x] Flat olive-green / bark-brown rgba (no textures).
- [x] `scene.xml` includes `grove.xml`. MJX loads with `nbody=26` (no shift),
      `ngeom=54` (was 39, +15 tree geoms).
- **Note**: switched from "OBJ mesh" to MuJoCo primitives for Phase 1 — same
  visual outcome, far less code, faster to iterate. OBJ-based detailed shapes
  deferred to Phase 4 (realism pass).

### Phase 2 — Realistic, flexible, interactive trees   `Status: in progress`

Reorganised: tree-gen code lives in `trees/` package. CLI is
`python -m trees.generate`. Full plan in `.claude/plans/...` — broken
into four shippable sub-tracks:

#### 2a — Browser hot-reload pipe   `Status: DONE`
- [x] `/grove` POST endpoint in `serve_web.py`, mirrors `/terrain`.
- [x] Render-thread reload: subprocess `python -m trees.generate ...`
      → `_swap_model()` helper (extracted, shared with terrain) →
      reloads `MjModel`, resets to keyframe, recreates renderer.
- [x] Request handling: terrain and grove blocks run sequentially in
      the render-thread loop. If both fire in the same tick, terrain
      reloads first then grove triggers a second swap. Correct, mildly
      wasteful — fine until we measure.
- [x] Tree-panel in `web/viewer.html`: variety chips (young / mature /
      ancient) + sliders for size / branches / leaves / detachment /
      seed. 400ms debounce. Header shows `building…` / `ready` /
      `error` from `grove_status` in `/state`.
- [x] Initial slider values sync from `/state.grove` on page load.

#### 2b — Blender + Sapling Tree Gen   `Status: TODO`
- [ ] One-time install: `apt install blender`.
- [ ] `trees/blender_sapling.py` — runs *inside* Blender via
      `blender --background --python ...`. Sapling Tree Gen produces
      olive-shaped mesh, exports `data/trees/<id>_wood.obj` +
      `<id>_leaves.obj` + `<id>.json` (branch midlines, olive points).
- [ ] `trees.generate` becomes orchestrator: invokes Blender, reads
      JSON sidecar, emits `grove.xml` referencing the OBJs.
- [ ] Graceful fallback to `trees.primitives` if Blender absent.
- [ ] Cache: `data/trees/cache_index.json` skips re-runs for repeated
      params.

#### 2c — Articulated flexible branches   `Status: TODO`
- [ ] `trees/articulate.py` — branches as nested `<body>` chains with
      **ball joints** (sway in all directions, not just one plane) +
      stiffness + damping. Capsule per segment with collision so the
      robot can both bend and climb.
- [ ] Investigate `<composite type="cable">` as a shortcut: it auto-
      generates the chain of cylinders + ball joints. Use if the API
      gives us per-segment collision/material control, else hand-emit.
- [ ] **MJX edit**: `envs/stand_env_mjx.py:197` slices `qpos[7:]` →
      `qpos[7:7+24]` (robot joints only). Same for `qvel`.
- [ ] `--rigid-branches` flag for training scenes (skip articulation
      to keep DOF count down). Default articulated for play / eval.

#### 2d — Dynamic olives with breakable welds   `Status: TODO`
- [ ] `trees/olives.py` — emits `<body><freejoint>` + sphere geom +
      `<equality><weld>` to parent branch.
- [ ] Add a `<force>` (and optionally `<torque>`) sensor on each weld
      so we read stem load by sensor name instead of decoding
      `data.efc_force` indices.
- [ ] Step hook in env: when sensor magnitude > stem strength,
      `data.eq_active[i] = 0` → olive becomes free.
- [ ] `--olives` flag, default off for training.
- [ ] Olive collision bitmask: only matches robot hands (avoid contact
      explosion with branches/ground while welded).

### Phase 6 (new) — Shaker harvest curriculum   `Status: TODO`
Real olive harvesting is mostly mechanical shakers, not climbing.
A shaker mode is also a much easier first RL task than climb-and-pick.
- [ ] `<velocity>` or `<position>` actuator at trunk base. Driven by
      `target = A * sin(2π f t)` with f ≈ 10–25 Hz, A small.
- [ ] Reward = number of olives detached per second.
- [ ] Curriculum: shaker mode → climb to specific olive → manual pick.

### Phase 3 — Grove   `Status: TODO`
- [ ] `--count N --layout grid|poisson|cluster` distributes trees over the
      heightfield. Spawn z = sample terrain height at (x, y).
- [ ] Per-tree variety/size jitter so the grove looks natural.
- [ ] LOD: distant trees use a single-billboard mesh (3–4 quads) instead of
      full geometry. Threshold by distance to robot spawn.
- [ ] Test: 100 trees compiles in MJX without OOM.

### Phase 4 — Realism pass   `Status: TODO`
- [ ] Bark texture (procedural noise → PNG) on trunk.
- [ ] Leaf cluster material with subtle vertex-colour variation.
- [ ] Soft ground shadows from trees (MuJoCo casts shadows from `castshadow`
      lights — already on for `sun`).
- [ ] Match palette to soil preset (mud → darker bark? sandy → silvery leaves?).

### Phase 5 — Climbing rewards   `Status: TODO`
- [ ] Reward shaping in `envs/stand_env_mjx.py` for height gained while in
      contact with tree collision geoms.
- [ ] New env preset that drops the robot at base of a tree.

## MJX checklist (run before merging any phase)

1. `python train_mjx.py --envs 256` boots without error and survives 1k steps.
2. `python view.py` shows the new scene with a healthy robot pose.
3. `python serve_web.py` (on `html_viewer` branch) shows it in browser too.
4. No new hardcoded indices were introduced in `stand_env_mjx.py`.

## Cross-session resumption

When picking this up cold:
1. `git branch --show-current` — should be `grove_scene` for grove work,
   `html_viewer` for browser-viewer work. Two parallel tracks.
2. Read this file's **Status** lines. The next `TODO` phase is the next thing.
3. Skim `envs/stand_env_mjx.py` lines 30–90 to refresh on the index
   constraints.
4. Run the MJX checklist above before starting; if it's already broken,
   diagnose that first.

## Open questions (deferred until they bite)

- Wind / leaves rustling: would need MuJoCo dynamic bodies or shader tricks.
  Skip for now.
- Fruit (olives) as collision objects? Tiny static spheres are cheap, could
  add as Phase 6.
- Multi-grove maps (terraces, irrigation rows)? After Phase 3.
