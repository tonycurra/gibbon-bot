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

### Phase 2 — Parameterized trees   `Status: TODO`
- [ ] Variety presets (young / mature / ancient) tune: trunk diameter, height,
      branch count, canopy spread, gnarliness.
- [ ] CLI flags: `--variety`, `--size`, `--detachment` (canopy openness 0..1),
      `--leaves` (density 0..1), `--branches` (extra subdivision steps).
- [ ] Browser slider panel (in `html_viewer` branch later) for each.
- [ ] Seed flag for reproducibility.

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
