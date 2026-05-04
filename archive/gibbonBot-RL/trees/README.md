# trees/ — Olive tree generation pipeline

Self-contained package that emits `data/grove.xml`. Consumed by
`data/scene.xml` via `<include file="grove.xml"/>`. Hot-reload friendly:
`serve_web.py` POSTs `/grove` → invokes `python -m trees.generate ...`
→ MJX model is reloaded.

## Run

```bash
python -m trees.generate --variety mature
python -m trees.generate --variety ancient --size 1.3 --detachment 0.6
python -m trees.generate --variety young --leaves 0.8 --branches 3
```

CLI flags (see `generate.py`):

| flag | range | meaning |
|------|-------|---------|
| `--variety` | young / mature / ancient | preset for trunk thickness, height, gnarl, branch count |
| `--size` | scalar | overall multiplier on dimensions |
| `--branches` | int | override branch count (else variety preset) |
| `--leaves` | 0–1 | foliage cluster density |
| `--detachment` | 0–1 | canopy spread (0=tight, 1=open) |
| `--pos` | x y | base position (z is sampled from the live heightfield) |
| `--seed` | int | rng seed |

## Backends

`generate.py` automatically picks the best available backend:

1. **Blender + Sapling Tree Gen** (preferred) — invokes Blender headlessly
   via `blender --background --python trees/blender_sapling.py -- ...`.
   Produces real OBJ meshes saved under `data/trees/`.
2. **Primitives fallback** — pure-Python capsules + ellipsoids when
   Blender is missing on PATH or fails. Force with `--primitives`.

Install Blender once: `sudo apt install blender` (Blender 4.x recommended).

## Modules

| module | role |
|--------|------|
| `varieties.py` | `young` / `mature` / `ancient` preset dicts (used by both backends) |
| `primitives.py` | Pure-Python tree as MuJoCo capsules + ellipsoids; emits primitive grove.xml |
| `blender_sapling.py` | Runs *inside* Blender — Sapling tree → wood OBJ + leaves OBJ + skeleton JSON |
| `generate.py` | Orchestrator — picks backend, samples terrain height, emits `grove.xml` (mesh-based or primitive) |

## Outputs

- `data/grove.xml` (always) — MJCF fragment included by `data/scene.xml`.
- `data/trees/tree_<id>_wood.obj` (Blender path) — trunk + branches mesh.
- `data/trees/tree_<id>_leaves.obj` (Blender path) — leaf cards.
- `data/trees/tree_<id>.json` (Blender path) — branch midline coordinates
  + olive tip positions. Consumed by Track 2c (articulated branches) and
  Track 2d (dynamic olives).

`data/trees/*` is gitignored.

## Roadmap (matches `GROVE_MASTERPLAN.md`)

| status | track | adds |
|--------|-------|------|
| ✅ Phase 1 | primitives | capsule trunk + branches, ellipsoid foliage, trunk collision |
| ✅ 2a | hot-reload | `serve_web.py` `/grove` endpoint + browser slider panel |
| ✅ 2b | `blender_sapling.py` | Blender Sapling export → OBJ + JSON skeleton; primitive fallback intact |
| ✅ 2c | `articulate.py` | longest N branches become nested `<body>` chains with ball joints (stiffness + damping). Cap via `--articulate-max` (default 8). Leaf clusters attach to branch tips so foliage moves with bend. `--rigid-branches` opts out for training. |
| ⏳ 2d | `olives.py` | dynamic olive bodies + breakable weld constraints (force/torque sensor + `eq_active` toggle) |
