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

## Modules

| module | role |
|--------|------|
| `varieties.py` | `young` / `mature` / `ancient` preset dicts |
| `primitives.py` | builds a tree as MuJoCo capsules + ellipsoids and emits `grove.xml` |
| `generate.py` | CLI orchestrator — parses args, samples terrain height, dispatches to a builder, writes `grove.xml` |

## Roadmap (matches `GROVE_MASTERPLAN.md`)

| status | track | adds |
|--------|-------|------|
| ✅ Phase 1 | primitives | capsule trunk + branches, ellipsoid foliage, trunk collision |
| ⏳ 2a | hot-reload | `serve_web.py` `/grove` endpoint + browser slider panel |
| ⏳ 2b | `blender_sapling.py` | Blender Python script using Sapling Tree Gen, exports OBJ wood/leaves + branch JSON; `generate.py` invokes Blender as subprocess and falls back to primitives if absent |
| ⏳ 2c | `articulate.py` | branches as nested `<body>` chains with hinge stiffness so they bend on contact |
| ⏳ 2d | `olives.py` | dynamic olive bodies + breakable weld constraints |

Generated mesh files (when Blender lands) are saved under `data/trees/`,
which is gitignored. A `cache_index.json` maps `(params)` → file paths
so repeat slider drags reuse cached OBJs.
