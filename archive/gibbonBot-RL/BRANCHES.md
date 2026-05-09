# Branch & Version Atlas

Every branch in this repo is preserved. Nothing has been deleted or
force-merged. This file is the map: pick the right branch for the kind
of work you want to resume.

The repo went through a lot of back-and-forth (CPU vs GPU, capsule vs
mesh trees, browser vs native viewer, lean vs full UI). Each branch
represents a snapshot of one of those directions. Most have been
superseded by `walk` / `dlab-demo`, but their code and history are still
checkout-able for reference.

## Quick-pick

| Want to… | Use branch |
| --- | --- |
| Continue current work (climb-demo + olive grove) | **`dlab-demo`** |
| Train walking-to-tree on MJX/GPU | **`walk`** |
| Stable mainline (matches `walk` minus 2 perf commits) | `main` |
| Browse the project in a browser viewer (MJPEG over Flask) | `html_viewer` |
| Lean MJX training loop (GPU, minimal) | `mjx-gpu` |
| Even leaner: spawn near floor, short episodes | `lean-mjx` |
| Clean codebase without limb-capsule collisions | `no-capsules` / `clean-no-capsules` |
| Reproduce floor-penetration debugging | `clean/train-loop` |
| Tree-only navigation env | `origin/tree-navigation` |
| Simple stand-up reward (older) | `simple-reward` |
| Earliest split-scene grove work | `grove_scene` |
| Earliest portable-setup attempt | `setup-improvements` |

## Active branches

### `dlab-demo` *(current)*
Tip: `d985ad5` (latest)
- Adds an `olive` Sapling variety with realistic Mediterranean silhouette
  (multi-trunk burly base, slim proportional trunk, wide rounded canopy,
  narrow lanceolate leaves).
- `view.py` gains a scripted **`POLICY = "demo"`** mode: kinematic
  mocap (`mj_forward`) — robot trots toward the tree, climbs the trunk,
  holds at the top, loops. No physics, no training required.
- `data/tree_only.xml` for inspecting the tree alone:
  `python view_scene.py data/tree_only.xml`.

### `walk`
Tip: `d985ad5`
- 2 commits ahead of `main`: faster sim (plane collision + half substeps
  + shorter episodes) and a JAX-traced clamp fix (`jnp.maximum`).
- Underlying env `envs/walk_env_mjx.py` rewards walking to the tree.
- This is the launching point for `dlab-demo`.

### `main`
Tip: `bac08a2` — Walk-to-tree env + realistic motor damping +
capsule tree generator. Stable reference; `walk` adds 2 commits on top.

## Specialty branches (preserved, not stale)

### `grove_scene` (= `bac08a2`, same as `main`)
Where the procedural olive-grove work landed. The plan lives in
[`GROVE_MASTERPLAN.md`](GROVE_MASTERPLAN.md). The tree generator
package (`trees/`) and `gen_terrain.py` were finalised here.

### `html_viewer`
Tip: `a4c1136` — Browser-based MJPEG viewer (`serve_web.py` +
`web/viewer.html`). Useful for headless / cloud deployment where you
can't run the native MuJoCo viewer. Already includes `/grove` and
`/terrain` POST endpoints with hot-reload.

### `mjx-gpu`
Tip: `72c52b0` — Stripped-down MJX GPU training. Removed hot-reload,
removed tune mode, added `view.py` for live inspection. The leanest
version of the JAX/MJX path before the `walk` env was added.

### `lean-mjx`
Tip: `8d64d82` — Spawn near floor (curriculum), shorter episodes,
faster physics. Pre-walk era; still useful as an MJX baseline.

### `lean-training`
Tip: `a374948` — Adds a parameter-range UI on top of lean-mjx. Older
generation of the trainer.

### `no-capsules` / `clean-no-capsules`
- `no-capsules` (`c5ca157`): disables limb-self-collision via capsule
  geoms, plus an `--inherit` weights-load fix.
- `clean-no-capsules` (`de51aa5`): adds a portable setup pass on top.
  Useful if MJX-collision blow-ups bite you.

### `clean/train-loop`
Tip: `1299176` — Floor-penetration debugging variant. Worth checking
out only if you re-encounter the contact-explosion bug.

### `simple-reward`
Tip: `1c9900a` — Older simpler reward shaping (soft arm penalty),
flat directory layout, has an early `QUICKSTART.md`. Reference for
how the project looked before the `envs/` split.

### `setup-improvements`
Tip: `78af9f2` — Earliest pass on portable setup + `requirements.txt`.
Most of this is now in `main`, but the first attempt is preserved.

### `sim/mujoco-setup`
Tip: `c041951` — The original upstream remote head (default branch
on origin). Kept as the fork-from point.

### `origin/tree-navigation` (remote-only)
Tip: `cd1403c` — Tree navigation feature for MJX training, predecessor
of the current walk-to-tree env. Not checked out locally; access with
`git checkout origin/tree-navigation -b tree-navigation` if you want it.

## How to use this atlas

```bash
git branch -a                    # see every branch
git log <branch> --oneline -20   # what's on it
git diff main...<branch>         # what's different from main
git checkout <branch>            # work on it
```

If a commit ever goes "missing" (e.g. after a rebase elsewhere), every
branch tip is also tagged `archive/<branchname>` — see `git tag | grep
^archive/`. Tags don't get garbage-collected, so the history is safe.

## When you make a new direction

Don't force-merge into `walk`. Instead:

1. Branch from `walk`: `git checkout -b feature/<short-name> walk`.
2. Land your work as commits.
3. Update this file's "Quick-pick" table with the new entry.
4. Tag the tip when done: `git tag archive/<branchname>`.

That way the codebase keeps growing without losing any past direction.
