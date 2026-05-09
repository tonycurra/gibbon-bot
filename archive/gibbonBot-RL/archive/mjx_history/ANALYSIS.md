# mjx_history — what each version actually does

## TL;DR — the bug

**Working viewers (every version before `6ad6589`):**
```python
for _ in range(PHYSICS_STEPS_PER_ACTION):
    mj_step(model, data)
    viewer.sync()              # sync after EVERY step
    if sleep_dt > 0:
        time.sleep(sleep_dt)   # tiny per-step sleep = 0.002s
```

**Broken viewers (`6ad6589` and everything I wrote on top of it):**
```python
last_sync_t = time.time()
for _ in range(PHYSICS_STEPS_PER_ACTION):
    mj_step(model, data)
    if time.time() - last_sync_t >= 1/30:   # rate-limited sync
        viewer.sync()
        last_sync_t = time.time()
viewer.sync()
behind = action_sim_dt - (time.time() - action_wall_t0)
if behind > 0:
    time.sleep(behind)         # ONE big sleep at the end — viewer frozen during it
```

The "rate-limited sync + one big sleep at end" pattern is what makes the viewer feel like it's running at 2fps. With `PHYSICS_STEPS_PER_ACTION = 200` and timestep = 0.002s, the action represents 0.4s of sim time. Old code made 200 viewer.sync() calls and 200 tiny sleeps inside the action — perfectly smooth. New code makes ~12 sync calls then sleeps for hundreds of ms with the viewer frozen.

The fix isn't the contype/conaffinity stuff. It's reverting to **sync-and-sleep per step**.

---

## Watcher versions (oldest → newest)

### `cf6614d` — 106 lines — primordial
Uses the gym `StandEnv` to drive playback. No direct MuJoCo loop. JIT'd `act()`. Polls for new checkpoints. No contype changes (uses scene as-is). Relies on `StandEnv` for stepping — which runs `PHYSICS_STEPS_PER_ACTION` mj_steps internally then syncs once per env.step. At low PHYSICS_STEPS this was fine.

### `55b1a8b` — 122 lines
Same as cf6614d but slightly cleaner. Still goes through `StandEnv`. Adds `--no-settle` reset. No collision tweaks.

### `2ea623f` — 177 lines — **drives MuJoCo directly for the first time**
Direct `mj_step` loop, with the **per-step sync + per-step sleep** pattern. Speed via `REALTIME_FACTOR` param. **No contype changes — runs scene as-is.** This is the canonical good pattern.

### `dfb4c86` — 216 lines — adds motor speed model
Same per-step sync+sleep pattern. Adds `motor_pos` substep clamping (so joints can't teleport per action), reads `MOTOR_MAX_SPEED` from params. `VIEWER_SPEED` param replaces `REALTIME_FACTOR`. **No contype changes.**

### `6496c82` — 312 lines — adds live stats panel
Same loop. Adds rolling per-episode stats deque, banner printing, ep_real_s estimate. Per-step sync+sleep still intact. **No contype changes.**

### `dfae092` — 322 lines — adds JSON live_stats sidecar
Same loop. Writes JSON to `runs/live_stats.json` for the now-removed params UI. Adds reward estimate per step. Per-step sync+sleep intact. `sleep_dt = timestep` (always real-time). **No contype changes.**

### `6ad6589` — 342 lines — **FIRST broken version (the regression)**
- Switches to **time-based sync** (`if time.time() - last_sync_t >= 1/30`) inside the loop
- Switches to **single trailing sleep** (`time.sleep(behind)`)
- Adds the contype=2/conaffinity=1 fix for **mesh geoms only**

The contype change was correct (mesh-mesh self-collision was indeed slow). But the sync/sleep refactor broke real-time playback — that's why this version "looked like it worked at first" but actually became choppy at high PHYSICS_STEPS.

### `72c52b0` — 232 lines — my "simplification"
Same broken sync/sleep pattern from 6ad6589. I just stripped the live_stats writes. Inherits the bug.

### `c757466` — 234 lines — my "fix" that made it worse
Changed contype filter from `geom_type == MESH` to `geom_type != PLANE`, which now overrides capsules, spheres, and the box torso too. Didn't help (the actual bug was sync/sleep). Possibly contributed to the "skeleton" appearance the user saw — though contype shouldn't affect rendering, the sweeping override may have hit something I don't understand.

---

## stand_env_mjx versions (training env, separate from the viewer)

### `67f032a` — 126 lines — first MJX env
- `PHYSICS_STEPS_PER_ACTION = 12`
- Sets `contype=0, conaffinity=0` for mesh geoms (totally disables mesh collision in the training env — necessary for MJX, which would explode otherwise)

### `c5ca157` — 172 lines — same approach, more reward terms
Same `contype=0` mesh disabling. Adds richer reward terms.

### `cf6614d` — 172 lines — almost identical to c5ca157
Trivial differences. Same physics setup.

### `a5af3b5` — 271 lines — last big rewrite
Same `contype=0` mesh disabling. Adds proximity penalty, more reward shaping. Still `PHYSICS_STEPS_PER_ACTION = 12`.

**Key training-env detail:** these versions all use `contype=0, conaffinity=0` (no mesh collision at all in training). The current `envs/stand_env_mjx.py` does the same. This is correct — for MJX, you need to fully disable mesh collision to keep JIT compile time tractable.

---

## train_mjx versions

### `67f032a` — 337 lines — initial release
Single training mode. Auto-pick `N_ENVS = 1024`. Hardcoded hyperparameters.

### `c5ca157` — 342 lines — adds working `--inherit`
Fixes weight-loading for resume.

### `cf6614d` — 342 lines — adds watch_mjx subprocess spawn
Same as c5ca157. Spawns viewer.

### `train_mjx_.py` — 332 lines — intermediate
Different layout. Pre-major-refactor.

### `a5af3b5` — 0 lines
Empty placeholder.

---

## What to do

The viewer was working before `6ad6589`. Restore the **per-step sync + per-step sleep** pattern from `dfb4c86` or `dfae092`. The contype/conaffinity overrides in viewer code can stay or go — they're not the source of the lag. The core fix is two lines in the inner loop:

```python
# Replace this:
if time.time() - last_sync_t >= SYNC_INTERVAL: viewer.sync(); last_sync_t = time.time()
# ... and this:
behind = action_sim_dt - (time.time() - t0)
if behind > 0: time.sleep(behind)

# With this (per step, inside the loop):
viewer.sync()
time.sleep(model.opt.timestep)   # ≈ 0.002s, real-time pacing
```

That's it. With 200 PHYSICS_STEPS_PER_ACTION × 0.002s = 0.4s per action, you get 200 viewer frames per action — way more than smooth.
