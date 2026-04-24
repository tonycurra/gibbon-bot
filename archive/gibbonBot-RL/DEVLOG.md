# Dev Log — Robotic Ape Simulation

## Rules
- Never edit files under `roarm_ws/roarm_description/urdf/` or `meshes/` directly.
- All simulation-specific files live at the repo root or in `sim/`.
- Every meaningful change gets a git commit on a feature branch.

---

## 2026-04-22 — Initial MuJoCo setup (branch: `sim/mujoco-setup`)

### What was done
1. **Read** the existing ROS2 package (`roarm_description`) and `QUICKSTART.md`.
2. **Created `roarm_ws/roarm_description/ape.urdf`** — a standalone URDF (no ROS/xacro
   dependencies) assembling the full 4-limb ape robot:
   - Root link: `torso` (free-floating box, 30×20×10 cm, 1.0 kg)
   - 4 arms mounted with fixed joints at torso corners, prefixes `fl_` `fr_` `bl_` `br_`
   - All joint/link names from the original `roarm_m3.xacro` preserved verbatim, just prefixed
   - Mesh paths changed from `$(find roarm_description)/...` to relative `meshes/roarm_m3/...`
   - Motor limits filled in from STS3215 servo datasheet: **effort 2.94 Nm, velocity 6.28 rad/s**
     (was `effort="0" velocity="0"` in the original, which means no force simulation)
   - Gripper joints get half torque (1.47 Nm) since they're lighter-duty

3. **Created `view_robot.py`** — drop-in MuJoCo viewer. Run with `python view_robot.py`.

4. **Original files untouched**: `roarm_m3.xacro`, all STL meshes, launch files.

### Mount orientations (initial)
| Arm | xyz on torso | rpy |
|-----|-------------|-----|
| fl_ | 0.12  0.10 -0.05 | 0  0.785 0 |
| fr_ | 0.12 -0.10 -0.05 | 0  0.785 0 |
| bl_ |-0.12  0.10 -0.05 | 0 -0.785 0 |
| br_ |-0.12 -0.10 -0.05 | 0 -0.785 0 |

---

## 2026-04-22 — Fix spawn angle (commit `56cb352`)

**Problem**: torso spawned at z=0 (ground level); arms immediately collided with the floor,
producing a violent sideways shove — robot appeared at an angle on launch.

**Fix**: added `data.qpos[2] = 0.5` + `mj_forward()` in `view_robot.py` before the viewer
loop to lift the torso 0.5 m above ground at t=0.

---

## 2026-04-22 — Arm mount angle 45° → 90° (commit `ac41507`)

Changed `rpy` pitch on all 4 mount joints from `±0.785` rad (45°) to `±1.5708` rad (90°).
Arms now point straight down from the torso instead of angled forward/backward.

Joints changed: `torso_to_fl_base_link`, `torso_to_fr_base_link`, `torso_to_bl_base_link`,
`torso_to_br_base_link` in `ape.urdf`.

---

## 2026-04-22 — Joint damping added (commit `62e6b88`)

**Problem**: all joints had no `<dynamics>` element → zero damping in MuJoCo → frictionless
hinges that oscillate forever (energy never dissipates).

**Fix**: added `<dynamics damping="0.5"/>` to all 24 revolute joints via `replace_all` on the
`velocity="6.28"/>` pattern.

---

## 2026-04-22 — Floor, gravity, freejoint, build pipeline (commit `1e388fd`)

Three separate problems fixed together:

**1. No floor / robot falling forever**
MuJoCo has gravity by default but no ground plane unless you define one. Added a floor geom
(`type="plane"`) in `build_scene.py` post-processing.

**2. Torso fixed to world (not free-floating)**
MuJoCo's URDF loader welds the root link to the world by default. Confirmed by checking
`model.nq=24` (should be 31 for 6DOF base + 24 arm joints).
Fix: added `<link name="world"/>` + `<joint type="floating">` from world → torso in `ape.urdf`.
Result: `nq=31, nv=30` — torso is now free-floating.

**3. Damping raised 0.5 → 2.0**
0.5 was still too low; joints continued to oscillate visibly. Bumped to 2.0 Nm·s/rad.

**New file: `build_scene.py`**
One-shot converter: loads `ape.urdf` → saves MJCF → patches XML to add:
- floor geom with friction
- `meshdir` compiler attribute (fixes mesh path resolution when scene.xml is at repo root)
- `home` keyframe (torso z=0.5, qw=1, all arm joints=0)

`scene.xml` is the generated output. `view_robot.py` now loads `scene.xml` and resets to the
`home` keyframe instead of manually setting `data.qpos`.

**Workflow going forward**:
```
edit ape.urdf  →  python build_scene.py  →  python view_robot.py
```

---

## 2026-04-22 — Position actuators + checkerboard floor (commit `a404e0b`)

**Problem**: `model.nu=0` — no actuators, so the viewer's Ctrl+A panel was empty and joints
couldn't be driven manually.

**Fix**: `build_scene.py` now iterates all revolute joints and emits a `<position>` actuator
for each one:
- `kp=50` (proportional gain)
- `ctrlrange` from `jnt_range` (position limits)
- `forcerange` from `jnt_actfrcrange` (effort limits already stored from URDF `<limit effort>`)
- Result: `nu=24`, one actuator per joint, correct torque limits wired in

**Also**: replaced the grey plane with a black/white checkerboard texture for easier visual
reference. `scene.xml` gets a `<texture builtin="checker">` + `<material>` + floor geom with
`material="floor_mat"`.

---

## 2026-04-22 — Agricultural terrain + test_joints.py (commit `0dcff40`)

**New file: `terrain.py`**
Generates a 64×64 heightfield that looks like plowed farmland:
- `hills`: two overlapping low-frequency sine waves
- `furrows`: medium-frequency ridges along one axis (`0.5 * sin(X * 2.5)`)
- `noise`: small gaussian noise for surface texture
- Values normalized to [0, 1]; MuJoCo scales by `max_height=0.5 m`

`terrain.apply(model)` writes the array into `model.hfield_data` at runtime.
Must be called after `from_xml_path` and before `MjData` is created.

**`build_scene.py` changes**:
- Removed flat plane, added `<hfield name="terrain">` asset (64×64, 10×10 m, max 0.5 m)
- Added grass checker texture: two shades of green (`rgb1=0.18 0.42 0.09`, `rgb2=0.28 0.55 0.14`)
- Home keyframe z raised `0.5 → 1.0 m` (must clear 0.5 m max terrain height)

**New file: `test_joints.py`**
Dedicated manual joint tester:
- Gravity off (`model.opt.gravity[:] = 0`)
- `kp` overridden to 200 for crisp slider response
- Torso velocity zeroed each step (pins base in place)
- Prints all 24 actuator names + ranges on launch
- Usage: `python test_joints.py` → Ctrl+A in viewer → drag sliders

---

## 2026-04-22 — Friction fix: robot sliding on terrain (commit `ba8d127`)

**Problem**: robot was visibly sliding on the grass terrain even after landing.

**Root cause**: MuJoCo computes effective contact friction as `min(geom_A, geom_B)`.
The terrain had `friction=1.2` but robot geoms inherited MuJoCo's default of `1.0`,
so effective friction was `1.0`. At oblique contact angles (arms landing at ~90° from
torso, gripper tips hitting sloped terrain), lateral forces exceed what 1.0 can resist.

**Fix in `build_scene.py`**:
- Added `<default class="robot"><geom friction="2.0 0.01 0.001" condim="4"/></default>`
- Tagged all non-terrain geoms with `class="robot"` in post-processing loop
- Terrain friction also raised to `2.0 0.01 0.001`
- `condim` raised `3 → 4` on both sides: condim=4 adds spinning friction (resists torque
  at contact point), condim=3 only handles sliding

**Note**: some residual sliding remains — this is expected without an active stance
controller. The robot has no policy telling feet to stay planted. Friction fix reduces
passive drift; active control is needed to fully eliminate it (RL phase).

---

## 2026-04-22 — Static tree obstacle (commit after ba8d127)

Added a simple tree 5 m from the robot spawn at world pos (5, 0, 0).
Implemented as a static body (no joint = welded to world) with two geoms:
- `trunk`: brown cylinder, radius 0.12 m, 2 m tall (half-height 1.0), centre at z=1.0
- `canopy`: dark green sphere, radius 1.3 m, centre at z=3.0

Added in `build_scene.py` via `ET.SubElement(worldbody, "body", ...)` after the terrain geom.
The tree is a solid collision object — the robot will bump into it.

---

## 2026-04-22 — RL environment + PPO training + eval

**New files**: `ape_env.py`, `train.py`, `eval.py`

**`ape_env.py`**: Gymnasium environment wrapping MuJoCo.
- 63-dim observation: torso z, quat, lin/ang vel, 24 joint pos/vel, unit vector to tree, distance
- 24-dim action: joint position targets in [-1, 1], remapped to joint ranges
- Reward: `progress×2 + 0.1 alive - 1e-4×ctrl² - 10 fall + 50 success`
- Termination: torso < 0.15m (fallen) or dist < 1.5m (success), timeout 2000 steps
- 5 physics steps per RL action

**`train.py`**: SB3 PPO, 8 parallel envs, 10M steps. Checkpoint + EvalCallback to `runs/ape_reach/`.

**`eval.py`**: loads checkpoint, runs episodes silently (no viewer), prints closest approach + status.

---

## 2026-04-22 — Physics tuning: heavier torso, softer actuators, velocity penalty

**Problem**: robot movements were rapid and jerky — didn't look physical.

**Changes**:
1. `ape.urdf`: torso mass 1.0 kg → 2.0 kg, inertia tensors updated accordingly.
   Total robot mass: ~3.8 kg (2 kg torso + ~0.45 kg × 4 arms from URDF inertials).
2. `build_scene.py`: actuator kp 50 → 20. Joints now respond sluggishly, like real servo motors under load.
3. `ape_env.py`: energy penalty 1e-4 → 5e-4; added joint velocity penalty 1e-3×sum(qvel²).
   Velocity penalty directly discourages flailing — the dominant source of rapid movements.

Rebuilt `scene.xml` after these changes.

---

## 2026-04-22 — Persistent live viewer + split-screen multi-env viewer

**Problem**: `mujoco.viewer` not accessible as attribute — needed explicit `import mujoco.viewer`.
Also viewer window was closing and reopening between episodes (jarring UX).

**`watch.py`** (rewritten): opens one persistent MuJoCo passive viewer window, runs episodes
back-to-back, hot-swaps the PPO policy when a newer checkpoint appears. Window never closes.

**`watch_multi.py`** (new): off-screen renders N envs via `mujoco.Renderer`, tiles frames into
a single OpenCV window (split-screen). Policy is batched across all N envs simultaneously.
- Default: 4 envs in a 2×2 grid
- `--n 8` for 2×4, etc.
- Labels show per-env distance to tree, step count; colour-coded (green=success, red=fallen)
- Hot-swaps policy on new checkpoint, same as `watch.py`

---

## 2026-04-23 — Interactive reward hints + one-command launcher

**Goal**: be able to watch training live and inject behavioural hints mid-run without restarting.

**`hints.py`**: plain Python dict of named reward terms with float weights.
All weights start at 0.0 (disabled). Edit and save while training; picked up next episode.
Available hints: `upright`, `spread_legs`, `front_arms_down`, `back_arms_down`, `symmetry`, `stillness`.

**`stand_env.py` changes**:
- `_reload_hints()` called every `reset()` — checks `hints.py` file mtime, `importlib.reload`s only when changed. Prints active hints to terminal on change.
- `_hints_reward()` computes per-step bonus from active hints using body positions (`xpos`) and joint arrays.

**`run.py`**: one command launches both training (subprocess) and viewer (foreground).
Closing the viewer or Ctrl+C terminates the trainer cleanly.
```
python run.py              # stand task
python run.py --task reach
python run.py --resume
```

**Workflow**: `python run.py` → watch → tell Claude what to encourage → Claude edits `hints.py` → next episode picks it up automatically.

---

## 2026-04-23 — Stand task: lift and hold stable height

**Goal**: teach the robot to push itself off the ground and hold a comfortable standing height — a prerequisite for locomotion and reaching behaviour.

**New file: `stand_env.py`**
Separate Gymnasium environment for the standing task.
- 59-dim observation: torso z, quat, lin/ang vel, 24 joint pos/vel (no tree signal needed)
- 24-dim action: same joint position targets as `ape_env`
- Start pose: torso 7 cm above terrain surface, arms in splayed-down init pose (`link2` angle −0.4 rad, `link3` 1.2 rad) + uniform noise ±0.1
- Reward: `height_reward + upright_reward + 0.05 alive − 5e-4×ctrl² − 1e-3×jvel²`
  - `height_reward`: Gaussian centred on `TARGET_Z=0.30 m` above terrain, max 1.0 (σ=0.08 m)
  - `upright_reward`: `0.3 × w²` where w is torso quaternion scalar — penalises tilt
- Termination: torso < 3 cm above terrain (fallen); timeout 2000 steps
- No success terminal state — reward is dense, agent learns to hold height indefinitely

**Terrain-aware spawn (`_terrain_z`)**: the terrain heightfield reaches 0.43 m at origin (seed=42).
Without this fix the robot spawned 0.36 m underground. `_terrain_z(x, y)` reads `hfield_data`
directly, converting pixel coordinates from world XY. Fall detection is now relative to terrain
surface, not absolute world Z.

**`train.py`**: added `--task reach|stand` flag. Checkpoints go to `runs/ape_reach/` or
`runs/ape_stand/` depending on task. Defaults to `reach` so existing workflow is unchanged.

**`watch.py`** (rewritten): now task-aware (`--task stand|reach`). Uses the env directly
instead of duplicating obs-building logic — one source of truth. Added `viewer_alive()` check
inside both the episode and outer loops so closing the window exits cleanly rather than crashing.
Catches and reports exceptions instead of killing silently. Prints `height=` and `h_rew=` for
the stand task.

**`eval.py`** (rewritten): same task flag, same env-direct approach. Tracks `best_z` per episode
for the stand task.

**Two-terminal workflow for stand training**:
```
Terminal 1:  python train.py --task stand
Terminal 2:  python watch.py --task stand
```

---

## 2026-04-24 — MuJoCo MJX GPU migration (46× speedup)

**Motivation**: SB3+CPU training was GPU-idle, ~1M steps/hour. MJX runs thousands of
parallel environments entirely on the RTX 4070 via JAX.

**Result**: 1024 parallel envs → **46M steps/hour** (46× over CPU baseline).

**New files**:
- `stand_env_mjx.py` — pure-JAX environment. Stateless `reset()`/`step()` functions
  operating on pytree state. `jax.vmap` gives free N-env parallelism.
- `train_mjx.py` — JAX PPO training loop. Rollout collection with `jax.lax.scan`,
  GAE advantage estimation, minibatch updates all JIT-compiled on GPU.
  Saves checkpoints to `runs/ape_stand_mjx/` and exports SB3-compatible `best_model.zip`
  so `watch.py` can visualise progress.

**Key engineering decisions**:
- Mesh geom collisions disabled before `mjx.put_model()` — STL meshes cause combinatorial
  explosion of collision kernel compilation pairs. Contact handled by torso box geom only.
- Tree trunk changed cylinder→capsule (MJX doesn't implement cylinder-box collision).
- Network: Flax `ActorCritic` with tanh-squashed Gaussian policy (same architecture as SB3 MlpPolicy).
- Separate from `train.py` — CPU path untouched, both work independently.

**Usage**:
```
python train_mjx.py              # 1024 envs, ~46M steps/hr
python train_mjx.py --envs 2048  # more envs if VRAM allows
```

---

## 2026-04-24 — Generational knowledge transfer (--inherit)

**Motivation**: changing hardware params (kp, torque limits, physics steps) invalidates the old
policy's assumptions but the network still knows useful things ("push down to rise", "stay upright").
Starting from scratch wastes all that. `--inherit` copies network weights from the best checkpoint
into a fresh run — like teaching the next generation from the previous one's experience.

**New flag: `--inherit`**
- Loads best available checkpoint (prefers `best/best_model.zip`, falls back to latest step ckpt)
- Copies `policy.state_dict()` into a freshly constructed PPO model
- Resets step counter and opens a new TensorBoard curve (it's a new run, not a continuation)
- Contrast with `--resume`: that continues the *same* run with the same step count

**New file: `lineage.py`**
Every training run appends one JSON entry to `runs/<task>/lineage.json` recording:
- timestamp, n_envs, inherited_from path
- physics params read live from `scene.xml` (kp, forcerange)
- env constants read live from `stand_env.py` / `ape_env.py` (PHYSICS_STEPS_PER_ACTION, TARGET_Z, etc.)

View the full lineage table:
```
python train.py --lineage --task stand
```

**Workflow for hardware changes**:
```
# change kp or torque in scene.xml, then:
python train.py --task stand --inherit --watch
```
The agent starts with all prior knowledge and adapts to the new physics within thousands of
steps rather than millions.

---

## 2026-04-24 — Slower, wider movements

**Goal**: reduce the quick twitchy motions and encourage slower, more deliberate arm sweeps.

Three changes:
1. `PHYSICS_STEPS_PER_ACTION` 5 → 12: each policy decision plays out over more physics time, so the robot can't jerk between poses rapidly.
2. Action-delta penalty `−2e-2 × Σ(aₜ − aₜ₋₁)²` added to reward: directly penalises large jumps between consecutive actions. `_prev_action` tracked in env state, reset to zeros on episode start.
3. Servo `kp` 20 → 8 in scene.xml: position-control gains halved, so joints respond more gradually rather than snapping to targets.

---

## 2026-04-24 — Flat ground + one-command training

**Problems fixed**:
1. `watch.py` printed FALLEN constantly even when the robot looked fine visually. Root cause: the terrain heightfield peaks at ~0.43 m at the spawn point, so `_terrain_z` was returning a high value and `above_ground = torso_z - ground_z` was near zero immediately after spawn.
2. Bumpy grass terrain made early training harder with no benefit at this stage.

**Changes**:
- `scene.xml`: replaced the hfield + grass texture with a flat grey `plane` geom. Ground is exactly z=0 everywhere.
- `stand_env.py`: removed `terrain_mod` dependency entirely. `above_ground` is now just `torso_z` directly. Removed `_terrain_z()` method and `_ground_z` instance variable. Reset no longer looks up terrain height — just sets `qpos[2] = SPAWN_CLEARANCE`.
- `ape_env.py`: removed `terrain_mod.apply()` call (no hfield to fill).
- `sim.py`: removed `terrain_mod` import and call (added this session as a simple interactive viewer).
- `train.py`: `--envs` flag (default 1) replaces the hardcoded `N_ENVS=8`. Added `--watch` flag that spawns `watch.py` as a subprocess so training + live viewer runs from a single command:
  ```
  python train.py --task stand --watch
  ```

---

## 2026-04-22 — GPU enabled (PyTorch cu121 for CUDA 12.2 driver)

**Problem**: PyTorch 2.11.0+cu130 was installed (needs CUDA 13.0 driver) but WSL2 driver
only supports CUDA 12.2 (driver version 12020 = 12.2.0). CUDA unavailable, training on CPU.

**Fix**: reinstalled PyTorch 2.5.1+cu121, which is compatible with CUDA ≥ 12.1 drivers.
```
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```
Result: `cuda available: True`, `GPU: NVIDIA GeForce RTX 4070 Laptop GPU`.
Expected training speedup: 10–20× vs CPU.
