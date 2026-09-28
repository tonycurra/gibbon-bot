# Gibbon-bot — repositories

Atlas of folders and branches. Full history is in each `.git`. Technical result: `EVALUATION.md`. Plan: `NEXT_STEPS.md`.

---

## Copying folders keeps git history

**Yes — if you copy the whole folder, including the hidden `.git` directory.**

That `.git` folder **is** the repository. The files you see are only the checked-out snapshot.

| What you copy | What you keep |
|---|---|
| Entire folder **with** `.git` | All branches, commits, tags |
| Folder **without** `.git` | Only the current checkout |
| `git clone --mirror` or `git bundle create repo.bundle --all` | History-only backup |

Confirm `.git` is inside (`ls -a folder | grep .git`). After restore: `git branch -a` and `git log --all --oneline`.

Do **not** `git remote remove origin` then `git gc --prune=now` unless local branches exist for every remote tip.

---

## Snapshot of this folder

| Folder | Kind | Codeberg | Commits | Branches | Contributors | Span |
|---|---|---|---|---|---|---|
| `gibbonBot-RL` | git | tonycurra/gibbonBot-RL | 399 | 23 | 389 Tony, 10 Jade Leurs | 2026-04-23 → 2026-07-13 |
| `gibbon-rl-mujoco` | git | tonycurra/gibbon-rl-mujoco | 3 | 1 | Tony (3) | 2026-04-26 → 2026-04-26 |
| `olivabot-mujoco-cpu` | git | tonycurra/olivabot-mujoco-cpu | 3 | 2 | Jade Leurs, jajayumei, tony_olivabot | 2026-04-27 → 2026-05-01 |
| `Roarm-m3pro-busServoDriverHAT_A-battery` | git | olivabot/Roarm-… | 1 | 1 | tony_olivabot | 2026-05-30 |
| `gibbon-brain` | **not a git repo** | — | — | — | local files (Tony) + GitHub MuJoCo clone | ~2026-07-14 |

---

## 1. `gibbonBot-RL` — main RL / simulation stack

**What it is.** Reinforcement-learning training stack for a 4-limbed robotic ape
(4× Waveshare RoArm-M3 arms on a shared torso) in MuJoCo + MJX (Warp GPU backend).
North star: robot walks to the olive tree on command.
**LIANA** = Live Interactive Ape Neural Architecture — dashboard + control over training
(`./liana go` → localhost:8000).

**Remote:** `ssh://git@codeberg.org/tonycurra/gibbonBot-RL.git`

**Checked out now:** local `main` = `9e2474a` (2026-05-15) — this is Codeberg’s `brain-modeling`,
**not** Codeberg’s `main`. Codeberg `main` is saved locally as `codeberg/main` (`f9eedc2`).

**Contributors (all history):**

| Author | Commits |
|---|---|
| Tony | 389 |
| Jade Leurs | 10 |

**Activity:** 2026-04-23 → 2026-07-13. Peak in late April–mid May (Tony), then Jade’s later branches in June–July.

**What’s in the current checkout (`main` / brain-modeling):**
scene + meshes (`data/`), env (`envs/env.py`), PPO trainer (`train_mjx.py`),
bioinspired brain modules (`brain/`), LIANA dashboard (`dashboard/`, `./liana`),
viewer (`view.py`), archive of older experiments (`archive/`).

### How to read the branches

Most branches are **experiments that forked off the early MuJoCo setup** (`sim/mujoco-setup`, 2026-04-26)
and were later tagged `archive/*` when work moved on. They are **not** linear steps on today’s `main`.
Jade’s June–July branches (`quadruped_training`, `unigree-go2`, `apenapepose_integration`) are separate lines.

| Branch | Tip date | Tip | Author of tip | Unique vs local `main` | What it was |
|---|---|---|---|---|---|
| `unigree-go2` | 2026-07-13 | `0580087` | Jade Leurs | 3 | Jade: Unitree Go2 training setup (Jul 13). Latest commit in the whole repo. |
| `quadruped_training` | 2026-06-28 | `fc4a84a` | Jade Leurs | 2 | Jade: custom Ant-v5-based quadruped env (Jun 28). Separate from ape stack. |
| `apenapepose_integration` | 2026-06-17 | `21170cb` | Jade Leurs | 7 | Jade: ape/nape/pose integration updates (Jun 17). Long history (316 commits) — likely i… |
| `codeberg/brain-modeling` | 2026-05-15 | `9e2474a` | Tony | 0 | Backup of Codeberg `brain-modeling` — same commit as local `main`. |
| `main` | 2026-05-15 | `9e2474a` | Tony | — | Current working line in this checkout. Same tip as Codeberg `brain-modeling`: LIANA das… |
| `codeberg/main` | 2026-05-14 | `f9eedc2` | Tony | 0 | Backup of Codeberg default `main` (`f9eedc2`, 2026-05-14): scene contact exclusions. Ol… |
| `brain/training-integration` | 2026-05-14 | `0536145` | Tony | 0 | Wire ApeBrain into the MJX training loop; persist/resume; MLflow. |
| `ape_brain` | 2026-05-11 | `ea48877` | Tony | 0 | Standalone ApeBrain architecture (8 modules + quantum stubs) before full training integ… |
| `dlab-demo` | 2026-05-09 | `99d1adc` | Tony | 0 | D-Lab demo snapshot. |
| `walk` | 2026-05-05 | `6a32ee8` | Tony | 0 | Walk-to-tree task / locomotion reward work. |
| `grove_scene` | 2026-05-05 | `87aaae4` | Tony | 0 | Grove / multi-tree scene variant. |
| `html_viewer` | 2026-05-04 | `3f296f5` | Tony | 0 | HTML/web viewer for the sim. |
| `lean-mjx` | 2026-05-04 | `d607a1e` | Tony | 0 | Slim MJX training path. |
| `mjx-gpu` | 2026-04-28 | `aac4981` | Tony | 0 | MJX GPU / Warp backend experiments. |
| `simple-reward` | 2026-04-27 | `a46b219` | Tony | 0 | Simplified reward function experiment. |
| `lean-training` | 2026-04-27 | `e4d2f30` | Tony | 0 | Slimmer training loop. |
| `clean/train-loop` | 2026-04-27 | `ce6eb6b` | Tony | 0 | Cleaned-up train loop. |
| `brain-modeling` | 2026-04-26 | `b859915` | Tony | 0 | LOCAL NAME MISMATCH: this branch tip is the old `sim/mujoco-setup` commit, not the Code… |
| `sim/mujoco-setup` | 2026-04-26 | `b859915` | Tony | 0 | Early MuJoCo/MJX project setup (Apr 26). Common ancestor of most archived experiments. |
| `clean-no-capsules` | 2026-04-26 | `e0c0b6e` | Tony | 1 | Scene without capsule collision proxies. |
| `setup-improvements` | 2026-04-26 | `84f3328` | Tony | 1 | Project setup / tooling improvements on top of mujoco-setup. |
| `tree-navigation` | 2026-04-25 | `a0ce6a0` | Tony | 0 | Tree as navigation target in the scene. |
| `no-capsules` | 2026-04-25 | `64e0a1b` | Tony | 0 | Earlier no-capsule collision work. |

### Branch notes and commits (unique vs current `main`, or latest on `main`)

#### `main`

Current working line in this checkout. Same tip as Codeberg `brain-modeling`: LIANA dashboard, walk/stand env, MJX trainer, brain modules. North star: walk to the olive tree on command.

Latest commits on this line:

```
9e2474a 2026-05-15 Tony — docs(liana): match current dashboard surface
8fbd355 2026-05-15 Tony — feat: one-button training, per-task progress card, notes DB
de38272 2026-05-15 Tony — fix(ui): per-task trial seconds, drop Speed input, Envs default 512
2dcde5e 2026-05-15 Tony — feat(tasks): task registry + progress bar + one-button Train toggle
f393009 2026-05-15 Tony — feat(sync): ./liana sync — object-storage mirror via rclone
a1284a2 2026-05-15 Tony — docs: bring LIANA.md / README / ARCHITECTURE / LOG up to date
80a3729 2026-05-15 Tony — fix(task): stand vs walk now train different rewards
08def18 2026-05-15 Tony — feat(viewer): trial survives Stop train, half-speed default, speed input
be0e36e 2026-05-15 Tony — feat: live env.py params view + trial honors smoothing & speed
ecd9b10 2026-05-15 Tony — feat(library): named saved policies — save/use/delete from UI
6dfce4b 2026-05-15 Tony — ui: remove Params card, lock params to known-good
e929b81 2026-05-15 Tony — feat: scripted Demo walk button — visual reference for 'good motion'
```

#### `codeberg/main`

Backup of Codeberg default `main` (`f9eedc2`, 2026-05-14): scene contact exclusions. Older than local `main`.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `codeberg/brain-modeling`

Backup of Codeberg `brain-modeling` — same commit as local `main`.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `brain-modeling`

LOCAL NAME MISMATCH: this branch tip is the old `sim/mujoco-setup` commit, not the Codeberg brain-modeling work. Use `main` or `codeberg/brain-modeling` for the real brain-modeling line.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `sim/mujoco-setup`

Early MuJoCo/MJX project setup (Apr 26). Common ancestor of most archived experiments.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `brain/training-integration`

Wire ApeBrain into the MJX training loop; persist/resume; MLflow.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `ape_brain`

Standalone ApeBrain architecture (8 modules + quantum stubs) before full training integration.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `walk`

Walk-to-tree task / locomotion reward work.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `tree-navigation`

Tree as navigation target in the scene.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `grove_scene`

Grove / multi-tree scene variant.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `html_viewer`

HTML/web viewer for the sim.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `dlab-demo`

D-Lab demo snapshot.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `mjx-gpu`

MJX GPU / Warp backend experiments.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `lean-mjx`

Slim MJX training path.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `lean-training`

Slimmer training loop.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `clean-no-capsules`

Scene without capsule collision proxies.

1 commit(s) not on local `main`:

```
e0c0b6e 2026-04-26 Tony — feat: add portable setup to clean no-capsules branch
```

#### `no-capsules`

Earlier no-capsule collision work.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `clean/train-loop`

Cleaned-up train loop.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `simple-reward`

Simplified reward function experiment.

No commits that are not already on local `main` (this tip is an ancestor of `main`, or equal).

#### `setup-improvements`

Project setup / tooling improvements on top of mujoco-setup.

1 commit(s) not on local `main`:

```
84f3328 2026-04-26 Tony — feat: add setup instructions and requirements.txt for portable installation
```

#### `quadruped_training`

Jade: custom Ant-v5-based quadruped env (Jun 28). Separate from ape stack.

2 commit(s) not on local `main`:

```
fc4a84a 2026-06-28 Jade Leurs — added custom (Ant-v5 based) environment
e4c87ef 2026-06-26 Jade Leurs — Initial commit
```

#### `unigree-go2`

Jade: Unitree Go2 training setup (Jul 13). Latest commit in the whole repo.

3 commit(s) not on local `main`:

```
0580087 2026-07-13 Jade Leurs — unitree_go2 training setup
fc4a84a 2026-06-28 Jade Leurs — added custom (Ant-v5 based) environment
e4c87ef 2026-06-26 Jade Leurs — Initial commit
```

#### `apenapepose_integration`

Jade: ape/nape/pose integration updates (Jun 17). Long history (316 commits) — likely includes much of earlier mainline plus her integration work.

7 commit(s) not on local `main`:

```
21170cb 2026-06-17 Jade Leurs — some updates
36ec54b 2026-05-22 Jade Leurs — integrated pose priors into training loop
08d141a 2026-05-21 Jade Leurs — added docs for clustering and integration options
6c6538b 2026-05-21 Jade Leurs — derive approximate joint angle limits from 2D OpenApePose keypoints
fe244e7 2026-05-21 Jade Leurs — added GMM pose priors
7729580 2026-05-21 Jade Leurs — added openapepose clustering
7a2fe9e 2026-05-21 Jade Leurs — fixed formatting
```

### Tags

Archive tags (Tony, 2026-05-15) marking old experiment tips so they could be deleted as branches later:

`archive/clean-no-capsules` → `e0c0b6e` (2026-04-26) feat: add portable setup to clean no-capsules branch
`archive/clean-train-loop` → `ce6eb6b` (2026-04-27) Fix floor penetration issue with proper collision detection
`archive/dlab-demo` → `a9b1d4c` (2026-05-09) dlab-demo: olive variety + scripted climb-demo + branch atlas
`archive/grove_scene` → `87aaae4` (2026-05-05) Walk-to-tree env + realistic motor damping + capsule tree generator
`archive/html_viewer` → `3f296f5` (2026-05-04) Terrain mesh: emit reversed-winding triangles so surface is visible from below
`archive/lean-mjx` → `d607a1e` (2026-05-04) changing parametres, evaluating results
`archive/lean-training` → `e4d2f30` (2026-04-27) Restore UI functionality with improved parameter ranges
`archive/main` → `87aaae4` (2026-05-05) Walk-to-tree env + realistic motor damping + capsule tree generator
`archive/mjx-gpu` → `aac4981` (2026-04-28) Simplify training: remove hot-reload/tune mode, add view.py for physics check
`archive/no-capsules` → `64e0a1b` (2026-04-25) fix: --inherit now actually loads MJX weights (was a stub)
`archive/setup-improvements` → `84f3328` (2026-04-26) feat: add setup instructions and requirements.txt for portable installation
`archive/sim-mujoco-setup` → `b859915` (2026-04-26) updating codebase with llm
`archive/simple-reward` → `a46b219` (2026-04-27) Flatten structure, add test.py, QUICKSTART, soft arm penalty
`archive/tree-navigation` → `a0ce6a0` (2026-04-25) docs: update README with tree navigation features
`archive/walk` → `6a32ee8` (2026-05-05) walk: speed up sim — plane collision, half substeps, shorter episodes

### Commit volume by month

| Month | Commits |
|---|---|
| 2026-04 | 96 |
| 2026-05 | 299 |
| 2026-06 | 3 |
| 2026-07 | 1 |

---

## 2. `gibbon-rl-mujoco` — early MuJoCo / Jade ape URDF

**What it is.** Earlier, smaller RL+MuJoCo project: Waveshare `roarm_ws` descriptions
plus Jade’s four-arm “gibbon-like” ape (`simulation_code_jade/`: xacro, URDF, `ape_simulation.py`).
Predecessor / sibling of `gibbonBot-RL`, not the live training stack.

**Remote:** `ssh://git@codeberg.org/tonycurra/gibbon-rl-mujoco.git`
**Branch:** `main` only. **Contributor:** Tony (3 commits).

```
5e27bd4 2026-04-26 Tony — Fix simulation to work without xacro dependency
33eead7 2026-04-26 Tony — Fix Ruff F401 warning: remove unused os import
62b599e 2026-04-26 Tony — Add gibbon-like robotic ape simulation (JADE design)
```

Tip `5e27bd4` 2026-04-26 — last commit: “Fix simulation to work without xacro dependency”.

---

## 3. `olivabot-mujoco-cpu` — CPU PPO stand-up trainer

**What it is.** CPU-side PPO (Stable Baselines 3) + MuJoCo: lift torso to a target height and keep it level.
Live param sliders (`envs/params.py`), embedded viewer (`test.py`), replay (`utils/watch.py`).
Jade’s `jade` branch is the later/more complete line.

**Remote:** `ssh://git@codeberg.org/tonycurra/olivabot-mujoco-cpu.git`

| Branch | Tip | Date | Author | What |
|---|---|---|---|---|
| `main` | `ad198f0` | 2026-04-27 | tony_olivabot | Initial commit |
| `jade` | `2da11dd` | 2026-05-01 | Jade Leurs | Jade’s training/viewer/params work |

All commits:

```
2da11dd 2026-05-01 jajayumei — update requirements.txt to include tensorboard
5160c10 2026-05-01 Jade Leurs — add telemetry logs
ad198f0 2026-04-27 tony_olivabot — first commit
```

Checked out now: `main` (the initial commit). Jade’s work is on branch `jade`, not in the working tree until you `git checkout jade`.

---

## 4. `Roarm-m3pro-busServoDriverHAT_A-battery` — hardware notes

**What it is.** Local notes / PDF for RoArm-M3 Pro + bus servo driver HAT + battery
(Waveshare hardware used as the ape’s limbs). Single commit, no software stack.

**Remote:** `ssh://git@codeberg.org/olivabot/Roarm-m3pro-busServoDriverHAT_A-battery.git`

```
a1dbafd 2026-05-30 tony_olivabot — Upload files to "/"
```

Files at tip:

```
roarm_m3_pro_migration_guide.pdf
```

---

## 5. `gibbon-brain` — local experiments (no project git)

**Not a git repository.** Copying this folder keeps the files as they are today,
with **no commit history** of your scripts. The nested `mujoco/` clone is Google DeepMind’s
upstream git (GitHub), not your work.

Your files (mtime ~ 2026-07-14 … 2026-07-16):

| File | Role (from names) |
|---|---|
| `control_robot.py` | Robot control |
| `agent_control.py` | Agent control |
| `evaluate.py` | Evaluation |
| `view_scene.py` | Scene viewer |
| `QUICK_START.md`, `SESSION_SUMMARY.md`, `mujoco_robot_control_approaches.md`, `projectplan.md` | Notes / session writeups |
| `gibbonBot-RL-brain-modeling_data/` | Data pulled from the brain-modeling line |
| `logs/` | Run logs |
| `mujoco/` | Upstream DeepMind MuJoCo source clone |

If you want history for this folder later, `git init` here (excluding `.venv` and `mujoco/`) before you rely on it as a backup of *work*.

---

## How this maps to evaluation

| Line of work | Where to look | Who | Outcome to evaluate |
|---|---|---|---|
| Live ape RL + LIANA | `gibbonBot-RL` `main` | Tony (primary), Jade (later branches) | Walk-to-tree, MJX, brain, dashboard |
| Archived sim experiments | `gibbonBot-RL` `archive/*` tags + named branches | Tony | Capsules, lean MJX, grove, html viewer, dlab demo, rewards |
| Jade: Go2 / quadruped / pose | `gibbonBot-RL` `unigree-go2`, `quadruped_training`, `apenapepose_integration` | Jade Leurs | Separate from the ape demo; latest activity in the repo |
| Jade ape URDF (early) | `gibbon-rl-mujoco` `main` | Tony packing Jade’s sim code | Four-arm xacro/URDF |
| CPU stand-up PPO | `olivabot-mujoco-cpu` `jade` (not `main`) | tony_olivabot, Jade, jajayumei | Simpler CPU trainer |
| Hardware / arm | `Roarm-…battery` | tony_olivabot | Reference PDF |
| Brain control scripts | `gibbon-brain` files only | unversioned | Control/eval scripts + notes |

To inspect any branch later without changing files: `git log BRANCH` / `git show BRANCH:path`.
To open it: `git checkout BRANCH` (or `git worktree add ../gibbonBot-RL-BRANCH BRANCH`).

