# gibbonBot-RL

PPO + MuJoCo MJX (JAX, GPU-accelerated) for a 4-limbed robotic ape.
Currently learning to walk to a tree on flat ground.

## Quickstart

```bash
# train a multi-skill curriculum (everyday flow)
python scripts/curriculum.py
# Trains stand → walk_forward → turn_left → turn_right in sequence,
# each task transferring policy weights from the previous. See docs/CURRICULUM.md.

# train one task manually
python train_mjx.py --task stand --no-resume --auto-push --label "what changed"
python train_mjx.py --task walk --resume \
    --resume-from runs/ape_stand_mjx/keepers/auto_*.npz   # cross-task transfer
# --task        : walk, stand, walk_forward, turn_left, turn_right
# --auto-push   : scripts/sync.sh fires after every checkpoint save → pushes to Codeberg
# --label       : free-form note attached to the MLflow run as a tag
# --resume-from : explicit checkpoint, even from a different task

# legacy entry point (back-compat shim that calls train_mjx.py --task walk)
python train_walk_mjx.py --auto-push --label "..."

# train remote
scripts/remote.sh start         # H100, walk task, autopilot, watchdog, all in one
scripts/remote.sh status        # GPU util + log tail + tmux state
scripts/remote.sh log           # live tail of training log
scripts/remote.sh stop          # graceful save + exit

# pull artifacts down + watch policy
scripts/pull_checkpoints.sh &   # syncs runs/ + mlruns/ from remote, every 30s
python view.py --env walk        # viewer hot-swaps to newest checkpoint
python view.py --env walk --best # loads best-by-mlflow-mean_rew checkpoint

# evaluate
python -m utils.eval --task walk         # cross-run summary
python -m utils.eval --task walk --plot  # writes reward.png
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000

# attach a free-form note to the latest MLflow run
python scripts/note.py "rew jumped at update 200 — heading reward dominates"

# compare metrics across recent training runs side-by-side
python scripts/compare_runs.py --top 10
python scripts/compare_runs.py --label CPG --plot cpg_runs.png

# inspect the policy via offscreen renderer (works without GUI)
python scripts/snapshot.py --frames 6 --gap 1.0 --policy latest
# writes 5 views (side_L, side_R, front, back, top) per frame to _snapshots/,
# plus _snapshots/landmarks.json with 3D xyz of 13 keypoints (torso + 4 limbs ×
# shoulder/elbow/hand). Use to compare gait against ape pose datasets.

# probe each joint to see which world direction +action moves the limb tip
python scripts/calibrate_cpg.py
# critical for tuning the CPG when the URDF has mirrored quaternions per limb.

# audit physics + motors against data/motors.md
python scripts/verify_dynamics.py
# checks masses, damping, stall torque, floor friction, solref/solimp.
# Run after any scene.xml change to catch drift.

# back up before destroying instance
scripts/archive_runs.sh
```

## Layout

```
.
├── train_walk_mjx.py    main PPO trainer (the script you actually run)
├── view.py              live MuJoCo viewer (hot-swaps to newest checkpoint)
├── README.md            this file
├── requirements.txt
├── brain/               Bioinspired neural architecture (Neurotransmitters, Brainstem, Basal Ganglia)
├── envs/                MJX environments + their hyperparams
├── data/                scene.xml, robot meshes, motor specs, terrain
├── trees/               olive-tree generation pipeline
├── utils/               autopilot, eval, best-checkpoint loader, limb dynamics
├── scripts/             remote.sh (drive remote GPU), monitor.py (watchdog),
│                        pull_checkpoints.sh, archive_runs.sh, gen_terrain.py, gen_trees.py
├── runs/                training output (mostly gitignored, except keepers/)
├── mlruns/              mlflow tracking (gitignored, synced from remote)
├── docs/                long-form: PLAN, CORE, BIOLOGY, REMOTE_GPU, BRANCHES, etc.
└── archive/             defunct code paths kept for reference
```

## What gets preserved

| What | Where | In git? |
|---|---|---|
| Auto-promoted curated checkpoints | `runs/<task>/keepers/auto_*.npz` + `.json` | **yes** |
| Reproducibility sidecar (params + fingerprint) | `runs/<task>/keepers/auto_*.json` | **yes** |
| Streaming/intermediate checkpoints | `runs/<task>/mjx_*.npz` | **yes** (auto-pushed to Codeberg) |
| MLflow run/metric history (sqlite) | `mlflow.db` | **yes** |
| MLflow legacy filesystem store | `mlruns/` | **yes** (deprecated, kept for old runs) |
| Per-run lineage | `runs/<task>/lineage.json` | yes |
| Training log | `runs/<task>/train.log` | yes |
| Lessons learned (prose) | `LESSONS.md` | yes |
| Hypothesis backlog | `BACKLOG.md` | yes |
| Robot URDF meshes | `data/meshes/*.stl` | yes |
| Scene XML | `data/scene.xml` | yes |

**Auto-push**: when `--auto-push` is passed to `train_walk_mjx.py`, `scripts/sync.sh`
runs after every checkpoint save (~every 32K env steps, ~30 s on the 4070). It
commits the data paths above and pushes to **every configured git remote**.
Idempotent — bails if nothing changed. Worst-case data loss: ~30 s of training
compute.

**Adding a backup remote** (recommended — survives a single remote outage):
```bash
git remote add backup git@gitlab.com:USER/gibbonBot-RL.git
# from now on, scripts/sync.sh pushes to both origin and backup automatically
```
Free EU options for the second remote: GitLab.com (EU regions), self-hosted
Gitea on a Hetzner VPS (~€5/mo). Free non-EU: GitHub.

A keeper is auto-saved every PPO update where `min_dist < 1.5 m` OR
`success_frac > 5 %` is achieved. Each keeper carries a sidecar JSON with
the params, scene hash, and metric snapshot that produced it — full
reproducibility unit.

## Docs

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — paste-able commands, deeper reference
- **[docs/PLAN.md](docs/PLAN.md)** — autopilot training plan (phases A→D)
- **[docs/CORE.md](docs/CORE.md)** — multi-task framework + per-limb diagnostics
- **[docs/BIOLOGY.md](docs/BIOLOGY.md)** — evolutionary priors built into the policy
- **[docs/REMOTE_GPU.md](docs/REMOTE_GPU.md)** — remote H100 deep-dive
- **[docs/IMITATION_PLAN.md](docs/IMITATION_PLAN.md)** — ape-pose imitation roadmap
- **[docs/BRANCHES.md](docs/BRANCHES.md)** — branch atlas
- **[docs/GROVE_MASTERPLAN.md](docs/GROVE_MASTERPLAN.md)** — olive-grove environment
- **[BACKLOG.md](BACKLOG.md)** — prioritized list of unimplemented ideas
- **[LESSONS.md](LESSONS.md)** — cumulative training insights (prose; numbers in MLflow)
- **[docs/ROBOT_VS_APE.md](docs/ROBOT_VS_APE.md)** — joint-by-joint mapping from robot DOF to ape anatomy + what's missing
- **[docs/BRAIN_CORE.md](docs/BRAIN_CORE.md)** — long-term vision for instinct + memory architecture (skill library, recurrent policy, world model)
- **[docs/CURRICULUM.md](docs/CURRICULUM.md)** — multi-skill curriculum: stand → walk_forward → turn_left → turn_right
- **[docs/CLOUD_DEPLOY.md](docs/CLOUD_DEPLOY.md)** — Dockerfile + docker-compose for local or cloud GPU training
- **[docs/APEBRAIN.md](docs/APEBRAIN.md)** — Bioinspired neural architecture (Neurotransmitters, Brainstem, Basal Ganglia)

## Tooling cheat sheet

| Script | Purpose |
|---|---|
| `train_walk_mjx.py --auto-push --label "..."` | Train walk-to-tree policy, push every save |
| `view.py --env walk` | Live MuJoCo viewer (hot-swaps to newest checkpoint) |
| `view.py --env walk --best` | View best-mean_rew checkpoint per MLflow |
| `scripts/snapshot.py` | 5-view PNGs + 13-keypoint 3D landmarks JSON |
| `scripts/calibrate_cpg.py` | Probe each joint, output direction-of-effect table |
| `scripts/verify_dynamics.py` | Audit physics + motors against motors.md |
| `scripts/compare_runs.py` | MLflow cross-run metric comparison |
| `scripts/note.py "..."` | Attach prose note to current MLflow run |
| `scripts/sync.sh` | One-shot push to all configured git remotes |

## Motor model

Per-actuator velocity & acceleration limits are derived from `data/motors.md`
(Waveshare ST3235 / ST3215 / ST3215-HS specs):

- ST3235 / ST3215: 4.71 rad/s (45 RPM), 2.94 N·m
- ST3215-HS (gripper/foot): 11.10 rad/s (106 RPM), 1.96 N·m

Acceleration is bounded so each joint follows a trapezoidal velocity profile —
smooth start *and* stop. Tunable in `envs/params_walk_mjx.py`:

- `MOTOR_SPEED_FRAC` (default 1.0) — global scale; <1 = slower than reality
- `MOTOR_ACCEL_RATIO` (default 5.0/s) — how quickly joints reach max speed

## CPG (Central Pattern Generator)

`envs/walk_env_mjx.py` injects a sinusoidal gait baseline into every action,
so the policy learns *corrections* to a working gait instead of inventing one
from scratch. Currently configured for **pace** (left + right pairs alternate;
camel/giraffe-style), tuned via per-limb sign correction from
`scripts/calibrate_cpg.py`.

Knobs in `envs/params_walk_mjx.py`:
- `W_CPG_PITCH` — propulsive sweep amplitude (front yaw + back pitch)
- `W_CPG_ELBOW` — foot-lift amplitude during swing
- `GAIT_PERIOD_STEPS` (in `walk_env_mjx.py`) — full stride period in env steps

Phase pattern (`LIMB_PHASE` in `walk_env_mjx.py`):
- Pace (current): `[0, π, 0, π]` — same-side limbs in phase
- Trot (alternative): `[0, π, π, 0]` — diagonal pairs in phase

## Action mapping (rest pose)

Action `0` does not map to joint angle 0. Instead it maps to a configurable
**rest pose** (knees/elbows pre-bent at 30° in the standing direction). This
prevents the "dead spider" failure where joint=0 means all limbs splayed
horizontally. `action=±1` reaches the joint limits relative to that rest pose.

See `envs/walk_env_mjx.py:load_model()` (`PER_LIMB_REST` array).
