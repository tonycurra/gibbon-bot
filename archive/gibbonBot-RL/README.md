# gibbonBot-RL

Reinforcement-learning training stack for a 4-limbed robotic ape.
4 × roarm-M3 arms (the same physical arm, repeated as front-left,
front-right, back-left, back-right) on a shared torso, simulated in
MuJoCo + MJX with a Warp physics backend on GPU.

**Current focus**: `stand` — keep the torso upright on flat ground with
all 4 limbs. Other tasks (`walk`, `walk_forward`, `turn_left`,
`turn_right`) are parked but their env code remains in `envs/` for
future curricula.

---

## What's in the box

```
┌────────────────────────────────────────────────────────────────┐
│  data/scene.xml          ← robot + floor + light (the world)   │
│  data/meshes/*.stl       ← roarm-M3 STL parts (real geometry)  │
│  data/SCENE_GEOMETRY.md  ← required reading before scene edits │
└────────────────────────────────────────────────────────────────┘
                              ▲
                              │ loaded by
                              │
┌────────────────────────────────────────────────────────────────┐
│  envs/<task>_env_mjx.py       ← reset / step / reward / done   │
│  envs/params_<task>_mjx.py    ← hyperparams (lr, n_envs, etc.) │
│  envs/_shared_physics_backend ← MJX vs MJX-Warp dispatch       │
└────────────────────────────────────────────────────────────────┘
                              ▲
                              │ stepped by
                              │
┌────────────────────────────────────────────────────────────────┐
│  train_mjx.py            ← PPO trainer, task-parameterized     │
│   • auto-resumes from runs/ape_<task>_mjx/mjx_*.npz            │
│   • logs reward + losses to mlflow.db                          │
│   • optional ApeBrain episodic memory (--brain-persist)        │
└────────────────────────────────────────────────────────────────┘
                              ▲
                              │ optionally enriched by
                              │
┌────────────────────────────────────────────────────────────────┐
│  brain/        ← bioinspired neural modules (8 + quantum stubs)│
│  • neurotransmitters   • amygdala (fear, punishment signals)   │
│  • brainstem (reflex)  • hippocampus (episodic + spatial mem)  │
│  • basal_ganglia       • prefrontal_cortex (goals, planning)   │
│  • cerebellum (motor)  • motor_cortex (motor primitives)       │
│  • ape_brain.py integrates all 8 via a Basal-Ganglia router.   │
└────────────────────────────────────────────────────────────────┘

  look.py   ← static inspect (gravity-only, no policy, no checkpoint)
  view.py   ← live MuJoCo viewer; hot-swaps to newest checkpoint
```

---

## Quickstart — stand training in Docker

We run training in Docker so the CUDA / JAX / MuJoCo / Warp versions are
pinned to one known-good combination. Local Python in `.venv` is used
only for the GUI viewer (`look.py`, `view.py`) and one-off scripts.

```bash
# 1. ONE-TIME: build the image (~5 GB, ~10 min).
docker build -t gibbonbot:latest .

# 2. ONE-TIME: confirm the GPU shows up inside the container.
docker run --rm --gpus all gibbonbot:latest nvidia-smi

# 3. EVERY DAY: start a stand training run, detached.
scripts/train_bg.sh docker stand           # auto-resumes from runs/ape_stand_mjx/
# Warp backend is on by default (MUJOCO_BACKEND=mjwarp in docker-compose.yml).

# 4. Watch progress (separate terminal — Ctrl-C just stops the tail).
scripts/train_bg.sh tail 100
docker compose logs -f trainer
docker compose up -d mlflow_ui             # then open http://localhost:5000

# 5. Inspect the live policy in the GUI viewer (uses .venv, not Docker).
python view.py                             # newest checkpoint, hot-swaps
python view.py --best                      # best mean_rew per MLflow

# 6. Stop the trainer cleanly.
docker compose stop trainer
```

Just want to look at the rest pose, no training, no policy?

```bash
python look.py                             # data/scene.xml, gravity only
```

**Full Docker workflow + troubleshooting**: [docs/GET_STARTED.md](docs/GET_STARTED.md).

---

## Layout

```
.
├── train_mjx.py             PPO trainer; default --task=stand
├── view.py                  live policy viewer; hot-swap; default --env=stand
├── look.py                  static scene viewer (gravity only, no policy)
├── Dockerfile               CUDA 12 + JAX-cuda + MuJoCo MJX + Warp
├── docker-compose.yml       trainer + mlflow_ui services
├── requirements.txt
├── README.md                this file
│
├── data/
│   ├── scene.xml            ← THE scene; flush-mounted two-slab torso
│   ├── SCENE_GEOMETRY.md    ← required reading before edits
│   ├── motors.md            ST3235 / ST3215 / ST3215-HS specs
│   ├── controls.md          real-arm web-UI joint conventions
│   ├── meshes/              roarm-M3 STL parts (byte-identical to URDF)
│   └── scenes_archive/      historical scenes + ape_early.urdf (source of truth)
│
├── envs/                    one env+params pair per task
│   ├── stand_env_mjx.py     ← current focus
│   ├── walk_env_mjx.py      (parked)
│   ├── turn_{left,right}_env_mjx.py, walk_forward_env_mjx.py
│   ├── _shared_*.py         model load, obs builder, physics backend, dynamics
│   └── params_*_mjx.py
│
├── brain/                   bioinspired neural architecture
│   ├── ape_brain.py         integrator (Basal Ganglia routes between modules)
│   ├── neurotransmitters.py, brainstem.py, basal_ganglia.py
│   ├── cerebellum.py, amygdala.py, hippocampus.py
│   ├── prefrontal_cortex.py, motor_cortex.py
│   ├── episodic_store.py, semantic_map.py    (persistent memory)
│   └── quantum.py           quantum-ready interface (classical fallback)
│
├── scripts/                 one-shot tools — see "Tooling" below
├── utils/                   autopilot, eval, checkpoint, brain_hook
├── tests/                   pytest suite (imports, MJWarp integration)
├── runs/                    training output (gitignored; mirrored to Scaleway)
├── mlruns/, mlflow.db       MLflow tracking (gitignored)
├── docs/                    long-form documentation (see index below)
└── archive/                 parked code paths kept for reference
    ├── scripts/             train_walk_mjx.py shim, gen_terrain.py, gen_trees.py
    └── trees/               procedural olive-tree generator (grove_scene branch)
```

---

## Docs index

**Start here**

- **[docs/GET_STARTED.md](docs/GET_STARTED.md)** — canonical Docker workflow for stand training
- **[data/SCENE_GEOMETRY.md](data/SCENE_GEOMETRY.md)** — units, formulas, gotchas; required before editing `data/scene.xml`

**Robot & physics**

- **[data/motors.md](data/motors.md)** — Waveshare ST3235 / ST3215 / ST3215-HS specs and damping derivation
- **[data/controls.md](data/controls.md)** — real-arm web-UI joint conventions (sim-to-real mapping)
- **[docs/ROBOT_VS_APE.md](docs/ROBOT_VS_APE.md)** — joint-by-joint mapping from robot DOF to ape anatomy
- **[docs/MJWARP_INTEGRATION.md](docs/MJWARP_INTEGRATION.md)** — how MJX-Warp is dispatched

**Training**

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — remote-GPU paste-able commands
- **[docs/REMOTE_GPU.md](docs/REMOTE_GPU.md)** — H100 deep-dive (bootstrap, sync, teardown)
- **[docs/CLOUD_DEPLOY.md](docs/CLOUD_DEPLOY.md)** — Docker + cloud GPU deployment notes
- **[docs/OBJECT_STORAGE.md](docs/OBJECT_STORAGE.md)** — Scaleway sync for runs/, mlruns/, jax_cache/
- **[docs/CURRICULUM.md](docs/CURRICULUM.md)** — multi-skill curriculum (stand → walk → turn)
- **[docs/PLAN.md](docs/PLAN.md)** — autopilot training plan (phases A→D)
- **[docs/IMITATION_PLAN.md](docs/IMITATION_PLAN.md)** — ape-pose imitation roadmap
- **[docs/OPENAPEPOSE_INTEGRATION.md](docs/OPENAPEPOSE_INTEGRATION.md)** — pose dataset integration
- **[docs/PARAMS_HISTORY.md](docs/PARAMS_HISTORY.md)** — hyperparam evolution

**Bioinspired brain**

- **[docs/APEBRAIN.md](docs/APEBRAIN.md)** — 8-module overview
- **[docs/BRAIN_CORE.md](docs/BRAIN_CORE.md)** — long-term vision (instinct + memory architecture)
- **[docs/BIOLOGY.md](docs/BIOLOGY.md)** — evolutionary priors built into the policy
- **[docs/QUANTUM.md](docs/QUANTUM.md)** — quantum interface (classical fallback today)

**Future / world**

- **[docs/CORE.md](docs/CORE.md)** — multi-task framework + per-limb diagnostics
- **[docs/GROVE_MASTERPLAN.md](docs/GROVE_MASTERPLAN.md)** — virtual olive-grove environment
- **[docs/BRANCHES.md](docs/BRANCHES.md)** — branch atlas

**Notes**

- **[docs/LESSONS.md](docs/LESSONS.md)** — cumulative training insights (prose; numbers are in MLflow)
- **[docs/BACKLOG.md](docs/BACKLOG.md)** — prioritized list of unimplemented ideas

---

## Tooling cheat sheet

| Script | Purpose |
|---|---|
| `scripts/train_bg.sh docker stand` | Start stand training in Docker, detached, Warp backend |
| `scripts/train_bg.sh status\|tail\|stop` | Inspect / stop the active trainer |
| `look.py` | Static scene inspect (no policy, no checkpoint) |
| `view.py` | Live policy viewer; hot-swaps to newest checkpoint; default `--env stand` |
| `view.py --env <task> --best` | View best-mean_rew checkpoint per MLflow for any task |
| `scripts/dashboard.sh` | Local web dashboard (log tail, metrics, start/stop) → http://127.0.0.1:8765 |
| `scripts/snapshot.py` | 5-view PNGs + 13-keypoint 3D landmarks JSON |
| `scripts/calibrate_cpg.py` | Probe each joint; output direction-of-effect table |
| `scripts/verify_dynamics.py` | Audit physics + motors against `data/motors.md` |
| `scripts/compare_runs.py` | MLflow cross-run metric comparison |
| `scripts/note.py "msg"` | Attach prose note to current MLflow run |
| `scripts/ape_audit.sh` | List everything running in this repo (procs, GPU users, zombies) |
| `scripts/remote.sh {bootstrap\|start\|status\|log\|stop\|final_sync}` | Drive remote GPU training |
| `scripts/pull_checkpoints.sh` | Sync remote `runs/` + `mlruns/` to local every 30 s |
| `scripts/sync.sh` | One-shot push to all configured git remotes |

---

## Where things persist

| What | Where | In git? |
|---|---|---|
| Streaming checkpoints | `runs/<task>/mjx_*.npz` | no (local + Scaleway only) |
| Auto-promoted "keeper" checkpoints | `runs/<task>/keepers/auto_*.npz` + sidecar `.json` | no |
| MLflow run/metric history | `mlflow.db` + `mlruns/` | no (auto-pushed to Scaleway) |
| Per-run lineage | `runs/<task>/lineage.json` | no |
| Training log | `train.log`, `runs/<task>/train.log` | no |
| **Lessons learned (prose)** | `docs/LESSONS.md` | **yes** |
| **Hypothesis backlog** | `docs/BACKLOG.md` | **yes** |
| Robot meshes | `data/meshes/*.stl` | yes |
| Scene XML | `data/scene.xml` | yes |
| Source code | everything else under tracking | yes |

Training data is mirrored to **`scaleway:gibbonbot-training/`** (see
[docs/OBJECT_STORAGE.md](docs/OBJECT_STORAGE.md)) on a 10-minute tmux
loop while remote training is active. The expensive XLA autotune cache
(`~/.jax_cache/`, ~15 min cold compile) is preserved across remote
instance lifetimes via the same sync.

---

## Reference: motor model

Per-actuator velocity & torque limits come from `data/motors.md`
(Waveshare specs, derived as `damping = τ_stall / ω_no_load`):

- **ST3235** (shoulders/hips yaw+pitch, elbow/knee, forearm/shin roll) —
  2.94 N·m, 45 RPM (4.71 rad/s) → damping `0.62`
- **ST3215** (wrist, ankle) — same 2.94 N·m / 45 RPM (different gearset) → damping `0.62`
- **ST3215-HS** (gripper, foot) — 1.96 N·m, 106 RPM (11.10 rad/s) → damping `0.18`

Each joint follows a trapezoidal velocity profile (smooth start *and*
stop). Tunable in `envs/params_<task>_mjx.py`:

- `MOTOR_SPEED_FRAC` — global scale; `<1` = slower than real-hardware spec
- `MOTOR_ACCEL_RATIO` — how quickly joints reach max speed

## Reference: action mapping (rest pose)

Action `0` does **not** map to joint angle 0 — it maps to a configurable
**rest pose** (knees/elbows pre-bent in the standing direction). This
prevents the "dead spider" failure mode where all-zeros means all limbs
splayed horizontally. `action=±1` reaches the joint limits relative to
the rest pose. See `PER_LIMB_REST` in each `envs/<task>_env_mjx.py`.

## Reference: CPG baseline (walk task only)

`envs/walk_env_mjx.py` injects a sinusoidal gait baseline into every
action, so the policy learns *corrections* to a working gait instead of
inventing one from scratch. Currently pace gait
(`LIMB_PHASE = [0, π, 0, π]`); switch to trot via `[0, π, π, 0]`.
Tuned via per-limb sign correction from `scripts/calibrate_cpg.py`.
