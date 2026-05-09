# Gibbon-bot core — strategic build

A multi-task gym for a 4-limbed robot with morphologically identical limbs,
with per-limb dynamics readout, evaluation suite, and (later) ape pose
similarity grading. Keeps current `train_walk_mjx.py` working while we
scaffold a cleaner foundation alongside.

## What "core" means

A single Gym-style env with pluggable tasks. One physics model, one obs
space, swap reward function and reset distribution per task.

```
envs/
├── gibbon_core.py          ← physics, obs, action, base step machinery (NEW, extracted from walk_env_mjx.py)
├── tasks/
│   ├── stand.py            ← reward: hold torso z, no flip
│   ├── walk.py             ← reward: forward velocity + reach target
│   ├── reach.py            ← reward: extend named limb to target xyz
│   ├── balance.py          ← reward: recover from external push
│   ├── limb_lift.py        ← reward: lift limb i without falling
│   └── symmetric_gait.py   ← reward: matched left-right limb phase
└── walk_env_mjx.py         ← legacy, keep for now, deprecate after pivot
```

Trainer takes `--task walk` etc. Task modules expose:
- `compute_reward(data, prev_data, params) → (reward, done, info)`
- `reset_state(rng, model) → qpos, qvel`
- `obs_extra(data) → optional task-specific observations` (e.g. for reach: target xyz)

## Per-limb dynamics — already shipped

`utils/limb_dynamics.py` aggregates Mujoco joint-level fields into per-limb
metrics. Use cases:
1. **URDF sanity** — at rest, all 4 limbs should report identical bias
   force. Confirmed: 0.3826 N for all four.
2. **Gait analysis at inference time** — log torque/energy per limb each
   step, plot to see gait phase symmetry.
3. **Reward terms** — add `W_LIMB_SYM * |FL-FR|` to penalize lopsided gaits.

```python
from utils.limb_dynamics import per_limb, symmetry_score, energy
torques = per_limb(data, "qfrc_actuator", "abs_sum")
asym    = symmetry_score(torques)
```

## Test battery (planned)

`tests/body_control/`:
1. **limb_identity.py** — apply same trajectory to each limb, assert
   responses match within tolerance. Pure validation, no RL.
2. **stand_stability.py** — run trained stand policy 5 s, check torso z
   drift < 2 cm.
3. **reach_accuracy.py** — for each limb, command target xyz, check
   end-effector error < 5 cm.
4. **balance_recovery.py** — apply external impulse to torso, time to
   return upright.
5. **single_limb_lift.py** — lift each limb in turn, hold 1 s, no fall.
6. **forward_speed.py** — measured m/s on flat ground.
7. **walk_to_target.py** — time to reach 0.5 m radius of (3, 0).

Each test is a standalone script that loads a checkpoint and prints a pass/fail
+ numeric score. Aggregated by `tests/run_all.py` into a scorecard:

```
TASK                  CKPT      SCORE   TARGET   PASS
stand_stability       v3        0.012   < 0.020   ✓
reach_FL              v3        0.041   < 0.050   ✓
walk_speed            v3        0.18    > 0.30    ✗
walk_to_target        v3        18.4 s  < 30 s    ✓
limb_symmetry_walk    v3        0.18    < 0.30    ✓
```

## Evaluation — three layers

1. **Numeric** — task success rate, torque per limb, energy efficiency.
   Logged to mlflow per training run.
2. **Symmetry** — per-limb force balance via `limb_dynamics.symmetry_score`.
   Should be near-zero for a well-behaved gait.
3. **Pose similarity** — render robot to RGB → run pretrained 2D pose
   estimator → match keypoints to clips from OpenApePose / Dryad chimp /
   eLife biomech. Layer 1 from `docs/IMITATION_PLAN.md`. Defer until we have
   a policy that moves at all.

## Build order

1. **Already done:** `utils/limb_dynamics.py`
2. **Next cycle:** extract `envs/gibbon_core.py` from `walk_env_mjx.py`,
   port walk task to `envs/tasks/walk.py`. Old training keeps working from
   `walk_env_mjx.py` until we cut over.
3. **Then:** stand.py + reach.py tasks.
4. **Then:** test battery (1-2-3 first, the rest after).
5. **Then:** evaluation scorecard.
6. **Later:** ape pose similarity (Layer 1 of IMITATION_PLAN).

H100 keeps training the walk task throughout. Refactor doesn't block training.

## Lost knowledge from old .npz?

For the walk task, the 8.85M-step weights converged to "stand still cushy"
because of the broken reward shaping. They're an *anchor against new reward*,
not useful prior. Either reset to zero or keep one in `runs/ape_walk_mjx/keepers/`
as a curio. Not worth fighting to preserve.

For other tasks (stand, reach, etc.) we'll start fresh anyway — different
reward, different optimum.
