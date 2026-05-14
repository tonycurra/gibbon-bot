# Autopilot training plan — quadruped walk-to-tree

What this doc is: the playbook I follow on a wakeup loop. You start me with
"run the plan" or "continue plan" and I keep cycling until the robot reaches
the tree.

**Goal:** robot starts at ~3 m from the tree → walks on 4 legs to within
0.6 m of the trunk → stays stable. Inspired by quadrupedal mammals.

**Hardware:** rented H100, ~€2.73/hr. Throughput ~200K sps at 16384 envs →
50 M steps ≈ 4 min wall, 200 M ≈ 17 min. Iteration cost is cheap; **stalled
training is what wastes money**.

---

## Phases

Each phase has: knobs, **success criterion** (numeric, measured from
`train.log` metric lines), and **stall fallback** if not reached in budget.
Auto-resume across phase boundaries: weights are preserved, only the
reward shaping and episode length change.

### Phase 0 — Infra check (current)
**Goal:** train.log flowing, mlflow getting metrics, monitor ok.
**Success:** at least 5 metric lines in train.log with non-NaN sps/mean_rew.
**Budget:** 5 min after restart. If still no log → fix.

### Phase A — Stand & survive (steps 0 → 30 M)
**Knobs:** keep current params_walk_mjx.py defaults. Robot doesn't have to
move yet — just not flip and roughly hold torso near TARGET_Z=0.30 m.
**Success:** mean_rew > 3.0 sustained 10 updates AND mean_dist NOT
diverging (i.e. still in the same arena, < 5.0 m).
**Stall fallback (if not by 50 M):**
  - FLIP_PENALTY 200 → 80 (was punishing exploration too hard)
  - W_GROUND 50 → 25 (lower quadratic cost for low torso)
  - ENT_COEF 0.05 → 0.10 (more exploration)

### Phase B — Walk toward tree (steps 30 M → 120 M)
**Knobs:** progress reward dominant.
  - W_PROGRESS 80 → 100
  - W_HEIGHT 4 → 6 (don't drop torso for cheaper progress)
**Success:** mean_dist trending down, < 1.8 m sustained 10 updates.
**Stall fallback (if not by 200 M):**
  - W_PROGRESS 100 → 150
  - LEARNING_RATE 0.00037 → 0.00020 (refine, not explore)
  - ENT_COEF 0.10 → 0.03

### Phase C — Reach the base (steps 120 M → 250 M)
**Knobs:** tighten success trigger.
  - SUCCESS_DIST 0.6 → 0.5 (final tolerance)
  - W_SUCCESS 50 → 100 (sharper terminal reward)
**Success:** mean_dist < 0.7 m sustained 10 updates AND success rate
(episodes ending in `dones==True` from goal hit) > 30%.
**Stall fallback (if not by 350 M):**
  - Increase MAX_STEPS 300 → 500 (more time per episode)
  - Reduce CLIP_EPS 0.15 → 0.10 (smaller policy steps)

### Phase D — Polish
**Goal:** consistent, stable approach. Smooth gait.
  - W_SMOOTH 0.1 → 0.5
  - W_JVEL 0.005 → 0.02
**Success criterion:** success rate > 70%, no flips for 50 episodes.
**Budget:** open-ended. User decides when to stop.

---

## Operating loop (what I do per wakeup)

```
on each wakeup (every ~5-10 min):
  1. read autopilot_state.json from runs/ape_walk_mjx/
  2. fetch fresh train.log + health.json from remote
  3. parse last N metric lines → current update, sps, mean_rew, mean_dist, total_steps
  4. compute phase progress vs phase's success criterion
  5. if criterion met → advance phase:
        - edit envs/params_walk_mjx.py with new phase's knobs
        - rsync to remote
        - send Ctrl-C to train tmux (graceful save)
        - restart with --resume (force-load past fingerprint mismatch)
        - log phase transition to autopilot.log
  6. if stalled (success budget exhausted, no progress) → apply fallback:
        - edit params with fallback values
        - rsync + restart as above
        - log fallback applied
  7. if catastrophic (NaN, crash, no log advance for 5+ min) → restart fresh,
     keep latest checkpoint
  8. write updated autopilot_state.json
  9. ScheduleWakeup for next cycle (5-10 min)
```

State tracked in `runs/ape_walk_mjx/autopilot_state.json`:
```json
{
  "phase": "A",
  "phase_started_at": "2026-05-09T18:00:00",
  "phase_started_steps": 5709824,
  "fallback_applied": false,
  "best_mean_rew": 0.0,
  "best_mean_dist": 3.0,
  "last_updated": "2026-05-09T18:05:00"
}
```

## What you see
- `autopilot.log` (text, append-only) — one line per wakeup decision
- `autopilot_state.json` — current state, machine-readable
- mlflow UI — full metric curves
- `view.py` (defaults to `--env stand`) or `view.py --best` — live policy

## How to start me
- "run the plan" / "start autopilot" / "continue plan" → I begin cycle 1
- "stop autopilot" → I stop scheduling wakeups
- "advance phase manually" → I skip ahead one phase
- "reset to phase X" → I rewind state

## What I will NOT do without asking
- Buy more hardware (cost change)
- Change MAX_STEPS by more than 2x (heavy compile churn)
- Push to remote-of-record (codeberg)
- Delete checkpoints

Everything else is fair game. Time and reward shaping are mine to tune.
