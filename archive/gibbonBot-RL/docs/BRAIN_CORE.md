# Gibbon brain core — design vision

The long-term goal is a robot that **knows its body** the way a real ape does:
innate gait, innate posture, innate awareness of own limbs and gravity, with
learned skills *layered on top* of those instincts. This doc tracks where we
are, where we're going, and the concrete pieces needed to get there.

## What "instinct + memory" means concretely in RL terms

| Biological concept | RL/code translation | Status |
|---|---|---|
| Innate gait pattern | CPG (Central Pattern Generator) — sinusoidal baseline added to policy actions | ✅ implemented (pace, `envs/walk_env_mjx.py`) |
| Innate posture | Action-zero rest pose (knees/elbows pre-bent for weight-bearing) | ✅ implemented |
| Innate biomechanics | Per-actuator real-motor speed/torque limits + trapezoidal accel profile | ✅ implemented (from `data/motors.md`) |
| Body schema (proprioception) | Joint positions, velocities, actuator forces in observation | ✅ implemented |
| Awareness of gravity | Torso quaternion, height, upright reward | ✅ implemented |
| Vestibular sense | Torso angular velocity in obs | ✅ implemented |
| Spatial awareness | Tree-relative xy, distance, gait clock in obs | ✅ implemented |
| Skill memory across tasks | Multiple policy heads, one per skill, plus a meta-policy that selects | ❌ not yet — see "Skill library" below |
| Long-horizon memory | Recurrent policy (LSTM/transformer) carrying state across episodes | ❌ not yet — see "Recurrent policy" |
| Curriculum learning | Progressive task difficulty (e.g. flat → slope → uneven → grove) | ❌ not yet |

## Where we are now

- **Policy**: stateless feedforward MLP (88 → 256 → 256 → 128 → 24-dim Gaussian).
  No memory across action steps beyond what's encoded in the obs.
- **Innate priors**: CPG, rest pose, motor limits, joint damping calibrated from
  real motor specs. The robot has built-in gait rhythm and biomechanical sanity.
- **Reward shaping**: progress, fwd-velocity, heading, success, upright, height,
  smoothness, ground-clearance, flip penalty, energy penalty. Comprehensive but
  task-specific (walk-to-tree).
- **Skill set**: one — walk-to-tree. Climb, brachiate, reach, grasp are not
  trained yet.

## Brain-core roadmap (priority order)

### 1. Skill library + meta-policy

**What**: train multiple specialized policies (`walk`, `climb`, `reach`),
each with its own reward + env. Add a thin **meta-policy** that, given a goal
+ current state, selects which skill to invoke.

**Code shape**:
```
envs/walk_env_mjx.py       # already exists
envs/climb_env_mjx.py      # NEW — climb a tree to height threshold
envs/reach_env_mjx.py      # NEW — bring gripper to a target position
runs/<skill>/keepers/      # promoted policies per skill
utils/meta_policy.py       # NEW — given task, pick a skill
```

**Why this first**: most of the infrastructure (PPO, MJX, MLflow, auto-push,
viewer, snapshot) is task-agnostic — only the env + obs need swapping.

### 2. Recurrent policy (memory across action steps)

**What**: LSTM or 1-block transformer between obs encoder and action head.
Hidden state carries between steps so the policy can integrate history (e.g.
"I've been pushing forward for 3 steps and the body isn't moving — switch
strategy").

**Why second**: for complex behaviors like climbing or carrying, lack of memory
becomes a hard ceiling. For pure walking, feedforward is fine.

### 3. World model (predictive instinct)

**What**: a small network that learns to predict next-obs given current
obs + action. Used either for model-based RL (Dyna-style) or as auxiliary loss
to shape the encoder. The robot effectively learns "what happens if I do X" —
proto-imagination.

**Why third**: significant complexity; only worth it after skills + memory work.

### 4. Curriculum

**What**: train on progressively harder envs. flat ground → soft ground →
slopes → uneven terrain → grove with obstacles. Each stage uses the previous
stage's policy as a warm start.

**Why fourth**: scaffolds the hard final task (olive grove) by guaranteeing
each transition is small enough to be learnable.

### 5. Cross-skill transfer

**What**: shared encoder across skills, trained with auxiliary losses
(BYOL/SimSiam-style self-supervised representation learning on proprioception
sequences). Skills then become small task heads on top of a rich body-schema
representation. This is the closest thing to "biological brain": a single
"motor cortex" with task-specific output muscles.

## Comparing training data — how to evaluate progress

Run `scripts/compare_runs.py` to see metrics across recent runs side-by-side.
Useful columns:

- `mean_rew` — average reward per env-step at the end of the run
- `mean_dist` — average final distance to goal (lower = better walking)
- `mean_speed` — average forward speed in m/s
- `success_frac` — fraction of episodes that reached the tree
- `mean_upright` — torso-up alignment (1.0 = perfect)
- `gait_coherence` — diagonal-pair phase agreement (positive = coordinated trot/pace)

```bash
scripts/compare_runs.py --top 10                    # last 10 runs
scripts/compare_runs.py --label "CPG"               # only CPG-related
scripts/compare_runs.py --plot reward_curves.png    # save plot
```

For prose-level lessons, see `LESSONS.md`. Numbers in `mlflow.db`.

## Architecture analogy: what a real ape brain has

| Brain region | Role | Robot equivalent today | Roadmap |
|---|---|---|---|
| Cerebellum | Motor coordination, gait timing | CPG (`envs/walk_env_mjx.py`) | Tune per-skill |
| Motor cortex | Voluntary movement | Policy MLP | Recurrent + multi-head |
| Brainstem | Basic reflexes (balance, gravity) | Upright reward + motor limits | (sufficient) |
| Hippocampus | Spatial memory, navigation | obs has tree xy | Add map-of-grove obs once grove exists |
| Prefrontal cortex | Goal-directed planning | None — purely reactive policy | Meta-policy + world model |
| Vestibular system | Balance | Torso quat + qvel obs | (sufficient) |
| Proprioception | Knowing where your limbs are | Joint pos/vel/torque obs | (sufficient) |

## Minimum viable next step (concrete, one-week-ish)

1. Define `envs/climb_env_mjx.py` — same robot, reward = torso z height
   gained against tree (not distance to it).
2. Train a climb policy. Promote keeper.
3. `utils/meta_policy.py`: a hardcoded rule for now — if torso > 1m and tree
   in arm reach, switch to climb; else walk.
4. Demo: spawn far from tree, walk to it, then climb.

That's 2 envs, 2 policies, 1 hand-rolled selector. No new ML beyond the
existing PPO. It validates the "skill library + meta-policy" architecture
without committing to hierarchical RL or world models.
