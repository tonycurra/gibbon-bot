# Biological priors in walk_env_mjx

What evolutionary biology actually buys us, what's implemented, what's
not, and the references behind each piece.

## Implemented

### 1. Spinal CPG-style phase clock (obs)
Vertebrates inherit *Central Pattern Generators* from fish — neural
oscillators in the spinal cord that produce rhythmic limb movement
without descending control. Tetrapod gaits are phase patterns over this
basic clock.

We give the policy a sin/cos clock of period `GAIT_PERIOD_STEPS=25` env
steps (~0.8 s at 16 substeps × 0.002 s). The network sees:
```
obs[86] = sin(2π · t / 25)
obs[87] = cos(2π · t / 25)
```
This means the network can produce phase-locked output without learning
to track time internally. It's a strong inductive bias toward periodic
motion.

References:
- Grillner & Wallén 1985, *Central Pattern Generators for Locomotion*
- Ijspeert 2008, *Central pattern generators for locomotion control in animals and robots*
- Owaki & Ishiguro 2017, *A Quadruped Robot Exhibiting Spontaneous Gait Transitions*

### 2. Bilateral symmetry penalty (reward)
Tetrapods evolved bilaterally symmetric gaits — at any given gait
(walk/trot/pace/bound), the *magnitude* of left/right limb torques
matches even when the *phase* differs. Asymmetry usually means lameness
or chaos.

We penalize per-limb torque magnitude differences:
```python
W_SYMMETRY * (|FL - FR|² + |BL - BR|²)
```
Default `W_SYMMETRY=0.005`. Doesn't enforce phase (so trot and bound
are both fine), only that the two sides are *equally loaded*.

References:
- Hildebrand 1976, *Analysis of tetrapod gaits*
- Alexander 2003, *Principles of Animal Locomotion*

### 3. Forward velocity reward (already shipped)
Quadrupeds optimize for energy-efficient forward translation. We reward
body velocity dotted with the unit vector to the goal — directly pays
for forward motion even before progress accumulates.

### 4. Per-limb dynamics readout (`utils/limb_dynamics.py`)
Verified URDF symmetry: at rest, all 4 limbs report identical bias
force (0.3826 N). Available reductions: sum, mean, abs_sum, rms, max,
plus pairwise asymmetry scores. Use it to log gait quality during
training or inference.

## Not yet implemented (cheap wins)

### Actuator latency
Real Waveshare ST3215/3235 servos respond ~10-20 ms after a command
arrives. Mujoco currently has zero lag → policy learns to rely on
instantaneous response that won't transfer to hardware.

Fix: delay-buffer the action. ~10 lines, will add when training is
converging on the new reward.

### Foot/paw model
Real animal locomotion uses viscoelastic foot pads + toe articulation.
Our limbs end in capsules with simple friction. Adds compliance and
shock absorption that matter for both stability and contact realism.

Fix: replace end-capsules with multi-segment toe + soft pad geom.
Bigger URDF rework, defer.

### Tendon-driven joints
Vertebrates have springy musculotendon units that store and return
energy passively. Mujoco supports tendons via `<tendon>` elements.
Lossless gait improvements possible but requires URDF rework.

## Not implemented (research-grade, defer)

### Ape-pose imitation reward
Layer 2 from `IMITATION_PLAN.md`. Render robot to RGB on the H100, run
2D pose estimator, score similarity to OpenApePose / Dryad chimp clips.
Costs ~30% sim throughput. Wait until current policy actually walks.

### AMP (Adversarial Motion Priors)
Train a discriminator on whatever animal motion clips we can collect,
add as a style reward bonus. Tolerant to morphology mismatch.
Multi-week project. Right ROI once basic locomotion works.

### Curriculum across terrain types
We have `gen_terrain.py` with slope/roughness/soil presets. A real
curriculum: train on flat → patch in 5°-slope → random slopes → mud
patches → grass. Tetrapods evolved on highly varied terrain; the
policy should too.

Cheap to enable, just need a training loop variant. Defer until after
walk converges on flat.

## What we will NOT pretend to do

- Whole-genome reconstruction of a quadruped lineage. There's no
  database for that.
- "Evolutionary" RL (genetic algorithms etc). PPO + CPG prior is more
  efficient than evolving from scratch.
- Direct copying of *humanoid* locomotion. Different morphology;
  bipedal priors hurt more than help on a quadruped.

## Effect on training

These priors mean the network has fewer things to learn from scratch.
Specifically:
- Periodicity: built-in via phase clock. No need to discover that
  walking is rhythmic.
- Bilateral coupling: rewarded directly. Asymmetric gaits get a
  penalty signal regardless of whether they make progress.
- Phase patterns (trot vs walk vs bound): still learned, but on top of
  the clock signal so much faster.

Expected gain: walk should emerge in 30-60M steps instead of 200M.
Will measure and update.
