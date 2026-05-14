# Ape-imitation training plan

Status: **planning** — nothing implemented yet. This doc captures the
decision so we don't re-litigate it next session.

## Goal

Train the robot to walk and climb by imitating real ape movement,
rather than relying purely on hand-shaped reward.

## The morphology blocker

Today's URDF has roughly equal-length arms and legs. Apes have arms
~1.3-1.5× leg length, longer phalanges, and very different shoulder /
hip ROM. **Strict joint-angle imitation cannot work** on the current
robot — even a perfect retarget would fail at climbing because the
geometry doesn't reach.

**Decision**: rework the URDF to ape-like proportions before any
imitation training is started. Targets:
- Arm length ≈ 1.4× leg length
- Extended shoulder ROM (overhead reach)
- Longer fore-paw / phalange segments (grip on branches)

## Datasets surveyed

Most pose datasets in the wild are 2D-only. There is no DeepMimic-grade
3D ape mocap library. Triage:

| Source | Usable for |
|---|---|
| OpenApePose (2D photos) | Bootstrapping a pose-estimator on our own clips. **Not motion.** |
| OpenMonkey-style (arXiv 2212.00741) | 2D pose only. Useful for gait-phase features, not joint targets. |
| DataDryad chimp set (10.5061/dryad.c59zw3rds) | If 3D — use as AMP positive examples. Verify scope first. |
| eLife / PMC9160209 / bioRxiv 2021.09.08 | Treadmill / lab biomech, often 2D + sparse 3D. Good for gait timing priors. |

Quality > quantity — 30s of clean clips beats hours of noisy data.

## Path chosen: AMP-style discriminator (not full DeepMimic)

Adversarial Motion Priors. Train a discriminator to score "ape-like"
vs "robot-random" trajectories, add its score as a bonus reward
alongside the existing walk reward. Distribution-matching, not
joint-matching — tolerates remaining morphology mismatch.

**Why not DeepMimic / full retargeting**: 1-3 month project, would
spend weeks fighting morphology even after URDF fix, and 3D ape data
is too sparse to feed it well.

## Phased plan

1. **URDF rework** (~1 day). Ape-proportioned skeleton committed to
   a new branch (e.g. `ape-morphology`). Validate it can stand and
   move under existing walk reward before touching imitation.
2. **Motion clip prep** (~1 week). Pull 3D-where-available, lift 2D-
   to-3D where needed, normalize to a canonical skeleton, slice into
   ~1-3s windows. Land in `data/motion_clips/`.
3. **AMP integration** (~2 weeks). Add discriminator network, motion
   clip sampler, and AMP reward term to `train_mjx.py`. Keep
   task reward as the dominant signal; AMP as a style bonus.
4. **Tune** (~1 week). The discriminator can collapse or dominate —
   needs entropy / loss tuning.

Total: ~1 month focused work on top of URDF rework.

## What we are NOT doing (yet)

- Full 3D pose retargeting onto the robot skeleton
- Per-joint imitation reward (DeepMimic-style)
- Curriculum across multiple ape behaviors simultaneously

These remain options later if AMP plateaus.

## References

- [AMP: Adversarial Motion Priors](https://xbpeng.github.io/projects/AMP/) — the technique we're copying
- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/) — the heavier alternative we're rejecting
- Datasets: OpenApePose, OpenMonkey, DataDryad chimp set, eLife / PMC / bioRxiv biomech papers (links in commit history)
