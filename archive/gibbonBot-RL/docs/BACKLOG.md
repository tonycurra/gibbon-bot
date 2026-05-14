# Backlog

Things proposed in conversation that weren't implemented inline. Priority is my call (P0 = do next, P2 = nice-to-have). Remove a line when it's done or no longer relevant.

## P1 — high value, low cost

- **Brain × trainer deeper integration.** `BrainHook` ships the *companion-memory* layer: persistent episodic store + `ApeBrain.sleep()` consolidation. What's still missing: (a) feed `Hippocampus.recall_persistent` results into actor obs so PPO can condition on recall (requires OBS_DIM change, then a JIT-friendly host-callback or pre-baked recall tensor); (b) populate `SemanticMap` from rollouts — needs torso XY in obs or a separate env hook since current obs is target-relative; (c) wire BG action proposals into PPO exploration (bonus or mixed-action sampler). Each piece is its own small project — measure impact before stacking.
- **Brain-core: skill library + meta-policy.** First concrete step in `docs/BRAIN_CORE.md`. Add `envs/climb_env_mjx.py`, train a climb policy, hand-rolled meta-policy that switches walk → climb based on distance to tree. Validates the multi-skill architecture without committing to hierarchical RL. Cost: ~1 week. **STATUS**: foundation shipped 2026-05-10 — `train_mjx.py` is now task-parameterized, four atomic skills (stand, walk_forward, turn_left, turn_right) have env + params files, `scripts/curriculum.py` sequences them with cross-task weight transfer. See `docs/CURRICULUM.md`. Next: extend with climb + reach.
- **Auto-push training data to Codeberg.** Add a `scripts/sync.sh` that does `git add runs mlruns train.log && git commit -m "auto: $(date)" && git push`. Hook into the trainer's SIGTERM save path and/or a 30-min cron. Cost: ~15 lines.
- **Dev pipeline audit.** Walk through `scripts/`, `utils/eval.py`, `utils/autopilot.py`, `keepers/` promotion, MLflow tracking, lineage.json — write up gaps and a recommended end-to-end flow (idea → param tweak → train → eval → promote → view). Cost: ~30 min reading + a doc.

## P2 — durability / off-machine archive

- **Body-lift bootstrap reward.** CPG-only zero-policy motion doesn't translate the body forward; legs lift but stance phase doesn't generate friction because torso is on the ground. Add a strong reward for `torso_z > 0.20` so PPO discovers body lift earlier. Should accelerate locomotion emergence.
- **Cloud object store for runs/ archive (EU).** Hetzner Object Storage (~€5/mo/TB) or Backblaze B2 EU. Run `scripts/archive_runs.sh` on a schedule to push tarballs. Only worth doing once total artifact size approaches 1GB or you're worried about Codeberg outages. Currently 96MB → years of free-tier coverage.
- **Postgres-backed MLflow.** Only needed if multiple machines (laptop + remote H100) log to the same MLflow concurrently. With sqlite + git-push workflow you don't need this.

## P2 — curriculum / training strategy

- **Per-limb curriculum.** Train with one limb actuated at a time (others frozen at neutral), let the policy learn isolated arm/leg dynamics, then unfreeze and combine. Requires an actuator-mask hook in `walk_env_mjx.py` that zeros `ctrl[mask]` and a curriculum schedule in `train_walk_mjx.py`. User asked for this 2026-05-10; deferred because trapezoidal-motor + reward changes were higher-leverage first.

## P2 — bigger lifts, do if motivation/PPO progress demands

- **AMP / DeepMimic-style imitation reward.** Retarget one curated ape locomotion clip onto the URDF, add an imitation reward term to the env, retrain. Would unlock natural-looking gait beyond what pure PPO converges to. Cost: 1-2 weeks (pose extraction → skeleton retarget → physics validation → reward integration → tuning). Only worth it if reward-shaping plateaus.
- **`mujoco.Renderer` snapshot script.** A small `scripts/snapshot.py` that loads a checkpoint, runs N steps in a single env, dumps frames to PNG. Lets me read frames via image-read tool to spot pose bugs without needing live screen access. Cost: ~30 lines.
- **OpenApePose / DeepLabCut dataset combine for a reference dataset.** Run pose estimation on real-ape video, build a 3D keypoint dataset for retargeting (prereq for the AMP item above). Cost: multi-week.

## P3 — physics engine alternatives (track, don't migrate yet)

- **NVIDIA Isaac Lab / Isaac Sim / Omniverse** — full robotics platform with USD scenes, PhysX/Warp physics, photorealistic rendering, massively parallel RL. Heavy migration (3–6 weeks): rewrite scene as USD, redo envs in Isaac task format, lose calibrated CPG / motor profile, lock to RTX-only. Worth it ONLY if we add vision-based policies (cameras observing the tree / apples). Stay on MJX while observations are pure proprioception.


- **`mujoco_warp`** ([github.com/google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp)) — MuJoCo physics on NVIDIA Warp instead of JAX/XLA. **INTEGRATED** ✅ Production-ready (v3.8.1). 2–5× faster kernel launches than MJX. **Migration cost: 0** — transparent abstraction layer implemented in `envs/_shared_physics_backend.py`. Auto-detects MJWarp if installed, falls back to MJX. Same scene.xml, same physics semantics. Enable with `MUJOCO_BACKEND=mjwarp` or just install and it auto-detects.
- **NVIDIA Newton Physics** ([developer.nvidia.com/newton-physics](https://developer.nvidia.com/newton-physics)) — Warp-based engine with rigid + articulated + soft + cloth + fluids. USD scene format, Isaac Lab integration. **Production-ready (v1.1.0, April 2026)**. Migration cost: 2–4 weeks (scene rewrite to USD + new env layer). Concrete trigger: need soft soil deformation, ropes, or complex multi-physics. The abstraction layer in `envs/_shared_physics_backend.py` makes future integration straightforward.

## P3 — speculative

- **Real-time CV comparison of MuJoCo viewer vs. real-ape video.** Attempted earlier; declined because pose estimators trained on real animals don't transfer to low-poly schematic robots. Re-evaluate only if we render a textured high-fidelity robot.

## Reference — 2026 physics-engine landscape

Snapshot of the alternatives we could move to if MJX hits a hard ceiling.
Most of these are *not* on the roadmap; they're here so we remember the
options exist.

**High-throughput RL**
- **NVIDIA Isaac Lab** — successor to Isaac Gym, PhysX 5 + Isaac Sim,
  industry default for humanoid/quadruped locomotion on NVIDIA GPUs.
  RTX-locked. Already in P2 above with a 3–6 week migration estimate.
- **MJX** (current) — JAX-native MuJoCo, matches Isaac/Brax throughput
  while keeping the MuJoCo contact model.
- **Genesis** ([github.com/Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis))
  — claims "world's fastest" but the 43M-FPS number is a Franka-on-plane
  benchmark; for contact-heavy quadruped walking the lead shrinks or
  flips. Real reason to adopt: generative-simulation pipeline (4D scenes
  from natural language) — relevant when we want diverse training
  environments for the harvest task.

**Differentiable / open-source**
- **Newton** ([developer.nvidia.com/newton-physics](https://developer.nvidia.com/newton-physics))
  — already in P2. NVIDIA + DeepMind + Disney Research, Linux Foundation
  governance, Warp-based, designed for contact-rich manipulation.
  Concrete trigger: soft soil, ropes, or multi-physics.
- **Brax** — JAX-based, fully differentiable; pieces have merged into
  MJX, so adopting it would mostly mean dropping MJX-specific code paths.
- **DART** — mature multi-body dynamics, CPU-only, used in biomechanics.
  Not relevant for our GPU workflow.

**ROS 2 / hardware-in-loop**
- **Gazebo (Harmonic/Ionic)** — default ROS 2 simulator, rich sensor
  models. Use when we wire the trained policy into the real robot stack
  (cameras, LiDAR, IMU). Already targeted for the sim2real export step.
- **Webots** — lighter than Gazebo, good for swarms and education. Not
  for us.
- **O3DE** — AAA-grade graphics; only worth it if vision-based policies
  need photorealism.

**Specialized**
- **Drake** (TRI/MIT) — formal verification, complex contact analysis,
  not speed-focused. Would only matter if we need provable safety
  guarantees, e.g. before a real harvester goes near a person.
- **CoppeliaSim** — swappable backends (MuJoCo / Bullet / Newton in one
  scene). Useful for prototyping but slower than going native.

| Engine | Strength | License | Hardware |
|---|---|---|---|
| Isaac Lab | Humanoid/Quadruped RL | BSD-3 | NVIDIA GPU |
| MJX (current) | MuJoCo on JAX | Apache 2.0 | GPU/TPU |
| Genesis | Generative scenes, manipulation speed | Apache 2.0 | Multi-backend |
| Newton | Differentiable + contact-rich | Apache 2.0 | GPU/CPU |
| Gazebo | ROS 2 + sensors | Apache 2.0 | CPU/GPU |
| Drake | Formal verification | BSD-3 | CPU |
