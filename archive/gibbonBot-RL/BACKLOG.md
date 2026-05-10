# Backlog

Things proposed in conversation that weren't implemented inline. Priority is my call (P0 = do next, P2 = nice-to-have). Remove a line when it's done or no longer relevant.

## P1 — high value, low cost

- **Auto-push training data to Codeberg.** Add a `scripts/sync.sh` that does `git add runs mlruns train.log && git commit -m "auto: $(date)" && git push`. Hook into the trainer's SIGTERM save path and/or a 30-min cron. Cost: ~15 lines.
- **Dev pipeline audit.** Walk through `scripts/`, `utils/eval.py`, `utils/autopilot.py`, `keepers/` promotion, MLflow tracking, lineage.json — write up gaps and a recommended end-to-end flow (idea → param tweak → train → eval → promote → view). Cost: ~30 min reading + a doc.

## P2 — durability / off-machine archive

- **Cloud object store for runs/ archive (EU).** Hetzner Object Storage (~€5/mo/TB) or Backblaze B2 EU. Run `scripts/archive_runs.sh` on a schedule to push tarballs. Only worth doing once total artifact size approaches 1GB or you're worried about Codeberg outages. Currently 96MB → years of free-tier coverage.
- **Postgres-backed MLflow.** Only needed if multiple machines (laptop + remote H100) log to the same MLflow concurrently. With sqlite + git-push workflow you don't need this.

## P2 — curriculum / training strategy

- **Per-limb curriculum.** Train with one limb actuated at a time (others frozen at neutral), let the policy learn isolated arm/leg dynamics, then unfreeze and combine. Requires an actuator-mask hook in `walk_env_mjx.py` that zeros `ctrl[mask]` and a curriculum schedule in `train_walk_mjx.py`. User asked for this 2026-05-10; deferred because trapezoidal-motor + reward changes were higher-leverage first.

## P2 — bigger lifts, do if motivation/PPO progress demands

- **AMP / DeepMimic-style imitation reward.** Retarget one curated ape locomotion clip onto the URDF, add an imitation reward term to the env, retrain. Would unlock natural-looking gait beyond what pure PPO converges to. Cost: 1-2 weeks (pose extraction → skeleton retarget → physics validation → reward integration → tuning). Only worth it if reward-shaping plateaus.
- **`mujoco.Renderer` snapshot script.** A small `scripts/snapshot.py` that loads a checkpoint, runs N steps in a single env, dumps frames to PNG. Lets me read frames via image-read tool to spot pose bugs without needing live screen access. Cost: ~30 lines.
- **OpenApePose / DeepLabCut dataset combine for a reference dataset.** Run pose estimation on real-ape video, build a 3D keypoint dataset for retargeting (prereq for the AMP item above). Cost: multi-week.

## P3 — speculative

- **Real-time CV comparison of MuJoCo viewer vs. real-ape video.** Attempted earlier; declined because pose estimators trained on real animals don't transfer to low-poly schematic robots. Re-evaluate only if we render a textured high-fidelity robot.
