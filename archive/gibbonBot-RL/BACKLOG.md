# Backlog

Things proposed in conversation that weren't implemented inline. Priority is my call (P0 = do next, P2 = nice-to-have). Remove a line when it's done or no longer relevant.

## P1 — high value, low cost

- **Dev pipeline audit.** Walk through `scripts/`, `utils/eval.py`, `utils/autopilot.py`, `keepers/` promotion, MLflow tracking, lineage.json — write up gaps and a recommended end-to-end flow (idea → param tweak → train → eval → promote → view). Cost: ~30 min reading + a doc.

## P2 — bigger lifts, do if motivation/PPO progress demands

- **AMP / DeepMimic-style imitation reward.** Retarget one curated ape locomotion clip onto the URDF, add an imitation reward term to the env, retrain. Would unlock natural-looking gait beyond what pure PPO converges to. Cost: 1-2 weeks (pose extraction → skeleton retarget → physics validation → reward integration → tuning). Only worth it if reward-shaping plateaus.
- **`mujoco.Renderer` snapshot script.** A small `scripts/snapshot.py` that loads a checkpoint, runs N steps in a single env, dumps frames to PNG. Lets me read frames via image-read tool to spot pose bugs without needing live screen access. Cost: ~30 lines.
- **OpenApePose / DeepLabCut dataset combine for a reference dataset.** Run pose estimation on real-ape video, build a 3D keypoint dataset for retargeting (prereq for the AMP item above). Cost: multi-week.

## P3 — speculative

- **Real-time CV comparison of MuJoCo viewer vs. real-ape video.** Attempted earlier; declined because pose estimators trained on real animals don't transfer to low-poly schematic robots. Re-evaluate only if we render a textured high-fidelity robot.
