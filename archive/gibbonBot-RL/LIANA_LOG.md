# LIANA — Session Log

Append-only diary. Every `./liana go`, every param tweak, every fix gets
a dated entry here. Index lives in [LIANA.md](LIANA.md) — this file is
just the raw timeline.

---

## 2026-05-15 12:37
started: task=walk

## 2026-05-15 12:45
started: task=walk  (dashboard http://localhost:8000/static/index.html)

## 2026-05-15 12:46
fixed: re-injected t0_trunk capsule into data/scene.xml at (3,0) — walk env was failing on missing tree geom

## 2026-05-15 12:46
started: task=walk  (dashboard http://localhost:8000/static/index.html)

## 2026-05-15 12:49
flow change: ./liana go now launches dashboard only. Training is started from the web UI. Default envs=1 for real-time viewing; trainer dumps qpos every step when envs<=8.

## 2026-05-15 12:51
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:20
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:30
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:36
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:39
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:40
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:48
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:57
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 13:59
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 14:14
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 14:22
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 14:29
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 14:35
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 14:49
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 14:54
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 14:58
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 15:04
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 15:06
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 15:30 — refactor cycle
- Single env file `envs/env.py`; archived per-task envs to `envs/_archive/`.
- Library card: save/use/delete named policies.
- Inline Runs card reads `mlflow.db` directly — no separate MLflow UI.
- Trial loop honors `ACTION_SMOOTH_ALPHA` + `MOTOR_SPEED_FRAC`.
- Trial default speed → 0.5× wall clock; survives Stop train.
- `read_params(task)` switches walk vs stand rewards.

## 2026-05-15 15:15
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 15:40
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html

## 2026-05-15 15:56
started: dashboard only (no trainer)  → http://localhost:8000/static/index.html
