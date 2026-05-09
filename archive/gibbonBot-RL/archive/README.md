# archive/

Older code paths kept for reference. **Not part of the active workflow.**

| File / dir | Why archived |
|---|---|
| `train_mjx.py` | Stand-task trainer, superseded by `train_walk_mjx.py` at the repo root |
| `test.py` | CPU stable-baselines3 trainer. MJX is the primary path now. |
| `view_scene.py` | Passive MuJoCo viewer for posing the robot. Use `view.py` instead. |
| `serve_web.py` + `web/` | Browser-based MJPEG viewer. Parked on the `html_viewer` branch; develop against `view.py` only. |
| `mjx_history/` | Historical snapshots of viewer / training scripts at notable commits. Gitignored. |
| `training_output.log` | Stale stdout dump. Gitignored. |

If you ever need one of these, they're still here — just `cp archive/<file> .`. Or check `docs/BRANCHES.md` for which branch each direction lives on.
