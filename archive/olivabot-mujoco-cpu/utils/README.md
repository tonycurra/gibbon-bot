# utils/

Helper scripts — none of these are imported by the training loop.

| File | Purpose | How to run |
|------|---------|------------|
| `watch.py` | Live viewer — loads the latest checkpoint and replays episodes. Hot-swaps to newer checkpoints as they're saved. | `python utils/watch.py` |
| `sim.py` | Opens the MuJoCo scene with no policy — just physics. Useful for inspecting the robot model. | `python utils/sim.py` |
| `lineage.py` | Logs metadata about each training run to `runs/ape_stand/lineage.json` (timestamp, checkpoint inherited from, env params). | Called automatically by `train.py` |
