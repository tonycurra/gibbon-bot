# gibbonBot-RL

Reinforcement learning for a 4-limbed robotic ape. PPO + MuJoCo MJX (GPU-accelerated, JAX). Goal: lift the torso to a target height and stay upright.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [QUICKSTART.md](QUICKSTART.md) for the local workflow, or [REMOTE_GPU.md](REMOTE_GPU.md) for renting + training on a remote H100.

## Workflow

Two scripts. Run them in two terminals.

```bash
# Terminal 1 — train on GPU. Auto-detects VRAM and picks env count
# (256/512/1024/2048). The viewer is auto-spawned alongside.
python train_mjx.py

# Terminal 2 — only needed if you launched with --no-watch.
# Hot-swaps to the latest checkpoint each episode.
python view.py
```

Common variants:

```bash
python train_mjx.py --envs 1024     # explicit env count
python train_mjx.py --no-watch      # headless (no viewer subprocess)
python train_mjx.py --resume        # continue from latest checkpoint
```

## Visual sanity check (no training)

`view.py` is a standalone viewer. Edit `POLICY` at the top to choose what runs:

```python
POLICY = "zero"     # all joints at rest — see how gravity/damping/contacts behave
POLICY = "random"   # random actions — see full range of motion
POLICY = "latest"   # load the most recent training checkpoint
```

Then `python view.py`. Useful for verifying physics (friction, contact, motor limits) before committing to a long training run.

## Files

| Script | What it does |
|--------|-------------|
| `train_mjx.py` | MJX/JAX GPU training loop with vmapped parallel envs |
| `view.py` | Live MuJoCo viewer — runs latest checkpoint or `zero`/`random` policy |
| `envs/stand_env_mjx.py` | MJX environment (stateless, vmappable) |
| `envs/params_mjx.py` | Training parameters — edit values, restart training |
| `envs/stand_env.py` | Old CPU Gymnasium reference env (kept for reference, not the main path) |
| `data/scene.xml` | Robot MJCF (joint ranges/torques match real ST32xx motors) |
| `data/motors.md` | Motor specs (ST3235 / ST3215 / ST3215-HS) + per-joint mapping |
| `data/controls.md` | Real-robot web UI, safety, startup, sim-to-real notes |
| `utils/lineage.py` | Per-run hardware/env metadata logger |
| `utils/tag.py` | Tag checkpoints with notes ("this one was good") |
| `view_scene.py` | Standalone interactive viewer for `scene.xml` (pose the robot manually) |
| `serve_web.py` | Browser viewer — streams the live scene as MJPEG + remote camera/chat/terrain controls |
| `gen_terrain.py` | Procedural terrain generator (slope, roughness, soil type, sun) — patches `scene.xml` in place |
| `web/viewer.html` | Mobile-first UI consumed by `serve_web.py` |

## Browser viewer

```bash
source .venv/bin/activate
python serve_web.py
# open http://localhost:8080
```

What it does:
- Renders `data/scene.xml` offscreen at 1280×1280 @ 20 fps and streams JPEG frames over `multipart/x-mixed-replace`.
- Idles the render thread when no client is watching → 0% CPU when the tab is closed.
- Touch-first UI: drag = orbit, pinch / scroll = zoom, +/− buttons on mobile. Camera always orbits around the robot — lookat tracks the torso position continuously.
- Camera cannot clip through terrain: server computes the camera world position and bilinear-samples the heightfield at that point, reducing distance as needed to keep 20 cm clearance. Free tilt (any elevation, including looking up from below).
- Top-left **terrain panel** with live sliders (slope 0–60%, roughness, softness, grip, sun) and 7 soil presets: loam · rocky · sandy · clay · terra rossa · limestone · mud. Each preset sets roughness, softness, and grip — which maps to `solimp`/`solref`/`friction` in the physics. Hot-reloads gen_terrain.py and resets the robot to the home keyframe so it always spawns above the new surface.
- Chat input bottom posts to `/chat`; messages are echoed to all connected clients.
- Pause (Space), reset (R / ↺ button).

## Terrain generation

```bash
python gen_terrain.py --slope 15 --roughness 0.08 --softness 0.3 --grip 0.7 --sun-elev 60
```

| flag | range | meaning |
|------|-------|---------|
| `--slope` | 0–60 (%) | linear ramp across the field |
| `--roughness` | 0–0.30 m | bump amplitude |
| `--bump-scale` | grid cells | bump wavelength (default 20) |
| `--softness` | 0–1 | 0 = concrete, 1 = mud (sets `solimp` / `solref`) |
| `--grip` | 0–1 | 0 = ice, 1 = sticky soil (sets friction) |
| `--sun-azim` | deg | 0=+X, 90=+Y |
| `--sun-elev` | deg | 90=zenith, <0=night |
| `--brightness` | 0–1.5 | multiplier on top of time-of-day base |

The script writes `data/terrain.png` (256×256 heightfield) and patches `data/scene.xml` (hfield zmax, floor friction/solimp/solref, skybox + sun + ambient, torso spawn z, `home` keyframe). Re-run any time; it edits in place.

## Parameters

All MJX training params live in `envs/params_mjx.py` — plain Python config with comments above each value explaining the range. Edit and restart training.

## Tag checkpoints

```bash
python utils/tag.py "first stable stand"            # tag latest checkpoint
python utils/tag.py 262144 "before W_SMOOTH bump"   # tag by partial step match
python utils/tag.py --list                          # show notes
```

Notes go to `runs/ape_stand_mjx/notes.md` (git-tracked).

## Layout

```
envs/
  stand_env_mjx.py — MJX environment
  params_mjx.py    — MJX params
  stand_env.py     — CPU reference env
  params.py        — CPU params
data/
  scene.xml        — Robot MJCF
  motors.md        — Motor specs
  controls.md      — Real-robot notes
utils/
  lineage.py       — Run metadata logger
  tag.py           — Checkpoint tagger
runs/
  ape_stand_mjx/
    mjx_<steps>.npz   — policy weights
    lineage.json      — per-run metadata
    notes.md          — tagged checkpoints
mjx_history/
  ANALYSIS.md      — full history of viewer/training versions and what each did
```
