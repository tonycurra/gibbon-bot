# LIANA — Live Interactive Ape Neural Architecture

_The GibbonBot's running notebook + control interface._

> **Demo north star (locked 2026-05-15):** Robot walks to the olive tree
> on command, reliably. Customer hits a button, the robot does it.
> Anything that doesn't move this forward is a tangent — push back.

## How to use it

```bash
./liana go             # spawn dashboard at http://localhost:8000/static/index.html
```

Then from the dashboard:
- **Task** dropdown — `walk` or `stand`
- **Scene** dropdown — pick from `data/scene*.xml` + archive
- **Envs** — parallel envs on GPU (default 2048)
- **Trial s** + **Speed** — how long each playback episode runs / how fast
- **Start** → trainer + auto-trial spawn. **Stop train** kills only the
  trainer; the trial loop keeps showing the best policy
- **Replay** / **Stop trial** — manual control of the playback loop
- **Demo walk** — scripted sine-wave gait (no RL) as a sanity reference
- **Library** — save the current best with a name, recall it later
- **Env Params** — read-only live dump of every active constant in
  `envs/env.py`
- **Runs** — last 20 training runs from `mlflow.db` (succ, rew, scene
  fingerprint, headline params)

Ctrl-C in the terminal tears down all subprocesses (dashboard, trainer,
trial).

## Tasks

| Task    | Reward                                                                                | Folder                  |
|---------|---------------------------------------------------------------------------------------|-------------------------|
| `walk`  | progress toward tree + upright + height − flip / ground / jerk                        | `runs/ape_walk_mjx/`    |
| `stand` | upright + height (tree-progress rewards **zeroed**) − flip / ground / jerk            | `runs/ape_stand_mjx/`   |

Reward weights live in `envs/env.py`; `read_params(task)` returns a
different tuple for `stand` vs `walk`. Folder layout is segregated so
keepers never mix.

## Architecture & where knowledge lives

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full data-flow
diagram and which artifact answers which question. TL;DR:

| What | Where |
|---|---|
| Scene, robot, reward code | git: `data/scene.xml`, `envs/env.py` |
| Policy weights | `runs/ape_<task>_mjx/keepers/*.npz` |
| Per-run metrics + params | MLflow (`mlruns/` + `mlflow.db`) |
| Episodic memory | `data/brain/*.sqlite` |
| Human notes | `LIANA.md` (this file) + `LIANA_LOG.md` |

## Canonical scene

`data/gibbonbot.xml` is the **frozen, do-not-edit** reference robot
scene. If `data/scene.xml` gets corrupted by a swap or a bad edit:

```
cp data/gibbonbot.xml data/scene.xml
```

and we're back to a known-good world. No tree, no shadows, no visual
overrides — just robot + floor + light. Future agents: don't touch it.

## Lessons (one-liner per learned thing)

_(empty — add as we learn)_

---

**Session log lives in [LIANA_LOG.md](LIANA_LOG.md).** Append a new entry
with `./liana log "what you did and why"`.
