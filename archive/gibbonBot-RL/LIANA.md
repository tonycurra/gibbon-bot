# LIANA — Live Interactive Ape Neural Architecture

_The gibbonBot's running notebook + control interface._

> **Demo north star:** Customer sees the robot in MuJoCo, doing increasingly
> complex stuff (stand → walk → reach tree). You can pause and tell Claude
> to change something on the spot; the change shows up within seconds.

This file is the brain's diary. Append-only. Every meaningful tweak gets a
dated entry — what was changed, why, and (later) the result. Scrolling this
file should let anyone catch up on what we've tried and what we know.

## How to use it

- `./liana go` — start trainer + viewer (default task `walk`, phase A is stand-and-survive)
- `./liana go stand` — train pure stand
- `./liana stop` — kill both
- `./liana status` — what's running + last 20 log lines
- `./liana log "tweak X because Y"` — drop a note into this file

When you tell Claude "make it lean forward more" / "stop the wobble" /
"this is going nowhere", Claude edits the relevant
`envs/params_<task>_mjx.py` and appends a `## <date> — <task>` entry below.

## Current state

- Task focus: **walk** (autopilot phase A → D handles stand → walk → reach)
- Active branch: `brain-modeling`
- See `utils/autopilot.py` for phase knobs and success criteria.

## Tasks summary

| Task         | Env file                            | Best policy                          |
|--------------|-------------------------------------|--------------------------------------|
| stand        | `envs/stand_env_mjx.py`             | `runs/ape_stand_mjx/`                |
| walk         | `envs/walk_env_mjx.py`              | `runs/ape_walk_mjx/`                 |
| walk_forward | `envs/walk_forward_env_mjx.py`      | `runs/ape_walk_forward_mjx/`         |
| turn_left    | `envs/turn_left_env_mjx.py`         | `runs/ape_turn_left_mjx/` (if any)   |
| turn_right   | `envs/turn_right_env_mjx.py`        | `runs/ape_turn_right_mjx/` (if any)  |
| climb        | _not yet — see BACKLOG.md_          | —                                    |

## Lessons (one-liner per learned thing)

_(empty — add as we learn)_

---

# Session log

_New entries appended below by `./liana log "..."` and by Claude when changes are made._

## 2026-05-15 12:37
started: task=walk

## 2026-05-15 12:45
started: task=walk  (dashboard http://localhost:8000/static/index.html)

## 2026-05-15 12:46
fixed: re-injected t0_trunk capsule into data/scene.xml at (3,0) — walk env was failing on missing tree geom
