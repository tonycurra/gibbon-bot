# LIANA — Live Interactive Ape Neural Architecture

_The GibbonBot's running notebook + control interface._

> **Demo north star (locked 2026-05-15):** > Robot starts dumb, visibly improves while you talk *("look, it learns")
> Robot walks to the olive tree
 on command, reliably. Customer hits a button, the robot does it. Anything
> that doesn't move this forward is a tangent — push back, don't build it.
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
