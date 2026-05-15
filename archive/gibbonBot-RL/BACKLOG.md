# Backlog — parked, not started

Ideas that came up but are **out of scope** for the current demo loop.
Don't start any of these until the basic `./liana go` loop is paying off
(robot visibly improving, customer demo north star moving forward).

## Parked

- **Teleop / live motor sliders in viewer** — drive actuators from the MuJoCo
  panel, override or augment the policy live.
- **Demonstration recording + behavior cloning** — record `(obs, ctrl)` while
  human drives, BC-pretrain the policy before PPO.
- **Climb-tree task** — new env, reward = grip + height; needs trunk-contact
  sensors and curriculum.
- **Dashboard extensions** — `/api/liana/*` endpoints, web log viewer, in-browser
  param sliders.
- **New brain modules** — Quantum-Basal, Amygdala, etc. left untouched until
  they have a concrete demo role.
- **Real-robot policy export** — long-term goal per memory; not in current scope.

## Promotion rule

A backlog item gets promoted only when **the customer demo concretely needs it**
and the current step's success criterion is met. Otherwise it stays here.
