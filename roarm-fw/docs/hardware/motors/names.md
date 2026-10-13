# What to say (human names)

Stand **behind the base**, arm reaching **away from you** — like it is your right arm on a desk.

Bus IDs stay 11–17 in firmware. Console + info page use **yaw / pitch / roll** (axes) plus these room words. Direction vs `pos` (+/−) is filled in during the one-joint limit pass. Do not guess the sign.

## Axes (yaw pitch roll)

Stand behind the base. Thumb along the forearm toward the fingers:

| Axis | What rotates | This arm |
| --- | --- | --- |
| **Yaw** | around vertical | **shoulder yaw** — id **11** (the base). Turn left / right. |
| **Pitch** | nod, lift / drop | **shoulder pitch** 12+13, **elbow** 14, **wrist pitch** 15 |
| **Roll** | twist along the forearm | **wrist roll** — id **16** (twist) |
| (grip) | not a rotation axis | **hand** — id **17** |

Shoulder is **two joints**: yaw (one motor) and pitch (two motors on one horn, physically coupled — never drive 12 or 13 alone).

| Say | Joint | Axis | ID | Motor | Other names (do not use in the room) |
| --- | --- | --- | --- | --- |
| **turn** left / right | shoulder yaw | yaw | 11 | ST3235 | base, B L / B R |
| **raise** / **lower** | shoulder pitch | pitch | **12+13** | ST3235 ×2 | shoulder-A/B, S U / S D |
| **bend** / **straighten** | elbow | pitch | 14 | ST3235 | E U / E D |
| **wrist up** / **wrist down** | wrist pitch | pitch | 15 | ST3235 | wrist 1, W+ / W− |
| **twist** clockwise / counterclockwise | wrist roll | roll | 16 | ST3215 | Slack mid **4068**. Your CW − to **978**. CCW + to `un` **7126**. `docs/baselines/twist.md` |
| **open hand** / **close hand** | hand | grip | 17 | ST3215HS | gripper. Open − to **609** (cable wall). Close + to **2088**. Gold: `docs/baselines/hand-open-close.md` |

**Raise is one joint.** Two motors, one horn pair. Move the upper arm, not one servo by itself.

## Directions (how we mean them)

| Say | Picture |
| --- | --- |
| turn left / right | Base spins. Left = your left, standing behind the base. |
| raise / lower | Upper arm lifts off the table / drops toward it. |
| bend / straighten | Elbow folds the forearm in / opens it out. |
| wrist up / down | Hand nods. Fingers tip up / tip down. Not a twist. |
| twist clockwise / counterclockwise | Looking **along the forearm toward the fingers**. Palm/jaws spin. |
| open hand / close hand | Jaws spread / jaws meet. |

Clockwise is only for **twist**. Do not say “rotate” — it mixes twist and turn.

## Six moves, seven IDs

```
turn     11
raise    12 and 13 together
bend     14
wrist    15
twist    16
hand     17
```

## Next

Flash chip `ESTOP` before any more twist command. Then wrist up / down. Numbers: `joint-limits.md`. Sitting: `docs/baselines/twist-sitting-2026-09-05.md`.
