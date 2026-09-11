# Calibrate

`/calibrate/` is the six-joint index. `/calibrate/<slug>/` is one joint. Marks live in host `Limit` (`GET/POST /api/limits`). Chip `lim N min|max|home` is only for non-wrap IDs (raw 0–4095). Wrap walls stay host Δ.

Sitting 2026-09-11: `../roarm-fw/development-logs/2026-09-11-calibrate.md`.

## Wrap joints (yaw 11, wrist-roll 16)

Encoder: **360° = 0–4095**, resolution **360° / 4096**. Wrap is the magnet at 0/4095, not a cable wall. Host `wrapSess` unwraps so Δ stays monotonic through the seam. `turns` is signed magnet turns.

Stand behind the base. Wrist roll: looking along the forearm toward the fingers. **CW = counts fall (−Δ). CCW = counts rise (+Δ).**

1. Slack the loom.
2. **here** — Δ = 0° at this slack. Pack-off forgets chip turns; host origin is `Limit.home`.
3. **calibrate** — tq off, start learn.
4. Move both ways by hand. **− / now / +** show degrees; counts underneath.
5. **done** — saves the swept span if it is ≥ **80** counts (~7°). Does **not** require crossing 0° both ways.
6. Slider is that Δ range. Disabled during learn and before here. Speed bar on the wrap page.

Test/sweep stay off for wrap until a wrap-walk exists (`WritePos` will not hop across 0).

Leftover raw min/max that sit on the same pose (both >200, both <4096, `|a−b| < 80`) are cleared on wrap-page load. Twist **3108 / 3111** was that pattern — not walls.

Do not mix this sitting’s slack **960** with Sep-5 slack **4068**. Re-learn Δ from the **here** you just set.

## Other joints

**here** copies live pos into home (Δ=0). Then − / + as Δ from that origin, live or typed. Shoulder pitch: set pair (mirror − / same +) before any slider. Until firmware flash, do **not** jog 12+13 (`raise to` is unknown on the chip). Hold/free are `id 12 hold` + `id 13 hold`.

## Host vs chip

| | Host | Chip (after flash) |
| --- | --- | --- |
| Unwrap / degrees | yes | no |
| Wrap Δ walls | yes | no — wrap `to` is a legal raw on this side of 0/4095 |
| Non-wrap min/max/home | notebook + `lim N` | NVS clamp |
| 12+13 | pair sign | `raise to G12 G13` SyncWrite |

## UART

`send()` in `core.js` is one command at a time. `estop` drops the queue then goes. Calibrate uses **`stream 180`**. Never send `raise hold` / `raise free` until `raise` is on the metal.

Hard-refresh after JS/CSS.
