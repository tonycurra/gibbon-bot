# Calibrate

`/calibration/` is the six-joint table. `/calibration/<slug>/` is one joint. `/calibrate/` and `/motors/` redirect here.

Order: hand → wrist roll → wrist pitch → elbow → shoulder pitch (pair 12+13) → yaw.

## Table

Live **now°** (from unwrap, so wrap joints stay monotonic), raw **pos** (0–4095), **un** (unwrap). A wrap looks like pos `4000→12` while un goes `4000→4108`. now° only tracks after **home** is set.

Click **home** = this pose is 0° (hand: jaws closed). Click **min** / **max** = save now as that wall (Shift-click to type a degree). **test** drives min → max → min. Joint name opens the dedicated page.

The page polls `get` so compact `stream` skipping still IDs does not freeze the numbers.

## Joint page

One toolbar. The home / min / max **buttons show the saved °**. Click = this pose is that mark. **go** types an angle. **next** is the following joint in the order above.

Shoulder pitch: **12** and **13** side by side (now° + load). **mirror −** / **same +** before min/max/test/play. Opposite now° of similar size → mirror. Rec/play is under **rec / play**, not the main path. Replay is `raise to` (both motors, hops ≤80 until chip has both walls). If they fight: STOP.

Test hop-walks 12+13 (chip aborts a `raise to` bigger than 80 counts unless both IDs already have min and max on the chip). Writes still cut at pack **10.5 V**.

## Safety

STOP / Esc limp. LOCK holds pose. Never command 12 or 13 alone. Twist homes: this sitting’s slack, not Sep-5 **4068** mixed with Sep-11 **960**.

`Limit` is the live notebook. Chip `lim` for non-wrap IDs. Snapshots: `POST /api/calibrations`.
