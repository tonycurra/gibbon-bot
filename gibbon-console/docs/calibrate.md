# Calibrate

`/calibration/` is the six-joint table. `/calibration/<slug>/` is one joint. `/calibrate/` and `/motors/` redirect here.

Order: hand → wrist roll → wrist pitch → elbow → shoulder pitch (pair 12+13) → yaw.

## Table

Live **now°** (from unwrap, so wrap joints stay monotonic), raw **pos** (0–4095), **un** (unwrap). A wrap looks like pos `4000→12` while un goes `4000→4108`. now° only tracks after **home** is set.

Click **home** = this pose is 0° (hand: jaws closed). Click **min** / **max** = Enter to use now, or type a degree. **test** drives min → max → min (wrap hops ≤900). Joint name opens the dedicated page.

The page polls `get` so compact `stream` skipping still IDs does not freeze the numbers.

## Joint page

Same click-to-set. **go** to a typed angle. **next**. Shoulder pitch: set **mirror −** / **same +** before go/test/play.

**rec** / **play** (shoulder only): STOP, rec, move the upper arm by hand, stop rec, play. Replay is `raise to` on both 12 and 13 in one packet. Path stays in the browser (`localStorage`). If they fight: STOP.

## Safety

STOP / Esc limp. LOCK holds pose. Never command 12 or 13 alone. Twist homes: this sitting’s slack, not Sep-5 **4068** mixed with Sep-11 **960**. Pack writes cut at 10.5 V.

`Limit` is the live notebook. Chip `lim` for non-wrap IDs. Snapshots: `POST /api/calibrations`.
