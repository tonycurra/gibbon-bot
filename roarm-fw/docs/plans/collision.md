# Collision vs kinematics

A per-joint wall is a **box** in joint space. The real free set is not a box. Wrist (15) is the first joint where that shows up as **parts hitting parts**, not just a cable or the jaws.

Keep the three jobs in `docs/plans/now.md`. Collision is a fourth job that **uses** limits and later FK. It does not replace them.

Limits table: `docs/plans/joint-limits.md`. Words: `docs/hardware/motors/names.md`.

## Four different hits

| Layer | What it is | When |
| --- | --- | --- |
| **1D wall** | One joint, other joints parked. First taut loom or plastic. | Now. Wrist sitting. |
| **Config hit** | Same wrist `pos` is safe with the elbow open and illegal with it folded. | After one wall exists, one extra pose. Not a solver. |
| **FK / capsules** | Where each lump is in the room. Link vs link, link vs table. | After all six walls + a home/sign table. |
| **IK** | Go to this XYZ. Many joint sets reach it; some collide. | Last. Stock RoArm IK assumes factory homes and stock loom. We have neither trusted yet. |

`load` is not a map. With `tq=0` it stays 0 (hand and twist). With `tq=1` a spike past ~400 is a **stop**, not a geometry model. How to detect and freeze: `docs/plans/control.md`.

## Why FK/IK blows up the space

Six moves (seven IDs). A box `[min,max]` per joint is ~6 numbers. Real free space is a **non-convex blob**: wrist-down + bend can put the jaws on the forearm; wrist-up + lower can put the hand on the table; twist rotates the **jaw silhouette** so a pitch that cleared at center scrapes after a quarter turn.

Forward kinematics answers “given these counts, where is each lump?” Inverse kinematics answers “given a point, which counts?” IK is many-to-one (stock even exposes extra `T` / `R` while holding the tip). You cannot reject collisions in XYZ until FK is trusted, and you cannot trust FK until walls and signs exist.

Do not port stock Cartesian IK. Homes, signs, and this HAT loom are not the wiki arm.

## What will actually hit on this arm

Likely pairs, distal first:

1. **Gripper ↔ forearm / wrist housing** — wrist down, worse if jaws open, worse if twist is off-center.
2. **Loom at the wrist pitch** — same family as twist cables and hand-open **609**. HAT cables still short first.
3. **Forearm ↔ upper arm** — bend, later.
4. **Upper arm ↔ base / table** — lower, later.
5. **Whole chain ↔ table / you** — turn, later. Stand behind the base.

Hand was jaws + cable. Twist was cable wrap (same raw `pos`, different loom). Wrist is the first **link-vs-link** joint.

## Do not do this yet

- Mesh collision, MuJoCo capsules, or turning collisions off to train.
- A generic joint writer.
- IK, Cartesian jog, or “keep the tip still, change T.”
- Treating stock ±90° wrist as the wall. Measure this sitting.

## Next sitting — wrist (15) only

Same ritual: USB + pack, `tq=0`, `python3 scripts/watch-hand.py`. Support the arm. One phrase.

**Park first.** Leave twist near the current visual home (seam ~4050 / park **4033**, not raw 2460; earlier look-alike **1036** was another turn). Leave the hand closed (~2088) so the jaws are a small lump. Do not wander 11/12/13/14 while finding the first wall. Write the seven `pos` lines as the **named rest** for this baseline.

Then:

1. **wrist up** — slow, stop at first taut or plastic. Say the word. What hit?
2. **wrist down** — same.
3. Record: stop `pos`, Δ sign, hit type (cable / housing / forearm / table), soft band = wall minus ~80 counts.

That fills the two empty rows in `joint-limits.md` and starts `docs/baselines/wrist.md` the way `twist.md` started.

**One coupling check, not a survey.** After those two walls, bend a little (or twist ~45°), then wrist-down again. If the stop `pos` **moves**, write “pose-dependent” and the other joints’ counts. That is the first collision-coupling datum. If it does not move, the 1D box is good enough for commanded wrist later.

No `wrist to` until that wall exists. Commanded motion checks a wall; it does not find one.

## After wrist, keep the ladder

1. **Bend → raise → turn** at the same named rest. Distal first, less mass.
2. **Calibration table** when all six have walls: home, sign, deg/count. Then FK.
3. **Host FK** as a stick figure (yaw + planar 4-bar + twist). Capsules only after the stick matches a tape on the desk.
4. **Named bad poses** (folded, table-scrape, jaws-on-forearm) as a short list of 7-tuples to refuse. Cheaper than a full C-space map.
5. **Reactive `load`** stays forever on any `tq=1` move.
6. **IK** only after FK is boring.

Firmware clamp stays an axis-aligned box. Config hits live on the host until that box is proven too loose.
