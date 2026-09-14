# Calibrate

`/calibration/` is the six-joint table (live now / load / mA / min / max). `/calibration/hand/` (and the other slugs) is one joint. Old `/calibrate/` redirects.

Order: hand → wrist roll → wrist pitch → elbow → shoulder pitch (pair 12+13) → yaw.

Every joint, same page:

1. **FREE ALL** (header) if you need to move by hand.
2. Put the joint at 0° — hand: jaws closed. Wrap: this sitting’s slack (not Sep-5 4068).
3. **0°** / **closed = 0°**.
4. Move to the stops. **mark** min and max, or type degrees.
5. **go** to a typed angle (check). **test** drives min → max → min for real (full travel, not a 7° hop).
6. **next** joint. **LOCK POSE** holds everything.

Hand: closed = 0°, open = +°. Pitch 12+13: set pair before go/test. Wrap 11/16: chip hop ≤900; host walks.

`Limit` is the live notebook. Chip `lim` is for non-wrap IDs. Dated snapshots: `POST /api/calibrations` (board + fw). Header **rec sitting** is gone — that was a trial log, not walls.
