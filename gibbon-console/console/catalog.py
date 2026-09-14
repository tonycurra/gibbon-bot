"""Six joints, seven motors. Yaw / pitch / roll live here and on /info/."""

MOTORS = [
    {"id": 11, "name": "yaw"},
    {"id": 12, "name": "pitch A"},
    {"id": 13, "name": "pitch B"},
    {"id": 14, "name": "elbow"},
    {"id": 15, "name": "wrist pitch"},
    {"id": 16, "name": "wrist roll"},
    {"id": 17, "name": "hand"},
]

JOINTS = [
    {
        "slug": "yaw",
        "name": "shoulder yaw",
        "axis": "yaw",
        "ids": [11],
        "say": "turn left / right",
        "note": "One motor. Same wrap as wrist roll: raw pos jumps at 0/4095. Mark here, then − / + as Δ. Servo mode cannot drive through that seam — free, turn through, hold.",
    },
    {
        "slug": "shoulder-pitch",
        "name": "shoulder pitch",
        "axis": "pitch",
        "ids": [12, 13],
        "paired": True,
        "say": "raise / lower",
        "note": "Two motors on one horn. Never command 12 or 13 alone. Use raise to G12 G13 (SyncWrite). If encoders run opposite, pair is mirror (−).",
    },
    {
        "slug": "elbow",
        "name": "elbow",
        "axis": "pitch",
        "ids": [14],
        "say": "bend / straighten",
        "note": "",
    },
    {
        "slug": "wrist-pitch",
        "name": "wrist pitch",
        "axis": "pitch",
        "ids": [15],
        "say": "wrist up / down",
        "note": "Nod. Not a twist.",
    },
    {
        "slug": "wrist-roll",
        "name": "wrist roll",
        "axis": "roll",
        "ids": [16],
        "say": "twist clockwise / counterclockwise",
        "note": "Looking along the forearm toward the fingers. Clockwise = counts fall (−Δ). Counterclockwise = counts rise (+Δ). Same raw pos can be two cable states — mark Δ from here, not min/max. Chip hop ≤900.",
    },
    {
        "slug": "hand",
        "name": "hand",
        "axis": "grip",
        "ids": [17],
        "say": "open / close",
        "note": "Soft 689–2008. Open wall 609, close wall 2088.",
    },
]

HAND = {"open": 689, "close": 2008, "to_max": 2075, "wall_open": 609, "wall_close": 2088}

# Distal → proximal. Shoulder yaw last (wrap, after the paired raise).
WIZARD = [
    {
        "slug": "hand",
        "title": "Hand",
        "do": "Close until the jaws meet. That is 0°. Then open until the cable just taut.",
        "minus": "closed",
        "plus": "open",
    },
    {
        "slug": "wrist-roll",
        "title": "Wrist roll",
        "do": "Slack the loom. Here = 0° at that slack. Looking along the forearm toward the fingers: clockwise is −, counterclockwise is +. Do not mix this sitting’s slack with an old home.",
        "minus": "clockwise −",
        "plus": "ccw +",
    },
    {
        "slug": "wrist-pitch",
        "title": "Wrist pitch",
        "do": "Nod the wrist by hand. Stop at the first taut cable — well before the plastic end. If load spikes, you are at the wall.",
        "minus": "nod −",
        "plus": "nod +",
    },
    {
        "slug": "elbow",
        "title": "Elbow",
        "do": "Support the forearm. Free the elbow. Bend and straighten slowly. Mark the first taut loom, not the hard stop.",
        "minus": "bend −",
        "plus": "straight +",
    },
    {
        "slug": "shoulder-pitch",
        "title": "Shoulder pitch",
        "do": "Two motors, one horn. Free both, lift the upper arm by hand. Watch load and mA on 12 and 13 — they should track. Opposite loads while held = they fight: STOP. Set pair (mirror or same) before any jog.",
        "minus": "lower −",
        "plus": "raise +",
    },
    {
        "slug": "yaw",
        "title": "Shoulder yaw",
        "do": "Stand behind the base. Slack the yaw loom. Here = 0°. Turn left and right by hand. Same wrap as wrist roll: through 0/4095 is fine while free.",
        "minus": "turn −",
        "plus": "turn +",
    },
]

CATALOG = {"motors": MOTORS, "joints": JOINTS, "hand": HAND, "wizard": WIZARD}

SLUGS = {j["slug"]: j for j in JOINTS}
