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

CATALOG = {"motors": MOTORS, "joints": JOINTS, "hand": HAND}

SLUGS = {j["slug"]: j for j in JOINTS}
