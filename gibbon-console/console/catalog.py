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
        "note": "One motor. The base is this yaw joint.",
    },
    {
        "slug": "shoulder-pitch",
        "name": "shoulder pitch",
        "axis": "pitch",
        "ids": [12, 13],
        "paired": True,
        "say": "raise / lower",
        "note": "Two motors on one horn, physically coupled. Never command 12 or 13 alone — they fight. When you move the upper arm by hand, watch both positions.",
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
        "note": "Looking along the forearm toward the fingers. Slack home ~4068. Chip hop ≤900. Do not command twist until ESTOP is flashed.",
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

CATALOG = {"motors": MOTORS, "joints": JOINTS}

SLUGS = {j["slug"]: j for j in JOINTS}
