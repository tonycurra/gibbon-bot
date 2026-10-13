# gibbon-digital-twin

One Waveshare RoArm-M3 in CPU MuJoCo, bolted to a table. Part of [Gibbon Bot](https://olivabot.com/en/projects/gibbon-bot.html). There is no torso, no second limb, and no policy in this repository.

The mesh and the joint names come from Waveshare [`roarm_description`](https://github.com/waveshareteam/roarm_ws) (`src/roarm_main/roarm_description`, MIT). The scene file that welds the arm to the floor is Olivabot’s. Damping is datasheet damping. Homes and signs are not matched to a sitting CSV yet, so this is not a digital twin of the metal until that identification is done.

Live firmware: [gibbon-roarm-fw](https://github.com/tonycurra/gibbon-roarm-fw). Sitting UI: [gibbon-console](https://github.com/tonycurra/gibbon-console). The earlier four-limb simulation is parked: [gibbon-rl](https://github.com/tonycurra/gibbon-rl), with the result written in [gibbon-archive](https://github.com/tonycurra/gibbon-archive).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python look.py --check   # headless, exits 0
.venv/bin/python look.py           # viewer, arm up
.venv/bin/python look.py --home    # stock elbow about 90°
```

Code glue is [Apache-2.0](LICENSE). Meshes and the URDF kinematics stay MIT; see [NOTICE](NOTICE).

## Files

| Path | What |
| --- | --- |
| `scene.xml` | MuJoCo: floor + welded base + 6 hinges |
| `urdf/roarm_m3.urdf` | Same chain, Waveshare joint names, no ROS |
| `meshes/*.stl` | Link CAD (mm; XML scales ×0.001) |
| `look.py` | CPU viewer / `--check` |

`gripper_left_link.stl` is in the dump; the official URDF only drives `gripper_link`.

## Joints

| Sitting word | Bus | URDF joint | Range (rad) |
| --- | --- | --- | --- |
| turn | 11 | `base_link_to_link1` | ±π |
| raise | 12+13 | `link1_to_link2` | ±π/2 |
| bend | 14 | `link2_to_link3` | −1 … 2.95 |
| wrist | 15 | `link3_to_link4` | ±π/2 |
| twist | 16 | `link4_to_link5` | ±π |
| hand | 17 | `link5_to_gripper_link` | 0 … 1.5 |

Raise is **one DOF** (dual-drive). Factory home in the XML: bend ≈ 1.57, hand closed.

This is **not** identified on our sitting yet (homes, signs, loom walls). Counts → rad comes later.

CPU only. Keep this model off the phone page.

Next: match a sitting CSV, then explore collisions so a safe box can go back to chip NVS.
