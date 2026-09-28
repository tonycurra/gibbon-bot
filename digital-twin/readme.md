# One-arm digital twin (CPU MuJoCo)

One Waveshare **RoArm-M3** bolted to a **table plane**. No torso, no four limbs, no RL. Not a git repo yet; `.gitignore` is ready (`.venv` stays local, meshes stay in the tree). Workspace: [`../STATUS.md`](../STATUS.md). Live robot: `../roarm-fw`. Parked four-limb RL: `../archive/`.

Meshes + kinematics from official [`roarm_description`](https://github.com/waveshareteam/roarm_ws) (`src/roarm_main/roarm_description`, **MIT**). CAD STLs live in `meshes/`.

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

## Run

```bash
cd digital-twin
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python look.py --check   # headless
.venv/bin/python look.py           # viewer, arm up (URDF zeros)
.venv/bin/python look.py --home    # stock elbow ~90°
```

CPU only. Do not put this in the phone page.

Next: match a sitting CSV, then random-explore collisions so the safe box can go back to chip NVS.
