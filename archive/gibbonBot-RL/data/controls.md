# Real-robot controls

This documents the **single-arm web interface** that ships with the roarm M3 (each limb on the gibbon is one of these arms). Useful for understanding the manufacturer's joint conventions, home-position offsets, and operating modes — important for sim-to-real transfer.

## Safety & power

- **Voltage**: 7–12.6 V. Use 12 V 5 A supply or a 3S Li-Po. Anything outside this range will damage the bus servos.
- **Torque**: motors are strong (up to 30 kg·cm at 12 V) — keep eyes/head/hands out of swept volumes during testing.
- **Pre-assembled**: factory-built; avoid disassembly given the number of bus servos and their daisy-chain wiring.
- **No impacts**: drops or collisions can damage the magnetic encoders and gears.
- **Default demo speed is intentionally slow**; very low speeds may cause shaking near singular configurations.

## Startup procedure

1. Plug 12 V 5 A into the power port on the arm and flip the switch. All joints auto-home to their power-on midpoints (see table below — note the elbow homes at 1.57 rad and the gripper at 3.14 rad, not 0).
2. The OLED on the controller shows three lines:
   - **Line 1** — WiFi mode (AP by default, SSID `RoArm-M3`)
   - **Line 2** — STA mode status; if connected to a router, displays the assigned IP
   - **Line 3** — MAC address (used for ESP-NOW peer-to-peer comms)
3. Connect phone/laptop to WiFi `RoArm-M3` (password `12345678`).
4. In Chrome, open `http://192.168.4.1` — the web UI loads with the AngleCtrl panel and other controls described below.

## Joint-control buttons (AngleCtrl)

Buttons display the commanded angle in **radians**. Numerical feedback updates only when the button is released.

| Button pair | Joint | Range | Range (rad) | Power-on home | Notes |
|---|---|---|---|---|---|
| **B L / B R** | base (yaw) | 360° | −3.14 to +3.14 | **0** | B L → +rad (left), B R → −rad (right) |
| **S D / S U** | shoulder | 180° | −1.57 to +1.57 | **0** | S D → −1.57 (forward), S U → +1.57 (back) |
| **E D / E U** | elbow | 180° | varies | **1.57** | Home offset by π/2 — E D increases, E U decreases |
| **W+DG / W-UG** | wrist 1 (pitch) | 180° | −1.57 to +1.57 | **0** | W+DG → +1.57 (down), W-UG → −1.57 (up) |
| **R+DG / R-UG** | wrist 2 (roll) | 360° | −3.14 to +3.14 | **0** | R+DG → +3.14 (right), R-UG → −3.14 (left) |
| **G+DG / G-UG** | gripper | 135° | 1.08 to 3.14 | **3.14** | G+DG → 3.14 (closed, default), G-UG → 1.08 (open) |
| **INIT** | all | — | — | — | Reset every joint to its power-on home |

### Per-joint to gibbon mapping

| Real-arm joint | Gibbon arm joint | Gibbon leg joint |
|---|---|---|
| base | `*_shoulder_yaw` | `*_hip_yaw` |
| shoulder | `*_shoulder_pitch` | `*_hip_pitch` |
| elbow | `*_elbow` | `*_knee` |
| wrist 1 | `*_forearm_roll` | `*_shin_roll` |
| wrist 2 | `*_wrist` | `*_ankle` |
| gripper | `*_gripper` | `*_foot` |

### Notable offsets vs. scene.xml

The real arm's **power-on home is not all-zeros**. In `scene.xml` we have `qpos = 0` for every joint, but the physical robot starts with:

- Elbow at **1.57 rad** (≈90°) — bent, not straight
- Gripper at **3.14 rad** (closed) — but this is a *firmware-displayed* value, not the underlying joint angle.

The scene.xml `gripper`/`foot` range is `0–1.5 rad` (`0–86°`), which matches the URDF and the actual mesh geometry. The web UI's `1.08–3.14 rad` range is a **firmware display offset** (the controller reports angles shifted by π/2-ish so the user sees the gripper as "closed at 3.14, open at 1.08") — but the underlying joint moves through `0–1.5 rad`. To map between them: `joint_rad ≈ 3.14 − web_ui_rad`.

## Torque-lock mode

| Button | Effect |
|---|---|
| **Torque OFF** | Disables torque lock — joints can be back-driven by hand even with power on |
| **Torque ON** | Enables torque lock — joints resist manual rotation (default) |

Sending a position command **automatically re-enables** the torque lock, even if it was off.

Useful for kinesthetic teaching / recording demonstrations on the physical robot. No direct simulation analogue — in MuJoCo you'd disable the position actuator (`ctrl=0` with `kp=0`) to mimic free-back-drive.

## DEFA — Dynamic External Force Adaptive Control

| Button | Effect |
|---|---|
| **DEFA ON** | After external displacement, joints spring back to the pre-disturbance pose |
| **DEFA OFF** | Disabled (default) |

Effectively a high-impedance hold mode. In sim this corresponds to a position controller with high `kp` and the target locked to the last commanded pose.

## Implications for sim-to-real

1. **Home pose mismatch** — if a learned policy outputs `qpos = 0` and the real robot powers on with elbow at π/2 and gripper at π, executing zero commands will sweep through a large joint motion. Either:
   - Match `scene.xml` keyframe to the real home (`elbow = 1.57`, `gripper = 3.14`), or
   - Always send an explicit "go-to-zero" trajectory before policy rollout.
2. **Gripper display offset** — the web UI shows the gripper at `3.14` (closed) / `1.08` (open). In the URDF/sim that's `0` / `1.5`. Same physical motion, just shifted by ≈π/2 in firmware-reported coordinates. Subtract from π when going sim→web or vice versa.
3. **Torque lock** is implicit in MuJoCo's position actuators (always on while `ctrl` is set). Power-cycling the real robot or sending `Torque OFF` is needed to allow manual repositioning — there's no equivalent state in the simulator.
