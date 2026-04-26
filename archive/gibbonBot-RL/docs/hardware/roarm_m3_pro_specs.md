# RoArm-M3-Pro Hardware Specifications

Source: https://www.waveshare.com/roarm-m3.htm

## Arm Overview

| Property | Value |
|---|---|
| Model | RoArm-M3-Pro |
| DOF | 5 + 1 (5 arm joints + gripper) |
| Arm weight | 1020.8 ± 15 g (excl. table clamp) |
| Workspace | Horizontal ⌀ 1120 mm, Vertical 798 mm |
| Load capacity | 0.2 kg @ 0.5 m |
| Absolute positioning | ≈ ±5 mm (same load) |
| Operating voltage | 12V 5A (or 3S LiPo) |
| Main controller | ESP32-WROOM-32 |

## Servo Motors

### ST3235 — Main Joints (base, shoulder×2, elbow, wrist-pitch, wrist-roll)

Source: https://www.waveshare.com/st3235-servo.htm

| Property | Value |
|---|---|
| Torque | 30 kg.cm @ 12V = **2.943 Nm** |
| Weight | **70.5 g** |
| Speed (no-load) | 45 RPM @ 12V = **4.71 rad/s** |
| Gear ratio | 1:345 (steel gears) |
| Encoder | 12-bit magnetic, 360°, resolution 0.088° |
| Stall current | 2.7 A |
| Torque constant (Kt) | 11 kg.cm/A |
| Voltage range | 6–12.6 V |
| Feedback | Position, Load, Speed, Voltage, Current, Temperature |
| Case | Aluminum alloy (CNC) |

### ST3215-HS — Gripper/EoAT

Source: https://www.waveshare.com/st3215-hs-servo-motor.htm

| Property | Value |
|---|---|
| Torque | 20 kg.cm @ 12V = **1.962 Nm** |
| Speed (no-load) | 106 RPM @ 12V = **11.1 rad/s** |
| Stall current | 2.4 A |
| Encoder | 12-bit magnetic, 360° |
| Feedback | Position, Load, Speed, Voltage |

## Joint Layout & Simulation Parameters

The RoArm-M3-Pro has **7 servos per arm**: the shoulder uses **dual-drive** (2× ST3235 coordinated), doubling its effective torque.

| Joint | Servo | Torque (Nm) | Range | scene.xml actuator |
|---|---|---|---|---|
| base (link1) | ST3235 | 2.943 | ±180° | `act_*_base_link_to_link1` |
| shoulder (link2) | **2× ST3235 dual-drive** | **5.886** | ±90° | `act_*_link1_to_link2` |
| elbow (link3) | ST3235 | 2.943 | −57°/+169° | `act_*_link2_to_link3` |
| wrist-pitch (link4) | ST3235 | 2.943 | ±90° | `act_*_link3_to_link4` |
| wrist-roll (link5) | ST3235 | 2.943 | ±180° | `act_*_link4_to_link5` |
| gripper | ST3215-HS | 1.962 | 0°–86° | `act_*_link5_to_gripper_link` |

## Mass Budget (per arm, simulation)

The 6 link bodies in scene.xml sum to **1020.7 g** (matching spec). Scale factor applied to URDF CAD masses: **5.288×**.

| Body | Simulated mass |
|---|---|
| link1 | 385.7 g |
| link2 | 372.0 g |
| link3 | 114.3 g |
| link4 | 52.9 g |
| link5 | 81.4 g |
| gripper_link | 15.2 g |
| **Total per arm** | **1021.5 g** |

Torso body (central mounting frame, 4 arm bases, no servos): **1.0 kg**.

## Feasibility Check (static torque)

With real masses (torso 1.0 kg + 4 × 1.02 kg = 5.08 kg total):
- Each arm must support ~1.27 kg at ~0.19 m moment arm ≈ **2.37 Nm** at the shoulder
- Dual-drive shoulder limit = **5.886 Nm → ~40% utilization → FEASIBLE**

Run: `python scripts/check_static_torque.py`

## Communication

- Protocol: TTL serial bus (half-duplex UART)
- Baud rate: 1 Mbps (servo bus), 115200 baud (ROS2 host interface)
- Command format: JSON over serial or HTTP
- Feedback fields available per servo: position (0–4095), load, speed, voltage, current, temperature
- ROS2 driver: `roarm_ws/roarm_ws/src/roarm_main/roarm_driver/roarm_driver/roarm_driver.py`
  - `feedback_get()` returns all feedback fields above (currently not used in RL obs)
