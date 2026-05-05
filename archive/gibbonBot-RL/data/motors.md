# Motor specifications

The robot uses three variants of the Waveshare ST32xx serial-bus servo family. All share the same mechanical envelope (45.22 × 35 × 24.72 mm), 360° magnetic encoder (resolution 360°/4096), 1 Mbps TTL bus, and 6–12.6 V input. They differ in gearing, peak torque, and speed.

## Per-joint assignment

| Joint (per limb) | Motor | Stall torque @12V | No-load speed @12V |
|---|---|---|---|
| shoulder_yaw / hip_yaw | ST3235 | 30 kg·cm (2.94 N·m) | 45 RPM |
| shoulder_pitch / hip_pitch | ST3235 | 30 kg·cm (2.94 N·m) | 45 RPM |
| elbow / knee | ST3235 | 30 kg·cm (2.94 N·m) | 45 RPM |
| forearm_roll / shin_roll | ST3235 | 30 kg·cm (2.94 N·m) | 45 RPM |
| wrist / ankle | ST3215 | 30 kg·cm (2.94 N·m) | 45 RPM |
| gripper / foot | ST3215-HS | 20 kg·cm (1.96 N·m) | 106 RPM |

Total: 24 motors (6 per limb × 4 limbs). The `-HS` (high-speed) variant trades torque for speed and is used at the end-effectors where fast actuation matters more than holding force.

## Mapping to scene.xml

The MuJoCo `forcerange` and `actuatorfrcrange` are set to the stall torque values above (in N·m):

- ST3235 / ST3215 joints → `forcerange="-2.94 2.94"`
- ST3215-HS gripper/foot → `forcerange="-1.96 1.96"`

Joint damping is set per-class in `<default>` blocks at the top of `scene.xml`,
derived from the linear DC motor model `damping ≈ τ_stall / ω_no_load`:

- ST3235 / ST3215 → `damping="0.62"` (2.94 N·m / 4.71 rad/s)
- ST3215-HS      → `damping="0.18"` (1.96 N·m / 11.10 rad/s)

## ST3235 — main joints (16 of 24)

Used for: shoulder yaw/pitch, elbow, forearm roll, hip yaw/pitch, knee, shin roll.

| Parameter | Value |
|---|---|
| Input voltage | 6–12.6 V |
| Stall torque @ 12 V | **30 kg·cm (2.94 N·m)** |
| No-load speed @ 12 V | 0.222 s/60° (≈45 RPM) |
| Idling current | 190 mA |
| Locked-rotor (stall) current | 2.7 A |
| Gear | High-precision steel, **1:345 reduction** |
| Encoder | 360° magnetic, 12-bit (4096 steps/rev) |
| Mechanical limit | None (continuous rotation possible) |
| Feedback | Position, load, speed, voltage, current, temperature |
| Bus | TTL serial, 1 Mbps, daisy-chain up to 253 IDs |

## ST3215 — wrist / ankle (4 of 24)

Used for: wrist, ankle.

| Parameter | Value |
|---|---|
| Input voltage | 6–12.6 V |
| Stall torque @ 12 V | **30 kg·cm (2.94 N·m)** |
| No-load speed @ 12 V | 0.222 s/60° (≈45 RPM) |
| No-load current | 200 mA |
| Locked-rotor (stall) current | 2.7 A |
| Gear | High-precision metal |
| Encoder | 360° magnetic, 12-bit (4096 steps/rev) |
| Mechanical limit | None |
| Feedback | Position, load, speed, voltage |
| Bus | TTL serial, 1 Mbps, daisy-chain up to 253 IDs |

Mechanically near-identical to ST3235 but without the high-reduction steel gearset, so somewhat lower precision.

## ST3215-HS — gripper / foot (4 of 24)

Used for: gripper, foot. *HS = high speed.*

| Parameter | Value |
|---|---|
| Input voltage | 6–12.6 V |
| Stall torque @ 12 V | **20 kg·cm (1.96 N·m)** |
| No-load speed @ 12 V | 0.094 s/60° (≈**106 RPM**) |
| Load current | 240 mA |
| Locked-rotor (stall) current | 2.4 A |
| Gear | High-precision metal |
| Encoder | 360° magnetic, 12-bit (4096 steps/rev) |
| Mechanical limit | None |
| Feedback | Position, load, speed, voltage |
| Bus | TTL serial, 1 Mbps, daisy-chain up to 253 IDs |

Trades ~33% of stall torque for ~2.4× the rotation speed.

## Common features

- Programmable servo / motor / stepper modes
- Built-in acceleration / deceleration profiles for smooth motion
- Any angle settable as the zero/midpoint
- Daisy-chainable on a shared TTL bus (up to 253 motors per bus)
- Direct power from 2S or 3S Li-ion packs

## Implications for simulation and policy

- **Torque budget per arm**: 5 × ST3235/ST3215 (2.94 N·m) + 1 × ST3215-HS (1.96 N·m) = 16.66 N·m peak across 6 joints — but only momentarily; sustained operation is limited by thermal/current draw (stall = 2.7 A per motor → significant battery drain).
- **Speed budget**: distal joints (gripper/foot) are 2.4× faster than proximal ones. Policies that demand fast hip/shoulder motion will saturate the velocity limit before the torque limit.
- **No mechanical end-stops**: limits in `scene.xml` (`range`, `ctrlrange`) are software-imposed, not physical. The real motor can rotate freely 360° if commanded outside the configured range.
