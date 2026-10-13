# Gibbon platform — ROS2 vs ours, rates, four limbs

Deep cut for the current hardware (one modified RoArm-M3 Pro, SBSD-HAT, `roarm-fw` + `gibbon-console`) and the goal: **two arms + two legs**, phone/BLE at the edge, cloud only for heavy jobs, agricultural latency. Not a sitting checklist — that stays `now.md`.

Founder/operator evaluation (why this stack, not official ROS 2, staged spend): [`ros2-vs-our-stack.md`](ros2-vs-our-stack.md). This page stays the technical map.

Official ROS2: [waveshareteam/roarm_ws](https://github.com/waveshareteam/roarm_ws) (`ros2-humble`), wiki [ROS2 workspace](https://www.waveshare.com/wiki/RoArm-M3_ROS2_Workspace_Description), [MoveIt drag](https://www.waveshare.com/wiki/RoArm-M3_Moveit2_Drag-and-drop_Interaction_in_Moveit2). URDF lives in `roarm_description`. CAD dump on the [RoArm-M3 wiki](https://www.waveshare.com/wiki/RoArm-M3) (“3D mode file”). LeRobot fork: Waveshare’s [lerobot](https://github.com/waveshareteam/lerobot) for the **AI-Kit / Jetson** line.

---

## 1. Safety rates (force cut, not battery)

The inner PID is **inside each servo** (kHz). We cannot beat that from BLE. Unexpected force on *this* arm is **`load` / `Δload` / `cur` / VIN sag / bus miss` while `tq=1`**. Voltage *spike* is rare (charger); brownout / sag on a stall is the real pack signal. `load` is 0 with torque off — teach-by-hand does not show a taut cable.

| Loop | Period | Who | Why |
| --- | --- | --- | --- |
| Servo PID | ~kHz | servo MCU | track goal |
| **Guard** (IDs with `tq=1`) | **40 ms** (25 Hz) | ESP32 | cut on load/cur/miss/VIN. Must work if the phone dies |
| Compact present stream (USB sitting) | **80 ms** | ESP32 → laptop | watch; skip still IDs |
| BLE notify → phone/console | **10 Hz moving limb**, **2 Hz idle** | ESP32 | human + page. ESTOP **immediate**, not queued behind a frame |
| Web paint | event-driven, cap ~10 Hz | browser | do not poll `get` of 7 IDs from JS |

Firmware already refuses `stream` &lt; **40 ms**. That is the floor we keep. 50 Hz for *all seven* present+tq reads is **not** free: the bus is half-duplex, one master.

Do **not** wait for BLE to stop a motor. Chip owns `estop`. Console/phone is a second button.

---

## 2. The bus is serial. There is no simultaneous servo telemetry

One DATA wire. ESP32 talks, one ID answers, then the next. SyncWrite can **command** many IDs in one packet (no replies). Reads are always round-robin.

| Pattern | Rough time | Use |
| --- | --- | --- |
| READ present, one ID | ~1–3 ms typical (12 ms timeout cap) | guard the moving joint |
| present + `tq`, seven IDs | ~14 hops, **~40–80 ms** if lucky, worse if misses | idle snapshot |
| `get` of 7 as fat ASCII | we measured ~1 s with host pauses | **starves** a live move (twist 2026-09-05) |

**Best collection now:**

1. While pushing: poll **only those IDs** at 40 ms (`get 16`, or 12+13 together). Never a full 7-way `get` in that window.
2. Still IDs: every 200–400 ms, or skip until they move (`stream` compact, **on chip 2026-09-12**).
3. Teach / imitation: **ring buffer on the ESP32** (e.g. 2–5 s × 25 Hz × 7 × ~16 B ≈ **6–14 kB**). Unload on BLE at 5–10 Hz or dump once after `trial stop`. Do not stream the raw 25 Hz ring over BLE while guarding.

50 Hz of *one* moving ID is fine. 50 Hz of *seven* is fighting the PHY. Parallel motors on one UART is **time-multiplex**, not MIMO.

Four **limbs** are four ESP32s, four buses. That *is* parallel — at the radio/USB, not inside one daisy chain.

---

## 3. ROS 2 — worth building on, or not?

**Do not replace `roarm-fw` + `gibbon-console` with `roarm_ws`.** Steal the URDF/meshes. Optionally add a thin ROS2 *bridge* later in the lab. Not in the grove.

### What Waveshare actually shipped

| Piece | What it is |
| --- | --- |
| `roarm_driver` | Laptop node. USB serial to **stock** firmware. JSON lines (`{"T":…}`), 115200, often **CP210x** on the **General Driver Board** |
| `roarm_description` | **URDF + meshes** for RViz |
| `roarm_moveit*` | MoveIt2, IKFast, MTC, keyboard servo — **Ubuntu 22.04 + Humble + VirtualBox** in their tutorial |
| LeRobot / AI-Kit | Imitation + RL on a **Jetson / GPU PC**, stock or their fork, WiFi/JSON era |

Stock stack assumes: WiFi AP or USB JSON, IMU/INA219/OLED on the old board, factory homes, Cartesian IK they published. Display launch **moves the real arm** to a vertical pose when RViz sliders connect.

### What we actually have

HAT swap, XT60, no IMU/INA219, custom ESP-IDF text (`get` / `id N` / BLE NUS), CH343 `/dev/ttyACM0`, dual-drive 12+13 as we defined, wrap Δ, chip ESTOP (**flashed 2026-09-12**, `info raise=1`). JSON driver will not speak this protocol. MoveIt’s collision world does not know our loom walls or `load` cuts.

### Fit to the product

| Need | ROS2+MoveIt on a laptop | Our edge split |
| --- | --- | --- |
| Stop on unexpected `load` if radio dies | planner is on the PC → **unsafe** | chip guard |
| Phone in olive grove | RViz/Humble VM, WiFi | BLE NUS + one Django page |
| Four independent ESP32s | one `roarm_driver` per USB; no grove coordinator | same image, role `arm_r` / `arm_l` / `leg_r` / `leg_l` |
| Cloud | people dump the whole stack in the cloud | train / vision only; **no** closed loop |
| Latency | MoveIt plan+execute is hundreds of ms–seconds | servo kHz + ESP 40 ms |
| URDF / pretty arm | **yes, take it** | web stick or glTF later |
| Imitation (neural) | LeRobot on GPU | after 1D walls + replay buffer on **our** sensors |

**Verdict:** keep building this stack. ROS2 is a **lab accessory** (visualize, maybe MoveIt once FK matches metal). It is not the safety computer, not the grove UI, not the four-limb coordinator. Porting MoveIt onto our firmware first would pause walls and ESTOP for months and still leave force-cut on the wrong side of the radio.

Cloud: datasets, RL training, maybe a heavy vision model. Commands and `estop` stay on the limb. If the orchard has no link, the robot must still freeze on `load`.

---

## 4. Immediate path to a connected gibbon (2+2)

Same firmware image on every HAT. Role is a name + BLE suffix, not four repos (`naming.md`).

```
phone / laptop  --WiFi-->  gibbon-console (one process)
                              |  BLE NUS ×4  or USB×n
                     arm_r  arm_l  leg_r  leg_l
                     (each: ESP32 + UART1 + IDs 11–17)
```

**Now (15 Sep — four limbs bolted, two 3S short):**

1. **Done 2026-09-12 on metal:** ESTOP + compact `stream` + `raise` SyncWrite (`info fw=20260912 raise=1`). In-tree sitting stamp **20260915** (`sweep`, `mat` ASCII, `info board=`).
2. Last hours: **teleop with two 3S**. Two packs = two limbs, body supported. Buy two more **only after** ESTOP + hold/nudge on both limbs look good. Then four packs, southern Italy. Do not write a stand/walk controller. Do not hang the chassis. Packs: `docs/hardware/lipo.md`.
3. Finish 1D walls on a limb that has a pack (`now.md`). Hand is done. Twist: unwrap + hops, **no wheel for range**. Sensitivity on `/raw/`; chip `guard *` is the fuse.

**Then one limb is “a gibbon part”:**

4. Teach-replay (imitation **without** a net): host path is live at `gibbon-console` `/imitation/` (browser rec/play, hop-walk, |load| slider). Chip ring-buffer dump is still later. That *is* the I/O translation: sensors = present block, actuators = `WritePos` + speed/acc.
5. Console: limb picker **gibbon / right arm / left arm / right leg / left leg**. Today only **one** USB/BLE bridge. Overview = four `found=7/7` + pack V comes after the second/third/fourth HAT talks.
6. Same FW image on every HAT, `ble=gibbon-XXXX` from MAC. No ROS to “connect” them. Four-at-once waits on four healthy 3S + N pipes.

**Then twin, then four on a chassis:**

7. `digital-twin/`: **one arm** MuJoCo from `roarm_description` URDF (not Jade’s ape from `archive/`). Validate *counts → deg* against a sitting CSV. Collision **explore in sim** after walls exist (`digital-twin/readme.md`). Safe box goes back to chip NVS — planner cannot override.
8. RL for energy/collision: **sim only**, after the twin matches. Do not PPO on metal. Archive `gibbon-rl-mujoco` / `olivabot-mujoco-cpu` are parked experiments, not this firmware.
9. Four HATs on the torso, **four 3S packs**, console overview — **southern Italy** after the two-pack lab is good enough. Navigate **top-down** (gibbon → limb → joint). Build **bottom-up** (motor → joint → limb → body). One pack = one limb. Do not Y-split. Do not buy the extra two 3S until two limbs pass ESTOP + hold/nudge. `docs/hardware/lipo.md`. `docs/plans/now.md`.

Do not wait for RL to walk. Do not wait for a web MuJoCo. A connected gibbon is four talking chips + ESTOP + walls.

---

## 5. CAD / web “twin”

| Source | Use |
| --- | --- |
| [roarm_ws `roarm_description`](https://github.com/waveshareteam/roarm_ws) | URDF + meshes. **Best steal.** |
| Wiki “RoArm-M3 3D mode file” | CAD download; heavy for the phone page |
| `archive/gibbonBot-RL` torso `.obj` / `.step` | **our** chassis, not the RoArm |
| `archive/gibbon-rl-mujoco` ape URDF | early four-arm toy; equal arm/leg, not this metal |
| `digital-twin/` | one RoArm-M3 on a table plane (`scene.xml`, `urdf/`, `meshes/`) |

**Web, fast:** do not embed MuJoCo in the browser. Stick figure or three.js boxes from FK (six angles) is enough to “see what’s going on” on a phone. Optional: convert URDF meshes to one glTF, load once, drive joints from `un`. MuJoCo viewer stays on the laptop for training.

---

## 6. Imitation vs RL vs cloud (order)

| Stage | Where | Notes |
| --- | --- | --- |
| Teach-replay | ESP32 buffer + console | no GPU. Limits clamp playback |
| LeRobot | lab PC / Jetson | only if we accept their JSON/stack **or** write a dataset adapter from our `Sample` rows. Not grove |
| MuJoCo RL | laptop | collision/energy in sim. Deploy as **more walls / slower speed / max-torque**, not a 50 Hz policy over BLE |
| Cloud | train, dataset, optional vision | never `to` / `estop` |

Input/output for IL: human `un[7]` trajectory → same `un[7]` goals. Sensors we have are listed on `/hat/`. There is no force cell; contact is `load` with `tq=1`.

---

## 7. What not to do in the next month

- Install Humble + MoveIt to “control the real arm” instead of finishing walls.
- Wheel mode for wrap.
- 50 Hz BLE of seven fat `get` lines.
- Four-limb RL on metal.
- Cloud in the force-cut path.
- MuJoCo in the phone page.

Last hours: `now.md` teleop on one powered limb. This page is the map after packs and walls.
