# Firmware + console vs ROS 2 — business and operational evaluation

**Verdict in one line:** keep building `roarm-fw` + `gibbon-console`. Official Waveshare ROS 2 is a lab accessory later, not the grove safety computer and not the product UI.

Sitting checklist stays in `now.md`. Technical rates / four-limb map stays in `gibbon-platform.md`. This page is the founder/operator decision: where months of engineering go, and why.

Written 2026-09-12. Hardware and protocol facts checked against this workspace. Official-stack facts checked against Waveshare wiki/GitHub and ROS/MoveIt docs (URLs at the end).

---

## 1. Decision

Spend the next months on **chip ESTOP, joint walls, one firmware image, phone/web console, then a second limb**. Do not install Humble + MoveIt to “control the real arm.” Do not put a ROS laptop in the orchard. Do not buy a Jetson per limb to get LeRobot for free.

That is not “ROS is bad.” MoveIt, URDF, and later a thin ROS 2 bridge are useful **in the lab**, after FK matches metal. They do not own `estop`, they do not survive radio death, and they do not match who will operate this machine.

| If we spend 3–6 months on… | We get | We do not get |
| --- | --- | --- |
| **Our stack** (current path) | Chip guard, walls, BLE/phone, four identical HATs, IP we own | Out-of-box Cartesian drag in RViz |
| **Official `roarm_ws`** | RViz, MoveIt IKFast, keyboard servo, community demos | Safety on the limb, grove radio, HAT protocol, four-limb product |
| **Stock JSON + Waveshare web** | Fast one-arm demo on the *old* driver board | Our HAT, our walls, BLE grove, no-WiFi-with-BLE, liability story |
| **LeRobot / Jetson AI-Kit** | Imitation on GPU in the lab | Field cost, four-limb edge, radio-death safety |
| **micro-ROS on the ESP32** | DDS topics if an agent is up | A simpler protocol, a smaller safety loop, grove autonomy |

The product is a **gibbon** (two arms + two legs) for **traditional / steep / mixed-canopy groves**, operated from a **phone**, with **edge compute on the limb**.

---

## 2. What we are building

Gibbon-inspired arboreal robot (Olivabot). Laboratory R&D since May 2025. It does not walk yet. Public line: [olivabot.com/en/projects/gibbon-bot.html](https://olivabot.com/en/projects/gibbon-bot.html).

### Product

| | |
| --- | --- |
| Morphology | Four limbs on a torso: `arm_r` `arm_l` `leg_r` `leg_l` |
| Terrain | Olive / similar traditional groves: trees, slope, radio drop, no reliable broadband |
| Operator | Person with a **phone** (or laptop on the same WiFi). Not a ROS 2 Humble VM |
| Safety owner | **The ESP32 on that limb.** Phone STOP is backup |
| Cloud | Datasets, training, optional heavy vision. **Never** `to` / `estop` / closed-loop force cut |
| Twin | One-arm **MuJoCo on a laptop** later (`digital-twin/`). Not in the phone browser |

Navigate **top-down** (gibbon → limb → joint). Build **bottom-up** (motor → joint → limb → body). Same firmware image on every HAT; role is a name + BLE suffix, not four repos (`docs/framework/naming.md`).

### Hardware we actually have (verify, do not mix with stock kit)

One modified **Waveshare RoArm-M3 Pro** + **Bus Servo Driver HAT (A)**, **ESP32-WROOM-32**, XT60 3S pack. UART switch **ESP32**.

| | This limb |
| --- | --- |
| Servos | ST3235 IDs 11–15, ST3215 ID 16, ST3215-HS ID 17. Protocol family ST/SC serial bus |
| Bus | UART1 GPIO19 TX / GPIO18 RX, **1 Mbps** 8N1, **half-duplex TTL** daisy chain. Not CAN, not parallel |
| USB | WCH CH343 `1a86:55d3`, **`/dev/ttyACM0`**, 115200 text. USB powers the **chip only** |
| BLE | Nordic UART. Advertise `gibbon` / scan `gibbon-XXXX`. This board: **`gibbon-3f54`**, BLE **`44:1D:64:86:3F:56`** |
| Missing vs old General Driver Board | No IMU, no INA219, no OLED, no user LED. Pack V from servo register 62 |
| Dual-drive | Shoulder pitch = IDs **12+13**, SyncWrite both or not at all |

Stock Waveshare ROS 2 tutorials assume the **General Driver Board**: CP2102/CP210x → **`/dev/ttyUSB0`**, JSON `{"T":…}` to **Arduino stock firmware**, WiFi AP `RoArm-M3` / `192.168.4.1`, IMU/INA219/OLED on that PCB. That is a different machine than this HAT.

### Software we actually have

```
browser ──HTTP/SSE──► gibbon-console :8080 ──USB / BLE──► ESP32 (roarm-fw) ──1 Mbps──► IDs 11–17
```

- **Chip:** native ESP-IDF C (`main/main.c`, `ble_nus.c`). Text protocol (`get` / `ping` / `batt` / `stream` / `who` / `id N …` / `estop`). **No move on boot.** Same bytes on USB and NimBLE. **On metal 2026-09-12:** compact `stream`, move ESTOP, `raise` SyncWrite, `info fw=20260912 raise=1`. Still owed: BLE ghost terminate / slow advertise.
- **Host:** Django `gibbon-console`, Firefox `http://127.0.0.1:8080`. Laptop owns ACM0 and BLE (Firefox has no Web Serial / Web Bluetooth). Phone uses the **same WiFi**. Pages: overview, joint, calibrate, trials, hat, raw, info, advanced. **STOP** / Escape = `estop`. Console sends `raise …` only after `info raise=1`.

Four limbs are **four ESP32s, four buses**. Parallelism is at radio/USB, not inside one daisy chain.

---

## 3. Operational requirements (must-haves)

These are product constraints, not preferences. An option that fails one of them is not “slower” — it is the wrong architecture.

### 3.1 Safety owner is on the chip

Unexpected force on a taut cable is `load` / `Δload` / `cur` / VIN sag / bus miss while `tq=1`. Inner PID is **inside each servo** (~kHz). We cannot beat that from BLE or from a laptop planner.

| Loop | Period | Who | Why |
| --- | --- | --- | --- |
| Servo PID | ~kHz | servo MCU | track goal |
| **Guard** (IDs with `tq=1`) | **40 ms** (25 Hz) | ESP32 | cut on load/cur/miss/VIN. Must work if the phone dies |
| Compact present stream (USB sitting) | **80 ms** | ESP32 → laptop | watch; skip still IDs |
| BLE notify | **10 Hz moving**, **2 Hz idle** (grove later **1–5 Hz**) | ESP32 | human + page. ESTOP **immediate**, not queued behind a frame |
| Web paint | event-driven, cap ~10 Hz | browser | do not poll `get` of 7 IDs from JS |

Firmware already **refuses `stream` &lt; 40 ms**. That floor stays. 50 Hz of all seven present+tq reads is not free: half-duplex, one master. Full `get` of 7 as fat ASCII measured ~**1 s** with host pauses and **starved** a live move (twist 2026-09-05; you cut XT60).

**Do not wait for BLE to stop a motor.** Chip owns `estop`. Console/phone is a second button.

Agricultural machinery design principles (ISO 18497 family) require a **defined safe state on communication loss**, and a way for a local/remote operator to stop automated functions. We are not claiming CE or ISO conformance today. We *are* claiming the architecture must not make “radio died” into “motors keep tracking last goal with nobody watching `load`.” That is what happened on twist when the chip did not guard during wheel.

### 3.2 Phone in the grove, not a ROS laptop in the orchard

Operator walks trees. Device is a phone (or a small tablet). Link is BLE NUS to the limb, or WiFi to a pocket host running Django.

Not: Ubuntu 22.04 + Humble + RViz + VirtualBox USB passthrough of a CP210x. Waveshare’s own tutorial is that stack. It is a **bench** workflow.

### 3.3 Radio death is normal

Olive canopy, distance, 3S pack, BLE supervision ~2 s. NimBLE `MAX_CONNECTIONS=1`. WiFi + BLE together on this **WROOM-32** is unstable — we do not run both.

If notify buffer is full: **drop the telemetry frame**, never block the guard. If the phone is gone: **chip still polls the moving ID** and cuts.

### 3.4 Multi-HAT, one image

Same `roarm-fw` on every board. Role `arm_r` / `arm_l` / `leg_r` / `leg_l` is config + BLE name, not four forks. Console later: limb picker **gibbon / right arm / left arm / right leg / left leg**. Overview = four `found=7/7` + pack V.

Official `roarm_driver` is **one USB JSON pipe to one stock arm**. Four of those still have no grove coordinator, no shared ESTOP story, and no HAT protocol.

### 3.5 Agricultural connectivity and latency budget

Rural groves: intermittent LTE, no 5G mmWave, trees block. Cloud round-trip is **not** in the force-cut path even if 5G were perfect — blockages still add delay ([IEEE ISEC 2024 edge vs cloud agriculture](https://doi.org/10.1109/isec61299.2024.10664737); [cloud–edge–device in smart agriculture](https://doi.org/10.3389/fpls.2025.1668545)).

Budget we already chose:

| Layer | Allowed | Forbidden |
| --- | --- | --- |
| Servo | kHz PID | host PID fighting servo PID |
| ESP32 | 20–40 ms guard, VIN refuse &lt; **10.5 V** | FK, unwrap, battery % guess on the hot path |
| BLE / phone | 1–10 Hz pose/energy; commands on demand | 50 Hz fat `get` of seven IDs |
| Cloud | train, dataset, optional vision | `to`, `estop`, load fuse |

### 3.6 Who can operate it

A mill contact, a technician, later an orchard worker with a phone. Not an engineer who sourced `ros-humble-desktop` and remembered Cyclone DDS.

Time-to-first-hosted-trial (order €30–80k if anyone pays — `path-to-revenue.md`) is **teleop climb + kg in a net**, not RViz sliders.

---

## 4. Options on the table

| Option | What it is | Grove safety | Phone UX | Four HAT | Cost at 4 limbs | Time-to-field | Use it? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A. Ours** — ESP-IDF + Django | Text/NUS, chip guard, walls, `:8080` | Yes (after flash ESTOP) | Yes (WiFi now; Web Bluetooth later) | Same image, role config | 4× HAT already on the limb | Months we already started | **Now** |
| **B. Official ROS 2** `roarm_ws` Humble + MoveIt2 | Laptop node, JSON to **stock** FW, RViz | Planner on PC → **no** | RViz / VM | One driver per USB; no coordinator | Ubuntu laptop + time | Demo in a week on *stock* board; months to retarget *this* HAT and still wrong safety split | Lab **later** |
| **C. Stock Waveshare JSON + web** | Arduino FW, `{"T":…}`, WiFi AP `192.168.4.1` | Their torque/dynamic-adapt JSON, not our loom walls | Their web, WiFi | Four APs / ESP-NOW toy | Cheap if we still had that board | Fast on General Driver Board; **wrong board** | No (HAT swap already done) |
| **D. LeRobot + Jetson / AI-Kit** | Hugging Face IL, ACT, leader–follower | Policy on GPU; radio still in the loop | Lab PC / Jetson | Dual-arm kit is a different SKU | Jetson Orin Nano Super **~$249** each; Waveshare Orin AI-Kit **~$1050–1530** for a *desktop dual-arm* | Lab imitation after we have datasets | Lab **after** teach-replay on our sensors |
| **E. micro-ROS on ESP32** | XRCE-DDS client; **agent on Linux** | Agent/WiFi death = no topics | None | Four agents | RAM on WROOM is the tax | Months rewriting a working UART | **No** |
| **F. Hybrid** | Keep A. Optional ROS 2 *bridge* in lab; steal URDF | Chip still owns metal | Phone still owns grove UI | Same | Cheap overlay | After FK matches | **Later**, not instead of A |

Arduino sketches (Waveshare `RoArm-M3_example.ino`, Arduino-ESP32 3.0.7, Huge APP + PSRAM) are option **C’s firmware**, not a fifth architecture. We already rejected Arduino vs IDF: NimBLE, RAM, dual-core bus+radio, four identical limbs, later Secure Boot (`docs/framework/framework.md`).

---

## 5. Deep evaluation of official ROS 2 for RoArm-M3

Sources: [waveshareteam/roarm_ws](https://github.com/waveshareteam/roarm_ws) (`ros2-humble`), wiki [ROS2 workspace](https://www.waveshare.com/wiki/RoArm-M3_ROS2_Workspace_Description), [drive node](https://www.waveshare.com/wiki/RoArm-M3_Drive_Node_to_Control_Real-world_Robotic_Arm), [MoveIt drag](https://www.waveshare.com/wiki/RoArm-M3_Moveit2_Drag-and-drop_Interaction_in_Moveit2), [JSON commands](https://www.waveshare.com/wiki/RoArm-M3-S_JSON_Command_Meaning). ROS 2 Humble: [install](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html), [Humble is LTS until May 2027](https://www.openrobotics.org/blog/2022/5/24/ros-2-humble-hawksbill-release). MoveIt Humble: [getting started](https://moveit.picknik.ai/humble/doc/tutorials/getting_started/getting_started.html) (source build: they recommend **32 GB RAM**; switch to Cyclone DDS).

### 5.1 What they actually shipped

| Package | Job |
| --- | --- |
| `roarm_driver` | Laptop Python node. USB serial → **stock** firmware. JSON lines. Default **`/dev/ttyUSB0`**. Depends on `roarm-sdk==0.1.0` |
| `roarm_description` | **URDF + meshes** for RViz |
| `roarm_moveit` + `roarm_moveit_ikfast_plugins` | MoveIt2 config, IKFast |
| `roarm_moveit_cmd` | Services: get pose, move joint/line/circle |
| `roarm_moveit_servo` | Keyboard / gamepad jog |
| `roarm_moveit_mtc_demo` | MoveIt Task Constructor pick/place demos |
| `roarm_msgs` | Custom messages |

Install path in their README: Ubuntu **22.04** ISO, often **Oracle VM VirtualBox**, `ros-humble-desktop`, `ros-humble-moveit-*`, `colcon build` of the workspace. Wiki: enter a container via `ros_humble.sh`, password `ws`, USB filter for **CP210x**.

Serial to stock firmware: **115200**, newline JSON, `{"T": 1041, "x": …}` etc. (`json_cmd.h` on the Arduino lower computer). Wireless in *their* demo is **WiFi HTTP** `/js?json=` or ESP-NOW — not BLE NUS.

### 5.2 Protocol mismatch

Our chip speaks:

```
get id=17 pos=… load=… v=…
id 17 to 1200
estop
stream 80
```

Their driver speaks:

```
{"T":210,"cmd":1}
{"T":1041,"x":235,"y":0,"z":234}
```

JSON `roarm_driver` **will not speak this protocol**. Porting us *to* JSON would throw away walls, compact stream, raise SyncWrite, and the ESTOP we still owe a flash. Writing a ROS 2 node that speaks *our* text is a **thin bridge** (option F), not “use official software.”

### 5.3 Board mismatch

| | Official tutorial | This robot |
| --- | --- | --- |
| USB chip | CP2102/CP210x | WCH CH343 |
| Device node | `/dev/ttyUSB0` | `/dev/ttyACM0` |
| PCB | General Driver Board (IMU, INA219, OLED, lidar Type-C) | Bus Servo Driver HAT (A), XT60, none of those sensors |
| Firmware | Arduino JSON + WiFi AP | ESP-IDF text + NimBLE |
| Cartesian IK | Their published JSON IK on the ESP | Not on our chip; host later, after walls |

Their MoveIt collision world does not know our loom walls, wrap Δ homes (Sep-5 twist slack **4068** vs 11 Sep **960**), or `load` cuts.

### 5.4 Compute location

Humble desktop + MoveIt + RViz wants a **laptop or IPC**: Ubuntu 22.04 amd64/arm64, typically 16 GB RAM (32 GB if you build MoveIt from source). Default RMW is Fast DDS; MoveIt docs still tell many users to `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` so every node matches.

DDS discovery, `ROS_DOMAIN_ID`, multicast, and firewalls are a **LAN problem**. They are not a grove-phone problem.

MoveIt **plan + execute** is the right tool for Cartesian goals in a known collision scene. It is the wrong tool for “cable just taut, cut in 40 ms.” Planning time is workload-dependent: unconstrained plans often **100 ms–1 s**; constrained plans in Humble have been reported at **tens of seconds** ([moveit2#2604](https://github.com/moveit/moveit2/issues/2604)). That is lab IK, not field safety.

NXP’s RoArm demo on i.MX ([Robotics Edge Platform](https://mcuxpresso.nxp.com/RoboticsEdgePlatform/latest/html/example/roarm.html)) is the same split: MoveIt Servo + `ros2_control` **on the Linux board**, JSON **to the driver board**. The safety computer is still not the servo bus master in the way we need.

### 5.5 Display launch moves hardware

Waveshare, in their own words ([drive node wiki](https://www.waveshare.com/wiki/RoArm-M3_Drive_Node_to_Control_Real-world_Robotic_Arm) and [README](https://github.com/waveshareteam/roarm_ws/blob/ros2-humble/README.md)):

> After starting Rviz2, the robot URDF in Rviz2 publishes each joint angle of the robotic arm to the drive node, and the drive node will edit these joint angles into the corresponding JSON format control commands and send them to the robotic arm, so the robotic arm will rotate.

`ros2 launch roarm_description display.launch.py` — arm goes **vertical**. Keep children and glass away.

`ros2 launch roarm_moveit roarm_moveit.launch.py` — arm moves, **forearm forward, parallel to the ground**. Dragging the marker does *not* move metal until **Plan & Execute** — but **launch itself already moved the real arm**.

That is the opposite of our boot rule: **no auto-move**. Hold pose or `tq=0` until a command arrives.

### 5.6 VirtualBox as the supported path

Their install is an Ubuntu ISO in VirtualBox, USB 3.0 xHCI, CP210x filter, sometimes a Humble **container**. USB passthrough + a real arm is a known class of “it moved and I did not mean it” bugs. Fine for a shop demo. Unacceptable as the only way a mill technician talks to a climbing robot.

### 5.7 Licensing and community

ROS 2 Humble and MoveIt 2 Humble are real, maintained ecosystems (Humble LTS **to May 2027**; MoveIt Humble same window per [moveit.ai/about/distribution](https://moveit.ai/about/distribution/)). That is an asset **if** you are already on Ubuntu + DDS.

`roarm_ws` itself: as of this writing the `waveshareteam/roarm_ws` tree has **no `LICENSE` file** at repo root (fetch 404). README license badge points at an older fork naming (`dudulrx/roarm_ws`). **Do not assume MIT.** Before copying meshes into `digital-twin/`, clone and read `LICENSE` / `package.xml`. Keep attribution. Prefer their URDF over inventing link lengths.

Community size: Waveshare wiki + a small GitHub repo. It is not Universal Robots’ `ur_robot_driver`. Forking it does not buy a field support org.

### 5.8 What we SHOULD steal

| Steal | Why |
| --- | --- |
| `roarm_description` URDF + meshes | Best kinematics/visual starting point for one-arm MuJoCo and later a stick/glTF phone view |
| Wiki “3D mode file” CAD | Heavy; laptop twin, not the phone page |
| Published joint names / approx link lengths | Cross-check against our count→deg table after walls exist |
| Nothing else as runtime | Not JSON driver, not their WiFi AP, not VirtualBox, not display.launch as a control path |

Do **not** use the four-arm archive scene as the metal model. Do **not** put MuJoCo in the phone.

---

## 6. Why our firmware + web console matches the business

### 6.1 Differentiation and IP

Anyone can clone `roarm_ws` and drag an M3 in RViz. Nobody else has **our** HAT + XT60 + chip guard + loom walls + four-limb role scheme + sitting log. That stack *is* the portable asset if door A in `path-to-revenue.md` (get paid for skills) happens. It is also the only thing a mill can host without hiring a ROS engineer.

Stock JSON firmware is Waveshare’s product. Shipping it as “Gibbon” is a demo of their kit.

### 6.2 Safety liability

Who is responsible when a wrist winds a cable? On 2026-09-05 the honest answer was: **host ignored `load`, chip did not watch during wheel, bus died, motor kept spinning.** Fix is a **chip loop**, not a better planner.

If ESTOP lives in MoveIt on a laptop, then USB unplug, VM freeze, DDS discovery stall, and BLE drop are all **hazardous**. Insurers and, later, EU machinery rules care about that split even for operator-present trials. Cloud in the force-cut path is worse: connectivity in agriculture is the documented failure mode, not the exception ([Sensors 2021 survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC8434609/); [edge vs cloud IoT agriculture](https://doi.org/10.33425/3066-1226.1180)).

### 6.3 Time-to-field vs time-to-RViz

We already have USB bring-up, pack-on gold, hand walls, twist walls (commanded check failed once), calibrate pages, trial JSONL. Next sitting is **flash ESTOP + raise + compact stream**, then finish wrap Δ. That is days-to-weeks of metal work.

Humble+MoveIt on *this* HAT is: rewrite or shim `roarm_driver`, invent JSON we do not speak, fight 12+13 dual-drive, still put the fuse on the wrong side of the radio. That pauses walls for months and does not climb a trunk.

### 6.4 Operator UX is a phone

`gibbon-console` is already a phone-capable web UI on `:8080` (same LAN). Joint words a human can say (yaw / raise / bend / twist / hand). STOP in the header. Calibrate is a sitting, not a URDF joint_state publisher.

RViz is the right UI for a kinematicist. It is the wrong UI for a person standing under a tree.

### 6.5 Four-limb product, not a one-arm demo

Official software is a **desktop 5+1 DoF arm**. Waveshare LeRobot kits are **leader–follower dual arm** on a table, optional Jetson. Our bet is **2+2 on a torso in canopy**. The bus is serial; four limbs = four masters. One firmware + role config is how that stays maintainable (`naming.md`). Four ROS workspaces is how it does not.

### 6.6 Edge-first cost

HAT already has the ESP32. Adding **four Jetson Orin Nanos** (~$249 × 4, plus 7–25 W each, plus enclosures, plus Ubuntu images) is a different BOM and a different failure surface. Use **one** GPU machine in the **lab** for training. Do not hang a Jetson on each wrist.

Cloud GPU for training is cheaper than four field GPUs and does not need to be in the grove.

### 6.7 Maintainability

One C image, one Python console, one text protocol. Chip vs host split is written down (`firmware-minimal.md`): chip = raw + walls + raise + estop; host = unwrap, degrees, calibrate, %.

ROS 2 Humble will EOL in **May 2027**. A field product that *is* Humble becomes a distro-migration project in the middle of trials. Our protocol does not.

---

## 7. What ROS 2 is still useful for later

Hybrid is real. It is **not now**.

| When | Use ROS 2 / MoveIt for | Still not |
| --- | --- | --- |
| After walls + count→deg you trust | Visualize our `un[]` in RViz via a **read-only** joint_state publisher | Sending goals until chip guard is boring |
| After one-arm MuJoCo matches CSV | Optional MoveIt planning **offline**; dump a hop list; play on chip inside walls | Planner overriding NVS walls |
| Lab IPC on a bench | Keyboard servo, MTC pick-place experiments on a **spare** stock-firmware arm if we keep one | The grove robot |
| Dataset glue | rosbag2 of camera + our Sample rows if a partner already lives in ROS | Replacing Django trials |

**Thin bridge (later):** a node that opens ACM0/BLE, speaks **our** lines, publishes `sensor_msgs/JointState`, subscribes to a *gated* command topic that the chip still clamps. Do not replace the protocol. Do not run the bridge as the ESTOP owner.

LeRobot: only with a **dataset adapter** from our `Sample` rows (or teach-replay dumps), or if we accept their JSON stack on a *different* arm. Not grove. Not before 1D walls + ESP32 ring buffer.

---

## 8. Cloud vs edge: what may leave the grove, what must never

```
must never leave the limb     may leave (eventually)
─────────────────────────     ──────────────────────────
estop / tq=0                  sitting CSV / JSONL
load/cur/VIN guard            count→deg tables
WritePos / SyncWrite          URDF / MuJoCo
NVS walls (non-wrap)          RL / ACT weights → become slower speed / max-torque / extra walls
pack cutoff 10.5 V            vision model (optional, high latency)
BLE bind (later)              mill reports, kg/h
```

**Device (ESP32):** bus master, fuse, radio. Survives phone death.

**Edge (phone / pocket Django):** operator, named poses, energy display, second STOP. May die.

**Cloud:** train, store, maybe a slow suggestion (“try this grasp”). If the orchard has no link, the robot must still **freeze on `load`**.

Do not run a 50 Hz neural policy over BLE. Teach-replay is `un[7]` → `un[7]` on the chip buffer. RL in sim deploys as **more walls / slower speed / max-torque (reg 16)**, not as closed-loop cloud inference.

---

## 9. Risks of staying custom — and mitigations

| Risk | Why it is real | Mitigation |
| --- | --- | --- |
| **We own bus bugs** | Half-duplex echo, 12 ms timeout, dual-drive 12+13, wrap 0/4095. Twist already wound a cable | Gold sittings, `esp32-bus.md`, flash the guard, never solo 12/13, hop limits until NVS walls |
| **No MoveIt out of box** | Cartesian drag and collision planning are delayed | We do not need Cartesian to wall joints or to teleop climb. FK after walls; IK last (`usb-then-ble.md`) |
| **URDF import work** | Their model ≠ our loom, wrap homes, dual-drive sign | Steal meshes; identify on **sitting CSV**; safe box goes **back to chip NVS** (`digital-twin/readme.md`) |
| **Fewer contributors** | ROS has more GitHub stars than we do | Our constraint is unpaid calendar, not staffing a ROS org (`path-to-revenue.md`) |
| **BLE security** | No pairing today (`near-future.md`) | Fine in lab. Field: bind/passkey before motion writes; `estop` not open to any phone in range |
| **Host still in the loop for wrap Δ** | Chip does not store wrap min/max | Host clamps Δ; chip still cuts on `load` while moving |
| **Flash owed** | BLE advertise / ghost terminate | Not ESTOP/`raise` — those are on metal 2026-09-12. Do not start a ROS install instead |

Risk of **switching** to official ROS 2 now: pause ESTOP/walls, talk JSON to the wrong board, put force-cut on the PC, and still have no four-limb grove UI. That is the more expensive risk.

---

## 10. Recommendation and staged spend

Aligned with the bottom-up plan in `gibbon-platform.md` and `now.md`. Do not wait for RL to walk. Do not wait for a web MuJoCo.

### Now (this arm, USB) — weeks, not a new stack

1. **Done 2026-09-12:** ESTOP + compact `stream` + `raise` SyncWrite (`info fw=20260912 raise=1`). Console sends `raise …` only after that line.
2. Finish wrap Δ walls on **16** and **11** from a known slack `here`. Do not mix Sep-5 `un` 1058…7046 with home **960**.
3. Soft-band wrist pitch inside 1126 … well short of 3134. Elbow host 995…2997, hop ≤80.
4. Pair sign on `/calibrate/shoulder-pitch/`, one SyncWrite nudge. If they fight: STOP.
5. Sensitivity slider on `/raw/` is host backup. **Chip `guard` is the fuse.**

Spend: sittings we already scheduled. **Not** Humble.

### Next (one limb is a gibbon part)

6. Teach-replay without a net: `tq=0` you move; ESP32 ring-buffers `un,load,v,cur` at 25 Hz (~2–5 s × 7 × ~16 B ≈ **6–14 kB**); BLE dump; play back as hops inside walls.
7. Console limb picker. Today only `right arm` is live.
8. Second physical arm: same FW, different `gibbon-XXXX` MAC, second BLE bridge. **No ROS required to connect them.**
9. Optional: stick figure / three.js boxes from six angles on the phone. Not MuJoCo WASM.

### Later (twin, then four HATs, then lab ROS)

10. `digital-twin/`: one arm MuJoCo from **`roarm_description` URDF**. Validate counts → deg against a sitting CSV. Collision explore in sim; safe workspace **returns to chip**.
11. RL for energy/collision: **sim only**. Deploy as walls / speed / max-torque. Archive `gibbon-rl-mujoco` stays parked.
12. Four HATs on a torso, four packs, console overview.
13. **Then** a read-only ROS 2 bridge or MoveIt offline planning if a partner needs it. Optional LeRobot adapter from our Sample rows on a lab GPU.

### Explicitly do not spend months on

- Humble + MoveIt as the way to move this HAT
- Wheel mode as wrap (already burned us)
- 50 Hz BLE of seven fat `get` lines
- Four-limb RL on metal
- Cloud in the force-cut path
- MuJoCo in the phone page
- micro-ROS rewrite of a working UART
- Jetson per limb
- Replacing `roarm-fw` protocol with JSON `{"T":…}`

---

## Fit table (product need vs stack)

| Need | ROS 2 + MoveIt on a laptop | Our edge split |
| --- | --- | --- |
| Stop on unexpected `load` if radio dies | planner is on the PC → **unsafe** | chip guard |
| Phone in olive grove | RViz / Humble VM / WiFi JSON | BLE NUS + one Django page |
| Four independent ESP32s | one `roarm_driver` per USB; no grove coordinator | same image, role name |
| Cloud | people dump the whole stack in the cloud | train / vision only; **no** closed loop |
| Latency | plan+execute hundreds of ms–seconds | servo kHz + ESP **40 ms** |
| URDF / pretty arm | **yes, take it** | web stick or glTF later |
| Imitation (neural) | LeRobot on GPU | after 1D walls + replay buffer on **our** sensors |
| Operator | ROS engineer | mill tech with a phone |
| Boot | display launch **moves metal** | **no move on boot** |

**Verdict:** keep building this stack. ROS 2 is a **lab accessory** once FK matches metal. It is not the safety computer, not the grove UI, not the four-limb coordinator.

---

## Sources

### Ours (this workspace)

- `gibbon-platform.md` — rates, serial bus, four-limb path
- `now.md` — sitting state; ESTOP + raise flashed 2026-09-12
- `near-future.md` — registers, BLE cadence, energy, language
- `firmware-minimal.md` — chip vs host
- `docs/framework/framework.md` — why ESP-IDF not Arduino
- `docs/framework/naming.md` — one image, four roles
- `docs/hardware.md`, `docs/hardware/motors/esp32-bus.md`
- `gibbon-console/docs/architecture.md`, `link.md`
- `digital-twin/readme.md`

### Official Waveshare / ROS / agriculture (fetched 2026-09-12)

- [waveshareteam/roarm_ws](https://github.com/waveshareteam/roarm_ws) (`ros2-humble` README: VirtualBox, CP210x, `/dev/ttyUSB0`, display launch moves arm vertical)
- [RoArm-M3 ROS2 Workspace Description](https://www.waveshare.com/wiki/RoArm-M3_ROS2_Workspace_Description)
- [Drive Node to Control Real-world Robotic Arm](https://www.waveshare.com/wiki/RoArm-M3_Drive_Node_to_Control_Real-world_Robotic_Arm)
- [MoveIt2 drag-and-drop](https://www.waveshare.com/wiki/RoArm-M3_Moveit2_Drag-and-drop_Interaction_in_Moveit2)
- [RoArm-M3 wiki](https://www.waveshare.com/wiki/RoArm-M3) (stock WiFi AP, JSON, CP2102, Arduino ESP32)
- [JSON command meaning](https://www.waveshare.com/wiki/RoArm-M3-S_JSON_Command_Meaning) (115200 serial, HTTP `/js?json=`)
- [Secondary development (Arduino)](https://www.waveshare.com/wiki/RoArm-M3-S_Secondary_Development_Tool_Usage)
- [RoArm-M3-AI-Kit](https://www.waveshare.com/wiki/RoArm-M3-AI-Kit) / [waveshareteam/lerobot](https://github.com/waveshareteam/lerobot)
- [Orin Nano AI-Kit product](https://www.waveshare.com/roarm-m3-s-orin-nano-ai-kit.htm) (~$1050–1531 with Jetson)
- [NVIDIA Jetson Orin Nano Super](https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) (list **$249**)
- [ROS 2 Humble Ubuntu debs](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Humble LTS to May 2027](https://www.openrobotics.org/blog/2022/5/24/ros-2-humble-hawksbill-release)
- [MoveIt Humble getting started](https://moveit.picknik.ai/humble/doc/tutorials/getting_started/getting_started.html) (32 GB RAM note, Cyclone DDS)
- [MoveIt distributions / EOL](https://moveit.ai/about/distribution/)
- [NXP RoArm ROS 2 / MoveIt architecture](https://mcuxpresso.nxp.com/RoboticsEdgePlatform/latest/html/example/roarm.html)
- [ISO 18497-1:2024](https://www.iso.org/standard/82684.html) (automated agricultural machinery design principles; communication-loss safe state is in this family — we are not claiming certification)
- [Cloud–edge–device in smart agriculture](https://doi.org/10.3389/fpls.2025.1668545)
- [Edge vs cloud for agricultural automation](https://doi.org/10.1109/isec61299.2024.10664737)
- [micro-ROS abandoned for UART (field report)](https://thenewtoncapstone.github.io/blog/posts/issues-with-microros/)
