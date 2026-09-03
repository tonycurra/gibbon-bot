# USB first, BLE later

## Order

1. USB only. Hello on serial (done), then ping IDs 11–17 (code on chip; battery-off = all miss), then `get` telemetry, then nudge one joint. If this fails, BLE will not help.
2. One text protocol on USB. Reuse it on BLE (wireless serial pipe). Same lines, two transports.
3. FK after joint numbers match reality.
4. IK last. Do it on the laptop first. Dual-drive shoulder (IDs 12+13) is easy to get wrong on the chip.

Boot: bus up → USB log → (later) BLE advertise. **Do not auto-move.** Hold pose or torque-off until a command arrives. No WiFi on these chips when BLE is on (ESP32-WROOM-32 coexistence is unstable).

## USB lines (enough)

```
ping
get
move 11 2048
speed 400
torque 0
torque 1
```

`get` dumps every servo: id, pos, speed, load, V, temp, current, moving. That is all the ST3215 reports.

## BLE (when USB is boring)

Each board is a **peripheral**. The four-limb host is a **central** later (laptop/Pi, not this firmware).

Simplest: Nordic-UART-style characteristic, same text as USB. Phone: nRF Connect. Laptop: Python `bleak`. Custom GATT only after that is stable.

**NimBLE** = IDF’s small BLE radio stack (not Bluedroid). Turn it on in `menuconfig` after USB `get`/`move` works.

## FK / IK

| | Where | When |
| --- | --- | --- |
| FK | ESP32 or laptop | After `get` is trusted. Gripper xyz from 7 angles. |
| IK | Laptop first | After FK matches a tape measure. |

Gotchas from stock firmware (`old docs to review/controls.md`): elbow home ≈ 1.57 rad, not 0. Gripper web-UI numbers are offset vs sim. Shoulder is two motors.
