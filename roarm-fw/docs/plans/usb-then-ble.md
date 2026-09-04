# USB first, BLE later

## Order

1. USB only. Hello on serial (done). Ping IDs 11–17 with pack off = all miss (done). Listen firmware on chip; USB-only gold `docs/baselines/usb-start.md` (done 2026-09-04). Next: plug XT60 while listening (`docs/plans/characterize-hat.md`), then `get`, then nudge one joint. If this fails, BLE will not help.
2. One text protocol on USB. Reuse it on BLE (wireless serial pipe). Same lines, two transports.
3. FK after joint numbers match reality.
4. IK last. Do it on the laptop first. Dual-drive shoulder (IDs 12+13) is easy to get wrong on the chip.

Boot: bus up → USB log → (later) BLE advertise. **Do not auto-move.** Hold pose or torque-off until a command arrives. No WiFi on these chips when BLE is on (ESP32-WROOM-32 coexistence is unstable).

## USB lines (enough)

```
state
gpio
info
ping
bench
get
raw 11
```

`state` / heartbeats are the every-start check (`docs/baselines/usb-start.md`). `bench` is the connect-the-board ping: pack off `found=0/7` (`usb-no-battery.md`). Pack on later: `found=7/7`, then `get`. `get` is on the chip (READ only) but not proven on metal until XT60.

Do not type `move` / `torque 1` until `get` is trusted.

## BLE (when USB `get` is trusted)

Each board is a **peripheral**. The four-limb host is a **central** later (laptop/Pi, not this firmware).

Simple HTML + JS page, Web Bluetooth, same text as USB. Power-state matrix (USB-only / USB+battery / battery unplug USB): `docs/plans/ble-html.md`.

**NimBLE**, not Bluedroid. Advertise after USB `get` works once. Phone: that page or nRF Connect. Custom GATT only after the text pipe is boring.

## FK / IK

| | Where | When |
| --- | --- | --- |
| FK | ESP32 or laptop | After `get` is trusted. Gripper xyz from 7 angles. |
| IK | Laptop first | After FK matches a tape measure. |

Gotchas from stock firmware (`old docs to review/controls.md`): elbow home ≈ 1.57 rad, not 0. Gripper web-UI numbers are offset vs sim. Shoulder is two motors.
