# USB first, BLE later

## Order

1. USB only. Hello, pack-off ping, listen + XT60 + `get` (done). Hand-move + hand walls + clamped `hand open/close` (done). Gold: `usb-start.md`, `usb-battery-on.md`, `hand-move-usb-battery-on.md`, `hand-open-close.md`. Next: **other joint walls** (`docs/plans/joint-limits.md`, `docs/plans/now.md`). Then BLE. If USB `get` is dead, BLE will not help.
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
batt
raw 11
```

`state` / heartbeats are the every-start check (`docs/baselines/usb-start.md`). `bench` is the connect-the-board ping: pack off `found=0/7` (`usb-no-battery.md`). Pack on: `found=7/7`, then `get` / `batt`. Proven on metal 2026-09-04.

Do not type `move` / `torque 1` until hand-move `pos` is trusted.

## BLE (when USB `get` is trusted)

Each board is a **peripheral**. The four-limb host is a **central** later (laptop/Pi, not this firmware).

Simple HTML is done and grew into Django: `../../gibbon-console/docs/README.md`. NimBLE details: `../ble.md`. Power-state matrix: `docs/plans/ble-html.md`.

**NimBLE**, not Bluedroid. Advertise after USB `get` works once. Listen-only matrix first: `bringup/02-ble-monitor` + Chromium `/monitor/`. Phone: that page or nRF Connect. Custom GATT only after the text pipe is boring.

Later (do not do this while walls are still hand-marked): see `firmware-minimal.md`.

Chip = raw present + walls + SyncWrite 12+13 + estop/guard. Host = unwrap, degrees, calibrate, %. Same bytes on USB and NimBLE. No `get` loop from the browser.

## FK / IK

| | Where | When |
| --- | --- | --- |
| FK | ESP32 or laptop | After `get` is trusted. Gripper xyz from 7 angles. |
| IK | Laptop first | After FK matches a tape measure. |

Gotchas from stock firmware (`old docs to review/controls.md`): elbow home ≈ 1.57 rad, not 0. Gripper web-UI numbers are offset vs sim. Shoulder is two motors.
