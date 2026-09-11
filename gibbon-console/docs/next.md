# Next

Do not start a new protocol. Parse the text the chip already prints (compact or fat `get`).

## Do not invent

HAT IMU, INA219, OLED, user LED, unique motor serials, BMS. `batt` is servo VIN + 3S rest-% guess. USB is not a charger.

## Firmware flash still owed (ACM0)

Stop Django first. One flash should cover:

1. Compact `stream` (omit zeros, skip still IDs).
2. Move guard `ESTOP` (`Δload` / cur / bus miss) including raise 12+13.
3. `raise to G12 G13` / `raise hold` / `raise free` SyncWrite. Solo 12/13 motion abort. `id N here`.
4. BLE energy advertise + ghost terminate in `ble_nus.c`. Until then, `bluetoothctl disconnect` recovers a stuck radio.

Until (3) is on the metal: **never send `raise …`**. Hold/free 12+13 as `id 12` / `id 13`. Do not jog shoulder pitch from home.

Confirm after flash: `stream` quiet until you move; `found=7/7`; `raise to` acks; pack V in the header.

Chip plan: `../roarm-fw/docs/plans/firmware-minimal.md`.

## Productive next

1. Finish wrap Δ on wrist-roll (16) and yaw (11) from a known slack **here**. Sep-5 twist gold is slack **4068**; 11 Sep **here** was **960** — do not mix. Procedure: `calibrate.md`.
2. Soft-band wrist pitch well inside 1126…3134 (3134 tripped `ESTOP id=15 dload`). Then elbow (host 995…2997).
3. After flash: pair on `/calibrate/shoulder-pitch/`, one SyncWrite nudge. If they fight: STOP.
4. Push host `Limit` rows into chip clamps for **non-wrap** IDs once a wall exists (today only hand/twist are hardcoded on chip).
5. Wrap-walk so test/sweep can cross 0/4095. Until then wrap test stays off.
6. MySQL when mariadb is installed (`docs/run.md`).
7. Delete stale root `index.html` / `console.js` / `console.css` / `sw.js` once nobody uses the old PWA.
8. Trial graphs: overlay Command marks on Sample traces.
9. Phone Chromium can Web Bluetooth later; Firefox stays on this WiFi bridge.

Arm-side order: `../roarm-fw/docs/plans/now.md`.

## Safety

Chip owns twist ESTOP (source; flash owed). Host **STOP** is backup and now queued so it is not starved. No `twist free` during wheel checks. Raise is SyncWrite both, or not at all — and not `raise …` until flash. Hop: twist ≤900; 11–15 ≤80 until both NVS walls exist.
