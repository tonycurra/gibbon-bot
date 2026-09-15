# Architecture

Django project `config/` + app `console/`. Firmware protocol: text lines, newline-terminated, USB 115200 and BLE Nordic UART. Compact `get` (omit zeros) and fat gold dumps both parse.

```
browser  ──HTTP/SSE──►  Django :8080  ──USB ACM0 and/or BLE NUS──►  ESP32  ──1 Mbps──►  servos 11–17
                         console/bridge.py
                         Trial / Command / Sample / Note / Limit
```

Firefox has no Web Serial / Web Bluetooth. The host bridge is the real link for the sitting UI. `/monitor/` is the exception: Chromium Web Bluetooth, packed `g1` frames, no commands.

## Tree

```
gibbon-console/
  manage.py  serve.py          runserver :8080
  config/                      Django settings (SQLite or GIBBON_MYSQL)
  console/
    catalog.py                 six joints, yaw/pitch/roll
    bridge.py                  USB termios + bleak NUS + ingest queue
    models.py  views.py  urls.py
    templates/console/         base, home, joint, calibrate, calibrate_joint, imitation, monitor, hat, raw, info, advanced, trials
    static/console/            app.css, js/core.js home.js joint.js calibrate.js imitate.js raw.js …
    hat.py                     gold USB captures for /hat/
  docs/
```

Root `index.html` / `console.js` are stale PWA leftovers.

## Pages

Nav: **home · calibration · imitation**. More: monitor, trials, hat, hand, info, flash, advanced.

| URL | |
| --- | --- |
| `/` | calibration + imitation. Jog table after walls exist |
| `/calibration/` | six-joint table: now° / pos / un / home / min / max / test |
| `/calibration/<slug>/` | one joint. Buttons save this pose. Shoulder: pair, then test. Rec/play under details |
| `/j/<slug>/` | jog one joint: sensors + actuate (after walls) |
| `/imitation/` | teach-replay: rec limp path, play hops, \|load\| STOP slider |
| `/monitor/` | Chromium Web Bluetooth. Packed `g1` matrix. No commands. Unplug USB |
| `/raw/` | hand: open / close / hold / free, effort trip, live graphs |
| `/hat/` | gold serial + bus timing, servo chip limits, unwrap vs wheel |
| `/info/` | axes, servo vs wheel, PID, 12+13, walls |
| `/flash/` | USB flash plan (Web Serial next) |
| `/advanced/` | chip / bus / EEPROM `who` + full serial |
| `/trials/` | sittings |
| `/trials/<id>/` | recorded samples + commands |
| `/calibrate/` `/motors/` | → `/calibration/` |
| `/graphs/` | → `/raw/` |
| `/act/` | → `/` |

Header: **usb** / **ble**, **pack %** or **pack off**, **LOCK**, **STOP**. Pack % is the 3S rest table from servo VIN (12.6 V = 100%, 10.8 V = 20%, 10.5 V = 10%, 9.0 V = 0%). USB without pack is the flash path. Trial recording is not in the header.

Slugs: `hand`, `wrist-roll`, `wrist-pitch`, `elbow`, `shoulder-pitch` (ids 12+13), `yaw`. Calibrate in that order.

## API

| | |
| --- | --- |
| `GET /api/status` | `{ ok, via, usb, ble_ok, port, bridge, ble, trial }` |
| `GET/POST /api/limits` | host notebook: home / amin / amax per servo 11–17 |
| `GET/POST /api/calibrations` | dated snapshots: board, fw, via, joints |
| `GET /api/rx` | SSE, `data: <line>` |
| `POST /api/tx` | raw command body, csrf exempt |
| `GET/POST /api/trial` | current / start (closes previous) |
| `POST /api/trial/stop` | |
| `POST /api/trial/note` | |
| `GET /api/trials/<id>/samples?servo=` | JSON for graphs |

`console/apps.py` starts `bridge.start()` only when `RUN_MAIN=true`. One serial/BLE owner.

## Protocol (chip)

Same lines USB and BLE. Parser: `core.js` / `bridge.py` (`key=value`). Compact `get` omits `un` when `un=pos`; the UI keeps the last unwrap so wrap joints do not snap back. Packed `g1` hex (bring-up monitor) also parses in `core.js`; the `/monitor/` page parses it itself.

Connect burst (quiet): `info` `get` `stream 180` `lim` `guard` `batt`. If `window.GIBBON_MONITOR`, burst is `info` + `mat 50` only. `who` / `bench` / `state` / `gpio` are on demand (`/hat/`, `/advanced/`, hand “servo dump”). Header `fw old` until the chip prints `info fw=… raise=1` — then `raise …` is allowed. `send()` drops `raise` until that line. Calibration polls `get` every 400 ms so pos/un/now° keep moving (compact stream skips still IDs). Footer is one line until you click **log**.

`mat` TX and `g1 ` RX are not trial Commands. Sitting chip `mat` is an ASCII 7×8 number dump (`vec` + `m t` + rows). Bring-up `02-ble-monitor` is packed `g1` hex — that is what `/monitor/` paints.

`send()` (`core.js`) queues UART writes. One at a time. `estop` drops pending then goes. Django is single-threaded; hold-all + STOP must not flood unknown text.

Live: `stream` → `get id=N pos=…` plus only fields that are not default zero. Stream skips still IDs **after flash**. READ present @56. See `../roarm-fw/docs/hardware/motors/sensors.md`.

Empty cell = that sensor is not alive (pack off). Do not invent values.
