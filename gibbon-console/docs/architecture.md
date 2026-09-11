# Architecture

Django project `config/` + app `console/`. Firmware protocol: text lines, newline-terminated, USB 115200 and BLE Nordic UART. Compact `get` (omit zeros) and fat gold dumps both parse.

```
browser  ──HTTP/SSE──►  Django :8080  ──USB ACM0 and/or BLE NUS──►  ESP32  ──1 Mbps──►  servos 11–17
                         console/bridge.py
                         Trial / Command / Sample / Note / Limit
```

Firefox has no Web Serial / Web Bluetooth. The host bridge is the real link.

## Tree

```
gibbon-console/
  manage.py  serve.py          runserver :8080
  config/                      Django settings (SQLite or GIBBON_MYSQL)
  console/
    catalog.py                 six joints, yaw/pitch/roll
    bridge.py                  USB termios + bleak NUS + ingest queue
    models.py  views.py  urls.py
    templates/console/         base, home, joint, calibrate, calibrate_joint, info, advanced, trials
    static/console/            app.css, js/core.js home.js joint.js calibrate.js …
  docs/
```

Root `index.html` / `console.js` are stale PWA leftovers.

## Pages

| URL | |
| --- | --- |
| `/` | overview — 6 joints (online / tq / pos). Click a row |
| `/j/<slug>/` | one joint: present table, pos/load graph, actuate |
| `/calibrate/` | joint index — here / − / + per row |
| `/calibrate/<slug>/` | one joint. Wrap (yaw, wrist-roll): degrees + Δ slider |
| `/info/` | docs: axes, servo vs wheel, PID layers, 12+13, walls |
| `/advanced/` | chip / bus / EEPROM `who` + full serial |
| `/trials/` | sittings |
| `/trials/<id>/` | recorded samples + commands |

Header (every page): usb/ble, pack V or pack off, motors 7/7, tq, **STOP**. Footer: last ~5 deltas; click **log** to expand.

Slugs: `yaw`, `shoulder-pitch` (ids 12+13), `elbow`, `wrist-pitch`, `wrist-roll`, `hand`.

## API

| | |
| --- | --- |
| `GET /api/status` | `{ ok, via, port, bridge, ble, trial }` |
| `GET /api/rx` | SSE, `data: <line>` |
| `POST /api/tx` | raw command body, csrf exempt |
| `GET/POST /api/trial` | current / start (closes previous) |
| `POST /api/trial/stop` | |
| `POST /api/trial/note` | |
| `GET /api/trials/<id>/samples?servo=` | JSON for graphs |
| `GET/POST /api/limits` | host notebook: home / amin / amax per servo 11–17 |

`console/apps.py` starts `bridge.start()` only when `RUN_MAIN=true`. One serial/BLE owner.

## Protocol (chip)

Same lines USB and BLE. Parser: `core.js` / `bridge.py` (`key=value`; missing = 0 / `un=pos`).

Connect burst (quiet): `info` `state` `gpio` `bench` `batt` `stream 180` `who`.

`send()` (`core.js`) queues UART writes. One at a time. `estop` drops pending then goes. Django is single-threaded; hold-all + STOP must not flood unknown text.

Live: `stream` → `get id=N pos=…` plus only fields that are not default zero. Stream skips still IDs **after flash**. READ present @56. See `../roarm-fw/docs/hardware/motors/sensors.md`.

Empty cell = that sensor is not alive (pack off). Do not invent values.
