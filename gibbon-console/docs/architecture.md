# Architecture

Django project `config/` + app `console/`. Firmware protocol does not change: text lines, newline-terminated, USB 115200 and BLE Nordic UART.

```
browser  ──HTTP/SSE──►  Django :8080  ──USB ACM0 and/or BLE NUS──►  ESP32  ──1 Mbps──►  servos 11–17
                         console/bridge.py
                         Trial / Command / Sample / Note
```

Firefox has no Web Serial / Web Bluetooth. Direct BLE/USB in-page is leftover Chrome-only code paths; the host bridge is the real link.

## Tree

```
gibbon-console/
  manage.py  serve.py          runserver :8080
  config/                      Django settings (SQLite or GIBBON_MYSQL)
  console/
    bridge.py                  USB termios + bleak NUS + ingest queue
    models.py  views.py  urls.py
    templates/console/         base + live, graphs, act, trials, trial
    static/console/            app.css, js/core.js live.js graphs.js act.js trials.js
  docs/
```

Root `index.html` / `console.js` are stale PWA leftovers. The live UI is Django templates.

## Pages

| URL | |
| --- | --- |
| `/` | live — chip / link / pack + present table + log |
| `/graphs/` | live traces from `get` (pos, un, speed, load, cur, V, temp) |
| `/act/` | one-shot commands (no duplicate ops) |
| `/trials/` | start/stop sitting, marks |
| `/trials/<id>/` | recorded samples + commands |

Status bar: online/offline, pack, open trial, connect/drop, **STOP**. Nav + breadcrumb on every page.

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

`console/apps.py` starts `bridge.start()` only when `RUN_MAIN=true` (Django reloader child). One serial/BLE owner.

## Protocol (chip)

Same lines USB and BLE. Parser: `console/static/console/js/core.js` and `console/bridge.py` (`GET_RE`).

Connect burst (quiet): `info` `state` `gpio` `bench` `batt` `stream 180` `who`.

Live motors: `stream` → `get id=N pos=… speed=… load=… v_raw=… v=… temp=… cur=… moving=… tq=… un=… turns=… st=… mode=…` from **READ present @56**, 15 bytes. See `../roarm-fw/docs/hardware/motors/sensors.md`.

Empty cell = that sensor is not alive (pack off). Do not invent values.
