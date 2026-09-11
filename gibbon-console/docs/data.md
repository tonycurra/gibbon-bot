# Data (trials)

A **trial** is one sitting: title, notes, commands you sent, `get` rows the motors sent back.

Start on `/trials/` before you move, or just click: the first live `get` or non-poll command opens a sitting titled `sitting YYYY-MM-DD HH:MM` if none is open. Stop when done. Marks (wall, slack, contact) are `Note` rows and append to `Trial.note`. `ESTOP` / `guard` / `lim id=` lines are Notes too (they open a sitting if recording was off).

## Models (`console/models.py`)

| | |
| --- | --- |
| **Trial** | started, ended, title, note, via (`usb`/`ble`), pack_v |
| **Command** | t_ms, text, source=`ui` — skip auto `info`/`state`/`bench`/`batt`/`who`/`stream` |
| **Sample** | t_ms, servo 11–17, pos, un, turns, speed, load, v, temp, cur, moving, tq, mode |
| **Note** | t_ms, text |
| **Limit** | servo 11–17, home, amin, amax — host notebook from `/calibrate/` |

Ingest: `bridge.db_loop` from the same RX lines the page parses. Miss-spam with pack off is not stored. Pack V set from first `batt pack=…V`.

Index: `(trial, servo, t_ms)`. Cap read API 8000 samples.

## Graphs

Live `/graphs/`: ring buffer in the browser from SSE `get` lines.  
Recorded `/trials/<id>/`: `GET /api/trials/<id>/samples?servo=`.

Traces: pos@56, un, speed@58, load@60, cur@69, V@62, temp@63. Load is PWM effort, 0 if `tq=0`.

## Files

- SQLite: `db.sqlite3` (gitignored)
- Old line dumps: `sittings/*.jsonl` (pre-Django)
- BLE addr cache: `.ble-addr`

Django admin can browse rows after `createsuperuser`.
