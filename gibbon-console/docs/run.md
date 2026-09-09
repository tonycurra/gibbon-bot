# Run

```bash
cd /home/t/code/gibbon-bot/gibbon-console
.venv/bin/pip install -r requirements.txt
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver 0.0.0.0:8080
```

Same as `.venv/bin/python serve.py` (thin wrapper around `runserver`).

- Laptop: `http://127.0.0.1:8080`
- Phone: `http://<lan-ip>:8080`
- **STOP** / Escape = `estop`

One process owns `/dev/ttyACM0`. Do not run `idf.py monitor` or `live.py` at the same time.

User `t` is in `dialout`. Bluetooth: BlueZ powered on (`rfkill unblock bluetooth`). Not in group `bluetooth` is OK if DBus works.

## DB

Default: SQLite `db.sqlite3`.

MySQL when the server exists:

```bash
sudo apt install mariadb-server
sudo mysql -e "CREATE DATABASE gibbon CHARACTER SET utf8mb4; CREATE USER gibbon@localhost IDENTIFIED BY 'gibbon'; GRANT ALL ON gibbon.* TO gibbon@localhost;"
GIBBON_MYSQL=1 .venv/bin/python manage.py migrate
GIBBON_MYSQL=1 .venv/bin/python manage.py runserver 0.0.0.0:8080
```

Env: `GIBBON_DB`, `GIBBON_DB_USER`, `GIBBON_DB_PASSWORD`, `GIBBON_DB_HOST`, `GIBBON_DB_PORT`.

## Env (link)

| | default |
| --- | --- |
| `GIBBON_TTY` | `/dev/ttyACM0` |
| `GIBBON_BLE` | `44:1D:64:86:3F:56` (also `.ble-addr`) |
| `PORT` | unused — Django is `:8080` |

Admin: `/admin/` after `createsuperuser`.
