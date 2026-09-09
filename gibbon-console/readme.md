Laptop UI + USB/BLE bridge + trial log. Chip: `../roarm-fw`.

**Docs:** [`docs/README.md`](docs/README.md) (run, architecture, link, data, UI, next).

```bash
cd /home/t/code/gibbon-bot/gibbon-console
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver 0.0.0.0:8080
```

Firefox `http://127.0.0.1:8080` — **live · graphs · act · trials**. **STOP** / Escape = estop. Phone: same WiFi. One process owns ACM0.
