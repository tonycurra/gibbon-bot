# Contributing

Issues and small patches are welcome: a page that disagrees with the chip protocol, a trial field, a test for a route you already use.

- The STOP control and the bridge that talks to the arm are the patches to send first.
- Do not expose this process to the public internet in a patch. It owns the serial port.
- Sign off commits with `Signed-off-by: Your Name <email>` (Developer Certificate of Origin).
- Keep `db.sqlite3`, `sittings/`, `.ble-addr`, and `.env` out of the diff.

By contributing you agree your patch is under the Apache-2.0 license in `LICENSE`, except documentation, which is CC BY 4.0.
