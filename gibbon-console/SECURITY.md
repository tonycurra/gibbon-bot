# Security

This process can send motion commands to a robot arm over USB or BLE. A bug that skips STOP, or that moves servo 12 or 13 alone, is a safety report.

Email [services@olivabot.com](mailto:services@olivabot.com). Say which page, which command left the browser, and what the chip printed.

Run it on localhost or your own LAN. Do not publish port 8080. The default `SECRET_KEY` is for a local sitting only. Set `SECRET_KEY` in the environment before any shared deployment.

There is no bug bounty. Research equipment, no warranty.
