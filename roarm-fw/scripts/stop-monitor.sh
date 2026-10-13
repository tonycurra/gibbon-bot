#!/bin/sh
# Stop idf.py monitor from a *second* terminal. No ] key. No IDF env.
# Real process name is "esp_idf_monitor", not idf_monitor.py.

pkill -f esp_idf_monitor
pkill -f 'idf.py .*monitor'
sleep 0.2
if pgrep -af esp_idf_monitor >/dev/null 2>&1 || pgrep -af 'idf.py .*monitor' >/dev/null 2>&1; then
    echo "still running — try: kill -9 \$(pgrep -f esp_idf_monitor)"
    pgrep -af esp_idf_monitor || true
    exit 1
fi
echo "monitor stopped"
