#!/usr/bin/env python3
"""Backdrive poller. READ get in a loop. You turn the joint.

Prints when pos changes or speed/load/cur/moving ticks.
Warns and exits if tq=1 on a joint we are not commanding (unexpected hold).

For commanded hand + CSV table use watch-csv.py. Docs: docs/software.md.

  python3 scripts/watch-hand.py
"""
import argparse
import re
import sys
import time

import serial

BAUD = 115200
POLL_S = 0.35
DRAIN_S = 0.85
JOINT = {
    11: "turn",
    12: "raise-A",
    13: "raise-B",
    14: "bend",
    15: "wrist",
    16: "twist",
    17: "hand",
}

GET_KV = re.compile(r"(\w+)=(\S+)")
PING_FOUND = re.compile(r"ping FOUND id=(\d+)")


def send(ser, cmd):
    ser.write((cmd + "\n").encode("ascii"))
    ser.flush()


def drain(ser, seconds):
    end = time.time() + seconds
    lines = []
    buf = b""
    while time.time() < end:
        n = ser.in_waiting
        chunk = ser.read(n or 1)
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8", "replace").strip()
            if line:
                lines.append(line)
    return lines


def parse_get(lines):
    rows = {}
    for line in lines:
        if line.startswith("get id=") and " miss" in line:
            mid = re.search(r"id=(\d+)", line)
            if mid:
                rows[int(mid.group(1))] = {"miss": True, "raw": line}
            continue
        if not line.startswith("get id="):
            continue
        kv = dict(GET_KV.findall(line))
        if "pos" not in kv:
            continue
        pos = int(kv["pos"])
        rows[int(kv["id"])] = {
            "pos": pos,
            "speed": int(kv.get("speed", 0)),
            "load": int(kv.get("load", 0)),
            "v": kv.get("v", ""),
            "temp": int(kv["temp"]) if "temp" in kv else 0,
            "cur": int(kv.get("cur", 0)),
            "moving": int(kv.get("moving", 0)),
            "tq": kv.get("tq", "0"),
            "raw": line,
        }
    return rows


def main():
    ap = argparse.ArgumentParser(description="Poll listen-v1 get while you backdrive joints")
    ap.add_argument("--port", default="/dev/ttyACM0")
    args = ap.parse_args()

    ser = serial.Serial(args.port, BAUD, timeout=0.15)
    time.sleep(0.3)
    ser.reset_input_buffer()

    print("=== open %s ===" % args.port, flush=True)
    send(ser, "state")
    time.sleep(0.2)
    send(ser, "gpio")
    for line in drain(ser, 2.4):
        print(line, flush=True)

    print("=== ping ===", flush=True)
    send(ser, "ping")
    for line in drain(ser, 1.2):
        print(line, flush=True)

    print("=== batt ===", flush=True)
    send(ser, "batt")
    for line in drain(ser, 1.0):
        print(line, flush=True)

    print("=== get rest ===", flush=True)
    send(ser, "get")
    last = parse_get(drain(ser, 1.2))
    for id_ in range(11, 18):
        row = last.get(id_)
        if not row:
            print("get id=%d (no reply)" % id_, flush=True)
        elif row.get("miss"):
            print(row["raw"], flush=True)
        else:
            print(
                "id=%d %-12s pos=%4d speed=%d load=%d v=%s temp=%d cur=%d moving=%d tq=%s"
                % (
                    id_,
                    JOINT[id_],
                    row["pos"],
                    row["speed"],
                    row["load"],
                    row["v"],
                    row["temp"],
                    row["cur"],
                    row["moving"],
                    row["tq"],
                ),
                flush=True,
            )
            if row["tq"] == "1":
                print("STOP tq=1 on id=%d — unplug XT60" % id_, flush=True)
                return 2

    print(
        "=== watching get every ~%.1fs — move joints by hand ==="
        % (POLL_S + DRAIN_S),
        flush=True,
    )
    t0 = time.time()
    n = 0
    while True:
        send(ser, "get")
        rows = parse_get(drain(ser, DRAIN_S))
        n += 1
        now = time.time() - t0
        changed = []
        tq_hot = []
        for id_ in range(11, 18):
            row = rows.get(id_)
            prev = last.get(id_)
            if not row:
                continue
            if row.get("miss"):
                if not (prev and prev.get("miss")):
                    changed.append("id=%d %s MISS" % (id_, JOINT[id_]))
                continue
            if row["tq"] == "1":
                tq_hot.append(id_)
            if prev and not prev.get("miss"):
                dpos = row["pos"] - prev["pos"]
                extras = []
                if row["speed"] != 0:
                    extras.append("speed=%d" % row["speed"])
                if row["load"] != 0:
                    extras.append("load=%d" % row["load"])
                if row["cur"] != 0:
                    extras.append("cur=%d" % row["cur"])
                if row["moving"]:
                    extras.append("moving=1")
                if abs(dpos) >= 2 or extras:
                    extra = (" " + " ".join(extras)) if extras else ""
                    changed.append(
                        "id=%d %-12s pos=%4d Δ%+d%s"
                        % (id_, JOINT[id_], row["pos"], dpos, extra)
                    )
            else:
                changed.append(
                    "id=%d %s pos=%d (first live)" % (id_, JOINT[id_], row["pos"])
                )
        if rows:
            last = rows
        if tq_hot:
            print(
                "t=%6.1fs STOP tq=1 on %s — unplug XT60" % (now, tq_hot),
                flush=True,
            )
            return 2
        if changed:
            print("t=%6.1fs %s" % (now, " | ".join(changed)), flush=True)
        elif n % 8 == 0:
            poss = []
            for id_ in range(11, 18):
                row = last.get(id_)
                if row and not row.get("miss"):
                    poss.append("%d:%d" % (id_, row["pos"]))
                else:
                    poss.append("%d:miss" % id_)
            print("t=%6.1fs still %s" % (now, " ".join(poss)), flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        sys.exit(0)
