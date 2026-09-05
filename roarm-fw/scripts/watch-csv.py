#!/usr/bin/env python3
"""Live get table + CSV. Optional hand-to steps for contact probe.

  python3 scripts/watch-csv.py --out docs/baselines/data/hand-contact.csv
  python3 scripts/watch-csv.py --out ... --steps 2008,2030,2050,2065,2075

CSV now. MySQL later if you want queries across many sessions.
Do not run idf.py monitor at the same time.
"""
import argparse
import csv
import os
import re
import sys
import time

import serial

BAUD = 115200
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
GET_RE = re.compile(
    r"get id=(\d+) pos=(-?\d+) speed=(-?\d+) load=(-?\d+) "
    r"v_raw=(\d+) v=([\d.]+) temp=(\d+) cur=(-?\d+) moving=(\d+) tq=(\S+)"
)
FIELDS = [
    "t_s", "note", "id", "name", "pos", "speed", "load", "cur",
    "moving", "tq", "v", "temp",
]


def send(ser, cmd):
    ser.write((cmd + "\n").encode("ascii"))
    ser.flush()


def drain(ser, seconds):
    end = time.time() + seconds
    lines = []
    buf = b""
    while time.time() < end:
        chunk = ser.read(ser.in_waiting or 1)
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
        m = GET_RE.search(line)
        if m:
            rows[int(m.group(1))] = {
                "pos": int(m.group(2)),
                "speed": int(m.group(3)),
                "load": int(m.group(4)),
                "v": m.group(6),
                "temp": int(m.group(7)),
                "cur": int(m.group(8)),
                "moving": int(m.group(9)),
                "tq": m.group(10),
            }
    return rows


def print_table(t_s, rows, note):
    print("\nt=%6.1fs %s" % (t_s, note), flush=True)
    print("id  name      pos  speed   load   cur mv tq     v temp", flush=True)
    for id_ in range(11, 18):
        r = rows.get(id_)
        if not r:
            print("%2d  %-8s   —" % (id_, JOINT[id_]), flush=True)
            continue
        print(
            "%2d  %-8s %4d  %5d  %5d  %4d  %d  %s  %4s  %3d"
            % (
                id_,
                JOINT[id_],
                r["pos"],
                r["speed"],
                r["load"],
                r["cur"],
                r["moving"],
                r["tq"],
                r["v"],
                r["temp"],
            ),
            flush=True,
        )


def write_rows(w, t_s, rows, note):
    for id_ in range(11, 18):
        r = rows.get(id_)
        if not r:
            continue
        w.writerow({
            "t_s": "%.2f" % t_s,
            "note": note,
            "id": id_,
            "name": JOINT[id_],
            "pos": r["pos"],
            "speed": r["speed"],
            "load": r["load"],
            "cur": r["cur"],
            "moving": r["moving"],
            "tq": r["tq"],
            "v": r["v"],
            "temp": r["temp"],
        })


def sample(ser, t0):
    send(ser, "get")
    rows = parse_get(drain(ser, DRAIN_S))
    return time.time() - t0, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="", help="comma pos list for hand to, then free")
    ap.add_argument("--dwell", type=float, default=2.2)
    args = ap.parse_args()

    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    ser = serial.Serial(args.port, BAUD, timeout=0.15)
    time.sleep(0.4)
    ser.reset_input_buffer()
    t0 = time.time()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        f.flush()

        t, rows = sample(ser, t0)
        print_table(t, rows, "rest")
        write_rows(w, t, rows, "rest")
        f.flush()

        if not steps:
            n = 0
            while True:
                t, rows = sample(ser, t0)
                n += 1
                print_table(t, rows, "watch")
                write_rows(w, t, rows, "watch")
                f.flush()
        else:
            for goal in steps:
                note = "to-%d" % goal
                send(ser, "hand to %d" % goal)
                print(">> hand to %d" % goal, flush=True)
                drain(ser, 0.4)
                end = time.time() + args.dwell
                last = rows
                while time.time() < end:
                    t, rows = sample(ser, t0)
                    last = rows or last
                    print_table(t, last, note)
                    write_rows(w, t, last, note)
                    f.flush()
                    h = last.get(17)
                    if h and abs(h["load"]) > 400 and h["speed"] == 0 and h["moving"] == 0:
                        print("load settled |load|>400 at pos=%d — likely contact" % h["pos"], flush=True)
                h = last.get(17)
                if h and abs(h["pos"] - goal) > 80 and abs(h["load"]) > 400:
                    print("STOP short of %d (pos=%d load=%d)" % (goal, h["pos"], h["load"]), flush=True)
                    break

            send(ser, "hand free")
            print(">> hand free", flush=True)
            drain(ser, 0.5)
            t, rows = sample(ser, t0)
            print_table(t, rows, "free")
            write_rows(w, t, rows, "free")
            f.flush()

    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        sys.exit(0)
