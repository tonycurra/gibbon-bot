#!/usr/bin/env python3
"""Live get table + CSV. Unwraps encoder wraps. Flags load spikes.

  python3 scripts/watch-csv.py --out docs/baselines/data/hand-contact.csv
  python3 scripts/watch-csv.py --out ... --steps 2008,2030,2050,2065,2075

Twist absolute (turns) is saved to docs/baselines/data/twist-absolute.json
so a 0–4095 wrap does not look like the same pose. Chip RAM loses turns
on reset; this file is the memory.

CSV now. MySQL later if you want queries across many sessions.
Do not run idf.py monitor at the same time.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import serial

BAUD = 115200
DRAIN_S = 0.85
TWIST_ABS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "docs", "baselines", "data", "twist-absolute.json",
)
LOAD_SPIKE = 400
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
FIELDS = [
    "t_s", "note", "id", "name", "pos", "un", "turns", "speed", "load", "cur",
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
        if not line.startswith("get id=") or " miss" in line:
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
        }
    return rows


class Unwrap:
    def __init__(self):
        self.turns = {i: 0 for i in range(11, 18)}
        self.last = {}
        self.last_load = {}

    def load_twist_abs(self, path):
        if not os.path.isfile(path):
            return
        try:
            st = json.load(open(path))
        except (OSError, ValueError) as e:
            print("UNWRAP file unreadable %s: %s" % (path, e), flush=True)
            return
        self.turns[16] = int(st.get("turns", 0))
        self.last[16] = int(st.get("raw", 0))
        print(
            "UNWRAP loaded twist raw=%s turns=%s un=%s center_raw=%s"
            % (st.get("raw"), st.get("turns"), st.get("un"), st.get("center_raw")),
            flush=True,
        )

    def apply(self, rows):
        events = []
        for id_, r in rows.items():
            p = r["pos"]
            prev = self.last.get(id_)
            if prev is not None:
                d = p - prev
                if d > 2048:
                    self.turns[id_] -= 1
                    events.append((id_, "WRAP", prev, p, self.turns[id_]))
                elif d < -2048:
                    self.turns[id_] += 1
                    events.append((id_, "WRAP", prev, p, self.turns[id_]))
            r["turns"] = self.turns[id_]
            r["un"] = p + self.turns[id_] * 4096
            prev_l = self.last_load.get(id_, 0)
            if abs(r["load"]) >= LOAD_SPIKE and abs(r["load"]) - abs(prev_l) >= 200:
                events.append((id_, "LOAD", r["load"], r["pos"], r["un"]))
            self.last[id_] = p
            self.last_load[id_] = r["load"]
        return events

    def save_twist(self, path, rows):
        r = rows.get(16)
        if not r:
            return
        st = {}
        if os.path.isfile(path):
            try:
                st = json.load(open(path))
            except (OSError, ValueError):
                st = {}
        # Pose only. Never rewrite home / walls from a coasted park.
        st["id"] = 16
        st["name"] = "twist"
        st["raw"] = r["pos"]
        st["turns"] = r["turns"]
        st["un"] = r["un"]
        st.setdefault("center_raw", 1036)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(st, f, indent=2)
            f.write("\n")


def print_table(t_s, rows, note):
    print("\nt=%6.1fs %s" % (t_s, note), flush=True)
    print("id  name      pos     un  tn  speed   load   cur mv tq     v temp", flush=True)
    for id_ in range(11, 18):
        r = rows.get(id_)
        if not r:
            print("%2d  %-8s   —" % (id_, JOINT[id_]), flush=True)
            continue
        print(
            "%2d  %-8s %4d  %6d  %2d  %5d  %5d  %4d  %d  %s  %4s  %3d"
            % (
                id_,
                JOINT[id_],
                r["pos"],
                r.get("un", r["pos"]),
                r.get("turns", 0),
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
            "un": r.get("un", r["pos"]),
            "turns": r.get("turns", 0),
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
    track = Unwrap()
    track.load_twist_abs(TWIST_ABS)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        f.flush()

        t, rows = sample(ser, t0)
        ev = track.apply(rows)
        for e in ev:
            print("%s id=%d %s" % (e[1], e[0], e[2:]), flush=True)
        print_table(t, rows, "rest")
        write_rows(w, t, rows, "rest")
        track.save_twist(TWIST_ABS, rows)
        f.flush()

        if not steps:
            n = 0
            while True:
                t, rows = sample(ser, t0)
                n += 1
                ev = track.apply(rows)
                for e in ev:
                    print("%s id=%d %s" % (e[1], e[0], e[2:]), flush=True)
                print_table(t, rows, "watch")
                write_rows(w, t, rows, "watch")
                track.save_twist(TWIST_ABS, rows)
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
                    ev = track.apply(last)
                    for e in ev:
                        print("%s id=%d %s" % (e[1], e[0], e[2:]), flush=True)
                    print_table(t, last, note)
                    write_rows(w, t, last, note)
                    track.save_twist(TWIST_ABS, last)
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
            track.apply(rows)
            print_table(t, rows, "free")
            write_rows(w, t, rows, "free")
            track.save_twist(TWIST_ABS, rows)
            f.flush()

    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        sys.exit(0)
