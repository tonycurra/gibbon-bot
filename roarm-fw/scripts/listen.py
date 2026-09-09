#!/usr/bin/env python3
"""Live sensor table. All registers, overwrites in place. READ only.

  python3 scripts/listen.py

pos = magnet 0-4095 (wraps). un = pos + turns×4096 (cable).
Do not run another ACM0 process at the same time.
"""
import re
import sys
import time

import serial

PORT = "/dev/ttyACM0"
BAUD = 115200
HOME = {16: 4068}  # slack twist; others blank
GET_RE = re.compile(
    r"get id=(\d+) pos=(-?\d+) speed=(-?\d+) load=(-?\d+) "
    r"v_raw=(\d+) v=([\d.]+) temp=(\d+) cur=(-?\d+) moving=(\d+) tq=(\S+)"
    r"(?: un=(-?\d+) turns=(-?\d+))?"
    r"(?: st=(-?\d+))?"
    r"(?: mode=(\S+))?"
)
JOINT = {
    11: "turn",
    12: "raise-A",
    13: "raise-B",
    14: "bend",
    15: "wrist",
    16: "twist",
    17: "hand",
}

HIDE = "\033[?25l"
SHOW = "\033[?25h"
HOME_CUR = "\033[H"
CLEAR_TAIL = "\033[J"
RST = "\033[0m"
DIM = "\033[90m"
GRN = "\033[32m"
YEL = "\033[33m"
RED = "\033[31m"


def send(ser, cmd):
    ser.write((cmd + "\n").encode("ascii"))
    ser.flush()


def drain(ser, seconds):
    end = time.time() + seconds
    buf, lines = b"", []
    while time.time() < end:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            ln = raw.decode("utf-8", "replace").strip()
            if ln:
                lines.append(ln)
    return lines


def parse(lines):
    rows = {}
    dump = None
    for ln in lines:
        if ln.startswith("listen dump="):
            dump = ln.endswith("=1")
        m = GET_RE.search(ln)
        if not m:
            continue
        id_ = int(m.group(1))
        pos = int(m.group(2))
        un = int(m.group(11)) if m.group(11) else pos
        tn = int(m.group(12)) if m.group(12) else 0
        rows[id_] = {
            "pos": pos,
            "speed": int(m.group(3)),
            "load": int(m.group(4)),
            "v": m.group(6),
            "temp": int(m.group(7)),
            "cur": int(m.group(8)),
            "moving": int(m.group(9)),
            "tq": m.group(10),
            "un": un,
            "turns": tn,
            "st": m.group(13) if m.group(13) else "?",
            "mode": m.group(14) if m.group(14) else "?",
        }
    return rows, dump


class Unwrap:
    def __init__(self):
        self.last = {}
        self.turns = {i: 0 for i in range(11, 18)}

    def apply(self, rows):
        for id_, r in rows.items():
            p = r["pos"]
            prev = self.last.get(id_)
            if prev is not None:
                d = p - prev
                if d > 2048:
                    self.turns[id_] -= 1
                elif d < -2048:
                    self.turns[id_] += 1
            self.last[id_] = p
            if r["un"] == r["pos"]:
                r["un"] = p + self.turns[id_] * 4096
                r["turns"] = self.turns[id_]
            else:
                self.turns[id_] = r["turns"]


def deg_of(id_, un):
    if id_ not in HOME:
        return ""
    return "%+6.1f" % ((un - HOME[id_]) * 360.0 / 4096.0)


def color_row(r, prev):
    if r["tq"] == "1":
        return RED
    if abs(r["load"]) >= 400 or abs(r["cur"]) >= 80:
        return YEL
    if prev and abs(r["pos"] - prev["pos"]) >= 2:
        return GRN
    if r["speed"] != 0:
        return GRN
    return RST


def draw(t_s, rows, prev, miss):
    vs = [rows[i]["v"] for i in range(11, 18) if i in rows]
    pack = vs[0] if vs else "?"
    tq_on = [i for i in range(11, 18) if i in rows and rows[i]["tq"] == "1"]
    tq_s = ("tq ON " + ",".join(str(i) for i in tq_on)) if tq_on else "tq all 0"
    miss_s = ("  MISS " + ",".join(str(i) for i in miss)) if miss else ""
    lines = [
        "  listen  t=%5.1fs  pack=%sV  %s%s" % (t_s, pack, tq_s, miss_s),
        DIM
        + "  pos 0–4095 magnet.  un = pos + turns×4096.  same bytes as get."
        + RST,
        "  ID  joint      pos     un  tn      °   spd  load  cur mv tq     V  °C  st mode",
    ]
    for id_ in range(11, 18):
        r = rows.get(id_)
        if not r:
            lines.append("  %2d  %-8s   —" % (id_, JOINT[id_]))
            continue
        c = color_row(r, prev.get(id_) if prev else None)
        lines.append(
            "%s  %2d  %-8s  %4d  %5d %+3d %7s  %4d  %4d  %3d  %d  %s  %4s  %3d  %2s %4s%s"
            % (
                c,
                id_,
                JOINT[id_],
                r["pos"],
                r["un"],
                r["turns"],
                deg_of(id_, r["un"]),
                r["speed"],
                r["load"],
                r["cur"],
                r["moving"],
                r["tq"],
                r["v"],
                r["temp"],
                r["st"],
                r["mode"],
                RST,
            )
        )
    lines.append("")
    lines.append(DIM + "  green = pos/speed   yellow = load/cur   red = torque on" + RST)
    lines.append(DIM + "  backdrive ok.  Ctrl+C stop." + RST)
    sys.stdout.write(HOME_CUR + "\n".join(lines) + "\n" + CLEAR_TAIL)
    sys.stdout.flush()


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.05)
    time.sleep(0.25)
    ser.reset_input_buffer()
    send(ser, "")
    drain(ser, 0.4)
    send(ser, "listen")
    extra = drain(ser, 0.35)
    _, dump = parse(extra)
    if dump is True:
        send(ser, "listen")
        drain(ser, 0.25)

    track = Unwrap()
    prev = {}
    t0 = time.time()
    sys.stdout.write(HIDE + "\033[2J")
    sys.stdout.flush()
    try:
        while True:
            send(ser, "get")
            raw = drain(ser, 1.5)
            rows, _ = parse(raw)
            track.apply(rows)
            miss = [i for i in range(11, 18) if i not in rows]
            draw(time.time() - t0, rows, prev, miss)
            if rows:
                prev = {i: dict(rows[i]) for i in rows}
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW + "\n")
        sys.stdout.flush()
        ser.close()


if __name__ == "__main__":
    main()
