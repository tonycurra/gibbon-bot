#!/usr/bin/env python3
"""Motor matrix from the chip. Labels once, numbers every --ms.

  python3 scripts/listen.py

Wire (after `mat 50`):
  vec row=11,12,13,14,15,16,17
  vec col=pos,spd,load,cur,mv,tq,v10,temp
  m <uptime_ms>
  <8 ints>   × 7   or  -  if that ID missed
"""
import argparse
import os
import sys
import time

import serial

PORT = "/dev/ttyACM0"
BAUD = 115200
PERIOD_MS = 50
NAME = {
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
CUR_HOME = "\033[H"
CLEAR_TAIL = "\033[J"
RST = "\033[0m"
DIM = "\033[90m"
GRN = "\033[32m"
YEL = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def send(ser, cmd):
    ser.write((cmd + "\n").encode("ascii"))
    ser.flush()


def color(vals, prev):
    if not vals:
        return DIM
    pos, spd, load, cur, _mv, tq = vals[:6]
    if tq:
        return RED
    if abs(load) >= 400 or abs(cur) >= 80:
        return YEL
    if prev and abs(pos - prev[0]) >= 2:
        return GRN
    if spd:
        return GRN
    return RST


def draw(t_s, ms, rows, cols, frame, prev, chip_t, note):
    n_ok = sum(1 for r in frame if r)
    vs = [r[6] / 10.0 for r in frame if r]
    if n_ok == 0:
        pack_s = CYAN + "pack off — plug XT60" + RST
    else:
        pack_s = "pack=%.1fV  found=%d/%d" % (vs[len(vs) // 2], n_ok, len(rows) or 7)
    labels = [("V" if c == "v10" else c) for c in (cols or [])]
    hdr = "  ID  joint      " + "  ".join("%4s" % c for c in labels)
    lines = [
        "  mat  %s  loop=%dms  t=%5.1fs  chip=%s" % (pack_s, ms, t_s, chip_t),
        DIM + "  labels once on the wire. matrix every %d ms. v10 = tenths of a volt." % ms + RST,
        hdr if cols else "  (waiting vec col=)",
    ]
    for i, id_ in enumerate(rows or range(11, 18)):
        name = NAME.get(id_, str(id_))
        vals = frame[i] if i < len(frame) else None
        if not vals:
            lines.append(DIM + "  %2s  %-8s   —" % (id_, name) + RST)
            continue
        prev_i = prev[i] if i < len(prev) else None
        cells = []
        for j, v in enumerate(vals):
            if cols and j < len(cols) and cols[j] == "v10":
                cells.append("%4.1f" % (v / 10.0))
            else:
                cells.append("%4d" % v)
        lines.append(
            "%s  %2s  %-8s  %s%s"
            % (color(vals, prev_i), id_, name, "  ".join(cells), RST)
        )
    lines.append("")
    lines.append(DIM + "  " + note + "  Ctrl+C stop." + RST)
    sys.stdout.write(CUR_HOME + "\n".join(lines) + "\n" + CLEAR_TAIL)
    sys.stdout.flush()


def parse_line(ln, st):
    if ln.startswith("vec row="):
        st["rows"] = [int(x) for x in ln[8:].split(",") if x]
        return "hdr"
    if ln.startswith("vec col="):
        st["cols"] = ln[8:].split(",")
        return "hdr"
    if ln.startswith("m "):
        st["chip_t"] = ln[2:].strip()
        st["pending"] = []
        st["need"] = len(st["rows"]) if st["rows"] else 7
        return "m"
    if ln.startswith("mat ms="):
        st["mat_ack"] = True
        return "ack"
    if st.get("need", 0) > 0:
        if ln == "-":
            st["pending"].append(None)
        else:
            try:
                nums = [int(x) for x in ln.split()]
            except ValueError:
                return None
            st["pending"].append(nums)
        st["need"] -= 1
        if st["need"] == 0:
            st["frame"] = st["pending"]
            st["pending"] = []
            return "frame"
        return "row"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", default=PORT)
    ap.add_argument("--ms", type=int, default=PERIOD_MS)
    args = ap.parse_args()
    ms = args.ms
    if ms < 40:
        ms = 40
    if ms > 2000:
        ms = 2000
    period = ms / 1000.0

    ser = None
    t0 = time.time()
    st = {
        "rows": [],
        "cols": [],
        "frame": [],
        "pending": [],
        "need": 0,
        "chip_t": "-",
        "mat_ack": False,
    }
    prev = []
    last_draw = last_cmd = 0.0
    note = "waiting"

    sys.stdout.write(HIDE + "\033[2J" + CUR_HOME)
    sys.stdout.flush()
    try:
        while True:
            if ser is None:
                if not os.path.exists(args.port):
                    draw(time.time() - t0, ms, st["rows"], st["cols"],
                         st["frame"], prev, st["chip_t"],
                         "USB gone — plug ESP32 debug USB-C")
                    time.sleep(period)
                    continue
                try:
                    ser = serial.Serial(args.port, BAUD, timeout=period)
                    time.sleep(0.2)
                    ser.reset_input_buffer()
                except serial.SerialException:
                    ser = None
                    time.sleep(period)
                    continue
                st = {
                    "rows": [],
                    "cols": [],
                    "frame": [],
                    "pending": [],
                    "need": 0,
                    "chip_t": "-",
                    "mat_ack": False,
                }
                prev, buf = [], b""
                last_cmd = 0.0
                note = "waiting for chip"
            try:
                chunk = ser.read(ser.in_waiting or 1)
            except serial.SerialException:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                continue
            if chunk:
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    ln = raw.decode("utf-8", "replace").strip()
                    if not ln:
                        continue
                    kind = parse_line(ln, st)
                    if kind == "hdr":
                        note = "labels ok"
                    elif kind == "ack":
                        note = ln
                    elif kind == "frame":
                        note = "matrix"
                    elif ln.startswith("err unknown") and "mat" in ln:
                        note = "chip has no mat — flash this tree"
            now = time.time()
            if (not st["mat_ack"]) and (now - last_cmd) >= 1.0:
                try:
                    send(ser, "mat %d" % ms)
                except serial.SerialException:
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                    continue
                last_cmd = now
                note = "sent mat %d" % ms
            if now - last_draw >= period:
                if st["frame"] and any(st["frame"]):
                    note = "backdrive ok, tq=0"
                elif st["frame"]:
                    note = "servos silent — plug XT60"
                draw(now - t0, ms, st["rows"], st["cols"],
                     st["frame"], prev, st["chip_t"], note)
                prev = [r[:] if r else None for r in st["frame"]]
                last_draw = now
    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None:
            try:
                send(ser, "mat 0")
            except Exception:
                pass
            try:
                ser.close()
            except Exception:
                pass
        sys.stdout.write(SHOW + CUR_HOME + CLEAR_TAIL)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
