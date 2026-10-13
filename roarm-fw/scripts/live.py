#!/usr/bin/env python3
"""One ACM0 process: listen + control. Same unwrap for hand-turn and moves.

Sitting (you type):
  python3 scripts/live.py
  python3 scripts/live.py --out docs/baselines/data/twist-live.csv

Commanded twist check (host listens the whole time, no free):
  python3 scripts/live.py --check twist --out docs/baselines/data/twist-commanded.csv

Host commands (not sent to the chip):
  here   this twist pose is slack home, turns=0 (saves home_raw / home_un)
  table  print all seven joints
  q      restore servo, keep torque (no free / no coast)

Everything else goes to the chip: twist servo, twist wheel, twist speed N,
twist to N, twist free, hand …, get, batt, ping, state, help.

Do not run idf.py monitor or watch-csv.py at the same time.
Test catalog: docs/software.md
"""
import argparse
import csv
import json
import os
import re
import select
import sys
import time

import serial

BAUD = 115200
POLL_S = 0.45
GET_WAIT = 1.1
TWIST_ABS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "docs", "baselines", "data", "twist-absolute.json",
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
CHIP_PREFIX = (
    "twist ", "hand ", "get", "batt", "ping", "pingall", "state",
    "help", "bench", "gpio", "listen", "raw ",
)
GET_KV = re.compile(r"(\w+)=(\S+)")
FIELDS = [
    "t_s", "note", "id", "name", "pos", "un", "turns", "speed", "load", "cur",
    "moving", "tq", "v", "temp",
]


def send(ser, cmd, echo=True):
    ser.write((cmd + "\n").encode("ascii"))
    ser.flush()
    if echo:
        print(">>", cmd, flush=True)


def read_lines(ser, seconds):
    end = time.time() + seconds
    lines, buf = [], b""
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


def parse_get_line(line):
    if not line.startswith("get id=") or " miss" in line or " pos=" not in line:
        return None
    kv = dict(GET_KV.findall(line))
    if "id" not in kv or "pos" not in kv:
        return None
    pos = int(kv["pos"])
    return {
        "pos": pos,
        "speed": int(kv.get("speed", 0)),
        "load": int(kv.get("load", 0)),
        "v": kv.get("v", ""),
        "temp": int(kv["temp"]) if "temp" in kv else 0,
        "cur": int(kv.get("cur", 0)),
        "moving": int(kv.get("moving", 0)),
        "tq": kv.get("tq", "0"),
        "id": int(kv["id"]),
    }


def parse_get(lines):
    rows = {}
    extra = []
    for line in lines:
        rec = parse_get_line(line)
        if rec:
            rows[rec["id"]] = rec
            continue
        if line.startswith(("twist", "hand", "err", "get LOAD")):
            extra.append(line)
    return rows, extra


class Unwrap:
    def __init__(self):
        self.turns = {i: 0 for i in range(11, 18)}
        self.last = {}
        self.home_raw = 4068
        self.home_un = 4068
        self.meta = {}

    def load_abs(self, path):
        if not os.path.isfile(path):
            return
        try:
            st = json.load(open(path))
        except (OSError, ValueError) as e:
            print("abs unreadable:", e, flush=True)
            return
        self.meta = st
        if "home_raw" in st:
            self.home_raw = int(st["home_raw"])
        if "home_un" in st:
            self.home_un = int(st["home_un"])
        print(
            "abs home_raw=%s home_un=%s walls cw=%s ccw=%s (session turns start 0 — type here if slack mid)"
            % (self.home_raw, self.home_un, st.get("cw"), st.get("ccw")),
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
            self.last[id_] = p
        return events

    def save_pose(self, path, rows, home=False):
        r = rows.get(16)
        if not r:
            return
        st = {}
        if os.path.isfile(path):
            try:
                st = json.load(open(path))
            except (OSError, ValueError):
                st = {}
        st.update({
            "id": 16,
            "name": "twist",
            "raw": r["pos"],
            "turns": r["turns"],
            "un": r["un"],
        })
        if home:
            st["home_raw"] = r["pos"]
            st["home_un"] = r["un"]
            st["cable"] = "slack"
            st["stale"] = False
            st["note"] = "here: slack home turns=0"
            self.home_raw = r["pos"]
            self.home_un = r["un"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(st, f, indent=2)
            f.write("\n")


def print_table(t_s, rows):
    print("\nt=%6.1fs" % t_s, flush=True)
    print("id  name      pos     un  tn  speed   load   cur mv tq     v temp", flush=True)
    for id_ in range(11, 18):
        r = rows.get(id_)
        if not r:
            print("%2d  %-8s   —" % (id_, JOINT[id_]), flush=True)
            continue
        print(
            "%2d  %-8s %4d  %6d  %2d  %5d  %5d  %4d  %d  %s  %4s  %3d"
            % (
                id_, JOINT[id_], r["pos"], r.get("un", r["pos"]),
                r.get("turns", 0), r["speed"], r["load"], r["cur"],
                r["moving"], r["tq"], r["v"], r["temp"],
            ),
            flush=True,
        )


def print_twist(t_s, rows, track):
    r = rows.get(16)
    if not r:
        print("t=%5.1f  twist miss" % t_s, flush=True)
        return
    dhome = r["un"] - track.home_un
    print(
        "t=%5.1f  twist pos=%4d un=%6d tn=%+d spd=%4d load=%5d tq=%s  Δhome=%+d"
        % (t_s, r["pos"], r["un"], r["turns"], r["speed"], r["load"], r["tq"], dhome),
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
    send(ser, "get", echo=False)
    rows, extra = parse_get(read_lines(ser, GET_WAIT))
    return time.time() - t0, rows, extra


def restore_servo(ser):
    send(ser, "twist speed 0")
    for line in read_lines(ser, 0.25):
        if line.startswith("twist"):
            print(line, flush=True)
    for _ in range(2):
        send(ser, "twist servo")
        for line in read_lines(ser, 0.3):
            if line.startswith("twist"):
                print(line, flush=True)
                if "mode=0" in line:
                    return


def is_chip_cmd(s):
    if s in ("get", "batt", "ping", "pingall", "state", "help", "bench", "gpio", "listen"):
        return True
    return s.startswith(CHIP_PREFIX)


# --- twist commanded check (listen while moving) ---
# Walls you set: CW raw 978, CCW un 7126, slack mid raw 4068.
# Soft targets stay ~80+ inside. Wheel jog (WritePos cannot cross 0).
# Mid is nearest raw 4068, not a stale un from a previous park.
HOME_RAW = 4068
CW_SOFT = 1200
CCW_SOFT_UN = 6900
WHEEL_SPD = 300
BRAKE = 200
LOAD_HI = 1350
DLOAD = 100
HELD_N = 3
HOP_MAX = 900


def chip_lines(ser, cmd, wait=0.3):
    send(ser, cmd)
    out = []
    for ln in read_lines(ser, wait):
        if parse_get_line(ln):
            continue
        print(ln, flush=True)
        out.append(ln)
    return out


def ensure_wheel(ser):
    for _ in range(4):
        lines = chip_lines(ser, "twist wheel", 0.3)
        if any("mode=1" in ln for ln in lines):
            return True
    print("WHEEL_MISS", flush=True)
    return False


def ensure_servo(ser):
    chip_lines(ser, "twist speed 0", 0.2)
    for _ in range(4):
        lines = chip_lines(ser, "twist servo", 0.3)
        if any("mode=0" in ln for ln in lines):
            return True
    print("SERVO_MISS", flush=True)
    return False


def poll(ser, t0, track, csv_w, csv_f, note, last):
    t, rows, extra = sample(ser, t0)
    for ln in extra:
        print(ln, flush=True)
    ev = track.apply(rows)
    for e in ev:
        print("%s id=%d %s" % (e[1], e[0], e[2:]), flush=True)
    last = rows or last
    print_twist(t, last, track)
    if csv_w:
        write_rows(csv_w, t, last, note)
        csv_f.flush()
    track.save_pose(TWIST_ABS, last)
    return last


def nearest_home_un(r):
    cur = r["un"]
    tn = r["turns"]
    cands = [HOME_RAW + k * 4096 for k in range(tn - 1, tn + 2)]
    return min(cands, key=lambda u: abs(u - cur))


def wheel_to(ser, t0, track, csv_w, csv_f, last, target_un, note):
    r = last.get(16)
    if not r:
        return last, "NO_POS"
    sign = 1 if target_un >= r["un"] else -1
    if not ensure_wheel(ser):
        return last, "WHEEL_MISS"
    chip_lines(ser, "twist speed %d" % (WHEEL_SPD if sign > 0 else -WHEEL_SPD), 0.15)
    held = 0
    prev = None
    cruise = []
    miss = 0
    t_end = time.time() + 22
    why = "ok"
    while time.time() < t_end:
        last = poll(ser, t0, track, csv_w, csv_f, note, last)
        r = last.get(16)
        if not r:
            miss += 1
            if miss >= 3:
                chip_lines(ser, "estop", 0.3)
                return last, "BUS_MISS"
            continue
        miss = 0
        al = abs(r["load"])
        if abs(r["speed"]) > 40 and len(cruise) < 4:
            cruise.append(al)
        cruise_avg = sum(cruise) / len(cruise) if cruise else 0
        if len(cruise) >= 4 and al - cruise_avg >= DLOAD:
            chip_lines(ser, "estop", 0.3)
            return last, "DLOAD"
        if al >= LOAD_HI or abs(r["cur"]) >= 80:
            chip_lines(ser, "estop", 0.3)
            why = "LOAD" if al >= LOAD_HI else "CUR"
            break
        if prev and abs(r["un"] - prev["un"]) < 20 and abs(r["speed"]) < 50:
            held += 1
            if held >= HELD_N:
                why = "HELD"
                break
        else:
            held = 0
        prev = r
        if sign > 0 and r["un"] >= target_un - BRAKE:
            break
        if sign < 0 and r["un"] <= target_un + BRAKE:
            break
    chip_lines(ser, "twist speed 0", 0.2)
    time.sleep(0.12)
    last = poll(ser, t0, track, csv_w, csv_f, note + "-stop", last)
    return last, why


def hold_mid(ser, t0, track, csv_w, csv_f, last):
    last = poll(ser, t0, track, csv_w, csv_f, "hold-mid", last)
    r = last.get(16)
    if not r:
        return last, "NO_POS"
    goal_un = nearest_home_un(r)
    if abs(r["un"] - goal_un) > 40:
        last, why = wheel_to(ser, t0, track, csv_w, csv_f, last, goal_un, "to-mid")
        if why not in ("ok", "HELD"):
            ensure_servo(ser)
            return last, why
    if not ensure_servo(ser):
        return last, "SERVO_MISS"
    last = poll(ser, t0, track, csv_w, csv_f, "hold-mid", last)
    r = last.get(16)
    hop = HOME_RAW - r["pos"]
    if abs(hop) <= HOP_MAX and abs(hop) > 8:
        chip_lines(ser, "twist to %d" % HOME_RAW, 0.3)
        time.sleep(0.45)
        last = poll(ser, t0, track, csv_w, csv_f, "servo-4068", last)
    return last, "ok"


def declare_here(ser, t0, track, csv_w, csv_f, last):
    send(ser, "twist here")
    for ln in read_lines(ser, 0.35):
        print(ln, flush=True)
    last = poll(ser, t0, track, csv_w, csv_f, "here", last)
    if 16 in last:
        track.turns[16] = 0
        track.last[16] = last[16]["pos"]
        last[16]["turns"] = 0
        last[16]["un"] = last[16]["pos"]
    track.apply(last)
    track.save_pose(TWIST_ABS, last, home=True)
    print("home set raw=%d un=%d turns=0" % (track.home_raw, track.home_un), flush=True)
    return last


def check_twist(ser, t0, track, csv_w, csv_f, last):
    print("CHECK twist listen+move  home raw=%d  CW soft=%d (wall 978)  CCW un soft=%d (wall 7126)"
          % (HOME_RAW, CW_SOFT, CCW_SOFT_UN), flush=True)
    print("Look at the loom — going to slack mid, then both soft stops, then hold mid. No free.",
          flush=True)
    last, why = hold_mid(ser, t0, track, csv_w, csv_f, last)
    print("AT mid-before-here", last.get(16), why, flush=True)
    r = last.get(16) or {}
    pos = r.get("pos", -1)
    # 40 and 4068 are the same seam neighborhood — do not rewrite home to 40.
    # CW from 40 toward 1200 is the wrong way (your CCW).
    on_4068 = pos >= HOME_RAW - 80
    if on_4068:
        last = declare_here(ser, t0, track, csv_w, csv_f, last)
    else:
        print("skip here pos=%s — not on 4068 side of seam (home stays %d)"
              % (pos, track.home_raw), flush=True)
    last, a = wheel_to(ser, t0, track, csv_w, csv_f, last, CW_SOFT, "cw")
    print("AT CW", last.get(16), a, flush=True)
    last, b = hold_mid(ser, t0, track, csv_w, csv_f, last)
    print("AT mid", last.get(16), b, flush=True)
    last, c = wheel_to(ser, t0, track, csv_w, csv_f, last, CCW_SOFT_UN, "ccw")
    print("AT CCW", last.get(16), c, flush=True)
    last, d = hold_mid(ser, t0, track, csv_w, csv_f, last)
    print("AT mid-hold", last.get(16), d, flush=True)
    print("LEGS", a, b, c, d, flush=True)
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="", help="twist = commanded roll check, listen while moving")
    args = ap.parse_args()

    ser = serial.Serial(args.port, BAUD, timeout=0.12)
    time.sleep(0.4)
    ser.reset_input_buffer()
    t0 = time.time()
    track = Unwrap()
    track.load_abs(TWIST_ABS)

    csv_f = csv_w = None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        csv_f = open(args.out, "w", newline="")
        csv_w = csv.DictWriter(csv_f, fieldnames=FIELDS)
        csv_w.writeheader()
        csv_f.flush()

    last = {}
    t, rows, extra = sample(ser, t0)
    for line in extra:
        print(line, flush=True)
    ev = track.apply(rows)
    for e in ev:
        print("%s id=%d %s" % (e[1], e[0], e[2:]), flush=True)
    print_table(t, rows)
    print_twist(t, rows, track)
    if csv_w:
        write_rows(csv_w, t, rows, "rest")
        csv_f.flush()
    last = rows or last
    track.save_pose(TWIST_ABS, last)

    if args.check == "twist":
        if not args.out:
            args.out = os.path.join(
                os.path.dirname(TWIST_ABS), "twist-commanded.csv")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            csv_f = open(args.out, "w", newline="")
            csv_w = csv.DictWriter(csv_f, fieldnames=FIELDS)
            csv_w.writeheader()
            csv_f.flush()
            write_rows(csv_w, t, last, "rest")
            csv_f.flush()
        try:
            check_twist(ser, t0, track, csv_w, csv_f, last)
        finally:
            restore_servo(ser)
            if csv_f:
                csv_f.close()
            ser.close()
            print("wrote %s (servo hold, no free)" % args.out, flush=True)
        return 0

    next_poll = time.time() + POLL_S
    n = 0

    print(
        "live. host: here | table | q    chip: twist servo/wheel/speed N/to N/free   (not twist mid)",
        flush=True,
    )

    try:
        while True:
            wait = max(0.0, next_poll - time.time())
            ready, _, _ = select.select([sys.stdin], [], [], wait)
            if ready:
                line = sys.stdin.readline()
                if line == "":
                    break
                cmd = line.strip()
                if not cmd:
                    continue
                if cmd in ("q", "quit"):
                    break
                if cmd == "table":
                    print_table(time.time() - t0, last)
                    continue
                if cmd == "here":
                    send(ser, "twist here")
                    for ln in read_lines(ser, 0.35):
                        print(ln, flush=True)
                    t, rows, extra = sample(ser, t0)
                    last = rows or last
                    if 16 in last:
                        track.turns[16] = 0
                        track.last[16] = last[16]["pos"]
                        last[16]["turns"] = 0
                        last[16]["un"] = last[16]["pos"]
                    track.apply(last)
                    track.save_pose(TWIST_ABS, last, home=True)
                    print_twist(t, last, track)
                    print("home set raw=%d un=%d turns=0" % (
                        track.home_raw, track.home_un), flush=True)
                    next_poll = time.time() + POLL_S
                    continue
                if cmd == "twist mid":
                    print("refusing twist mid — firmware goal is 4050, not slack 4068. use twist to 4068 or here",
                          flush=True)
                    continue
                if not is_chip_cmd(cmd):
                    print("unknown: %s  (here | table | q | twist … | hand …)" % cmd, flush=True)
                    continue
                send(ser, cmd)
                for ln in read_lines(ser, 0.35):
                    if parse_get_line(ln):
                        continue
                    print(ln, flush=True)
                t, rows, extra = sample(ser, t0)
                for ln in extra:
                    print(ln, flush=True)
                ev = track.apply(rows)
                for e in ev:
                    print("%s id=%d %s" % (e[1], e[0], e[2:]), flush=True)
                last = rows or last
                print_twist(t, last, track)
                if csv_w:
                    write_rows(csv_w, t, last, cmd.replace(" ", "-")[:24])
                    csv_f.flush()
                track.save_pose(TWIST_ABS, last)
                next_poll = time.time() + POLL_S
                continue

            t, rows, extra = sample(ser, t0)
            n += 1
            ev = track.apply(rows)
            for e in ev:
                print("%s id=%d %s" % (e[1], e[0], e[2:]), flush=True)
            last = rows or last
            print_twist(t, last, track)
            if n % 8 == 0:
                print_table(t, last)
            if csv_w:
                write_rows(csv_w, t, last, "watch")
                csv_f.flush()
            track.save_pose(TWIST_ABS, last)
            next_poll = time.time() + POLL_S
    finally:
        restore_servo(ser)
        if csv_f:
            csv_f.close()
        ser.close()
        print("stopped (servo, no free)", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        sys.exit(0)
