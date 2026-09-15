#!/usr/bin/env python3
"""Wait for the HAT on USB, print every serial line, track board status.

  python3 listen.py
  python3 listen.py --log /tmp/01-usb-listen.txt

No HTML. One process owns ACM0 — stop idf.py monitor first.
Plug the ESP32 debug USB-C (not the bus Type-C). Pack may be off.
"""
import argparse
import glob
import os
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("need pyserial: python3 -m pip install pyserial")

BAUD = 115200
SILENT_S = 4.0


def now():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%H:%M:%S", time.localtime(t)) + ".%03d" % ms


def find_port(prefer):
    if prefer and os.path.exists(prefer):
        return prefer
    ports = sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def emit(out, text):
    line = "%s  %s" % (now(), text)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if out:
        out.write(line + "\n")
        out.flush()


def status(out, name):
    emit(out, "# status=%s" % name)


def open_serial(port):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = BAUD
    ser.timeout = 0.2
    ser.dtr = True
    ser.rts = True
    ser.open()
    return ser


def session(ser, port, out):
    buf = b""
    last_rx = time.time()
    alive = False
    fw = None
    mac = None
    status(out, "open %s" % port)
    try:
        while os.path.exists(port):
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                last_rx = time.time()
                if not alive:
                    alive = True
                    status(out, "alive")
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    ln = raw.decode("utf-8", "replace").strip("\r")
                    if not ln:
                        continue
                    emit(out, ln)
                    if ln.startswith("bringup "):
                        fw = ln.split(" ", 1)[1].strip()
                        status(out, "fw=%s" % fw)
                    elif ln.startswith("mac=") and mac is None:
                        mac = ln[4:].strip()
                        status(out, "mac=%s" % mac)
            elif alive and (time.time() - last_rx) >= SILENT_S:
                alive = False
                status(out, "silent")
    except serial.SerialException as e:
        emit(out, "# serial %s" % e)
    finally:
        try:
            ser.close()
        except Exception:
            pass
    status(out, "gone")


def main():
    ap = argparse.ArgumentParser(description="USB serial listen for bring-up")
    ap.add_argument("-p", "--port", default="/dev/ttyACM0")
    ap.add_argument("--log", help="append timestamped lines to this file")
    args = ap.parse_args()

    out = open(args.log, "a", encoding="utf-8") if args.log else None
    if out:
        emit(out, "# log start port=%s" % args.port)

    status(out, "waiting")
    try:
        while True:
            port = find_port(args.port)
            if not port:
                time.sleep(0.2)
                continue
            try:
                ser = open_serial(port)
            except serial.SerialException:
                time.sleep(0.2)
                continue
            emit(out, "+ %s" % port)
            session(ser, port, out)
            emit(out, "- %s" % port)
            status(out, "waiting")
            time.sleep(0.3)
    except KeyboardInterrupt:
        status(out, "stop")
    finally:
        if out:
            out.close()


if __name__ == "__main__":
    main()
