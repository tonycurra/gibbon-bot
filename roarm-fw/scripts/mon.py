#!/usr/bin/env python3
"""Raw USB text pipe to the chip. No IDF, no Django, no reset dance.

  ./scripts/mon.sh
  python3 scripts/mon.py
  python3 scripts/mon.py /dev/ttyACM0

Type firmware commands, Enter. Ctrl+C quits. estop still works on the chip.
Do not run this with Django, idf.py monitor, or live.py.
"""
import os
import select
import sys
import time

import serial

PORT = "/dev/ttyACM0"
BAUD = 115200

SITTING = """\
chip serial  {port} {baud}  Ctrl+C quits
  ping / get N / mat 50 / stream          look
  id N free                               limp (raise = id 12 free)
  lim N home                              mark 0° at live pos
  lim N min   /  lim N max                mark stops
  sweep N                                 min→max→min, 80 inside walls
  sweep 12                                raise pair (both marked)
  sweep stop  /  estop                    cancel / all tq=0
"""


def open_port(port):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = BAUD
    ser.timeout = 0.05
    ser.write_timeout = 0.5
    ser.dsrdtr = False
    ser.rtscts = False
    ser.open()
    try:
        ser.setDTR(False)
        ser.setRTS(False)
    except Exception:
        pass
    return ser


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    if not os.path.exists(port):
        sys.exit(f"no {port} — plug the HAT, UART switch = ESP32")
    try:
        ser = open_port(port)
    except serial.SerialException as e:
        sys.exit(
            f"{port}: {e}\nstop Django / idf monitor first, then: sudo chmod 666 {port}"
        )

    sys.stdout.write(SITTING.format(port=port, baud=BAUD))
    sys.stdout.flush()
    try:
        while True:
            r, _, _ = select.select([ser, sys.stdin], [], [], 0.1)
            if ser in r:
                chunk = ser.read(ser.in_waiting or 1)
                if chunk:
                    sys.stdout.write(chunk.decode("utf-8", "replace"))
                    sys.stdout.flush()
            if sys.stdin in r:
                line = sys.stdin.readline()
                if line == "":
                    break
                ser.write(line.encode("ascii", "replace").replace(b"\r", b""))
                if not line.endswith("\n"):
                    ser.write(b"\n")
                ser.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.05)


if __name__ == "__main__":
    main()
