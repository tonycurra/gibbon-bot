"""Gold serial from this Waveshare Bus Servo Driver HAT (A).

Captures: roarm-fw/docs/baselines/usb-start.md, usb-no-battery.md,
usb-battery-on.md, hand-move-usb-battery-on.md. Firmware listen v1,
print_get_id / cmd_bench / cmd_batt in roarm-fw/main/main.c.
"""

CHIP = {
    "sku": "27577",
    "board": "Waveshare Bus Servo Driver HAT (A)",
    "mcu": "ESP32-WROOM-32 (ESP32-D0WD-V3 rev v3.1)",
    "mac": "44:1d:64:86:3f:54",
    "chip": "ESP32 cores=2 rev=301 flash=4096KB",
    "idf": "v5.4.2",
    "usb": "/dev/ttyACM0 @ 115200",
    "bus": "gpio18/19 @ 1 Mbps, ids 11–17",
    "fw": "listen v1",
}

MOTORS = [
    {"id": 11, "joint": "turn / yaw", "model": "ST3235"},
    {"id": 12, "joint": "raise A / pitch", "model": "ST3235"},
    {"id": 13, "joint": "raise B / pitch", "model": "ST3235"},
    {"id": 14, "joint": "bend / elbow", "model": "ST3235"},
    {"id": 15, "joint": "wrist pitch", "model": "ST3235"},
    {"id": 16, "joint": "twist / roll", "model": "ST3215"},
    {"id": 17, "joint": "hand / grip", "model": "ST3215HS"},
]

# Present block @56, 15 bytes — same READ on every ID. Firmware print_get_id().
FIELDS = [
    {"addr": "56–57", "key": "pos", "kind": "encoder", "unit": "0–4095 / turn",
     "off": "miss", "on": "live", "hand": "tracks the horn"},
    {"addr": "58–59", "key": "speed", "kind": "from encoder", "unit": "steps/s",
     "off": "miss", "on": "0 if still", "hand": "lives if you spin"},
    {"addr": "60–61", "key": "load", "kind": "PWM effort", "unit": "~±1000",
     "off": "miss", "on": "0 (tq off)", "hand": "always 0"},
    {"addr": "62", "key": "v", "kind": "VIN at this servo", "unit": "0.1 V LSB",
     "off": "miss", "on": "~11.4 V", "hand": "pack rail"},
    {"addr": "63", "key": "temp", "kind": "case / driver", "unit": "°C",
     "off": "miss", "on": "26–29", "hand": "live"},
    {"addr": "65", "key": "st", "kind": "status byte", "unit": "raw",
     "off": "miss", "on": "0 typical", "hand": "0"},
    {"addr": "66", "key": "moving", "kind": "flag", "unit": "0/1",
     "off": "miss", "on": "0", "hand": "stays 0 — not “I touched it”"},
    {"addr": "69–70", "key": "cur", "kind": "winding estimate", "unit": "mA",
     "off": "miss", "on": "0", "hand": "0 or 1 noise"},
    {"addr": "40", "key": "tq", "kind": "torque enable", "unit": "0/1",
     "off": "miss", "on": "0", "hand": "must stay 0"},
    {"addr": "33", "key": "mode", "kind": "EEPROM", "unit": "0 servo / 1 wheel",
     "off": "—", "on": "cached", "hand": "servo"},
]

GET_GOLD = [
    {"id": 11, "pos": 1900, "v": "11.5", "temp": 26, "note": "near mid (2048 ≈ 180°)"},
    {"id": 12, "pos": 2025, "v": "11.4", "temp": 28, "note": "paired with 13"},
    {"id": 13, "pos": 2073, "v": "11.4", "temp": 29, "note": "48 counts from 12"},
    {"id": 14, "pos": 2821, "v": "11.4", "temp": 28, "note": "stock home is not 2048"},
    {"id": 15, "pos": 2176, "v": "11.4", "temp": 29, "note": ""},
    {"id": 16, "pos": 4024, "v": "11.4", "temp": 26, "note": "near 0/4095 wrap"},
    {"id": 17, "pos": 2088, "v": "11.4", "temp": 28, "note": "jaws closed wall"},
]

BATT_GOLD = [
    {"id": 11, "v": "11.5"},
    {"id": 12, "v": "11.4"},
    {"id": 13, "v": "11.4"},
    {"id": 14, "v": "11.4"},
    {"id": 15, "v": "11.4"},
    {"id": 16, "v": "11.4"},
    {"id": 17, "v": "11.3"},
]

HAND_REST = [
    {"id": 11, "name": "base", "pos": 1131, "v": "11.5", "temp": 25},
    {"id": 12, "name": "shoulder-A", "pos": 1921, "v": "11.4", "temp": 26},
    {"id": 13, "name": "shoulder-B", "pos": 2173, "v": "11.4", "temp": 28},
    {"id": 14, "name": "elbow", "pos": 1904, "v": "11.4", "temp": 27},
    {"id": 15, "name": "wrist-pitch", "pos": 2147, "v": "11.4", "temp": 27},
    {"id": 16, "name": "wrist-roll", "pos": 2893, "v": "11.3", "temp": 25},
    {"id": 17, "name": "gripper", "pos": 2088, "v": "11.4", "temp": 27},
]

HAND_NET = [
    {"id": 11, "rest": 1131, "last": 1238, "d": "+107", "deg": "+9°"},
    {"id": 12, "rest": 1921, "last": 1984, "d": "+63", "deg": "+6°"},
    {"id": 13, "rest": 2173, "last": 2110, "d": "−63", "deg": "−6°"},
    {"id": 14, "rest": 1904, "last": 2836, "d": "+932", "deg": "+82°"},
    {"id": 15, "rest": 2147, "last": 1478, "d": "−669", "deg": "−59°"},
    {"id": 16, "rest": 2893, "last": 3033, "d": "+140", "deg": "+12°"},
    {"id": 17, "rest": 2088, "last": 2086, "d": "−2", "deg": "opened and closed"},
]

BENCH_OFF = """bench v1
chip=ESP32 cores=2 rev=301 flash=4096KB
mac=44:1d:64:86:3f:54
bus=gpio18/19@1Mbps ids=11-17
ping 11=miss 12=miss 13=miss 14=miss 15=miss 16=miss 17=miss
found=0/7
guess=usb-no-battery"""

BENCH_ON = """bench v1
chip=ESP32 cores=2 rev=301 flash=4096KB
mac=44:1d:64:86:3f:54
bus=gpio18/19@1Mbps ids=11-17
ping 11=found 12=found 13=found 14=found 15=found 16=found 17=found
found=7/7
guess=usb-battery-on"""

GET_ON = """get id=11 pos=1900 speed=0 load=0 v_raw=115 v=11.5 temp=26 cur=0 moving=0 tq=0
get id=12 pos=2025 speed=0 load=0 v_raw=114 v=11.4 temp=28 cur=0 moving=0 tq=0
get id=13 pos=2073 speed=0 load=0 v_raw=114 v=11.4 temp=29 cur=0 moving=0 tq=0
get id=14 pos=2821 speed=0 load=0 v_raw=114 v=11.4 temp=28 cur=0 moving=0 tq=0
get id=15 pos=2176 speed=0 load=0 v_raw=114 v=11.4 temp=29 cur=0 moving=0 tq=0
get id=16 pos=4024 speed=0 load=0 v_raw=114 v=11.4 temp=26 cur=0 moving=0 tq=0
get id=17 pos=2088 speed=0 load=0 v_raw=114 v=11.4 temp=28 cur=0 moving=0 tq=0"""

BATT_ON = """batt id=11 v=11.5
batt id=12 v=11.4
batt id=13 v=11.4
batt id=14 v=11.4
batt id=15 v=11.4
batt id=16 v=11.4
batt id=17 v=11.3
batt pack=11.4V est=25% cells=3 n=7 ok"""

BATT_OFF = """batt pack=miss est=? note=servos need XT60"""

INFO_OFF = """info chip=ESP32 cores=2 rev=301 flash=4096KB reset=poweron
info mac=44:1d:64:86:3f:54 heap=297940 uptime_ms=123760
info idf=v5.4.2 usb=115200 bus=gpio18/19@1Mbps ids=11-17
info note=listen-only until ping/get. no move. servos need XT60"""

STATE_OFF = """state t=37780 heap=297940 reset=poweron tasks=5 bus_rx=0 dump=1
state mac=44:1d:64:86:3f:54
state bus=uart1 gpio18/19 1Mbps rx-only until ping
gpio 18(rx)=1 19(tx)=1 (uart1, not remuxed)"""

UNPLUG = """ping none 11-17 — expected if battery is off
found=0/7
guess=usb-no-battery
batt pack=miss est=? note=servos need XT60"""

# Wire: 1 Mbps 8N1 = 10 µs/byte. Firmware stream floor is 40 ms (cmd_stream).
TIMING = [
    {"what": "bit on UART1", "dt": "10 µs/byte", "note": "1 Mbps 8N1"},
    {"what": "one present READ, on the wire", "dt": "~0.3 ms", "note": "8 TX + ~21 RX"},
    {"what": "one ID get (present + tq)", "dt": "~2–12 ms", "note": "wait until reply; timeout 12 ms"},
    {"what": "firmware stream floor", "dt": "40 ms", "note": "`stream N` refuses faster. ~25 Hz one moving ID"},
    {"what": "console default stream", "dt": "180 ms", "note": "afterConnect. sitting, not a grove radio"},
    {"what": "full `get` 11–17", "dt": "~80–170 ms", "note": "14 transactions. starves a live wheel"},
    {"what": "BLE to a phone in the grove", "dt": "200–1000 ms", "note": "pose + energy. not the inner loop"},
]

# ST3215 series memory table (Waveshare FAQ points here; values from the STS table).
# Addr 33: 0 servo, 1 constant-speed (reg 46), 2 PWM open-loop, 3 step.
MEMORY = [
    {"addr": "5", "name": "ID", "rw": "EE", "note": "1–253. this arm 11–17"},
    {"addr": "6", "name": "baud", "rw": "EE", "note": "default 1 Mbps. do not change live"},
    {"addr": "9–12", "name": "amin / amax", "rw": "EE", "note": "often 0–4095 = no wall"},
    {"addr": "16 / 48", "name": "max torque", "rw": "EE/RAM", "note": "0–1000 = 0–100% stall. not wired yet"},
    {"addr": "21–23", "name": "Kp Kd Ki", "rw": "EE", "note": "`who` reads. do not write first"},
    {"addr": "33", "name": "mode", "rw": "EE", "note": "0 servo · 1 wheel · 2 PWM · 3 step"},
    {"addr": "40", "name": "torque en", "rw": "RAM", "note": "0 limp · 1 hold/follow"},
    {"addr": "41", "name": "acc", "rw": "RAM", "note": "0=step, 1–254. unit ~100 step/s². default 20"},
    {"addr": "42–43", "name": "goal pos", "rw": "RAM", "note": "0–4095 in servo mode"},
    {"addr": "44–45", "name": "goal time", "rw": "RAM", "note": "we write 0. not a spline"},
    {"addr": "46–47", "name": "goal speed", "rw": "RAM", "note": "steps/s. 50 ≈ 0.732 RPM. bit15 dir in wheel"},
    {"addr": "56–70", "name": "present", "rw": "RAM", "note": "the 15-byte `get` block"},
]

MODES = [
    {"n": "0", "name": "servo", "cmd": "goal pos + speed + acc", "loop": "position PID", "wrap": "stops at 0 and 4095", "use": "default. all joints"},
    {"n": "1", "name": "wheel", "cmd": "signed speed @46", "loop": "speed", "wrap": "spins forever; encoder still wraps", "use": "do not need this for unwrap"},
    {"n": "2", "name": "PWM", "cmd": "open-loop duty", "loop": "none", "wrap": "no position goal", "use": "no"},
    {"n": "3", "name": "step", "cmd": "relative steps", "loop": "position, ±~7 turns RAM", "wrap": "lost on power-off", "use": "no — we unwrap instead"},
]

LINKS = [
    {"title": "ST3215 wiki", "url": "https://www.waveshare.com/wiki/ST3215_Servo"},
    {"title": "Bus servo drive circuit (HAT PHY, not the servo PCB)",
     "url": "https://files.waveshare.com/upload/d/d3/Bus_servo_control_circuit.pdf"},
    {"title": "Protocol manual (packet / question-answer)",
     "url": "https://files.waveshare.com/upload/2/27/Communication_Protocol_User_Manual-EN%28191218-0923%29.pdf"},
    {"title": "ST3215 user manual",
     "url": "https://files.waveshare.com/upload/f/f4/ST3215_Servo_User_Manual.pdf"},
]
