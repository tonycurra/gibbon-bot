"""USB serial + BLE Nordic-UART. Same text pipe as the chip."""
import atexit
import asyncio
import os
import queue
import re
import subprocess
import threading
import time

import termios

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTY = os.environ.get("GIBBON_TTY", "/dev/ttyACM0")
BAUD = termios.B115200
NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_SVC = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
ADDR_PATH = os.path.join(ROOT, ".ble-addr")
DEFAULT_ADDR = os.environ.get("GIBBON_BLE", "44:1D:64:86:3F:56")
GET_KV = re.compile(r"(\w+)=(\S+)")
MISS_RE = re.compile(r"get id=(\d+) miss")
BATT_RE = re.compile(r"batt pack=([\d.]+)V")


def parse_get(line):
    if not line.startswith("get id=") or " miss" in line:
        return None
    kv = dict(GET_KV.findall(line))
    if "pos" not in kv or "id" not in kv:
        return None
    pos = int(kv["pos"])
    return {
        "id": int(kv["id"]),
        "pos": pos,
        "speed": int(kv.get("speed", 0)),
        "load": int(kv.get("load", 0)),
        "v": float(kv["v"]) if "v" in kv else None,
        "temp": int(kv["temp"]) if "temp" in kv else None,
        "cur": int(kv.get("cur", 0)),
        "moving": int(kv.get("moving", 0)),
        "tq": (kv.get("tq") or "0")[:4],
        "un": int(kv.get("un", pos)),
        "turns": int(kv.get("turns", 0)),
        "mode": (kv.get("mode") or "0")[:8],
    }

clients = []
clients_lock = threading.Lock()
serial_fd = None
serial_ok = False
serial_lock = threading.Lock()
ble_ok = False
ble_name = ""
ble_loop = None
ble_write_q = None
dbq = queue.Queue(maxsize=2000)
_started = False
stopping = False


def load_addr():
    try:
        with open(ADDR_PATH) as f:
            a = f.read().strip()
            if a:
                return a
    except OSError:
        pass
    return DEFAULT_ADDR


def save_addr(addr):
    if not addr:
        return
    try:
        with open(ADDR_PATH, "w") as f:
            f.write(addr.strip() + "\n")
    except OSError:
        pass


def bluez_drop(addr):
    if not addr:
        return
    try:
        subprocess.run(["bluetoothctl", "disconnect", addr], timeout=4, capture_output=True)
    except Exception:
        pass


def subscribe():
    q = queue.Queue(maxsize=200)
    with clients_lock:
        clients.append(q)
    return q


def unsubscribe(q):
    with clients_lock:
        if q in clients:
            clients.remove(q)


def broadcast(line):
    if not line:
        return
    if line.startswith(("ESTOP", "guard ", "err", "hand ")) or " abort " in line:
        print("rx", line, flush=True)
    try:
        dbq.put_nowait(("rx", line))
    except queue.Full:
        pass
    with clients_lock:
        dead = []
        for q in clients:
            try:
                q.put_nowait(line)
            except queue.Full:
                dead.append(q)
        for q in dead:
            clients.remove(q)


def close_tty():
    global serial_fd, serial_ok
    with serial_lock:
        serial_ok = False
        if serial_fd is not None:
            try:
                os.close(serial_fd)
            except OSError:
                pass
            serial_fd = None


def open_tty():
    global serial_fd, serial_ok
    if not os.path.exists(TTY):
        return False
    try:
        fd = os.open(TTY, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(fd)
        a[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK
                  | termios.ISTRIP | termios.INLCR | termios.IGNCR
                  | termios.ICRNL | termios.IXON)
        a[1] &= ~termios.OPOST
        a[2] &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)
        a[2] |= termios.CS8 | termios.CREAD | termios.CLOCAL
        a[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
        a[4] = BAUD
        a[5] = BAUD
        a[6][termios.VMIN] = 0
        a[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, a)
        with serial_lock:
            if serial_fd is not None:
                try:
                    os.close(serial_fd)
                except OSError:
                    pass
            serial_fd = fd
            serial_ok = True
        return True
    except OSError:
        serial_ok = False
        return False


def serial_loop():
    buf = b""
    while True:
        if not os.path.exists(TTY):
            if serial_ok or serial_fd is not None:
                close_tty()
                broadcast("serial drop")
            time.sleep(0.4)
            continue
        if not serial_ok:
            if open_tty():
                buf = b""
                broadcast("serial up")
            else:
                time.sleep(0.5)
            continue
        try:
            chunk = os.read(serial_fd, 4096)
        except OSError:
            close_tty()
            broadcast("serial drop")
            time.sleep(0.3)
            continue
        if not chunk:
            time.sleep(0.02)
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            broadcast(line.decode("utf-8", "replace").replace("\r", "").strip())


def write_serial(text):
    data = text.encode("utf-8")
    if not data.endswith(b"\n"):
        data += b"\n"
    s = text.strip()
    if s and not s.startswith("get") and not s.startswith("stream") and s not in (
        "info", "state", "gpio", "bench", "batt", "who", "lim", "guard",
    ):
        print("tx", s, flush=True)
    with serial_lock:
        if serial_ok and serial_fd is not None:
            os.write(serial_fd, data)
            _db_tx(text)
            return
    if ble_ok and ble_loop is not None and ble_write_q is not None:
        ble_loop.call_soon_threadsafe(ble_write_q.put_nowait, data)
        _db_tx(text)
        return
    raise OSError("no link")


def _db_tx(text):
    try:
        dbq.put_nowait(("tx", text.strip()))
    except queue.Full:
        pass


def link_ok():
    return serial_ok or ble_ok


def via():
    if serial_ok:
        return "usb"
    if ble_ok:
        return "ble"
    return "none"


def status():
    return {
        "ok": link_ok(),
        "via": via(),
        "port": TTY if serial_ok else (ble_name or TTY),
        "bridge": True,
        "ble": ble_name,
    }


async def ble_task():
    global ble_ok, ble_name, ble_write_q
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError:
        print("bleak missing — USB only", flush=True)
        return
    ble_write_q = asyncio.Queue()
    buf = b""
    wait = 1.0

    def on_rx(_h, data):
        nonlocal buf
        buf += bytes(data)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            broadcast(line.decode("utf-8", "replace").replace("\r", "").strip())

    def is_gibbon(dev, adv=None):
        name = (dev.name or "") + " " + ((adv.local_name if adv else "") or "")
        if "gibbon" in name.lower():
            return True
        addr = load_addr()
        if addr and dev.address.upper() == addr.upper():
            return True
        uuids = [u.lower() for u in ((adv.service_uuids if adv else None) or [])]
        return NUS_SVC in uuids

    async def pick():
        addr = load_addr()
        if addr:
            return addr
        try:
            found = await BleakScanner.discover(timeout=2.0, return_adv=True)
        except Exception as e:
            print("ble scan", e, flush=True)
            return None
        items = found.items() if isinstance(found, dict) else ((d, None) for d in found)
        for dev, adv in items:
            if is_gibbon(dev, adv):
                save_addr(dev.address)
                return dev
        return None

    while not stopping:
        if serial_ok:
            if ble_ok:
                ble_ok = False
                ble_name = ""
                broadcast("ble drop")
            await asyncio.sleep(1)
            wait = 1.0
            continue
        addr = load_addr()
        bluez_drop(addr)
        await asyncio.sleep(0.4)
        target = await pick()
        if not target:
            await asyncio.sleep(wait)
            wait = min(wait * 1.6, 12.0)
            continue
        try:
            print("ble try", target, flush=True)
            async with BleakClient(target, timeout=10.0) as client:
                await client.start_notify(NUS_TX, on_rx)
                save_addr(client.address)
                while not ble_write_q.empty():
                    try:
                        ble_write_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                ble_ok = True
                ble_name = client.address
                try:
                    ble_name = next(
                        (n for n in (getattr(client, "name", None),) if n),
                        ble_name,
                    )
                except Exception:
                    pass
                if not str(ble_name).startswith("gibbon"):
                    ble_name = "gibbon-" + client.address.replace(":", "")[-4:].lower()
                broadcast("ble up " + ble_name)
                wait = 1.0
                while client.is_connected and not serial_ok and not stopping:
                    try:
                        payload = await asyncio.wait_for(ble_write_q.get(), timeout=0.4)
                        await client.write_gatt_char(NUS_RX, payload, response=False)
                    except asyncio.TimeoutError:
                        pass
                try:
                    await client.stop_notify(NUS_TX)
                except Exception:
                    pass
        except Exception as e:
            print("ble", e, flush=True)
            await asyncio.sleep(wait)
            wait = min(wait * 1.6, 12.0)
        if ble_ok:
            ble_ok = False
            ble_name = ""
            broadcast("ble drop")
        if not stopping:
            await asyncio.sleep(0.6)
    bluez_drop(load_addr())


def ble_thread():
    global ble_loop
    ble_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ble_loop)
    ble_loop.run_until_complete(ble_task())


SKIP_TX = frozenset({"info", "state", "gpio", "bench", "batt", "who", "lim", "guard"})


def skip_tx(line):
    if not line or line in SKIP_TX:
        return True
    if line.startswith("stream ") or line.startswith("get"):
        return True
    return False


def db_loop():
    from .models import Command, Note, Sample, Trial

    while True:
        kind, line = dbq.get()
        t_ms = int(time.time() * 1000)
        try:
            if kind == "tx":
                if skip_tx(line):
                    continue
                trial = Trial.objects.filter(ended__isnull=True).first()
                if not trial:
                    trial = Trial.objects.create(
                        title="sitting " + time.strftime("%Y-%m-%d %H:%M"),
                        via=via(),
                    )
                Command.objects.create(trial=trial, t_ms=t_ms, text=line[:240], source="ui")
                continue
            miss = MISS_RE.match(line)
            if miss:
                continue
            g = parse_get(line)
            if g:
                trial = Trial.objects.filter(ended__isnull=True).first()
                if not trial:
                    trial = Trial.objects.create(
                        title="sitting " + time.strftime("%Y-%m-%d %H:%M"),
                        via=via(),
                    )
                Sample.objects.create(
                    trial=trial, t_ms=t_ms, servo=g["id"],
                    pos=g["pos"], speed=g["speed"], load=g["load"],
                    v=g["v"], temp=g["temp"], cur=g["cur"],
                    moving=g["moving"], tq=g["tq"],
                    un=g["un"], turns=g["turns"], mode=g["mode"],
                )
                continue
            trial = Trial.objects.filter(ended__isnull=True).first()
            if line.startswith("ESTOP") or line.startswith("guard ") or line.startswith("lim id="):
                if not trial:
                    trial = Trial.objects.create(
                        title="sitting " + time.strftime("%Y-%m-%d %H:%M"),
                        via=via(),
                    )
                Note.objects.create(trial=trial, t_ms=t_ms, text=line[:500])
                continue
            bp = BATT_RE.search(line)
            if bp and trial:
                if trial.pack_v is None:
                    trial.pack_v = float(bp.group(1))
                    trial.via = via()
                    trial.save(update_fields=["pack_v", "via"])
        except Exception as e:
            print("db", e, flush=True)


def _atexit():
    global stopping
    stopping = True
    bluez_drop(load_addr())


def start():
    global _started
    if _started:
        return
    _started = True
    atexit.register(_atexit)
    open_tty()
    threading.Thread(target=serial_loop, daemon=True).start()
    threading.Thread(target=ble_thread, daemon=True).start()
    threading.Thread(target=db_loop, daemon=True).start()
    print("bridge serial", TTY, "ok" if serial_ok else "WAIT", flush=True)
    print("bridge ble   ", load_addr(), "direct, scan fallback", flush=True)
