const NUS_SVC = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e";
const NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";
const USB_VID = 0x1a86, USB_PID = 0x55d3;
const JOINTS = [
  { name: "base", id: 11 },
  { name: "shoulder a", id: 12 },
  { name: "shoulder b", id: 13 },
  { name: "elbow", id: 14 },
  { name: "wrist", id: 15 },
  { name: "twist", id: 16 },
  { name: "hand", id: 17 },
];
const GET_RE = /get id=(\d+) pos=(-?\d+) speed=(-?\d+) load=(-?\d+) v_raw=(\d+) v=([\d.]+) temp=(\d+) cur=(-?\d+) moving=(\d+) tq=(\S+) un=(-?\d+) turns=(-?\d+) st=(-?\d+) mode=(\S+)/;
const MISS_RE = /get id=(\d+) miss/;
const WHO_RE = /who id=(\d+) name=\S+ fw=(\S+) mode=(-?\d+) kp=(-?\d+) kd=(-?\d+) ki=(-?\d+) amin=(-?\d+) amax=(-?\d+) goal=(-?\d+) tqlim=(-?\d+) motor=(\S+)/;
const HB_RE = /^(?:t=)?(\d+)\s+heap=(\d+)\s+bus_rx=(\d+)\s+ble=(\d+)\s+stream=(\d+)/;
const $ = id => document.getElementById(id);
const dash = x => (x == null || x === "") ? "—" : x;

let transport = null, alive = false, wantBle = false, wantHost = true, hostEs = null;
let motorOpen = 0, paintQ = 0;
const rows = {};
const chip = {};
const batt = { pack: null, miss: true, est: null, cells: null, n: null, lvl: null };

function log(s) {
  const el = $("log");
  el.textContent += s + "\n";
  if (el.textContent.length > 8000) {
    const t = el.textContent.slice(-5000);
    const n = t.indexOf("\n");
    el.textContent = n >= 0 ? t.slice(n + 1) : t;
  }
  el.scrollTop = el.scrollHeight;
}

function fail(e) {
  const m = String(e && e.message ? e.message : e);
  if (/No port selected|cancelled/i.test(m)) return;
  log(m);
}

function foundN() {
  return JOINTS.filter(j => rows[j.id] && !rows[j.id].miss).length;
}

function modeOf(r) {
  if (!r || r.mode == null || r.mode === "?") return "—";
  if (r.mode === "0" || r.mode === 0) return "servo";
  if (r.mode === "1" || r.mode === 1) return "wheel";
  return String(r.mode);
}

function blk(title, pairs) {
  return `<div class="blk"><h2>${title}</h2>${pairs.map(([k, val, cls]) =>
    `<div><b>${k}</b><i class="${cls || ""}">${dash(val)}</i></div>`).join("")}</div>`;
}

function paint() {
  paintQ = 0;
  const n = foundN();
  const bleOn = chip.via === "ble" || chip.bleConn === 1 || chip.bleConn === "1";
  $("meta").innerHTML =
    blk("chip", [
      ["soc", chip.soc || "ESP32"],
      ["rev", chip.rev],
      ["cores", chip.cores],
      ["flash", chip.flash ? chip.flash + "KB" : null],
      ["heap", chip.heap],
      ["uptime", chip.uptime != null ? Math.floor(chip.uptime / 1000) + "s" : null],
      ["reset", chip.reset],
      ["tasks", chip.tasks],
      ["idf", chip.idf],
      ["mac", chip.mac],
    ]) +
    blk("link", [
      ["usb", chip.via === "usb" ? "in" : (alive ? "out" : null), chip.via === "usb" ? "ok" : ""],
      ["ble", bleOn ? "link " + (chip.ble || "") : (chip.ble ? "ready " + chip.ble : null), bleOn ? "ok" : ""],
      ["stream", chip.stream != null ? chip.stream + "ms" : null],
      ["dump", chip.dump],
      ["bus_rx", chip.bus_rx],
      ["bus", chip.bus],
      ["gpio", chip.gpio],
      ["via", chip.via],
    ]) +
    blk("pack", [
      ["pack", batt.miss ? "off" : (batt.pack != null ? batt.pack + "V" : null), batt.miss ? "bad" : "ok"],
      ["est", batt.est != null ? batt.est + "%" : null],
      ["cells", batt.cells],
      ["servos", batt.n],
      ["level", batt.lvl],
      ["found", alive ? n + "/7" : null, n === 7 ? "ok" : "bad"],
      ["guess", chip.guess],
    ]);
  $("hint").innerHTML = !alive
    ? "waiting for chip."
    : batt.miss && n === 0
      ? `<span class="next">no motors — plug XT60. usb only powers the chip.</span>`
      : n === 7 ? `<span class="ok">bus live — present @56 every ${dash(chip.stream)}ms.</span>`
        : n === 0 ? `<span class="next">pack on, motors still silent.</span>`
          : `<span class="next">${n}/7 on the bus — others miss.</span>`;
  $("armBody").innerHTML = JOINTS.map(j => {
    const r = rows[j.id];
    const miss = !r || r.miss;
    const hot = r && Math.abs(r.load) > 400;
    const on = motorOpen === j.id ? " on" : "";
    const cells = miss
      ? [j.id, "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"]
      : [j.id, r.pos, r.un, r.turns, r.speed, r.load, r.v, r.temp, r.cur, r.moving, r.tq, modeOf(r), r.st, r.fw, r.motor];
    const tds = cells.map((c, i) => `<td${i === 5 && hot ? ' class="hot"' : ""}>${dash(c)}</td>`).join("");
    return `<tr class="${miss ? "no" : "go"}${on}" data-id="${j.id}"><td>${j.name}</td>${tds}</tr>`;
  }).join("");
  const j = JOINTS.find(x => x.id === motorOpen);
  const r = j && rows[j.id];
  const box = $("acts");
  if (!j || !alive) { box.innerHTML = ""; return; }
  if (!r || r.miss) { box.innerHTML = `<span class="bad">${j.name} id ${j.id} — no present @56</span>`; return; }
  let extra = "";
  if (j.id === 17) extra = `<button data-cmd="hand open">open</button><button data-cmd="hand close">close</button>`;
  else if (j.id === 16) extra = `<button data-cmd="twist speed 0">stop</button>`;
  else extra = `<button data-act="nudge" data-dir="-1">−</button><button data-act="nudge" data-dir="1">+</button>`;
  box.innerHTML = `<span class="ok">${j.name} id ${j.id}</span>${extra}<button data-act="free">free</button><button data-act="hold">hold</button>` +
    (r.kp != null ? `<span>kp ${dash(r.kp)} kd ${dash(r.kd)} ki ${dash(r.ki)} amin ${dash(r.amin)} amax ${dash(r.amax)} goal ${dash(r.goal)} tqlim ${dash(r.tqlim)}</span>` : "") +
    (r.raw ? `<span class="raw">${r.raw}</span>` : "");
}

function requestPaint() {
  if (!paintQ) paintQ = requestAnimationFrame(paint);
}

function setLink(ok) {
  alive = ok;
  $("link").textContent = ok ? "online" : "offline";
  $("link").className = ok ? "ok" : "bad";
  $("dot").className = "dot " + (ok ? "on" : "off");
  $("btnHost").hidden = ok;
  $("btnDisc").hidden = !ok;
  $("btnStop").hidden = !ok;
  $("btnBle").hidden = ok || !navigator.bluetooth;
  $("btnUsb").hidden = ok || !navigator.serial;
  paint();
}

function onLine(line) {
  if (!line || line === "bridge") return;
  if (line === "serial drop" || line === "ble drop") { requestPaint(); return; }
  if (line === "serial up" || line.startsWith("ble up")) {
    if (wantHost) connectHost().catch(fail);
    return;
  }
  const miss = MISS_RE.exec(line);
  if (miss) {
    const id = +miss[1];
    const was = rows[id];
    rows[id] = Object.assign({}, was, { miss: true });
    if (was && !was.miss) log(line);
    requestPaint();
    return;
  }
  if (line.startsWith("get LOAD")) { log(line); return; }
  const g = GET_RE.exec(line);
  if (g) {
    const id = +g[1];
    const next = {
      miss: false, pos: +g[2], speed: +g[3], load: +g[4], v_raw: +g[5], v: g[6], temp: +g[7],
      cur: +g[8], moving: +g[9], tq: g[10], un: +g[11], turns: +g[12], st: +g[13], mode: g[14], raw: line,
    };
    const prev = rows[id];
    const live = !prev || prev.miss || next.moving
      || Math.abs((prev.pos || 0) - next.pos) >= 2
      || prev.speed !== next.speed || prev.load !== next.load
      || prev.cur !== next.cur || prev.tq !== next.tq;
    rows[id] = Object.assign({}, prev, next);
    if (live) log(line);
    requestPaint();
    return;
  }
  const who = WHO_RE.exec(line);
  if (who) {
    const id = +who[1];
    rows[id] = Object.assign(rows[id] || {}, {
      fw: who[2], mode: who[3], kp: +who[4], kd: +who[5], ki: +who[6],
      amin: +who[7], amax: +who[8], goal: +who[9], tqlim: +who[10], motor: who[11],
    });
    requestPaint();
    return;
  }
  const hb = HB_RE.exec(line);
  if (hb && (line.startsWith("t=") || /^\d+\s+heap=/.test(line))) {
    chip.uptime = +hb[1]; chip.heap = hb[2]; chip.bus_rx = hb[3];
    chip.bleConn = hb[4]; chip.stream = hb[5];
    requestPaint();
    return;
  }
  if (/batt pack=miss/.test(line)) {
    batt.miss = true; batt.pack = batt.est = batt.cells = batt.n = batt.lvl = null;
    log(line);
    requestPaint(); return;
  }
  const bp = /batt pack=([\d.]+)V est=(\d+)% cells=(\d+) n=(\d+) (\S+)/.exec(line);
  if (bp) {
    batt.miss = false; batt.pack = bp[1]; batt.est = +bp[2]; batt.cells = +bp[3]; batt.n = +bp[4]; batt.lvl = bp[5];
    log(line);
    requestPaint(); return;
  }
  const bv = /batt id=(\d+) v=([\d.]+)/.exec(line);
  if (bv) { rows[+bv[1]] = Object.assign(rows[+bv[1]] || {}, { v: bv[2] }); requestPaint(); return; }
  const inf = /info chip=(\S+) cores=(\d+) rev=(\d+) flash=(\d+)KB reset=(\S+)/.exec(line);
  if (inf) {
    chip.soc = inf[1]; chip.cores = inf[2]; chip.rev = inf[3]; chip.flash = inf[4]; chip.reset = inf[5];
    requestPaint(); return;
  }
  const im = /info mac=([0-9a-f:]+) heap=(\d+) uptime_ms=(\d+)/i.exec(line);
  if (im) { chip.mac = im[1]; chip.heap = im[2]; chip.uptime = +im[3]; requestPaint(); return; }
  const ii = /info idf=(\S+) usb=\S+ bus=(\S+) ids=\S+ ble=(\S+)/.exec(line);
  if (ii) { chip.idf = ii[1]; chip.bus = ii[2]; chip.ble = ii[3]; requestPaint(); return; }
  const st = /state t=(\d+) heap=(\d+) reset=(\S+) tasks=(\d+) bus_rx=(\d+) dump=(\d+) stream=(\d+) ble=(\d+)/.exec(line);
  if (st) {
    chip.uptime = +st[1]; chip.heap = st[2]; chip.reset = st[3]; chip.tasks = st[4];
    chip.bus_rx = st[5]; chip.dump = st[6]; chip.stream = st[7]; chip.bleConn = st[8];
    requestPaint(); return;
  }
  const sm = /state mac=([0-9a-f:]+) name=(\S+)/i.exec(line);
  if (sm) { chip.mac = sm[1]; chip.ble = sm[2]; requestPaint(); return; }
  const bc = /chip=(\S+) cores=(\d+) rev=(\d+) flash=(\d+)KB/.exec(line);
  if (bc) { chip.soc = bc[1]; chip.cores = bc[2]; chip.rev = bc[3]; chip.flash = bc[4]; requestPaint(); return; }
  const mac = /^mac=([0-9a-f:]+)$/i.exec(line);
  if (mac) { chip.mac = mac[1]; requestPaint(); return; }
  const gs = /guess=(\S+)/.exec(line);
  if (gs) { chip.guess = gs[1]; requestPaint(); return; }
  const fd = /^found=(\d+)\/(\d+)/.exec(line);
  if (fd) { chip.found = fd[1]; requestPaint(); return; }
  const bleName = /ble=(gibbon-\S+)/.exec(line);
  if (bleName) { chip.ble = bleName[1]; requestPaint(); return; }
  const gp = /gpio 18\(rx\)=(\d+) 19\(tx\)=(\d+)/.exec(line);
  if (gp) { chip.gpio = "18=" + gp[1] + " 19=" + gp[2]; requestPaint(); return; }
  const sms = /stream ms=(\d+) dump=(\d+)/.exec(line);
  if (sms) { chip.stream = sms[1]; chip.dump = sms[2]; requestPaint(); return; }
  if (/^(get present|batt id=|batt READ|ping |bench v|bus=|stream |listen|reset=|tasks|who EEPROM|state bus)/.test(line)) return;
  log(line);
}

async function send(cmd, quiet) {
  if (!transport) return;
  if (!quiet) log(">> " + cmd);
  await transport.write(cmd + "\n");
}

function feedText(chunk, state) {
  state.buf += chunk;
  let i;
  while ((i = state.buf.indexOf("\n")) >= 0) {
    onLine(state.buf.slice(0, i).replace(/\r$/, "").trim());
    state.buf = state.buf.slice(i + 1);
  }
}

async function afterConnect(kind) {
  batt.miss = true; batt.pack = batt.est = batt.cells = batt.n = batt.lvl = null;
  Object.keys(rows).forEach(k => delete rows[k]);
  chip.via = kind;
  setLink(true);
  await send("info", true);
  await send("state", true);
  await send("gpio", true);
  await send("bench", true);
  await send("batt", true);
  await send("stream 180", true);
  await send("who", true);
}

async function disconnect() {
  wantHost = false;
  wantBle = false;
  try { if (transport && transport.write) await transport.write("stream 0\n"); } catch {}
  try { if (transport && transport.close) await transport.close(); } catch {}
  transport = null;
  setLink(false);
}

function ensureSse() {
  if (hostEs) return;
  hostEs = new EventSource("api/rx");
  hostEs.onmessage = ev => onLine(ev.data);
}

function hostTransport() {
  return {
    write: async s => {
      const r = await fetch("api/tx", { method: "POST", body: s });
      if (!r.ok) throw new Error(await r.text());
    },
    close: async () => {},
  };
}

async function connectHost() {
  wantHost = true;
  const st = await fetch("api/status").then(r => r.ok ? r.json() : null).catch(() => null);
  if (!st || !st.bridge) { log("run: .venv/bin/python -u serve.py"); return; }
  ensureSse();
  transport = hostTransport();
  if (!st.ok) { setLink(false); return; }
  if (alive && chip.via === st.via) return;
  await afterConnect(st.via || "host");
}

async function connectUSB() {
  if (!navigator.serial) return;
  const ports = await navigator.serial.getPorts();
  let port = ports.find(p => p.getInfo().usbVendorId === USB_VID) || ports[0];
  if (!port) port = await navigator.serial.requestPort({ filters: [{ usbVendorId: USB_VID, usbProductId: USB_PID }] });
  await port.open({ baudRate: 115200 });
  const writer = port.writable.getWriter();
  const reader = port.readable.getReader();
  const dec = new TextDecoder(), enc = new TextEncoder(), state = { buf: "" };
  transport = {
    write: async s => writer.write(enc.encode(s)),
    close: async () => { try { reader.cancel(); } catch {} try { writer.releaseLock(); } catch {} try { await port.close(); } catch {} },
  };
  (async () => {
    try {
      while (alive) {
        const { value, done } = await reader.read();
        if (done) break;
        feedText(dec.decode(value, { stream: true }), state);
      }
    } catch (e) { if (alive) { setLink(false); log(String(e)); } }
  })();
  await afterConnect("usb");
}

async function attachBLE(dev) {
  wantBle = true;
  const server = await dev.gatt.connect();
  const svc = await server.getPrimaryService(NUS_SVC);
  const rx = await svc.getCharacteristic(NUS_RX);
  const tx = await svc.getCharacteristic(NUS_TX);
  const dec = new TextDecoder(), state = { buf: "" };
  await tx.startNotifications();
  tx.addEventListener("characteristicvaluechanged", ev => feedText(dec.decode(ev.target.value), state));
  const enc = new TextEncoder();
  transport = {
    write: async s => {
      const u8 = enc.encode(s);
      for (let i = 0; i < u8.length; i += 120) {
        const sl = u8.slice(i, i + 120);
        if (rx.properties.writeWithoutResponse) await rx.writeValueWithoutResponse(sl);
        else await rx.writeValue(sl);
      }
    },
    close: async () => { try { await tx.stopNotifications(); } catch {} try { server.disconnect(); } catch {} },
  };
  if (!dev._gibbonHook) {
    dev._gibbonHook = true;
    dev.addEventListener("gattserverdisconnected", () => {
      if (!wantBle) return;
      transport = null; setLink(false);
      setTimeout(() => { if (wantBle && !alive) attachBLE(dev).catch(fail); }, 800);
    });
  }
  await afterConnect("ble");
}

async function connectBLE() {
  if (!navigator.bluetooth) return;
  const known = (await navigator.bluetooth.getDevices?.() || []).find(d => (d.name || "").startsWith("gibbon"));
  const dev = known || await navigator.bluetooth.requestDevice({ filters: [{ namePrefix: "gibbon" }], optionalServices: [NUS_SVC] });
  await attachBLE(dev);
}

$("armBody").onclick = e => {
  const row = e.target.closest("tr[data-id]");
  if (!row) return;
  const id = +row.getAttribute("data-id");
  motorOpen = motorOpen === id ? 0 : id;
  paint();
};
$("acts").onclick = e => {
  const b = e.target.closest("button");
  if (!b || !motorOpen) return;
  const cmd = b.getAttribute("data-cmd");
  if (cmd) { send(cmd); return; }
  const act = b.getAttribute("data-act");
  if (act === "nudge") send("id " + motorOpen + " nudge " + (b.getAttribute("data-dir") === "-1" ? -40 : 40));
  if (act === "free" || act === "hold") send("id " + motorOpen + " " + act);
};
$("btnHost").onclick = () => connectHost().catch(fail);
$("btnBle").onclick = () => connectBLE().catch(fail);
$("btnUsb").onclick = () => connectUSB().catch(fail);
$("btnDisc").onclick = disconnect;
$("btnStop").onclick = () => send("estop");
window.addEventListener("keydown", e => { if (e.key === "Escape" && transport) send("estop"); });

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js");
setLink(false);
connectHost().catch(() => {});
setInterval(async () => {
  if (!wantHost) return;
  const st = await fetch("api/status").then(r => r.ok ? r.json() : null).catch(() => null);
  if (!st) return;
  if (st.ok && !alive) connectHost().catch(() => {});
  if (!st.ok && alive) setLink(false);
}, 2000);
