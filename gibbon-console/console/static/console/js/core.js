const CATALOG = window.CATALOG || { motors: [], joints: [] };
const MOTORS = CATALOG.motors;
const JOINTS = CATALOG.joints;
const GET_KV = /(\w+)=(\S+)/g;
const MISS_RE = /get id=(\d+) miss/;
const WHO_RE = /who id=(\d+) name=\S+ fw=(\S+) mode=(-?\d+) kp=(-?\d+) kd=(-?\d+) ki=(-?\d+) amin=(-?\d+) amax=(-?\d+) goal=(-?\d+) tqlim=(-?\d+) motor=(\S+)/;
const HB_RE = /^(?:t=)?(\d+)\s+heap=(\d+)\s+bus_rx=(\d+)\s+ble=(\d+)\s+stream=(\d+)/;
const $ = id => document.getElementById(id);
const dash = x => (x == null || x === "") ? "—" : x;
const G = {
  alive: false, wantHost: true, hostEs: null, transport: null,
  motorOpen: 0, paintQ: 0, rows: {}, chip: {},
  batt: { pack: null, miss: true, est: null, cells: null, n: null, lvl: null },
  link: { usb: false, ble: false },
  hooks: [], recent: [], lim: {}, guard: "", guardCfg: null,
  pair12: null,
  logFocus: null,
};

function footKeep(s) {
  const id = G.logFocus;
  if (id == null) return true;
  if (/^ESTOP|^>> |^guard |^raise blocked/.test(s)) return true;
  if (s.startsWith(id + "  ") || s.startsWith("get LOAD id=" + id)) return true;
  if (s.indexOf("id=" + id) >= 0 || /\bhand /.test(s) || s.startsWith("raw ")) return true;
  return false;
}

function paintFoot() {
  const el = $("footLog");
  if (!el) return;
  const lines = G.recent.filter(footKeep);
  const n = el.classList.contains("wide") ? 80 : 1;
  el.textContent = lines.slice(-n).join("\n") || "—";
  if (el.classList.contains("wide")) el.scrollTop = el.scrollHeight;
}

function log(s) {
  G.recent.push(s);
  if (G.recent.length > 200) G.recent.shift();
  paintFoot();
  const el = $("log");
  if (!el) return;
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

function parseGet(line) {
  if (!line.startsWith("get id=")) return null;
  const kv = {};
  line.replace(GET_KV, (_, k, v) => { kv[k] = v; return ""; });
  if (kv.pos == null) return null;
  const pos = +kv.pos;
  return {
    id: +kv.id, pos,
    speed: kv.speed != null ? +kv.speed : 0,
    load: kv.load != null ? +kv.load : 0,
    cur: kv.cur != null ? +kv.cur : 0,
    moving: kv.moving != null ? +kv.moving : 0,
    tq: kv.tq != null ? kv.tq : "0",
    un: kv.un != null ? +kv.un : null,
    turns: kv.turns != null ? +kv.turns : null,
    st: kv.st != null ? +kv.st : 0,
    mode: kv.mode != null ? kv.mode : "0",
    v: kv.v != null ? kv.v : null,
    temp: kv.temp != null ? +kv.temp : null,
    hex: kv.hex || null,
  };
}

function fmtDelta(prev, next) {
  const bits = [];
  if (!prev || prev.miss) {
    bits.push("pos " + next.pos);
  } else {
    if (Math.abs(next.pos - prev.pos) >= 2) bits.push("pos " + prev.pos + "→" + next.pos);
    if (next.un != null && prev.un != null && Math.abs(next.un - prev.un) >= 2) {
      bits.push("un " + prev.un + "→" + next.un);
    }
    if ((Math.abs(next.speed) >= 80 || Math.abs(prev.speed || 0) >= 80) && next.speed !== prev.speed) {
      bits.push("spd " + next.speed);
    }
    if (next.load !== prev.load && (Math.abs(next.load) >= 30 || Math.abs(prev.load) >= 30)) {
      bits.push("load " + next.load);
    }
    if (Math.abs((next.cur || 0) - (prev.cur || 0)) >= 5) bits.push("cur " + next.cur);
    if (next.tq !== prev.tq) bits.push("tq " + next.tq);
    if (next.moving !== prev.moving) bits.push("mv " + next.moving);
  }
  if (!bits.length) return null;
  return next.id + "  " + bits.join("  ");
}

function foundN() {
  return MOTORS.filter(m => G.rows[m.id] && !G.rows[m.id].miss).length;
}

function modeOf(r) {
  if (!r || r.mode == null || r.mode === "?") return "—";
  if (r.mode === "0" || r.mode === 0) return "servo";
  if (r.mode === "1" || r.mode === 1) return "wheel";
  return String(r.mode);
}

let sendQ = [];
let sendBusy = 0;

function raiseReady() {
  return String(G.chip.raise || "") === "1";
}

function send(cmd, quiet) {
  if (!G.transport) return Promise.resolve();
  if (/^raise\b/.test(cmd) && !raiseReady()) {
    log("raise blocked — chip missing info fw= raise=1. flash roarm-fw (stop Django first)");
    return Promise.resolve();
  }
  if (!quiet) log(">> " + cmd);
  if (cmd === "estop") sendQ = sendQ.filter(c => c === "estop");
  sendQ.push(cmd);
  return pumpSend();
}

async function pumpSend() {
  if (sendBusy) return;
  sendBusy = 1;
  try {
    while (sendQ.length && G.transport) {
      const cmd = sendQ.shift();
      await G.transport.write(cmd + "\n");
    }
  } finally {
    sendBusy = 0;
    if (sendQ.length) pumpSend();
  }
}

function boardId() {
  if (G.chip.board) return G.chip.board;
  const mac = G.chip.mac || "";
  const p = mac.toLowerCase().split(":");
  if (p.length >= 6) return "gibbon-" + p[4] + p[5];
  const ble = G.chip.ble || "";
  const m = /gibbon-[0-9a-f]+/i.exec(ble);
  return m ? m[0].toLowerCase() : "unknown";
}

function applyHostStatus(st) {
  if (!st) return;
  G.link.usb = st.usb != null ? !!st.usb : st.via === "usb";
  G.link.ble = st.ble_ok != null ? !!st.ble_ok : st.via === "ble";
  if (G.chip.bleConn && +G.chip.bleConn > 0) G.link.ble = true;
  paintLink();
}

function tqOn() {
  return MOTORS.filter(m => G.rows[m.id] && !G.rows[m.id].miss && G.rows[m.id].tq === "1").map(m => m.id);
}

/* 3S pack: 12.6 V = 4.2 V/cell full. Empty floor 10.0 V (9.9 is loaded LVC — keep a 0.1 V pad). Linear, not a BMS. */
const LIPO3S_FULL = 12.6;
const LIPO3S_EMPTY = 10.0;

function lipo3sPct(v) {
  if (v >= LIPO3S_FULL) return 100;
  if (v <= LIPO3S_EMPTY) return 0;
  return Math.round((v - LIPO3S_EMPTY) / (LIPO3S_FULL - LIPO3S_EMPTY) * 100);
}

function packVolts() {
  const vs = MOTORS.map(m => {
    const r = G.rows[m.id];
    return r && !r.miss && r.v != null ? +r.v : null;
  }).filter(v => v != null && v > 6).sort((a, b) => a - b);
  if (vs.length) return vs[vs.length >> 1];
  if (G.batt && !G.batt.miss && G.batt.pack != null) return +G.batt.pack;
  return null;
}

function paintStatus() {
  const pack = $("packEl");
  if (pack) {
    if (!G.alive) {
      pack.className = "mut";
      pack.textContent = "pack —";
      pack.title = "";
    } else if (G.batt.miss && packVolts() == null) {
      pack.className = "bad";
      pack.textContent = "pack off";
      pack.title = "servo VIN miss — XT60 pack off. USB does not power the motors";
    } else {
      const v = packVolts();
      const pct = (G.batt && !G.batt.miss && G.batt.est != null)
        ? G.batt.est
        : (v != null ? lipo3sPct(v) : null);
      const lvl = pct == null ? "miss"
        : (v != null && v < 10.5) ? "cutoff"
        : pct <= 20 ? "low"
        : pct <= 30 ? "recharge"
        : "ok";
      pack.className = lvl === "ok" ? "ok" : (lvl === "recharge" ? "warn" : "bad");
      pack.textContent = pct == null ? "pack —" : ("pack " + pct + "%");
      pack.title = v != null
        ? (v.toFixed(1) + " V  ·  3S linear from servo VIN, not a BMS. 12.6 V = 100%. 10.0 V = 0%. Chip refuses writes below 10.5 V.")
        : "";
    }
  }
  const board = $("boardEl");
  if (board) {
    const id = boardId();
    board.className = id === "unknown" ? "mut" : "ok";
    board.textContent = "board " + id;
  }
  const found = $("foundEl");
  if (found) {
    if (!G.alive) {
      found.className = "mut";
      found.textContent = "motors —";
    } else {
      const n = foundN();
      found.className = n === 7 ? "ok" : "bad";
      found.textContent = "motors " + n + "/7";
    }
  }
  const tq = $("tqEl");
  if (tq) {
    const on = tqOn();
    if (!G.alive || !on.length) {
      tq.className = "mut";
      tq.textContent = "tq off";
    } else {
      tq.className = "bad";
      tq.textContent = "tq " + on.join(",");
    }
  }
  const fw = $("fwEl");
  if (fw) {
    if (!G.alive) {
      fw.className = "mut";
      fw.textContent = "fw —";
      fw.title = "";
    } else if (raiseReady()) {
      fw.className = "ok";
      fw.textContent = "fw " + (G.chip.fw || "ok");
      fw.title = "SyncWrite raise on chip. compact stream + move guard.";
    } else {
      fw.className = "bad";
      fw.textContent = "fw old";
      fw.title = "chip did not print info fw= raise=1 — 12+13 jog blocked. flash roarm-fw, stop Django first.";
    }
  }
}

function paintLink() {
  const usb = $("usbEl");
  const ble = $("bleEl");
  if (usb) {
    usb.className = "pill " + (G.link.usb ? "ok" : "mut");
    usb.title = G.link.usb ? "USB connected" : "USB off";
  }
  if (ble) {
    ble.className = "pill " + (G.link.ble ? "ok" : "mut");
    ble.title = G.link.ble ? "Bluetooth connected" : "Bluetooth off";
  }
  const wired = G.link.usb || G.link.ble;
  $("dot").className = "dot " + (wired ? "on" : "off");
}

function setLink(ok) {
  G.alive = ok;
  paintLink();
  $("btnHost").hidden = ok;
  $("btnDisc").hidden = !ok;
  $("btnStop").disabled = !ok;
  if ($("btnLockPose")) $("btnLockPose").disabled = !ok;
  paintStatus();
  G.hooks.forEach(fn => fn("link"));
}

function onLine(line) {
  if (!line || line === "bridge") return;
  if (line === "serial drop") { G.link.usb = false; paintLink(); G.hooks.forEach(fn => fn("link")); return; }
  if (line === "ble drop") { G.link.ble = false; paintLink(); G.hooks.forEach(fn => fn("link")); return; }
  if (line === "serial up") {
    G.link.usb = true; paintLink();
    if (G.wantHost) connectHost().catch(fail);
    return;
  }
  if (line.startsWith("ble up")) {
    G.link.ble = true; paintLink();
    if (G.wantHost) connectHost().catch(fail);
    return;
  }
  const miss = MISS_RE.exec(line);
  if (miss) {
    const id = +miss[1];
    const was = G.rows[id];
    G.rows[id] = Object.assign({}, was, { miss: true });
    if (was && !was.miss) log(line);
    G.hooks.forEach(fn => fn("sample", G.rows[id]));
    return;
  }
  if (line.startsWith("get LOAD")) { log(line); return; }
  const parsed = parseGet(line);
  if (parsed) {
    const id = parsed.id;
    const prev = G.rows[id];
    const next = Object.assign({}, prev, parsed, { miss: false, raw: line, t: Date.now(), id });
    if (parsed.v == null && prev && prev.v != null) next.v = prev.v;
    if (parsed.temp == null && prev && prev.temp != null) next.temp = prev.temp;
    if (parsed.un == null && prev && prev.un != null) next.un = prev.un;
    if (parsed.un == null && next.un == null) next.un = parsed.pos;
    if (parsed.turns == null && prev && prev.turns != null) next.turns = prev.turns;
    if (parsed.turns == null && next.turns == null) next.turns = 0;
    const note = fmtDelta(prev, next);
    G.rows[id] = next;
    if (note) log(note);
    G.hooks.forEach(fn => fn("sample", next));
    return;
  }
  if (line.startsWith("lim id=")) {
    const kv = {};
    line.replace(GET_KV, (_, k, v) => { kv[k] = v; return ""; });
    const id = +kv.id;
    const num = v => (v == null || v === "-") ? null : +v;
    const rec = { id, amin: num(kv.min), amax: num(kv.max), home: num(kv.home), nv: kv.nv || "" };
    G.lim[id] = rec;
    log(line);
    G.hooks.forEach(fn => fn("lim", rec));
    return;
  }
  if (line.startsWith("lim ")) { log(line); return; }
  if (line.startsWith("guard dload=")) {
    const kv = {};
    line.replace(GET_KV, (_, k, v) => { kv[k] = v; return ""; });
    G.guardCfg = {
      dload: +kv.dload || 0,
      hand: kv.hand != null ? +kv.hand : 0,
      load: +kv.load || 0,
      cur: +kv.cur || 0,
    };
    log(line);
    G.hooks.forEach(fn => fn("guardcfg", G.guardCfg));
    return;
  }
  if (line.startsWith("ESTOP") || line.startsWith("guard on") || line.startsWith("guard idle")) {
    G.guard = line;
    log(line);
    G.hooks.forEach(fn => fn("guard", line));
    return;
  }
  const who = WHO_RE.exec(line);
  if (who) {
    const id = +who[1];
    G.rows[id] = Object.assign(G.rows[id] || {}, {
      fw: who[2], mode: who[3], kp: +who[4], kd: +who[5], ki: +who[6],
      amin: +who[7], amax: +who[8], goal: +who[9], tqlim: +who[10], motor: who[11],
    });
    G.hooks.forEach(fn => fn("who"));
    return;
  }
  const hb = HB_RE.exec(line);
  if (hb && (line.startsWith("t=") || /^\d+\s+heap=/.test(line))) {
    G.chip.uptime = +hb[1]; G.chip.heap = hb[2]; G.chip.bus_rx = hb[3];
    G.chip.bleConn = hb[4]; G.chip.stream = hb[5];
    if (+hb[4] > 0) G.link.ble = true;
    paintLink();
    G.hooks.forEach(fn => fn("chip"));
    return;
  }
  if (/batt pack=miss/.test(line)) {
    G.batt = { pack: null, miss: true, est: null, cells: null, n: null, lvl: null };
    log(line); paintStatus(); G.hooks.forEach(fn => fn("batt")); return;
  }
  const bp = /batt pack=([\d.]+)V est=(\d+)% cells=(\d+) n=(\d+) (\S+)/.exec(line);
  if (bp) {
    G.batt = { miss: false, pack: bp[1], est: +bp[2], cells: +bp[3], n: +bp[4], lvl: bp[5] };
    log(line); paintStatus(); G.hooks.forEach(fn => fn("batt")); return;
  }
  const inf = /info chip=(\S+) cores=(\d+) rev=(\d+) flash=(\d+)KB reset=(\S+)/.exec(line);
  if (inf) {
    G.chip.soc = inf[1]; G.chip.cores = inf[2]; G.chip.rev = inf[3]; G.chip.flash = inf[4]; G.chip.reset = inf[5];
    G.hooks.forEach(fn => fn("chip")); return;
  }
  const ifw = /^info fw=/.test(line);
  if (ifw) {
    const kv = {};
    line.replace(GET_KV, (_, k, v) => { kv[k] = v; return ""; });
    if (kv.fw) G.chip.fw = kv.fw;
    G.chip.raise = kv.raise || "0";
    paintStatus();
    G.hooks.forEach(fn => fn("chip"));
    return;
  }
  const ib = /^info board=(\S+)/.exec(line);
  if (ib) {
    G.chip.board = ib[1];
    paintStatus();
    G.hooks.forEach(fn => fn("chip"));
    return;
  }
  const im = /info mac=([0-9a-f:]+) heap=(\d+) uptime_ms=(\d+)/i.exec(line);
  if (im) { G.chip.mac = im[1]; G.chip.heap = im[2]; G.chip.uptime = +im[3]; G.hooks.forEach(fn => fn("chip")); return; }
  const ii = /info idf=(\S+) usb=\S+ bus=(\S+) ids=\S+ ble=(\S+)/.exec(line);
  if (ii) { G.chip.idf = ii[1]; G.chip.bus = ii[2]; G.chip.ble = ii[3]; G.hooks.forEach(fn => fn("chip")); return; }
  const st = /state t=(\d+) heap=(\d+) reset=(\S+) tasks=(\d+) bus_rx=(\d+) dump=(\d+) stream=(\d+) ble=(\d+)/.exec(line);
  if (st) {
    G.chip.uptime = +st[1]; G.chip.heap = st[2]; G.chip.reset = st[3]; G.chip.tasks = st[4];
    G.chip.bus_rx = st[5]; G.chip.dump = st[6]; G.chip.stream = st[7]; G.chip.bleConn = st[8];
    if (+st[8] > 0) G.link.ble = true;
    paintLink();
    G.hooks.forEach(fn => fn("chip")); return;
  }
  const sm = /state mac=([0-9a-f:]+) name=(\S+)/i.exec(line);
  if (sm) { G.chip.mac = sm[1]; G.chip.ble = sm[2]; G.hooks.forEach(fn => fn("chip")); return; }
  const bc = /chip=(\S+) cores=(\d+) rev=(\d+) flash=(\d+)KB/.exec(line);
  if (bc) { G.chip.soc = bc[1]; G.chip.cores = bc[2]; G.chip.rev = bc[3]; G.chip.flash = bc[4]; G.hooks.forEach(fn => fn("chip")); return; }
  const mac = /^mac=([0-9a-f:]+)$/i.exec(line);
  if (mac) { G.chip.mac = mac[1]; G.hooks.forEach(fn => fn("chip")); return; }
  const gs = /guess=(\S+)/.exec(line);
  if (gs) { G.chip.guess = gs[1]; G.hooks.forEach(fn => fn("chip")); return; }
  const bleName = /ble=(gibbon-\S+)/.exec(line);
  if (bleName) { G.chip.ble = bleName[1]; G.hooks.forEach(fn => fn("chip")); return; }
  const gp = /gpio 18\(rx\)=(\d+) 19\(tx\)=(\d+)/.exec(line);
  if (gp) { G.chip.gpio = "18=" + gp[1] + " 19=" + gp[2]; G.hooks.forEach(fn => fn("chip")); return; }
  const sms = /stream ms=(\d+) dump=(\d+)/.exec(line);
  if (sms) { G.chip.stream = sms[1]; G.chip.dump = sms[2]; G.hooks.forEach(fn => fn("chip")); return; }
  if (line.startsWith("raw ping") || line.startsWith("raw read")) {
    log(line);
    G.hooks.forEach(fn => fn("lab", line));
    return;
  }
  if (/^(get present|batt id=|batt READ|ping |bench v|bus=|stream |listen|reset=|tasks|who EEPROM|state bus|found=)/.test(line)) return;
  log(line);
}

async function afterConnect(kind) {
  G.batt = { pack: null, miss: true, est: null, cells: null, n: null, lvl: null };
  Object.keys(G.rows).forEach(k => delete G.rows[k]);
  G.lim = {};
  G.guard = "";
  G.guardCfg = null;
  G.chip.fw = null;
  G.chip.raise = null;
  G.chip.via = kind;
  setLink(true);
  await send("info", true);
  await send("get", true);
  await send("stream 180", true);
  await send("lim", true);
  await send("guard", true);
  await send("batt", true);
}

async function disconnect() {
  G.wantHost = false;
  try { if (G.transport && G.transport.write) await G.transport.write("stream 0\n"); } catch {}
  G.transport = null;
  setLink(false);
}

function ensureSse() {
  if (G.hostEs && G.hostEs.readyState !== 2) return;
  G.hostEs = new EventSource("/api/rx");
  G.hostEs.onmessage = ev => onLine(ev.data);
  G.hostEs.onerror = () => { try { G.hostEs.close(); } catch {} G.hostEs = null; };
}

function hostTransport() {
  return {
    write: async s => {
      const r = await fetch("/api/tx", { method: "POST", body: s });
      if (!r.ok) throw new Error(await r.text());
    },
    close: async () => {},
  };
}

async function connectHost() {
  G.wantHost = true;
  const st = await fetch("/api/status").then(r => r.ok ? r.json() : null).catch(() => null);
  if (!st || !st.bridge) { log("host not answering — hard-refresh, or restart serve.py"); return; }
  applyHostStatus(st);
  const chip = $("trialChip");
  if (chip) chip.textContent = st.trial ? ("rec " + (st.trial.title || ("trial " + st.trial.id))) : "not recording";
  ensureSse();
  G.transport = hostTransport();
  if (!st.ok) { setLink(false); return; }
  if (G.alive && G.chip.via === st.via) return;
  await afterConnect(st.via || "host");
}

$("btnHost").onclick = () => connectHost().catch(fail);
$("btnDisc").onclick = disconnect;
$("btnStop").onclick = () => { if (G.transport) send("estop"); };
if ($("btnLockPose")) $("btnLockPose").onclick = () => { if (G.transport) lockPose(); };
window.addEventListener("keydown", e => { if (e.key === "Escape" && G.transport) send("estop"); });
G.hooks.push(() => paintStatus());
const footBtn = $("footToggle");
if (footBtn) {
  footBtn.title = "bus log — click to expand";
  footBtn.onclick = () => {
    $("footLog").classList.toggle("wide");
    paintFoot();
  };
}
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.getRegistrations().then(rs => rs.forEach(r => r.unregister()));
}
setLink(false);
connectHost().catch(() => {});
setInterval(async () => {
  if (!G.wantHost) return;
  const st = await fetch("/api/status").then(r => r.ok ? r.json() : null).catch(() => null);
  if (!st) return;
  applyHostStatus(st);
  const chip = $("trialChip");
  if (chip) chip.textContent = st.trial ? ("rec " + (st.trial.title || ("trial " + st.trial.id))) : "not recording";
  if (st.ok && !G.alive) connectHost().catch(() => {});
  if (!st.ok && G.alive) setLink(false);
  if (G.wantHost) ensureSse();
}, 2000);
async function fetchLimits() {
  const d = await fetch("/api/limits").then(r => r.json());
  const m = {};
  (d.limits || []).forEach(x => { m[x.servo] = x; });
  return m;
}

async function saveLimit(id, key, pos, opts) {
  if (pos == null || pos === "") {
    throw new Error("no live pos — connect, pack on, wait for a get line");
  }
  const r = await fetch("/api/limits", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ servo: id, [key]: +pos }),
  });
  if (!r.ok) throw new Error(await r.text());
  const chip = { home: "home", amin: "min", amax: "max" }[key];
  const skipChip = opts && opts.chip === false;
  if (chip && G.transport && !skipChip) {
    await send("lim " + id + " " + chip + " " + (+pos));
  }
  return r.json();
}

function parsePairNote(note) {
  const m = /(?:^|\s)pair=(-?1)(?:\s|$)/.exec(note || "");
  return m ? +m[1] : null;
}

function pairSign() {
  if (G.pair12 != null) return G.pair12;
  try {
    const s = localStorage.getItem("gibbon.pair.12");
    if (s === "1" || s === "-1") return +s;
  } catch (e) {}
  return null;
}

function syncPairFromLimits(limits) {
  const L = limits && limits[12];
  const n = parsePairNote(L && L.note);
  if (n != null) G.pair12 = n;
}

async function savePairSign(sign) {
  G.pair12 = sign;
  try { localStorage.setItem("gibbon.pair.12", String(sign)); } catch (e) {}
  const L = (await fetchLimits())[12] || {};
  const rest = String(L.note || "").replace(/\s*pair=-?1\b/g, "").trim();
  const note = (rest + " pair=" + sign).trim();
  await fetch("/api/limits", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ servo: 12, note }),
  });
}

function cmdRaiseTo(g12, g13) {
  return "raise to " + Math.round(g12) + " " + Math.round(g13);
}

function sendRaiseTq(on) {
  if (raiseReady()) return send(on ? "raise hold" : "raise free");
  send(on ? "id 12 hold" : "id 12 free");
  return send(on ? "id 13 hold" : "id 13 free");
}

function lockPose() {
  MOTORS.forEach(m => {
    if (m.id === 12 || m.id === 13) return;
    send("id " + m.id + " hold");
  });
  sendRaiseTq(true);
}

window.G = G;
window.MOTORS = MOTORS;
window.JOINTS = JOINTS;
window.send = send;
window.log = log;
window.dash = dash;
window.boardId = boardId;
window.foundN = foundN;
window.modeOf = modeOf;
window.fetchLimits = fetchLimits;
window.saveLimit = saveLimit;
window.pairSign = pairSign;
window.syncPairFromLimits = syncPairFromLimits;
window.savePairSign = savePairSign;
window.raiseReady = raiseReady;
window.cmdRaiseTo = cmdRaiseTo;
window.sendRaiseTq = sendRaiseTq;
window.$ = $;
