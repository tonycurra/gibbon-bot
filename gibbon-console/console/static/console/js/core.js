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
  hooks: [], recent: [],
};

function paintFoot() {
  const el = $("footLog");
  if (!el) return;
  const n = el.classList.contains("wide") ? 16 : 5;
  el.textContent = G.recent.slice(-n).join("\n") || "—";
}

function log(s) {
  G.recent.push(s);
  if (G.recent.length > 80) G.recent.shift();
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
    un: kv.un != null ? +kv.un : pos,
    turns: kv.turns != null ? +kv.turns : 0,
    st: kv.st != null ? +kv.st : 0,
    mode: kv.mode != null ? kv.mode : "0",
    v: kv.v != null ? kv.v : null,
    temp: kv.temp != null ? +kv.temp : null,
  };
}

function fmtDelta(prev, next) {
  const bits = [];
  if (!prev || prev.miss) {
    bits.push("pos " + next.pos);
  } else {
    if (Math.abs(next.pos - prev.pos) >= 2) bits.push("pos " + prev.pos + "→" + next.pos);
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

async function send(cmd, quiet) {
  if (!G.transport) return;
  if (!quiet) log(">> " + cmd);
  await G.transport.write(cmd + "\n");
}

function viaLabel() {
  if (!G.alive) return "offline";
  if (G.chip.via === "usb") return "usb";
  if (G.chip.via === "ble") return "ble";
  return G.chip.via || "online";
}

function tqOn() {
  return MOTORS.filter(m => G.rows[m.id] && !G.rows[m.id].miss && G.rows[m.id].tq === "1").map(m => m.id);
}

function paintStatus() {
  const pack = $("packEl");
  if (pack) {
    if (!G.alive) {
      pack.className = "mut";
      pack.textContent = "pack —";
    } else if (G.batt.miss) {
      pack.className = "bad";
      pack.textContent = "pack off";
    } else {
      pack.className = "ok";
      pack.textContent = "pack " + G.batt.pack + "V";
    }
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
}

function setLink(ok) {
  G.alive = ok;
  $("link").textContent = viaLabel();
  $("link").className = ok ? "ok" : "bad";
  $("dot").className = "dot " + (ok ? "on" : "off");
  $("btnHost").hidden = ok;
  $("btnDisc").hidden = !ok;
  $("btnStop").disabled = !ok;
  paintStatus();
  G.hooks.forEach(fn => fn("link"));
}

function onLine(line) {
  if (!line || line === "bridge") return;
  if (line === "serial drop" || line === "ble drop") { G.hooks.forEach(fn => fn("link")); return; }
  if (line === "serial up" || line.startsWith("ble up")) {
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
    const note = fmtDelta(prev, next);
    G.rows[id] = next;
    if (note) log(note);
    G.hooks.forEach(fn => fn("sample", next));
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
  const im = /info mac=([0-9a-f:]+) heap=(\d+) uptime_ms=(\d+)/i.exec(line);
  if (im) { G.chip.mac = im[1]; G.chip.heap = im[2]; G.chip.uptime = +im[3]; G.hooks.forEach(fn => fn("chip")); return; }
  const ii = /info idf=(\S+) usb=\S+ bus=(\S+) ids=\S+ ble=(\S+)/.exec(line);
  if (ii) { G.chip.idf = ii[1]; G.chip.bus = ii[2]; G.chip.ble = ii[3]; G.hooks.forEach(fn => fn("chip")); return; }
  const st = /state t=(\d+) heap=(\d+) reset=(\S+) tasks=(\d+) bus_rx=(\d+) dump=(\d+) stream=(\d+) ble=(\d+)/.exec(line);
  if (st) {
    G.chip.uptime = +st[1]; G.chip.heap = st[2]; G.chip.reset = st[3]; G.chip.tasks = st[4];
    G.chip.bus_rx = st[5]; G.chip.dump = st[6]; G.chip.stream = st[7]; G.chip.bleConn = st[8];
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
  if (/^(get present|batt id=|batt READ|ping |bench v|bus=|stream |listen|reset=|tasks|who EEPROM|state bus|found=)/.test(line)) return;
  log(line);
}

async function afterConnect(kind) {
  G.batt = { pack: null, miss: true, est: null, cells: null, n: null, lvl: null };
  Object.keys(G.rows).forEach(k => delete G.rows[k]);
  G.chip.via = kind;
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
  if (!st || !st.bridge) { log("run: .venv/bin/python manage.py runserver 0.0.0.0:8080"); return; }
  const chip = $("trialChip");
  if (chip) chip.textContent = st.trial ? ("trial " + (st.trial.title || st.trial.id)) : "";
  ensureSse();
  G.transport = hostTransport();
  if (!st.ok) { setLink(false); return; }
  if (G.alive && G.chip.via === st.via) return;
  await afterConnect(st.via || "host");
}

$("btnHost").onclick = () => connectHost().catch(fail);
$("btnDisc").onclick = disconnect;
$("btnStop").onclick = () => { if (G.transport) send("estop"); };
window.addEventListener("keydown", e => { if (e.key === "Escape" && G.transport) send("estop"); });
G.hooks.push(() => paintStatus());
const footBtn = $("footToggle");
if (footBtn) {
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
  const chip = $("trialChip");
  if (chip) chip.textContent = st.trial ? ("trial " + (st.trial.title || st.trial.id)) : "";
  if (st.ok && !G.alive) connectHost().catch(() => {});
  if (!st.ok && G.alive) setLink(false);
  if (G.wantHost) ensureSse();
}, 2000);
window.G = G;
window.MOTORS = MOTORS;
window.JOINTS = JOINTS;
window.send = send;
window.log = log;
window.dash = dash;
window.foundN = foundN;
window.modeOf = modeOf;
window.$ = $;
