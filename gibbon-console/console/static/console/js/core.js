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
const G = {
  alive: false, wantHost: true, hostEs: null, transport: null,
  motorOpen: 0, paintQ: 0, rows: {}, chip: {},
  batt: { pack: null, miss: true, est: null, cells: null, n: null, lvl: null },
  hooks: [],
};

function log(s) {
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

function foundN() {
  return JOINTS.filter(j => G.rows[j.id] && !G.rows[j.id].miss).length;
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

function setLink(ok) {
  G.alive = ok;
  $("link").textContent = ok ? "online" : "offline";
  $("link").className = ok ? "ok" : "bad";
  $("dot").className = "dot " + (ok ? "on" : "off");
  $("btnHost").hidden = ok;
  $("btnDisc").hidden = !ok;
  $("btnStop").hidden = !ok;
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
  const g = GET_RE.exec(line);
  if (g) {
    const id = +g[1];
    const next = {
      miss: false, pos: +g[2], speed: +g[3], load: +g[4], v_raw: +g[5], v: +g[6], temp: +g[7],
      cur: +g[8], moving: +g[9], tq: g[10], un: +g[11], turns: +g[12], st: +g[13], mode: g[14],
      raw: line, t: Date.now(), id,
    };
    const prev = G.rows[id];
    const live = !prev || prev.miss || next.moving
      || Math.abs((prev.pos || 0) - next.pos) >= 2
      || prev.speed !== next.speed || prev.load !== next.load
      || prev.cur !== next.cur || prev.tq !== next.tq;
    G.rows[id] = Object.assign({}, prev, next);
    if (live) log(line);
    G.hooks.forEach(fn => fn("sample", G.rows[id]));
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
    log(line); paintPack(); G.hooks.forEach(fn => fn("batt")); return;
  }
  const bp = /batt pack=([\d.]+)V est=(\d+)% cells=(\d+) n=(\d+) (\S+)/.exec(line);
  if (bp) {
    G.batt = { miss: false, pack: bp[1], est: +bp[2], cells: +bp[3], n: +bp[4], lvl: bp[5] };
    log(line); paintPack(); G.hooks.forEach(fn => fn("batt")); return;
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
  if (gp) { G.chip.gpio = "18=" + gp[1] + " 19=" + gp[2]; return; }
  const sms = /stream ms=(\d+) dump=(\d+)/.exec(line);
  if (sms) { G.chip.stream = sms[1]; G.chip.dump = sms[2]; return; }
  if (/^(get present|batt id=|batt READ|ping |bench v|bus=|stream |listen|reset=|tasks|who EEPROM|state bus|found=)/.test(line)) return;
  log(line);
}

function paintPack() {
  const el = $("packEl");
  if (!el) return;
  el.className = G.batt.miss ? "bad" : "ok";
  el.textContent = !G.alive ? "" : G.batt.miss ? "pack off" : ("pack " + G.batt.pack + "V");
}

async function afterConnect(kind) {
  G.batt = { pack: null, miss: true, est: null, cells: null, n: null, lvl: null };
  Object.keys(G.rows).forEach(k => delete G.rows[k]);
  G.chip.via = kind;
  setLink(true);
  paintPack();
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
  paintPack();
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
$("btnStop").onclick = () => send("estop");
window.addEventListener("keydown", e => { if (e.key === "Escape" && G.transport) send("estop"); });
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
window.JOINTS = JOINTS;
window.send = send;
window.log = log;
window.dash = dash;
window.foundN = foundN;
window.modeOf = modeOf;
window.$ = $;
