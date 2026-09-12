const HAND = (window.CATALOG && window.CATALOG.hand) || { open: 689, close: 2008, to_max: 2075 };
const ID = 17;
const LO = HAND.open;
const HI = HAND.to_max;
const LOAD_SCALE = 1350;
const VARS = [
  { key: "pos", label: "position  encoder  counts", color: "#00f0c8", wide: 1 },
  { key: "load", label: "effort  PID load  counts", color: "#ff2bd6" },
  { key: "cur", label: "current  winding  mA", color: "#ff6bd6" },
  { key: "v", label: "pack  VIN  V", color: "#ffb000" },
  { key: "temp", label: "temp  case  °C", color: "#c8f060" },
  { key: "speed", label: "speed  encoder  counts/s", color: "#4ad4ff" },
];
const hist = { t: [] };
VARS.forEach(v => { hist[v.key] = []; });

let lastSpd = null;
let lastAcc = null;
let tripped = 0;
let poll = 0;
let sayLock = "";
let acting = "";
G.logFocus = ID;

function rec() {
  return G.rows[ID];
}

function clamp(n) {
  n = +n;
  if (n < LO) return LO;
  if (n > HI) return HI;
  return n;
}

function num(rec, k) {
  if (k === "v") return rec.v == null || rec.v === "" ? 0 : +rec.v;
  const x = rec[k];
  return x == null || x === "" ? 0 : +x;
}

function spark(canvas, ys, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const n = ys.length;
  if (n < 2) return;
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (lo === hi) { lo -= 1; hi += 1; }
  g.strokeStyle = color;
  g.lineWidth = 1.2;
  g.beginPath();
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1) * w;
    const y = h - (ys[i] - lo) / (hi - lo) * (h - 4) - 2;
    if (i) g.lineTo(x, y); else g.moveTo(x, y);
  }
  g.stroke();
  g.fillStyle = "#3d5c5a";
  g.font = "10px ui-monospace, monospace";
  g.fillText(String(Math.round(hi * 10) / 10), 4, 10);
  g.fillText(String(Math.round(lo * 10) / 10), 4, h - 4);
}

function ensureCharts() {
  const wrap = $("charts");
  if (!wrap || wrap.dataset.ready) return;
  wrap.dataset.ready = "1";
  wrap.innerHTML = VARS.map(v =>
    `<div class="chart${v.wide ? " wide" : ""}"><h3><span>${v.label}</span> <b data-live="${v.key}">—</b></h3><canvas data-k="${v.key}"></canvas></div>`
  ).join("");
}

function drawCharts() {
  ensureCharts();
  const wrap = $("charts");
  if (!wrap) return;
  wrap.querySelectorAll("canvas").forEach(c => {
    const k = c.getAttribute("data-k");
    const spec = VARS.find(v => v.key === k);
    spark(c, hist[k], spec.color);
    const live = wrap.querySelector(`[data-live="${k}"]`);
    if (live && hist[k].length) {
      const y = hist[k][hist[k].length - 1];
      live.textContent = k === "v" ? y.toFixed(1) : String(y);
    }
  });
}

function push(sample) {
  hist.t.push(sample.t || Date.now());
  VARS.forEach(v => hist[v.key].push(num(sample, v.key)));
  if (hist.t.length > 400) {
    hist.t.shift();
    VARS.forEach(v => hist[v.key].shift());
  }
}

let drawQ = 0;
function requestDraw() {
  if (!drawQ) drawQ = requestAnimationFrame(() => { drawQ = 0; drawCharts(); });
}

function say(s) {
  sayLock = s;
  const el = $("handSay");
  if (el) el.textContent = s;
}

function senseN() {
  return +$("sense").value;
}

function paintMeter(loadAbs, trip) {
  const fill = $("loadFill");
  const mark = $("loadTrip");
  if (fill) fill.style.width = Math.min(100, loadAbs / LOAD_SCALE * 100) + "%";
  if (mark) {
    mark.style.display = trip > 0 ? "block" : "none";
    mark.style.left = Math.min(100, trip / LOAD_SCALE * 100) + "%";
  }
}

function paint() {
  const r = rec();
  const miss = !r || r.miss;
  const loadAbs = miss ? 0 : Math.abs(+r.load || 0);
  const trip = senseN();
  const tqOn = r && (r.tq === "1" || r.tq === 1);
  $("lvPos").textContent = miss ? "—" : r.pos;
  $("lvBand").textContent = "open " + LO + "  ·  close " + HAND.close + "  ·  counts";
  $("lvLoad").textContent = miss ? "—" : r.load;
  $("lvTq").textContent = miss ? "—" : (tqOn ? (r.moving ? "moving" : "holding") : "free — you can turn it");
  $("lvCur").textContent = miss ? "—" : ((r.cur || 0) + " mA");
  $("lvVT").textContent = miss ? "—" : ((r.v != null ? r.v + " V" : "V —") + "  ·  " + (r.temp != null ? r.temp + " °C" : ""));
  if ($("lvSpd")) $("lvSpd").textContent = miss ? "—" : String(r.speed);
  if ($("lvMode")) $("lvMode").textContent = miss ? "—" : modeOf(r);
  if ($("lvPid")) $("lvPid").textContent = (r && r.kp != null) ? (r.kp + " / " + r.kd + " / " + r.ki) : "—";
  $("loadNow").textContent = String(loadAbs);
  $("senseVal").textContent = trip === 0 ? "off" : String(trip);
  const meter = $("loadMeter");
  if (meter) meter.classList.toggle("hot", trip > 0 && loadAbs >= trip * 0.7);
  paintMeter(loadAbs, trip);
  const el = $("tripEl");
  if (tripped) {
    el.className = "guard bad";
    el.textContent = "STOP — effort hit " + loadAbs + " (limit " + trip + "). jaws free. ease off, then move again";
  } else if (tqOn && trip > 0) {
    el.className = "guard on";
    el.textContent = "watching — STOP if |effort| ≥ " + trip;
  } else {
    el.className = "guard";
    el.textContent = trip === 0 ? "page trip off — chip guard can still cut" : "watching off — free or hold first";
  }
  if (!tripped && !acting) {
    const auto = miss ? "waiting for the hand"
      : tqOn && r.moving ? "hand moving"
      : tqOn ? "holding"
      : "free — turn by hand, or open / close";
    const s = $("handSay");
    if (s && s.textContent !== auto) s.textContent = auto;
  }
}

async function applyMove() {
  const spd = +$("mvSpeed").value;
  const acc = +$("mvAcc").value;
  if (spd !== lastSpd) { lastSpd = spd; await send("move speed " + spd); }
  if (acc !== lastAcc) { lastAcc = acc; await send("move acc " + acc); }
}

async function goTo(n, label) {
  tripped = 0;
  acting = label;
  say(label);
  await applyMove();
  await send("hand to " + clamp(n));
}

$("goal").oninput = () => { $("goalVal").textContent = $("goal").value; };
$("sense").oninput = paint;
$("sense").onchange = () => {
  const n = senseN();
  if (G.transport) send("guard hand " + n);
};

$("btnOpen").onclick = () => goTo(LO, "opening hand");
$("btnClose").onclick = () => goTo(HAND.close, "closing hand");
$("btnGo").onclick = () => goTo($("goal").value, "hand to " + clamp($("goal").value));
$("btnHold").onclick = () => {
  acting = "";
  tripped = 0;
  say("holding — torque on");
  send("id 17 hold");
};
$("btnFree").onclick = () => {
  acting = "";
  tripped = 0;
  say("free — turn by hand");
  send("hand free");
};
if ($("btnWho")) $("btnWho").onclick = () => send("who 17");
if ($("btnRaw")) $("btnRaw").onclick = () => send("raw 17");

G.hooks.push((ev, sample) => {
  if (ev === "sample" && sample && sample.id === ID) {
    const trip = senseN();
    const tqOn = sample.tq === "1" || sample.tq === 1;
    if (!tripped && tqOn && trip > 0 && Math.abs(+sample.load) >= trip) {
      tripped = 1;
      acting = "";
      say("STOP — hand pushed too hard");
      send("estop");
      send("hand free");
    }
    if (tripped && Math.abs(+sample.load || 0) < trip * 0.4) tripped = 0;
    if (acting && sample.moving === 0 && tqOn) acting = "";
    push(sample);
    requestDraw();
  }
  if (ev === "lab") {
    const dump = $("labDump");
    if (dump) {
      const prev = dump.textContent === "—" ? "" : dump.textContent + "\n";
      dump.textContent = (prev + sample).slice(-2000);
    }
  }
  if (ev === "who") {
    const dump = $("labDump");
    const r = rec();
    if (dump && r && r.kp != null) dump.textContent = "EEPROM  fw " + (r.fw || "—") + "  mode " + modeOf(r) + "  kp/kd/ki " + r.kp + "/" + r.kd + "/" + r.ki;
  }
  paint();
});

function tick() {
  if (!G.transport || poll) return;
  poll = 1;
  send("get 17", true).finally(() => { poll = 0; });
}

setInterval(tick, 400);
ensureCharts();
paint();
if (G.transport) send("guard", true);
