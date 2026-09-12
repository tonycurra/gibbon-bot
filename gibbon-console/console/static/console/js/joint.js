const slug = window.JOINT_SLUG;
const joint = JOINTS.find(j => j.slug === slug);
const HAND = (window.CATALOG && window.CATALOG.hand) || { open: 689, close: 2008, to_max: 2075 };
const OPS = {
  yaw: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  "shoulder-pitch": [["free", "free both"], ["hold", "hold both"], ["nudge", "nudge both Δ"]],
  elbow: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  "wrist-pitch": [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  "wrist-roll": [["free", "free"], ["hold", "hold"], ["to", "to pos"], ["servo", "servo"], ["wheel", "wheel"], ["speed", "wheel speed"], ["here", "here"], ["estop", "twist estop"]],
};
const NEED = new Set(["nudge", "to", "speed"]);
const VARS = [
  { key: "pos", label: "pos  encoder  sensor", color: "#00f0c8", wide: 1 },
  { key: "v", label: "VIN  V  sensor", color: "#ffb000" },
  { key: "temp", label: "temp  °C  sensor", color: "#c8f060" },
  { key: "cur", label: "current  mA  sensor", color: "#ff6bd6" },
  { key: "load", label: "load  PID  derived", color: "#ff2bd6" },
  { key: "speed", label: "speed  derived", color: "#4ad4ff" },
];
const hist = {};
let limits = {};
let gripDrag = 0;
joint.ids.forEach(id => {
  hist[id] = { t: [] };
  VARS.forEach(v => { hist[id][v.key] = []; });
});

function num(rec, k) {
  if (k === "tq") return rec.tq === "1" || rec.tq === 1 ? 1 : 0;
  const x = rec[k];
  return x == null || x === "" ? 0 : +x;
}

function fillOps() {
  if (!$("op")) return;
  $("op").innerHTML = (OPS[slug] || []).map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  $("actHint").textContent = joint.paired
    ? (raiseReady()
      ? "raise SyncWrite — both motors one packet. set pair (mirror/same) on calibrate first. if they fight: STOP."
      : "raise blocked — header fw old. flash first. hold/free still id 12 + id 13.")
    : "hold = tq on, watch load if you push. nudge = chip guards the move.";
  showVal();
}

function showVal() {
  if (!$("op")) return;
  const op = $("op").value;
  $("valWrap").hidden = !NEED.has(op);
  if (op === "nudge") { $("val").value = 40; $("val").min = -80; $("val").max = 80; }
  if (op === "to") { $("val").value = 2000; $("val").min = 0; $("val").max = 4095; }
  if (op === "speed") { $("val").value = 0; $("val").min = -3073; $("val").max = 3073; }
}

const txq = [];
let txBusy = 0;
let lastSpd = null;
let lastAcc = null;
let gripUntil = 0;
let gripGoal = null;
let gripTimer = 0;

function go(cmd, kind) {
  kind = kind || (String(cmd).startsWith("get ") ? "get" : "cmd");
  if (kind === "get") {
    if (gripDrag || Date.now() < gripUntil) return;
    if (txq.some(x => x.k !== "get")) return;
    if (txq.some(x => x.k === "get")) return;
    txq.push({ k: "get", cmd });
  } else if (kind === "to") {
    for (let i = txq.length - 1; i >= 0; i--) if (txq[i].k === "to") txq.splice(i, 1);
    txq.push({ k: "to", cmd });
  } else {
    for (let i = txq.length - 1; i >= 0; i--) if (txq[i].k === "to") txq.splice(i, 1);
    txq.push({ k: kind, cmd });
  }
  pump();
}

async function pump() {
  if (txBusy) return;
  txBusy = 1;
  try {
    while (txq.length) {
      const item = txq.shift();
      if (item.k === "get" && (gripDrag || txq.some(x => x.k !== "get"))) continue;
      await send(item.cmd, item.k === "get");
    }
  } finally {
    txBusy = 0;
    if (txq.length) pump();
  }
}

function prepSpeed() {
  const spd = $("mvSpeed") ? $("mvSpeed").value : "300";
  const acc = ($("mvAcc") && $("mvAcc").value) || "20";
  if (spd !== lastSpd) { lastSpd = spd; go("move speed " + spd); }
  if (acc !== lastAcc) { lastAcc = acc; go("move acc " + acc); }
}

async function prepMove() {
  prepSpeed();
}

function cmdFor(id, op, n) {
  if (id === 17 && (op === "open" || op === "close" || op === "to" || op === "free")) {
    return op === "to" ? "hand to " + n : "hand " + op;
  }
  if (id === 16 && op !== "nudge" && op !== "hold") {
    if (op === "to") return "twist to " + n;
    if (op === "speed") return "twist speed " + n;
    if (op === "estop") return "twist estop";
    return "twist " + op;
  }
  if (op === "to") return "id " + id + " to " + n;
  if (op === "nudge") return "id " + id + " nudge " + n;
  return "id " + id + " " + op;
}

function paintSense() {
  const el = $("sense");
  if (!el) return;
  el.innerHTML = joint.ids.map(id => {
    const r = G.rows[id];
    const miss = !r || r.miss;
    const load = miss ? "—" : r.load;
    const cur = miss ? "—" : r.cur;
    const pos = miss ? "—" : r.pos;
    const v = miss ? "—" : r.v;
    const temp = miss ? "—" : r.temp;
    const hotL = r && Math.abs(r.load) > 400;
    const hotC = r && Math.abs(r.cur) >= 5;
    return `<div class="senseRow">
      <span><b>pos</b> ${pos}</span>
      <span class="${hotL ? "hot" : ""}"><b>load</b> ${load}</span>
      <span class="${hotC ? "hot" : ""}"><b>mA</b> ${cur}</span>
      <span><b>V</b> ${v}</span>
      <span><b>°C</b> ${temp}</span>
      <span><b>tq</b> ${miss ? "—" : (r.tq === "1" ? "on" : "off")}</span>
      <span class="mut">${miss ? "miss" : (r.moving ? "moving" : "still")}</span>
    </div>`;
  }).join("");
}

function paintGuard() {
  const el = $("guardEl");
  if (!el) return;
  const s = G.guard || "";
  el.className = "guard" + (s.startsWith("ESTOP") ? " bad" : (s.startsWith("guard on") ? " on" : ""));
  el.textContent = s || (slug === "hand"
    ? "guard idle — trip Δload is settable. 0 = only hard |load| / mA"
    : "guard idle — a commanded move is cut if load jumps +100 above free cruise, or |current| ≥ 80 mA");
}

function paintTable() {
  paintSense();
  paintGuard();
  const body = $("armBody");
  if (!body) return;
  body.innerHTML = joint.ids.map(id => {
    const r = G.rows[id];
    const miss = !r || r.miss;
    const cells = miss
      ? [id, "—", "—", "—", "—", "—", "—", "—", "—", "—"]
      : [id, r.pos, r.un, r.speed, r.load, r.cur, r.v, r.temp, r.moving, r.tq];
    const hotLoad = r && Math.abs(r.load) > 400;
    const hotCur = r && Math.abs(r.cur) >= 5;
    return `<tr class="${miss ? "no" : "go"}">${
      cells.map((c, i) => {
        const cls = (i === 4 && hotLoad) || (i === 5 && hotCur) ? ' class="hot"' : "";
        return `<td${cls}>${dash(c)}</td>`;
      }).join("")
    }</tr>`;
  }).join("");
  const raw = $("rawGets");
  if (raw) {
    raw.textContent = joint.ids.map(id => {
      const r = G.rows[id];
      return r && r.raw ? r.raw : "get id=" + id + " —";
    }).join("\n");
  }
}

function spark(canvas, ys, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const n = ys.length;
  if (n < 2) return;
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (lo === hi) { lo -= 1; hi += 1; }
  g.strokeStyle = color; g.lineWidth = 1.2; g.beginPath();
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1) * w;
    const y = h - (ys[i] - lo) / (hi - lo) * (h - 4) - 2;
    if (i) g.lineTo(x, y); else g.moveTo(x, y);
  }
  g.stroke();
  g.fillStyle = "#3d5c5a"; g.font = "10px ui-monospace,monospace";
  g.fillText(String(hi), 4, 10);
  g.fillText(String(lo), 4, h - 4);
}

function ensureCharts() {
  const wrap = $("charts");
  if (wrap.dataset.ready) return;
  wrap.dataset.ready = "1";
  wrap.innerHTML = joint.ids.map(id => {
    const m = MOTORS.find(x => x.id === id);
    const who = m ? m.name : id;
    return VARS.map(v =>
      `<div class="chart${v.wide ? " wide" : ""}"><h3><span>${who} ${v.label}</span> <b data-live="${id}-${v.key}">—</b></h3><canvas data-id="${id}" data-k="${v.key}"></canvas></div>`
    ).join("");
  }).join("");
}

function drawCharts() {
  ensureCharts();
  $("charts").querySelectorAll("canvas").forEach(c => {
    const id = +c.getAttribute("data-id");
    const k = c.getAttribute("data-k");
    const spec = VARS.find(v => v.key === k);
    const ys = hist[id][k];
    spark(c, ys, spec.color);
    const live = document.querySelector(`[data-live="${id}-${k}"]`);
    if (live && ys.length) live.textContent = ys[ys.length - 1];
  });
}

function push(rec) {
  const h = hist[rec.id];
  if (!h) return;
  h.t.push(rec.t);
  VARS.forEach(v => h[v.key].push(num(rec, v.key)));
  if (h.t.length > 400) {
    h.t.shift();
    VARS.forEach(v => h[v.key].shift());
  }
}

let drawQ = 0;
function requestDraw() {
  if (!drawQ) drawQ = requestAnimationFrame(() => { drawQ = 0; drawCharts(); });
}

function pollPresent() {
  if (!G.alive || !G.transport || gripDrag || Date.now() < gripUntil) return;
  for (const id of joint.ids) go("get " + id, "get");
}

function livePos(id) {
  const r = G.rows[id];
  return r && !r.miss ? r.pos : null;
}

function handBand() {
  return { lo: HAND.open, hi: HAND.to_max };
}

function applyGripRange() {
  const g = $("grip");
  if (!g) return;
  const { lo, hi } = handBand();
  g.min = lo;
  g.max = hi;
  if (+g.value < lo) g.value = lo;
  if (+g.value > hi) g.value = hi;
}

function paintGrip(pos) {
  const g = $("grip");
  const v = $("gripVal");
  if (!g) return;
  const hold = gripDrag || (gripGoal != null && Date.now() < gripUntil);
  if (hold && pos != null && gripGoal != null && Math.abs(pos - gripGoal) < 20) gripGoal = null;
  if (!hold && pos != null) {
    g.value = Math.max(+g.min, Math.min(+g.max, pos));
  }
  if (v) v.textContent = hold ? g.value : (pos != null ? pos : g.value);
}

function paintMarks() {
  const bits = joint.ids.map(id => {
    const L = limits[id] || {};
    return id + " home " + dash(L.home) + " · min " + dash(L.amin) + " · max " + dash(L.amax);
  });
  $("markState").textContent = bits.join("  |  ");
  $("btnTest").disabled = !joint.ids.every(id => {
    const L = limits[id] || {};
    return L.amin != null && L.amax != null;
  });
}

function markSay(s) {
  const el = $("markMsg");
  if (el) el.textContent = s || "";
  if (s) log(s);
}

async function markKey(key) {
  for (const id of joint.ids) {
    const pos = livePos(id);
    if (pos == null) {
      markSay("no live pos on id " + id + " — connect, pack on");
      return;
    }
    try {
      await saveLimit(id, key, pos);
      markSay("marked " + key + "=" + pos + " on id " + id);
    } catch (e) {
      markSay(String(e && e.message ? e.message : e));
      return;
    }
  }
  limits = await fetchLimits();
  paintMarks();
}

function handGo(cmd) {
  prepSpeed();
  const n = cmd.startsWith("hand to ") ? +cmd.slice(8) : null;
  if (n != null && !Number.isNaN(n)) {
    gripGoal = n;
    gripUntil = Date.now() + 2500;
    go(cmd, "to");
    return;
  }
  if (cmd === "hand open") { gripGoal = HAND.open; gripUntil = Date.now() + 2500; }
  if (cmd === "hand close") { gripGoal = HAND.close; gripUntil = Date.now() + 2500; }
  go(cmd);
}

function sendGrip() {
  const g = $("grip");
  if (!g) return;
  $("gripVal").textContent = g.value;
  handGo("hand to " + Math.round(+g.value));
}

function paintGuardCfg() {
  const el = $("gdload");
  if (!el || !G.guardCfg || document.activeElement === el) return;
  el.value = G.guardCfg.hand;
}

function setupHand() {
  const g = $("grip");
  $("actHint").textContent = "open / close / drag the slider. trip Δload cuts a squeeze; 0 = off.";
  applyGripRange();
  paintGrip(livePos(17));
  paintGuardCfg();
  prepSpeed();
  $("btnOpen").onclick = () => handGo("hand open");
  $("btnClose").onclick = () => handGo("hand close");
  const spd = $("mvSpeed");
  if (spd) spd.onchange = () => prepSpeed();
  g.addEventListener("pointerdown", e => {
    gripDrag = 1;
    try { g.setPointerCapture(e.pointerId); } catch {}
  });
  g.addEventListener("input", () => {
    gripDrag = 1;
    $("gripVal").textContent = g.value;
    clearTimeout(gripTimer);
    gripTimer = setTimeout(sendGrip, 60);
  });
  const end = () => {
    if (!gripDrag) return;
    clearTimeout(gripTimer);
    sendGrip();
    gripDrag = 0;
  };
  g.addEventListener("pointerup", end);
  g.addEventListener("pointercancel", () => { gripDrag = 0; clearTimeout(gripTimer); });
  const gd = $("gdload");
  if (gd) {
    gd.onchange = () => go("guard hand " + Math.max(0, Math.min(4000, +gd.value || 0)));
  }
  const fac = $("btnFactory");
  if (fac) {
    fac.onclick = async () => {
      await saveLimit(17, "amin", HAND.open);
      await saveLimit(17, "amax", HAND.to_max);
      limits = await fetchLimits();
      paintMarks();
      markSay("walls " + HAND.open + "–" + HAND.to_max + ". open → " + HAND.open + ", close → " + HAND.close + ".");
    };
  }
}

let tableQ = 0;
function requestTable() {
  if (!tableQ) tableQ = requestAnimationFrame(() => { tableQ = 0; paintTable(); });
}

G.hooks.push((kind, rec) => {
  requestTable();
  if (kind === "lim" && rec && rec.id) {
    const cur = limits[rec.id] || {};
    if (rec.home != null) cur.home = rec.home;
    if (rec.amin != null) cur.amin = rec.amin;
    if (rec.amax != null) cur.amax = rec.amax;
    limits[rec.id] = cur;
    paintMarks();
  }
  if (kind === "sample" && rec && !rec.miss && hist[rec.id]) {
    push(rec);
    requestDraw();
    if (slug === "hand" && rec.id === 17) paintGrip(rec.pos);
  }
  if (kind === "guard") paintGuard();
  if (kind === "guardcfg") paintGuardCfg();
});

if ($("op")) {
  fillOps();
  $("op").onchange = showVal;
}
if ($("actForm")) {
  $("actForm").onsubmit = async e => {
    e.preventDefault();
    const op = $("op").value;
    const n = $("val").value;
    prepSpeed();
    if (joint.paired) {
      if (op === "free") { sendRaiseTq(false); return; }
      if (op === "hold") { sendRaiseTq(true); return; }
      if (op === "nudge") {
        if (!raiseReady()) {
          markSay("raise blocked — flash first (header fw old)");
          return;
        }
        const sign = pairSign();
        const p12 = livePos(12);
        const p13 = livePos(13);
        const d = +n;
        if (sign == null || p12 == null || p13 == null) {
          markSay("set pair on /calibrate/shoulder-pitch/ first");
          return;
        }
        go(cmdRaiseTo(p12 + d, p13 + sign * d));
        return;
      }
    }
    for (const id of joint.ids) go(cmdFor(id, op, n));
  };
}
if (slug === "hand") setupHand();

$("markRow").onclick = e => {
  const b = e.target.closest("button[data-k]");
  if (b) markKey(b.getAttribute("data-k"));
};
$("btnFree").onclick = () => {
  gripGoal = null;
  if (joint.paired) sendRaiseTq(false);
  else for (const id of joint.ids) go(cmdFor(id, "free"));
  markSay("free — turn by hand, watch pos, then mark");
};
$("btnTest").onclick = async () => {
  if (joint.paired) {
    markSay("test shoulder pitch from /calibrate/shoulder-pitch/ — needs pair + SyncWrite");
    return;
  }
  const spd = Math.max(1, Math.min(3073, +($("testSpeed") && $("testSpeed").value) || +$("mvSpeed").value || 300));
  const acc = Math.max(0, Math.min(150, +($("testAcc") && $("testAcc").value) || +$("mvAcc").value || 20));
  await send("move speed " + spd);
  await send("move acc " + acc);
  for (const id of joint.ids) {
    const L = limits[id] || {};
    if (L.amin == null || L.amax == null) {
      markSay("mark min and max first");
      return;
    }
    const lo = Math.min(+L.amin, +L.amax);
    const hi = Math.max(+L.amin, +L.amax);
    const pos = livePos(id);
    let goal = Math.round((lo + hi) / 2);
    if (id !== 17 && pos != null && Math.abs(goal - pos) > 80) {
      goal = pos + (goal > pos ? 80 : -80);
    }
    await send(cmdFor(id, "to", goal));
    markSay("test to " + goal + " @ speed " + spd + " acc " + acc + " inside " + lo + "–" + hi + ". STOP if it fights.");
  }
};

paintTable();
drawCharts();
fetchLimits().then(m => { limits = m; syncPairFromLimits(m); paintMarks(); paintGrip(livePos(17)); });
setInterval(pollPresent, 500);
pollPresent();
