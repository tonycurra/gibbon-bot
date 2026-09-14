const ENC = 4096;
const NEAR = 50;
const TEST_MS = 20000;
const WIZARD = (window.CATALOG && window.CATALOG.wizard) || [];
const SLUG = window.JOINT_SLUG || "";
const JOINT = JOINTS.find(j => j.slug === SLUG) || null;
const IDS = JOINT ? JOINT.ids : [];
const ID = IDS[0] || 0;

let limits = {};
const homeUn = {};
let test = null;
let lastSpd = null;

function isHand(id) { return id === 17; }
function isWrap(id) { return id === 11 || id === 16; }

function livePos(id) {
  const r = G.rows[id];
  return r && !r.miss ? r.pos : null;
}
function liveUn(id) {
  const r = G.rows[id];
  if (!r || r.miss) return null;
  return r.un != null ? r.un : r.pos;
}
function liveDelta(id) {
  const un = liveUn(id);
  if (un == null || homeUn[id] == null) return null;
  if (isHand(id)) return homeUn[id] - un;
  return un - homeUn[id];
}
function degOf(d) { return (+d) * 360 / ENC; }
function cntOf(deg) { return Math.round((+deg) * ENC / 360); }
function fmtDeg(d) {
  if (d == null || !isFinite(+d)) return "—";
  const n = degOf(d);
  return (n > 0 ? "+" : "") + n.toFixed(1) + "°";
}
function say(s) {
  const el = $("calMsg");
  if (el) el.textContent = s || "";
  if (s) log(s);
}
function cmdFree(id) {
  if (id === 17) return "hand free";
  if (id === 16) return "twist free";
  return "id " + id + " free";
}
function cmdHold(id) { return "id " + id + " hold"; }
function cmdHere(id) { return id === 16 ? "twist here" : ("id " + id + " here"); }
function cmdTo(id, goal) {
  if (id === 17) return "hand to " + goal;
  if (id === 16) return "twist to " + goal;
  return "id " + id + " to " + goal;
}
function rawFromDelta(id, d) {
  const origin = homeUn[id] != null ? homeUn[id] : (limits[id] && limits[id].home);
  if (origin == null || d == null) return null;
  const raw = isHand(id) ? (+origin - +d) : (+origin + +d);
  return Math.max(0, Math.min(4095, Math.round(raw)));
}
function syncHome() {
  MOTORS.forEach(m => {
    if (homeUn[m.id] != null) return;
    const L = limits[m.id];
    if (L && L.home != null) homeUn[m.id] = +L.home;
  });
}
function wallsOk(id) {
  const L = limits[id] || {};
  if (L.home == null || L.amin == null || L.amax == null) return false;
  return Math.abs(+L.amax - +L.amin) >= 80;
}
function spd(n) {
  const acc = n < 150 ? 8 : 16;
  const key = n + "/" + acc;
  if (key === lastSpd) return;
  lastSpd = key;
  send("move speed " + n);
  send("move acc " + acc);
}

async function markZero() {
  for (const id of IDS) {
    const pos = livePos(id);
    if (pos == null) { say("no live pos — pack on"); return; }
    await send(cmdHere(id));
    if (G.rows[id] && !G.rows[id].miss) {
      G.rows[id].un = pos;
      G.rows[id].turns = 0;
    }
    homeUn[id] = pos;
    await saveLimit(id, "home", pos);
    await fetch("/api/limits", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ servo: id, amin: isHand(id) ? 0 : null, amax: null }),
    });
  }
  limits = await fetchLimits();
  say(isHand(ID) ? "closed = 0°" : "0°");
  fillInputs();
}

async function markWall(key) {
  if (homeUn[ID] == null) { say("set 0° first"); return; }
  const d = liveDelta(ID);
  if (d == null) { say("no live pos"); return; }
  if (isHand(ID) && key === "amin") {
    if (Math.abs(d) > 40) { say("close the jaws first"); return; }
  }
  if (isHand(ID) && key === "amax" && d < 80) { say("open more"); return; }
  await saveLimit(ID, key, d, { chip: false });
  const raw = rawFromDelta(ID, d);
  if (raw != null && !isWrap(ID)) {
    await send("lim " + ID + " " + (key === "amin" ? "min" : "max") + " " + raw);
  }
  if (IDS[1]) {
    const d2 = liveDelta(IDS[1]);
    if (d2 != null) await saveLimit(IDS[1], key, d2, { chip: false });
  }
  limits = await fetchLimits();
  fillInputs();
  say((key === "amin" ? "min " : "max ") + fmtDeg(d));
}

async function goDeg(deg) {
  if (homeUn[ID] == null) { say("set 0° first"); return; }
  const raw = rawFromDelta(ID, cntOf(deg));
  if (raw == null) { say("no origin"); return; }
  spd(120);
  if (ID === 12) {
    if (pairSign() == null) { say("set pair first"); return; }
    const d = cntOf(deg);
    const a = rawFromDelta(12, d);
    const b = rawFromDelta(13, pairSign() * d);
    if (!raiseReady()) { say("fw old — flash"); return; }
    send(cmdRaiseTo(a, b));
    return;
  }
  send(cmdTo(ID, raw));
}

function fillInputs() {
  const L = limits[ID] || {};
  const minEl = $("inMin");
  const maxEl = $("inMax");
  if (minEl && document.activeElement !== minEl) {
    minEl.value = L.amin != null ? degOf(L.amin).toFixed(1) : "";
  }
  if (maxEl && document.activeElement !== maxEl) {
    maxEl.value = L.amax != null ? degOf(L.amax).toFixed(1) : "";
  }
}

function paintPair() {
  const s = pairSign();
  const st = $("pairState");
  if (st) st.textContent = s == null ? "unset" : (s < 0 ? "mirror −" : "same +");
  const m = $("btnPairMirror");
  const a = $("btnPairSame");
  if (m) m.classList.toggle("on", s === -1);
  if (a) a.classList.toggle("on", s === 1);
}

function paintIndex() {
  WIZARD.forEach(w => {
    const j = JOINTS.find(x => x.slug === w.slug);
    if (!j) return;
    const id = j.ids[0];
    const r = G.rows[id];
    const miss = !r || r.miss;
    const now = $("now-" + w.slug);
    const load = $("load-" + w.slug);
    const cur = $("cur-" + w.slug);
    const mn = $("min-" + w.slug);
    const mx = $("max-" + w.slug);
    const row = $("row-" + w.slug);
    if (now) now.textContent = homeUn[id] != null ? fmtDeg(liveDelta(id)) : (miss ? "—" : String(r.pos));
    if (load) {
      load.textContent = miss ? "—" : r.load;
      load.className = !miss && Math.abs(r.load) > 400 ? "hot" : "";
    }
    if (cur) {
      cur.textContent = miss ? "—" : (r.cur || 0);
      cur.className = !miss && Math.abs(r.cur) >= 5 ? "hot" : "";
    }
    const L = limits[id] || {};
    if (mn) mn.textContent = L.amin != null ? fmtDeg(L.amin) : "—";
    if (mx) mx.textContent = L.amax != null ? fmtDeg(L.amax) : "—";
    if (row) row.className = wallsOk(id) ? "ok" : "";
  });
}

function paintJoint() {
  if (!JOINT) return;
  const r = G.rows[ID];
  const miss = !r || r.miss;
  const deg = $("degNow");
  if (deg) deg.textContent = homeUn[ID] != null ? fmtDeg(liveDelta(ID)) : (miss ? "—" : String(r.pos));
  const load = $("loadLive");
  if (load) {
    load.textContent = miss ? "—" : r.load;
    load.className = !miss && Math.abs(r.load) > 400 ? "hot" : "";
  }
  const cur = $("curLive");
  if (cur) {
    cur.textContent = miss ? "—" : (r.cur || 0);
    cur.className = !miss && Math.abs(r.cur) >= 5 ? "hot" : "";
  }
  IDS.forEach(i => {
    const el = $("load-" + i);
    const rr = G.rows[i];
    if (el) el.textContent = !rr || rr.miss ? "—" : rr.load;
  });
  const t = $("btnTest");
  if (t && !test) t.disabled = !wallsOk(ID);
}

async function runTest() {
  if (test) { test = null; say("test stop"); return; }
  if (!wallsOk(ID)) { say("mark min and max first"); return; }
  const L = limits[ID];
  const lo = rawFromDelta(ID, L.amin);
  const hi = rawFromDelta(ID, L.amax);
  if (lo == null || hi == null) { say("need 0°"); return; }
  if (ID === 12 && pairSign() == null) { say("set pair first"); return; }
  if (!isWrap(ID)) {
    await send("lim " + ID + " min " + Math.min(lo, hi));
    await send("lim " + ID + " max " + Math.max(lo, hi));
  }
  const phases = [
    { name: "min", goal: lo },
    { name: "max", goal: hi },
    { name: "min", goal: lo },
  ];
  test = { i: 0, phases, t0: Date.now(), sent: 0 };
  spd(120);
  say("test " + fmtDeg(L.amin) + " ↔ " + fmtDeg(L.amax));
  await sendGoal(phases[0].goal);
}

async function sendGoal(goal) {
  const pos = livePos(ID);
  const hop = isWrap(ID) ? 900 : 0;
  let g = goal;
  if (hop && pos != null && Math.abs(goal - pos) > hop) {
    g = pos + (goal > pos ? hop : -hop);
  }
  if (ID === 12) {
    const origin = homeUn[12];
    const d = origin == null ? 0 : (g - origin);
    const b = homeUn[13] == null ? g : (homeUn[13] + pairSign() * d);
    await send(cmdRaiseTo(g, b));
    return;
  }
  await send(cmdTo(ID, g));
}

function tickTest() {
  if (!test) return;
  const ph = test.phases[test.i];
  if (!ph) return;
  if (Date.now() - test.t0 > TEST_MS) {
    test = null;
    say("test timeout on " + ph.name + " — STOP if it fights");
    return;
  }
  const pos = livePos(ID);
  if (pos == null) return;
  if (Math.abs(pos - ph.goal) > NEAR) {
    if (!test.sent || Date.now() - test.sent > 250) {
      test.sent = Date.now();
      sendGoal(ph.goal);
    }
    return;
  }
  test.i++;
  test.sent = 0;
  if (test.i >= test.phases.length) {
    test = null;
    send(cmdHold(ID));
    if (ID === 12) sendRaiseTq(true);
    say("test ok");
    const t = $("btnTest");
    if (t) t.textContent = "test ok";
    return;
  }
  test.t0 = Date.now();
  sendGoal(test.phases[test.i].goal);
  say("test → " + test.phases[test.i].name);
}

function paint() {
  syncHome();
  if (SLUG) paintJoint();
  else paintIndex();
  paintPair();
  if (test) tickTest();
}

if ($("btnZero")) $("btnZero").onclick = () => markZero().catch(e => say(String(e)));
if ($("btnMarkMin")) $("btnMarkMin").onclick = () => markWall("amin").catch(e => say(String(e)));
if ($("btnMarkMax")) $("btnMarkMax").onclick = () => markWall("amax").catch(e => say(String(e)));
if ($("inMin")) $("inMin").onchange = () => {
  if (homeUn[ID] == null) { say("set 0° first"); return; }
  const d = cntOf(+$("inMin").value);
  saveLimit(ID, "amin", d, { chip: false }).then(async () => {
    const raw = rawFromDelta(ID, d);
    if (raw != null && !isWrap(ID)) await send("lim " + ID + " min " + raw);
    limits = await fetchLimits();
  }).catch(e => say(String(e)));
};
if ($("inMax")) $("inMax").onchange = () => {
  if (homeUn[ID] == null) { say("set 0° first"); return; }
  const d = cntOf(+$("inMax").value);
  saveLimit(ID, "amax", d, { chip: false }).then(async () => {
    const raw = rawFromDelta(ID, d);
    if (raw != null && !isWrap(ID)) await send("lim " + ID + " max " + raw);
    limits = await fetchLimits();
  }).catch(e => say(String(e)));
};
if ($("btnGo")) $("btnGo").onclick = () => {
  const v = $("inGo") && $("inGo").value;
  if (v === "" || v == null) { say("type a degree"); return; }
  goDeg(+v).catch(e => say(String(e)));
};
if ($("btnTest")) $("btnTest").onclick = () => runTest().catch(e => say(String(e)));
if ($("btnPairMirror")) $("btnPairMirror").onclick = () => savePairSign(-1).then(paintPair);
if ($("btnPairSame")) $("btnPairSame").onclick = () => savePairSign(1).then(paintPair);

G.hooks.push(paint);
fetchLimits().then(m => {
  limits = m;
  syncPairFromLimits(m);
  syncHome();
  fillInputs();
  paint();
});
setInterval(() => { if (test) tickTest(); }, 80);
