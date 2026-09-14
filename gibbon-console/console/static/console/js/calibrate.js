const ENC = 4096;
const NEAR = 50;
const TEST_MS = 20000;
const SLUG = window.JOINT_SLUG || "";
const JOINT = JOINTS.find(j => j.slug === SLUG) || null;

let limits = {};
const homeUn = {};
let test = null;
let lastSpd = null;
let rec = null;
let playOn = false;
const PATH_KEY = "gibbon.shoulder.path";

function isHand(id) { return id === 17; }
function isWrap(id) { return id === 11 || id === 16; }
function primary(j) { return j.ids[0]; }
function originOf(id) {
  if (homeUn[id] != null) return homeUn[id];
  if (limits[id] && limits[id].home != null) return +limits[id].home;
  return null;
}
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
  const o = originOf(id);
  if (un == null || o == null) return null;
  return isHand(id) ? o - un : un - o;
}
function degOf(d) { return (+d) * 360 / ENC; }
function cntOf(deg) { return Math.round((+deg) * ENC / 360); }
function fmtDeg(d) {
  if (d == null || !isFinite(+d)) return "";
  const n = degOf(d);
  return (n > 0 ? "+" : "") + n.toFixed(1);
}
function say(s) {
  const el = $("calMsg");
  if (el) el.textContent = s || "";
  if (s) log(s);
}
function cmdHere(id) { return id === 16 ? "twist here" : ("id " + id + " here"); }
function cmdTo(id, goal) {
  if (id === 17) return "hand to " + Math.round(goal);
  if (id === 16) return "twist to " + Math.round(goal);
  return "id " + id + " to " + Math.round(goal);
}
function rawFromDelta(id, d) {
  const o = originOf(id);
  if (o == null || d == null) return null;
  const raw = isHand(id) ? +o - +d : +o + +d;
  return Math.max(0, Math.min(4095, Math.round(raw)));
}
function syncHome() {
  MOTORS.forEach(m => {
    if (homeUn[m.id] != null) return;
    const L = limits[m.id];
    if (L && L.home != null) homeUn[m.id] = +L.home;
  });
}
function wallsOk(j) {
  const L = limits[primary(j)] || {};
  if (originOf(primary(j)) == null || L.amin == null || L.amax == null) return false;
  return Math.abs(+L.amax - +L.amin) >= 80;
}
function wallDelta(id, d) {
  const s = pairSign();
  return id === 13 && s != null ? s * d : d;
}
function spd(n) {
  const acc = n < 150 ? 8 : 16;
  const key = n + "/" + acc;
  if (key === lastSpd) return;
  lastSpd = key;
  send("move speed " + n);
  send("move acc " + acc);
}
function root(j) {
  return $("row-" + j.slug) || (JOINT && j.slug === JOINT.slug ? $("calSet") : null);
}
function cell(j, cls) {
  const r = root(j);
  return r ? r.querySelector("." + cls) : null;
}

function fillWalls(j) {
  const id = primary(j);
  const L = limits[id] || {};
  const put = (cls, on, text) => {
    const el = cell(j, cls);
    if (!el || el.querySelector("input")) return;
    el.textContent = on ? text : (j.slug === "hand" && cls === "home" ? "closed" : "set");
    el.className = "set " + cls + (on ? " on" : "");
  };
  put("home", originOf(id) != null, "0");
  put("min", L.amin != null, fmtDeg(L.amin));
  put("max", L.amax != null, fmtDeg(L.amax));
  const btn = cell(j, "test");
  if (btn && (!test || test.j.slug !== j.slug)) {
    btn.disabled = !wallsOk(j);
    if (btn.textContent === "ok" || btn.textContent === "stop" || btn.textContent === "min" || btn.textContent === "max") {
      /* keep */
    } else {
      btn.textContent = "test";
    }
  }
}

async function saveHome(j) {
  for (const id of j.ids) {
    const pos = livePos(id);
    if (pos == null) { say("no live pos — pack on"); return; }
    await send(cmdHere(id));
    if (G.rows[id] && !G.rows[id].miss) {
      G.rows[id].un = pos;
      G.rows[id].turns = 0;
    }
    homeUn[id] = pos;
    await saveLimit(id, "home", pos);
    const raw = rawFromDelta(id, 0);
    if (raw != null && !isWrap(id)) await send("lim " + id + " home " + pos);
  }
  limits = await fetchLimits();
  syncPairFromLimits(limits);
  fillWalls(j);
  say(isHand(primary(j)) ? "closed = 0°" : "0°");
}

async function saveWall(j, key, deg) {
  if (originOf(primary(j)) == null) { say("click home first"); return; }
  if (deg === "" || deg == null || !isFinite(+deg)) return;
  const d = cntOf(+deg);
  for (const id of j.ids) {
    const dd = wallDelta(id, d);
    await saveLimit(id, key === "min" ? "amin" : "amax", dd, { chip: false });
    const raw = rawFromDelta(id, dd);
    if (raw != null && !isWrap(id)) {
      await send("lim " + id + " " + (key === "min" ? "min" : "max") + " " + raw);
    }
  }
  limits = await fetchLimits();
  fillWalls(j);
  say(j.name + " " + key + " " + (+deg) + "°");
}

function startEdit(j, field) {
  if (field === "home") {
    saveHome(j).catch(e => say(String(e)));
    return;
  }
  if (originOf(primary(j)) == null) { say("click home first"); return; }
  const td = cell(j, field);
  if (!td || td.querySelector("input")) return;
  const d = liveDelta(primary(j));
  const L = limits[primary(j)] || {};
  const saved = field === "min" ? L.amin : L.amax;
  const input = document.createElement("input");
  input.type = "number";
  input.step = "0.1";
  input.value = d != null ? degOf(d).toFixed(1) : (saved != null ? degOf(saved).toFixed(1) : "");
  let closed = false;
  const done = ok => {
    if (closed) return;
    closed = true;
    const v = input.value;
    if (input.parentNode) input.remove();
    fillWalls(j);
    if (ok) saveWall(j, field, v).catch(e => say(String(e)));
  };
  td.textContent = "";
  td.appendChild(input);
  input.focus();
  input.select();
  input.onclick = ev => ev.stopPropagation();
  input.onkeydown = ev => {
    if (ev.key === "Enter") { ev.preventDefault(); done(true); }
    if (ev.key === "Escape") { ev.stopPropagation(); done(false); }
  };
  input.onblur = () => done(true);
}

function idsText(j, key) {
  return j.ids.map(id => {
    const r = G.rows[id];
    if (!r || r.miss) return "—";
    if (key === "pos") return String(r.pos);
    const un = r.un != null ? r.un : r.pos;
    return String(un);
  }).join(" / ");
}
function wrapish(j) {
  return j.ids.some(id => {
    const r = G.rows[id];
    if (!r || r.miss) return false;
    const un = r.un != null ? r.un : r.pos;
    return Math.abs(un - r.pos) >= 2 || (r.turns || 0);
  });
}
function paintLive(j) {
  const id = primary(j);
  const r = G.rows[id];
  const miss = !r || r.miss;
  const d = liveDelta(id);
  if (cell(j, "now")) {
    cell(j, "now").textContent = d != null ? fmtDeg(d) : (miss ? "—" : String(r.pos));
  }
  const pos = cell(j, "pos");
  const un = cell(j, "un");
  if (pos) pos.textContent = miss ? "—" : idsText(j, "pos");
  if (un) {
    un.textContent = miss ? "—" : idsText(j, "un");
    un.className = "un" + (wrapish(j) ? " wrap" : "");
  }
}

function bindJoint(j) {
  const r = root(j);
  if (!r) return;
  const home = r.querySelector(".home");
  const min = r.querySelector(".min");
  const max = r.querySelector(".max");
  const btn = r.querySelector(".test");
  if (home) home.onclick = () => startEdit(j, "home");
  if (min) min.onclick = () => startEdit(j, "min");
  if (max) max.onclick = () => startEdit(j, "max");
  if (btn) btn.onclick = () => runTest(j).catch(e => say(String(e)));
}

async function goDeg(deg) {
  const j = JOINT;
  if (!j) return;
  if (originOf(primary(j)) == null) { say("click home first"); return; }
  const raw = rawFromDelta(primary(j), cntOf(deg));
  if (raw == null) { say("no origin"); return; }
  spd(120);
  await sendGoal(j, raw);
}

async function sendGoal(j, goal) {
  const id = primary(j);
  const now = livePos(id);
  const hop = isWrap(id) ? 900 : 0;
  let g = goal;
  if (hop && now != null && Math.abs(goal - now) > hop) g = now + (goal > now ? hop : -hop);
  if (id === 12) {
    if (pairSign() == null) { say("set pair first"); return; }
    if (!raiseReady()) { say("fw old — flash"); return; }
    const o12 = originOf(12);
    const d = o12 == null ? 0 : (g - o12);
    const b = originOf(13) == null ? g : (originOf(13) + pairSign() * d);
    await send(cmdRaiseTo(g, b));
    return;
  }
  await send(cmdTo(id, g));
}

async function runTest(j) {
  if (test && test.j.slug === j.slug) {
    test = null;
    const btn = cell(j, "test");
    if (btn) btn.textContent = "test";
    say("test stop");
    return;
  }
  if (test) { say("stop the other test first, or STOP"); return; }
  if (!wallsOk(j)) { say("set min and max first"); return; }
  if (j.paired && pairSign() == null) { say("set pair first"); return; }
  if (j.paired && !raiseReady()) { say("fw old — flash"); return; }
  const id = primary(j);
  const L = limits[id];
  const lo = rawFromDelta(id, L.amin);
  const hi = rawFromDelta(id, L.amax);
  if (lo == null || hi == null) { say("need home"); return; }
  if (!isWrap(id)) {
    await send("lim " + id + " min " + Math.min(lo, hi));
    await send("lim " + id + " max " + Math.max(lo, hi));
  }
  test = {
    j,
    phases: [{ name: "min", goal: lo }, { name: "max", goal: hi }, { name: "min", goal: lo }],
    i: 0, t0: Date.now(), sent: 0,
  };
  const btn = cell(j, "test");
  if (btn) { btn.disabled = false; btn.textContent = "stop"; }
  spd(120);
  say(j.name + " test " + fmtDeg(L.amin) + " → " + fmtDeg(L.amax));
  await sendGoal(j, lo);
}

function tickTest() {
  if (!test) return;
  const j = test.j;
  const ph = test.phases[test.i];
  if (!ph) return;
  const btn = cell(j, "test");
  if (Date.now() - test.t0 > TEST_MS) {
    test = null;
    if (btn) { btn.textContent = "test"; btn.disabled = !wallsOk(j); }
    say("test timeout on " + ph.name + " — STOP if it fights");
    return;
  }
  const now = livePos(primary(j));
  if (now == null) return;
  if (Math.abs(now - ph.goal) > NEAR) {
    if (!test.sent || Date.now() - test.sent > 250) {
      test.sent = Date.now();
      sendGoal(j, ph.goal);
    }
    return;
  }
  test.i++;
  test.sent = 0;
  if (test.i >= test.phases.length) {
    test = null;
    if (primary(j) === 12) sendRaiseTq(true);
    else send("id " + primary(j) + " hold");
    if (btn) btn.textContent = "ok";
    say("test ok");
    return;
  }
  test.t0 = Date.now();
  if (btn) btn.textContent = test.phases[test.i].name;
  sendGoal(j, test.phases[test.i].goal);
  say("test → " + test.phases[test.i].name);
}

function paintIndex() {
  document.querySelectorAll("table.calIdx tr[data-slug]").forEach(tr => {
    const j = JOINTS.find(x => x.slug === tr.dataset.slug);
    if (!j) return;
    const id = primary(j);
    const r = G.rows[id];
    const miss = !r || r.miss;
    paintLive(j);
    const load = cell(j, "load");
    const cur = cell(j, "cur");
    if (load) {
      load.textContent = miss ? "—" : r.load;
      load.className = "load" + (!miss && Math.abs(r.load) > 400 ? " hot" : "");
    }
    if (cur) {
      cur.textContent = miss ? "—" : (r.cur || 0);
      cur.className = "cur" + (!miss && Math.abs(r.cur || 0) >= 5 ? " hot" : "");
    }
    fillWalls(j);
    tr.className = wallsOk(j) ? "ok" : "";
  });
}

function paintJoint() {
  if (!JOINT) return;
  const id = primary(JOINT);
  const r = G.rows[id];
  const miss = !r || r.miss;
  const d = liveDelta(id);
  const deg = $("degNow");
  if (deg) deg.textContent = d != null ? fmtDeg(d) + "°" : (miss ? "—" : String(r.pos));
  const posEl = $("posLive");
  const unEl = $("unLive");
  if (posEl) posEl.textContent = miss ? "—" : idsText(JOINT, "pos");
  if (unEl) {
    unEl.textContent = miss ? "—" : idsText(JOINT, "un");
    unEl.className = wrapish(JOINT) ? "wrap" : "";
  }
  const load = $("loadLive");
  if (load) {
    load.textContent = miss ? "—" : r.load;
    load.className = !miss && Math.abs(r.load) > 400 ? "hot" : "";
  }
  const cur = $("curLive");
  if (cur) {
    cur.textContent = miss ? "—" : (r.cur || 0);
    cur.className = !miss && Math.abs(r.cur || 0) >= 5 ? "hot" : "";
  }
  JOINT.ids.forEach(i => {
    const el = $("load-" + i);
    const rr = G.rows[i];
    if (el) el.textContent = !rr || rr.miss ? "—" : rr.load;
  });
  fillWalls(JOINT);
  recTick();
}

function loadPath() {
  try { return JSON.parse(localStorage.getItem(PATH_KEY) || "null"); } catch (e) { return null; }
}
function savePath(samples) {
  try { localStorage.setItem(PATH_KEY, JSON.stringify(samples)); } catch (e) {}
}
function paintRec() {
  const n = $("recN");
  const play = $("btnPlay");
  const btn = $("btnRec");
  const samples = rec ? rec.samples : (loadPath() || []);
  if (n) n.textContent = samples.length + " pts";
  if (play) play.disabled = playOn || samples.length < 2;
  if (btn) btn.textContent = rec ? "stop" : "rec";
}
function recTick() {
  if (!rec) return;
  const a = G.rows[12], b = G.rows[13];
  if (!a || a.miss || !b || b.miss) return;
  const last = rec.samples[rec.samples.length - 1];
  if (last && Math.abs(last.p12 - a.pos) < 2 && Math.abs(last.p13 - b.pos) < 2) {
    paintRec();
    return;
  }
  rec.samples.push({
    t: Date.now() - rec.t0,
    p12: a.pos, p13: b.pos,
    u12: a.un != null ? a.un : a.pos,
    u13: b.un != null ? b.un : b.pos,
    load12: a.load || 0, load13: b.load || 0,
  });
  if (rec.samples.length > 2500) rec.samples.shift();
  paintRec();
}
function toggleRec() {
  if (rec) {
    savePath(rec.samples);
    say("saved " + rec.samples.length + " pts — play to replay 12+13");
    rec = null;
    paintRec();
    return;
  }
  rec = { t0: Date.now(), samples: [] };
  recTick();
  say("rec shoulder — STOP, move by hand, click stop");
  paintRec();
}
async function playPath() {
  if (playOn) { playOn = false; say("play stop"); return; }
  const samples = (rec && rec.samples.length >= 2) ? rec.samples : (loadPath() || []);
  if (samples.length < 2) { say("rec a move first"); return; }
  if (pairSign() == null) { say("set pair first"); return; }
  if (!raiseReady()) { say("fw old — flash"); return; }
  playOn = true;
  const btn = $("btnPlay");
  if (btn) { btn.disabled = false; btn.textContent = "stop"; }
  spd(80);
  say("play " + samples.length + " pts  raise to");
  for (let i = 0; i < samples.length && playOn; i++) {
    if (G.guard && String(G.guard).startsWith("ESTOP")) break;
    const s = samples[i];
    await send(cmdRaiseTo(s.p12, s.p13));
    if (i + 1 < samples.length) {
      const dt = Math.min(250, Math.max(20, samples[i + 1].t - s.t));
      await new Promise(r => setTimeout(r, dt));
    }
  }
  playOn = false;
  if (btn) btn.textContent = "play";
  if (G.guard && String(G.guard).startsWith("ESTOP")) { say("play cut"); paintRec(); return; }
  sendRaiseTq(true);
  say("play done");
  paintRec();
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

function paint() {
  if (G.guard && String(G.guard).startsWith("ESTOP")) {
    if (test) test = null;
    if (playOn) playOn = false;
  }
  syncHome();
  if (SLUG) paintJoint();
  else paintIndex();
  paintPair();
  paintRec();
  if (test) tickTest();
}

if ($("btnGo")) $("btnGo").onclick = () => {
  const v = $("inGo") && $("inGo").value;
  if (v === "" || v == null) { say("type a degree"); return; }
  goDeg(+v).catch(e => say(String(e)));
};
if ($("btnPairMirror")) $("btnPairMirror").onclick = () => savePairSign(-1).then(paintPair);
if ($("btnPairSame")) $("btnPairSame").onclick = () => savePairSign(1).then(paintPair);
if ($("btnRec")) $("btnRec").onclick = () => toggleRec();
if ($("btnPlay")) $("btnPlay").onclick = () => playPath().catch(e => say(String(e)));

if (JOINT) bindJoint(JOINT);
else document.querySelectorAll("table.calIdx tr[data-slug]").forEach(tr => {
  const j = JOINTS.find(x => x.slug === tr.dataset.slug);
  if (j) bindJoint(j);
});

G.hooks.push(paint);
fetchLimits().then(m => {
  limits = m;
  syncPairFromLimits(m);
  syncHome();
  if (JOINT) fillWalls(JOINT);
  else document.querySelectorAll("table.calIdx tr[data-slug]").forEach(tr => {
    const j = JOINTS.find(x => x.slug === tr.dataset.slug);
    if (j) fillWalls(j);
  });
  paint();
});
setInterval(() => { if (test) tickTest(); }, 80);
setInterval(() => { if (G.alive && G.transport) send("get", true); }, 400);
paintRec();
