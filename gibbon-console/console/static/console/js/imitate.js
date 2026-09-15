const PATH_KEY = "gibbon.imitation.path";
const SENSE_KEY = "gibbon.imitation.sense";
const LOAD_SCALE = 1350;
const WRAP_ABORT = 1800;
const MAX_PTS = 2500;
const HOP = { 11: 80, 12: 80, 13: 80, 14: 80, 15: 80, 16: 900, 17: 0 };
const IDS = MOTORS.map(m => m.id);

let rec = null;
let playOn = false;
let tripped = 0;
let poll = 0;
let lastGuard = null;
const on = {};

JOINTS.forEach(j => { on[j.slug] = true; });

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
function live(id) {
  const r = G.rows[id];
  return r && !r.miss ? r : null;
}
function livePos(id) {
  const r = live(id);
  return r ? r.pos : null;
}
function hopMax(id) {
  return HOP[id] != null ? HOP[id] : 80;
}
function cmdTo(id, goal) {
  if (id === 17) return "hand to " + Math.round(goal);
  if (id === 16) return "twist to " + Math.round(goal);
  return "id " + id + " to " + Math.round(goal);
}
function toward(cur, goal, hop) {
  if (cur == null || !hop) return goal;
  const d = goal - cur;
  if (d > hop) return cur + hop;
  if (d < -hop) return cur - hop;
  return goal;
}
function senseN() {
  return +$("sense").value;
}
function loadPath() {
  try { return JSON.parse(localStorage.getItem(PATH_KEY) || "null"); } catch (e) { return null; }
}
function savePath(blob) {
  try { localStorage.setItem(PATH_KEY, JSON.stringify(blob)); } catch (e) {}
}
function samplesNow() {
  if (rec) return rec.samples;
  const blob = loadPath();
  return (blob && blob.samples) || [];
}
function idsWanted() {
  const ids = [];
  JOINTS.forEach(j => {
    if (!on[j.slug]) return;
    j.ids.forEach(id => ids.push(id));
  });
  return ids;
}
function say(s) {
  const el = $("imiSay");
  if (el) el.textContent = s;
  if (s) log(s);
}
function stopped() {
  return !!(G.guard && String(G.guard).startsWith("ESTOP"));
}

function maxLoad(ids) {
  let abs = 0, who = null;
  (ids || IDS).forEach(id => {
    const r = live(id);
    if (!r) return;
    const a = Math.abs(+r.load || 0);
    if (a >= abs) { abs = a; who = id; }
  });
  return { abs, who };
}

function paintMeter() {
  const trip = senseN();
  const { abs, who } = maxLoad(idsWanted());
  const fill = $("loadFill");
  const mark = $("loadTrip");
  if (fill) fill.style.width = Math.min(100, abs / LOAD_SCALE * 100) + "%";
  if (mark) {
    mark.style.display = trip > 0 ? "block" : "none";
    mark.style.left = Math.min(100, trip / LOAD_SCALE * 100) + "%";
  }
  const meter = $("loadMeter");
  if (meter) meter.classList.toggle("hot", trip > 0 && abs >= trip * 0.7);
  $("loadNow").textContent = String(abs);
  $("senseVal").textContent = trip === 0 ? "off" : String(trip);
  const hot = $("hotJoint");
  if (hot) {
    const m = MOTORS.find(x => x.id === who);
    hot.textContent = who == null ? "—" : ((m ? m.name : "") + " " + who);
  }
  const el = $("tripEl");
  const anyTq = idsWanted().some(id => {
    const r = live(id);
    return r && (r.tq === "1" || r.tq === 1);
  });
  if (tripped) {
    el.className = "guard bad";
    el.textContent = "STOP — |load| " + abs + " (limit " + trip + "). arm limp";
  } else if (playOn && trip > 0) {
    el.className = "guard on";
    el.textContent = "watching — STOP if |load| ≥ " + trip + " on any included joint";
  } else if (anyTq && trip > 0) {
    el.className = "guard on";
    el.textContent = "watching off play — torque is on";
  } else {
    el.className = "guard";
    el.textContent = trip === 0
      ? "page trip off — chip guard can still cut"
      : "watching off — rec with torque free, play to arm the trip";
  }
}

function paintRec() {
  const samples = samplesNow();
  const n = $("recN");
  const play = $("btnPlay");
  const recBtn = $("btnRec");
  const clr = $("btnClear");
  const dur = samples.length >= 2 ? ((samples[samples.length - 1].t - samples[0].t) / 1000).toFixed(1) : "0.0";
  if (n) n.textContent = samples.length + " pts  ·  " + dur + " s";
  if (play) {
    play.disabled = rec != null || (!playOn && samples.length < 2);
    play.textContent = playOn ? "stop" : "play";
  }
  if (recBtn) {
    recBtn.disabled = playOn;
    recBtn.textContent = rec ? "stop" : "rec";
  }
  if (clr) clr.disabled = rec != null || playOn || samples.length < 1;
  drawPath(samples);
}

function paintRows() {
  JOINTS.forEach(j => {
    const id = j.ids[0];
    const r = live(id);
    const miss = !r;
    const posEl = $("pos-" + j.slug);
    const loadEl = $("load-" + j.slug);
    const tr = $("row-" + j.slug);
    if (posEl) {
      posEl.textContent = miss ? "—" : (j.paired
        ? (livePos(12) + " / " + (livePos(13) == null ? "—" : livePos(13)))
        : String(r.pos));
    }
    if (loadEl) {
      const loads = j.ids.map(i => {
        const rr = live(i);
        return rr ? rr.load : "—";
      }).join(" / ");
      loadEl.textContent = loads;
      const hot = j.ids.some(i => {
        const rr = live(i);
        return rr && Math.abs(+rr.load || 0) > 400;
      });
      loadEl.className = hot ? "hot" : "";
    }
    if (tr) tr.className = miss || !G.alive ? "no" : "go";
  });
}

function drawPath(samples) {
  const c = $("imiPath");
  if (!c) return;
  const ids = idsWanted();
  if (samples.length < 2 || !ids.length) {
    c.hidden = true;
    return;
  }
  c.hidden = false;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 640, h = c.clientHeight || 96;
  c.width = w * dpr;
  c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const t0 = samples[0].t, t1 = samples[samples.length - 1].t || t0 + 1;
  const span = Math.max(1, t1 - t0);
  const pal = ["#7ec4c8", "#d45a8c", "#d4b06a", "#6ab0d4", "#c87ec4", "#8cd47e", "#d48c6a"];
  ids.forEach((id, k) => {
    g.strokeStyle = pal[k % pal.length];
    g.globalAlpha = 0.85;
    g.lineWidth = 1.2;
    g.beginPath();
    let n = 0;
    samples.forEach(s => {
      const p = s.p && s.p[id];
      if (p == null) return;
      const x = (s.t - t0) / span * w;
      const y = h - (p / 4095) * (h - 6) - 3;
      if (!n) g.moveTo(x, y); else g.lineTo(x, y);
      n++;
    });
    if (n > 1) g.stroke();
  });
}

function snapshot() {
  const p = {}, u = {}, load = {};
  let n = 0;
  idsWanted().forEach(id => {
    const r = live(id);
    if (!r) return;
    p[id] = r.pos;
    u[id] = r.un != null ? r.un : r.pos;
    load[id] = r.load || 0;
    n++;
  });
  return n ? { p, u, load } : null;
}

function recTick() {
  if (!rec) return;
  const snap = snapshot();
  if (!snap) { paintRec(); return; }
  const last = rec.samples[rec.samples.length - 1];
  if (last) {
    let moved = 0;
    Object.keys(snap.p).forEach(id => {
      if (Math.abs(snap.p[id] - (last.p[id] || 0)) >= 2) moved = 1;
      if (last.p[id] != null && Math.abs(snap.p[id] - last.p[id]) > WRAP_ABORT) {
        rec.wrap = rec.wrap || [];
        if (rec.wrap.indexOf(+id) < 0) rec.wrap.push(+id);
      }
    });
    if (!moved) { paintRec(); return; }
  }
  rec.samples.push({ t: Date.now() - rec.t0, p: snap.p, u: snap.u, load: snap.load });
  if (rec.samples.length > MAX_PTS) rec.samples.shift();
  paintRec();
}

async function toggleRec() {
  if (playOn) { say("stop play first"); return; }
  if (rec) {
    const wrap = rec.wrap || [];
    savePath({ t: Date.now(), samples: rec.samples, wrap });
    let msg = "saved " + rec.samples.length + " pts — play to replay";
    if (wrap.length) msg += " — wrap on " + wrap.join(",") + ", play cannot cross 0/4095";
    say(msg);
    rec = null;
    paintRec();
    return;
  }
  if (!G.transport) { say("connect first"); return; }
  if (!idsWanted().length) { say("include a joint"); return; }
  tripped = 0;
  await send("estop");
  rec = { t0: Date.now(), samples: [], wrap: [] };
  recTick();
  say("rec — arm limp, support it, move by hand, click stop");
  paintRec();
}

function applyGuard(n) {
  if (!G.transport || n <= 0) return;
  const key = n + "/" + n;
  if (key === lastGuard) return;
  lastGuard = key;
  send("guard load " + n);
  send("guard dload " + n);
}

function forceTrip(why) {
  if (tripped) return;
  tripped = 1;
  playOn = false;
  say("STOP — " + why);
  send("estop");
}

async function hopWalk(goal) {
  for (let k = 0; k < 48 && playOn; k++) {
    if (stopped()) return "estop";
    const cmds = [];
    let more = false;
    const p12 = goal[12], p13 = goal[13];
    if (p12 != null && p13 != null) {
      const c12 = livePos(12), c13 = livePos(13);
      if (wrapAbort(c12, p12) || wrapAbort(c13, p13)) return "wrap 12+13";
      const n12 = toward(c12, p12, hopMax(12));
      const n13 = toward(c13, p13, hopMax(13));
      if (n12 !== p12 || n13 !== p13) more = true;
      cmds.push(cmdRaiseTo(n12, n13));
    } else if (p12 != null || p13 != null) {
      return "raise needs both 12 and 13";
    }
    const rest = [11, 14, 15, 16, 17];
    for (let i = 0; i < rest.length; i++) {
      const id = rest[i];
      if (goal[id] == null) continue;
      const cur = livePos(id);
      if (wrapAbort(cur, goal[id])) return "wrap " + id;
      const n = toward(cur, goal[id], hopMax(id));
      if (n !== goal[id]) more = true;
      cmds.push(cmdTo(id, n));
    }
    for (let i = 0; i < cmds.length; i++) await send(cmds[i], true);
    if (!more) return null;
    await sleep(Math.max(50, cmds.length * 30));
  }
  return playOn ? "hop timeout" : "stop";
}

function wrapAbort(cur, goal) {
  return cur != null && Math.abs(goal - cur) > WRAP_ABORT;
}

async function playPath() {
  if (playOn) { playOn = false; say("play stop"); paintRec(); return; }
  if (rec) { say("stop rec first"); return; }
  const samples = samplesNow();
  if (samples.length < 2) { say("rec a move first"); return; }
  if (!G.transport) { say("connect first"); return; }
  const needRaise = samples.some(s => s.p && (s.p[12] != null || s.p[13] != null));
  if (needRaise && !raiseReady()) { say("fw old — flash before 12+13 play"); return; }
  tripped = 0;
  G.guard = "";
  playOn = true;
  lastGuard = null;
  await send("get", true);
  applyGuard(senseN());
  send("move speed 80");
  send("move acc 8");
  paintRec();
  say("play " + samples.length + " pts");
  const prog = $("playN");
  for (let i = 0; i < samples.length && playOn; i++) {
    if (stopped()) break;
    const s = samples[i];
    if (prog) prog.textContent = (i + 1) + " / " + samples.length;
    const goal = {};
    idsWanted().forEach(id => {
      if (s.p && s.p[id] != null) goal[id] = s.p[id];
    });
    const err = await hopWalk(goal);
    if (err) {
      if (err !== "stop" && err !== "estop") say("play cut — " + err);
      break;
    }
    if (i + 1 < samples.length) {
      const dt = Math.min(400, Math.max(20, samples[i + 1].t - s.t));
      await sleep(dt);
    }
  }
  playOn = false;
  if (prog) prog.textContent = "";
  paintRec();
  if (tripped) return;
  if (stopped()) { say("play cut"); return; }
  lockPose();
  say("play done — held");
}

function checkForce(sample) {
  if (!playOn || tripped) return;
  const trip = senseN();
  if (trip <= 0) return;
  const id = sample && sample.id;
  if (id && idsWanted().indexOf(id) < 0) return;
  const a = Math.abs(+(sample && sample.load) || 0);
  if (a >= trip) forceTrip("id " + id + " load " + sample.load + " ≥ " + trip);
}

function build() {
  $("imiBody").innerHTML = JOINTS.map(j => `<tr id="row-${j.slug}" class="no">
    <td class="pick"><input type="checkbox" id="on-${j.slug}" ${on[j.slug] ? "checked" : ""}></td>
    <td>${j.name}${j.paired ? ' <span class="axis">both</span>' : ""}</td>
    <td id="pos-${j.slug}">—</td>
    <td id="load-${j.slug}">—</td>
  </tr>`).join("");
  JOINTS.forEach(j => {
    const el = $("on-" + j.slug);
    if (!el) return;
    el.onchange = () => {
      on[j.slug] = el.checked;
      if (j.paired) on[j.slug] = el.checked;
      paintRec();
    };
  });
  try {
    const s = localStorage.getItem(SENSE_KEY);
    if (s != null && +s >= 0) $("sense").value = s;
  } catch (e) {}
}

function paint() {
  if (stopped() && playOn) playOn = false;
  paintRows();
  paintMeter();
  paintRec();
}

$("btnRec").onclick = () => toggleRec().catch(e => say(String(e)));
$("btnPlay").onclick = () => playPath().catch(e => say(String(e)));
$("btnClear").onclick = () => {
  if (rec || playOn) return;
  try { localStorage.removeItem(PATH_KEY); } catch (e) {}
  say("cleared");
  paintRec();
};
$("sense").oninput = () => {
  try { localStorage.setItem(SENSE_KEY, String(senseN())); } catch (e) {}
  if (playOn) applyGuard(senseN());
  paintMeter();
};

build();
G.hooks.push((ev, sample) => {
  if (ev === "sample") {
    recTick();
    checkForce(sample);
  }
  if (ev === "guard" && stopped() && playOn) {
    playOn = false;
    say("play cut");
  }
  paint();
});
setInterval(() => {
  if (!G.alive || !G.transport || poll) return;
  if (!rec && !playOn) return;
  poll = 1;
  send("get", true).finally(() => { poll = 0; });
}, 200);
paint();
