let limits = {};
let built = 0;
let pollI = 0;
let pending = null;
let pendingT = 0;
let test = null;
let hopBusy = 0;
let lastSpd = null;
const homeUn = {};
const drag = {};
const gripUntil = {};
const gripTimer = {};
const HAND = (window.CATALOG && window.CATALOG.hand) || { open: 689, to_max: 2075 };
const CONFIRM_MS = 4000;
const NEAR = 40;
const HOP_MS = 14000;
const ENC = 4096; /* 360° / 4096. pos 0–4095 wraps at the magnet, not the cable */
const KEYS = { home: "here", amin: "−", amax: "+" };
let learn = null;
const wrapSess = {};
const CAL_SLUG = window.JOINT_SLUG || "";
const CAL_INDEX = !CAL_SLUG;
const pairVote = { last12: null, last13: null, mirror: 0, same: 0 };

function calJoints() {
  return CAL_SLUG ? JOINTS.filter(j => j.slug === CAL_SLUG) : JOINTS;
}

function isWrap(id) {
  return id === 11 || id === 16;
}

function wrapHop(id) {
  if (id === 16) return 900;
  if (id === 17) return 400;
  return 80;
}

function livePos(id) {
  const r = G.rows[id];
  return r && !r.miss ? r.pos : null;
}

function liveUn(id) {
  if (isWrap(id) && wrapSess[id] && wrapSess[id].un != null) return wrapSess[id].un;
  const r = G.rows[id];
  if (!r || r.miss) return null;
  return r.un != null ? r.un : r.pos;
}

function liveTurns(id) {
  if (isWrap(id) && wrapSess[id] && wrapSess[id].turns != null) return wrapSess[id].turns;
  const r = G.rows[id];
  if (!r || r.miss) return null;
  return r.turns != null ? r.turns : 0;
}

function keyLabel(_id, key) {
  return KEYS[key];
}

function fmtDelta(d) {
  if (d == null || d === "") return "—";
  const n = +d;
  if (!isFinite(n)) return "—";
  return (n > 0 ? "+" : "") + n;
}

function degOf(d) {
  return (+d) * 360 / ENC; /* 360° / 4096 */
}

function fmtDeg(d) {
  if (d == null || d === "" || !isFinite(+d)) return "—";
  const n = degOf(d);
  return (n > 0 ? "+" : "") + n.toFixed(1) + "°";
}

function liveDelta(id) {
  const un = liveUn(id);
  if (un == null || homeUn[id] == null) return null;
  return un - homeUn[id];
}

function tickWrapSess(id) {
  if (!isWrap(id)) return;
  const pos = livePos(id);
  if (pos == null) return;
  let s = wrapSess[id];
  if (!s) {
    wrapSess[id] = { lastPos: pos, n: 0, last: "", turns: 0, un: pos };
    return;
  }
  if (s.lastPos != null && Math.abs(pos - s.lastPos) > ENC / 2) {
    if (pos < s.lastPos) s.turns++;
    else s.turns--;
    s.n++;
    s.last = s.lastPos + "→" + pos;
  }
  s.lastPos = pos;
  s.un = pos + s.turns * ENC;
}

function resetWrapSess(id) {
  const pos = livePos(id);
  wrapSess[id] = { lastPos: pos, n: 0, last: "", turns: 0, un: pos };
}

function paintLearnBtn(id) {
  const b = $("learn-" + id);
  const k = $("keep-" + id);
  const on = !!(learn && learn.id === id);
  if (b) {
    b.textContent = "calibrate";
    b.classList.toggle("on", on);
    b.disabled = on;
  }
  if (k) {
    k.hidden = !on;
    k.disabled = !on;
    k.classList.toggle("on", on);
  }
  const card = $("cal-" + id);
  if (card) card.classList.toggle("learning", on);
}

function setCalStream(on) {
  send(on ? "stream 180" : "stream 180", true);
}

function startLearn(id) {
  if (learn && learn.id === id) return;
  if (learn) {
    setCalStream(false);
    learn = null;
  }
  if (!hasHere([id])) {
    say("slack the loom, then here — now is 0°. then calibrate and move both ways");
    return;
  }
  const pos = livePos(id);
  const un = liveUn(id);
  const d = liveDelta(id);
  if (pos == null || un == null || d == null) {
    say("no live pos on id " + id + " — pack on");
    return;
  }
  if (test) stopTest("stopped");
  setTq([id], false);
  setCalStream(true);
  learn = {
    id,
    minD: d,
    maxD: d,
    minUn: un,
    maxUn: un,
  };
  paintLearnBtn(id);
  paintRange(id);
  say("calibrating — move both ways. − and + should grow. then done");
}

function tickLearn() {
  if (!learn) return;
  const id = learn.id;
  const un = liveUn(id);
  const d = liveDelta(id);
  if (un == null || d == null) return;
  if (un < learn.minUn) learn.minUn = un;
  if (un > learn.maxUn) learn.maxUn = un;
  if (d < learn.minD) learn.minD = d;
  if (d > learn.maxD) learn.maxD = d;
  fillTyped(id, "amin", learn.minD);
  fillTyped(id, "amax", learn.maxD);
}

async function keepLearn() {
  if (!learn) {
    say("press calibrate first, then move, then done");
    return;
  }
  const id = learn.id;
  const span = learn.maxD - learn.minD;
  if (span < 80) {
    say("range is only " + fmtDeg(span) + " — move more, then done. now "
      + fmtDeg(learn.minD) + " … " + fmtDeg(learn.maxD));
    return;
  }
  const minD = learn.minD;
  const maxD = learn.maxD;
  learn = null;
  setCalStream(false);
  paintLearnBtn(id);
  await commitMark(id, "amin", { [id]: minD }, true);
  await commitMark(id, "amax", { [id]: maxD }, true);
  say("kept " + fmtDeg(minD) + " … " + fmtDeg(maxD)
    + "  (" + fmtDelta(minD) + " … " + fmtDelta(maxD) + "). slider is that range");
}

function paintRange(id) {
  const L = limits[id] || {};
  let lo = L.amin != null && !isLegacyRaw(L) ? +L.amin : null;
  let hi = L.amax != null && !isLegacyRaw(L) ? +L.amax : null;
  if (learn && learn.id === id) {
    lo = learn.minD;
    hi = learn.maxD;
  }
  const loEl = $("wallLo-" + id);
  const hiEl = $("wallHi-" + id);
  const loCnt = $("wallLoCnt-" + id);
  const hiCnt = $("wallHiCnt-" + id);
  if (loEl) loEl.textContent = fmtDeg(lo);
  if (hiEl) hiEl.textContent = fmtDeg(hi);
  if (loCnt) loCnt.textContent = fmtDelta(lo);
  if (hiCnt) hiCnt.textContent = fmtDelta(hi);
}

function hasHere(ids) {
  return ids.every(i => homeUn[i] != null);
}

function isLegacyRaw(L) {
  if (!L || L.amin == null || L.amax == null) return false;
  const a = +L.amin;
  const b = +L.amax;
  if (a < 0 || b < 0) return false;
  return a > 200 && b > 200 && a < 4096 && b < 4096 && Math.abs(a - b) < 80;
}

function usesDelta(id) {
  const L = limits[id] || {};
  if (L.amin != null && +L.amin < 0) return true;
  if (L.amax != null && +L.amax < 0) return true;
  if (homeUn[id] != null && (L.amin != null || L.amax != null) && !isLegacyRaw(L)) return true;
  return false;
}

function syncHomeUn() {
  MOTORS.forEach(m => {
    const id = m.id;
    if (homeUn[id] != null) return;
    const L = limits[id] || {};
    if (L.home == null) return;
    homeUn[id] = +L.home;
  });
}

function cmdFree(id) {
  if (id === 17) return "hand free";
  if (id === 16) return "twist free";
  return "id " + id + " free";
}

function cmdHold(id) {
  return "id " + id + " hold";
}

function cmdHere(id) {
  return id === 16 ? "twist here" : ("id " + id + " here");
}

function cmdTo(id, goal) {
  if (id === 17) return "hand to " + goal;
  if (id === 16) return "twist to " + goal;
  return "id " + id + " to " + goal;
}

function jointFor(id) {
  return JOINTS.find(j => j.ids.indexOf(id) >= 0);
}

function mates(id) {
  const j = jointFor(id);
  return j ? j.ids : [id];
}

function tqOn(id) {
  return !!(G.rows[id] && G.rows[id].tq === "1");
}

function namesOf(ids) {
  return ids.map(id => {
    const m = MOTORS.find(x => x.id === id);
    return m ? m.name : id;
  }).join("+");
}

function spdWord(n) {
  if (n < 90) return "crawl";
  if (n < 180) return "slow";
  if (n < 320) return "walk";
  return "brisk";
}

function spdNow() {
  const el = $("calSpd");
  return el ? Math.max(40, Math.min(480, +el.value || 120)) : 120;
}

function paintSpd() {
  const n = spdNow();
  const el = $("calSpdVal");
  if (el) el.textContent = n + " " + spdWord(n);
}

function applySpeed() {
  const n = spdNow();
  const acc = n < 150 ? 8 : n < 280 ? 16 : 24;
  const key = n + "/" + acc;
  if (key === lastSpd) return;
  lastSpd = key;
  send("move speed " + n);
  send("move acc " + acc);
}

function say(s) {
  const el = $("calMsg");
  if (el) el.textContent = s || "";
  if (s) log(s);
}

function wallsOf(id) {
  const L = limits[id] || {};
  if (L.amin == null || L.amax == null) return null;
  const a = +L.amin;
  const b = +L.amax;
  if (usesDelta(id)) {
    return { lo: Math.min(a, b), hi: Math.max(a, b), home: 0, delta: true };
  }
  if (isWrap(id)) return null;
  return {
    lo: Math.min(a, b),
    hi: Math.max(a, b),
    home: L.home != null ? +L.home : null,
    delta: false,
  };
}

function chipWall(id, key) {
  const L = limits[id] || {};
  if (key === "home") return L.home != null ? +L.home : null;
  const v = key === "amin" ? L.amin : L.amax;
  if (v == null) return null;
  if (isWrap(id)) return null;
  if (usesDelta(id) || +v < 0) {
    const origin = L.home != null ? +L.home : homeUn[id];
    if (origin == null) return null;
    return Math.max(0, Math.min(4095, origin + +v));
  }
  return +v;
}

function canTest(id) {
  if (isWrap(id)) return false;
  if ((id === 12 || mates(id).indexOf(12) >= 0) && pairSign() == null) return false;
  return mates(id).every(i => {
    const w = wallsOf(i);
    if (!w) return false;
    if (w.delta) return homeUn[i] != null;
    return true;
  });
}

function showMark(id, key) {
  const L = limits[id] || {};
  const v = L[key];
  if (v == null) return "—";
  if (key === "home") return String(v);
  if (usesDelta(id) || +v < 0) return fmtDeg(v) + "  " + fmtDelta(v);
  if (isWrap(id)) return "raw — re-mark";
  return String(v);
}

function motorName(id) {
  const m = MOTORS.find(x => x.id === id);
  return m ? m.name : ("id " + id);
}

function fillTyped(i, key, val) {
  const el = $("in-" + key + "-" + i);
  if (!el || document.activeElement === el) return;
  el.value = val == null || val === "" ? "" : val;
}

function fillMarks() {
  calJoints().forEach(j => {
    const id = j.ids[0];
    const h = $("mark-home-" + id);
    const n = $("mark-min-" + id);
    const x = $("mark-max-" + id);
    const show = (key) => j.ids.map(i => showMark(i, key)).join(" · ");
    if (h) h.textContent = show("home");
    if (n) n.textContent = show("amin");
    if (x) x.textContent = show("amax");
    j.ids.forEach(i => {
      const L = limits[i] || {};
      fillTyped(i, "home", L.home);
      if (!(learn && learn.id === i)) {
        fillTyped(i, "amin", L.amin);
        fillTyped(i, "amax", L.amax);
      }
    });
  });
}

function slideBand(id) {
  const hop = wrapHop(id);
  if (homeUn[id] != null) {
    const d = liveDelta(id);
    if (learn && learn.id === id) {
      const lo = Math.min(learn.minD, d == null ? 0 : d, 0);
      const hi = Math.max(learn.maxD, d == null ? 0 : d, 0);
      if (hi > lo) return { lo, hi, delta: true };
    }
    const w = wallsOf(id);
    if (w && w.delta) return { lo: w.lo, hi: w.hi, delta: true };
    if (d == null) return { lo: -hop, hi: hop, delta: true };
    return { lo: d - hop, hi: d + hop, delta: true };
  }
  if (id === 17) return { lo: HAND.open, hi: HAND.to_max, delta: false };
  const pos = livePos(id);
  const w = wallsOf(id);
  if (w && !w.delta) return { lo: w.lo, hi: w.hi, delta: false };
  if (pos == null) return { lo: 0, hi: 4095, delta: false };
  return { lo: pos - hop, hi: pos + hop, delta: false };
}

function paintSlide(id) {
  const sl = $("slide-" + id);
  if (!sl) return;
  sl.disabled = !!(learn && learn.id === id) || (isWrap(id) && homeUn[id] == null);
  const { lo, hi, delta } = slideBand(id);
  const dragging = mates(id).some(i => drag[i] || (gripUntil[i] && Date.now() < gripUntil[i]));
  if (!dragging) {
    sl.min = lo;
    sl.max = hi;
    sl.dataset.delta = delta ? "1" : "0";
    const val = delta ? liveDelta(id) : livePos(id);
    if (val != null) sl.value = Math.max(+sl.min, Math.min(+sl.max, val));
  }
  const loL = $("slideLo-" + id);
  const hiL = $("slideHi-" + id);
  const lab = $("slideLab-" + id);
  const shown = dragging ? +sl.value : (delta ? liveDelta(id) : livePos(id));
  if (loL) loL.textContent = delta ? fmtDeg(lo) : String(lo);
  if (hiL) hiL.textContent = delta ? fmtDeg(hi) : String(hi);
  if (lab) lab.textContent = delta ? fmtDeg(shown) : (shown == null ? "" : String(shown));
}

function bindSlide(id) {
  const g = $("slide-" + id);
  if (!g) return;
  const end = () => {
    if (!drag[id]) return;
    clearTimeout(gripTimer[id]);
    sendSlide(id);
    drag[id] = 0;
  };
  g.addEventListener("pointerdown", e => {
    drag[id] = 1;
    try { g.setPointerCapture(e.pointerId); } catch {}
  });
  g.addEventListener("input", () => {
    drag[id] = 1;
    clearTimeout(gripTimer[id]);
    gripTimer[id] = setTimeout(() => sendSlide(id), 60);
  });
  g.addEventListener("pointerup", end);
  g.addEventListener("pointercancel", () => { drag[id] = 0; clearTimeout(gripTimer[id]); });
}

function sendSlide(id) {
  const g = $("slide-" + id);
  if (!g || !G.alive) return;
  if (test) stopTest("stopped");
  const goal = Math.round(+g.value);
  const ids = mates(id);
  ids.forEach(i => { gripUntil[i] = Date.now() + 2000; });
  applySpeed();
  if (id === 16 && G.rows[16] && String(G.rows[16].mode) === "1") send("twist servo");

  if (homeUn[id] != null) {
    const curD = liveDelta(id);
    const pos = livePos(id);
    if (curD == null || pos == null) return;
    if (id === 17) {
      send(cmdTo(17, Math.round(homeUn[17] + goal)));
      return;
    }
    const hop = wrapHop(id);
    let d = Math.max(-hop, Math.min(hop, goal - curD));
    if (!d) return;
    driveDelta(ids, d);
    return;
  }

  if (id === 17) {
    send(cmdTo(17, goal));
    return;
  }
  const pos = livePos(id);
  if (pos == null) return;
  const hop = wrapHop(id);
  const d = Math.max(-hop, Math.min(hop, goal - pos));
  if (!d) return;
  driveDelta(ids, d);
}

function driveDelta(ids, d) {
  if (ids[0] === 12 || ids.indexOf(12) >= 0 || ids.indexOf(13) >= 0) {
    const sign = pairSign();
    const p12 = livePos(12);
    const p13 = livePos(13);
    if (p12 == null || p13 == null) return;
    if (sign == null) {
      say("set pair first (mirror or same). do not jog 12+13 until then — they will fight");
      return;
    }
    let n12 = p12 + d;
    let n13 = p13 + sign * d;
    if (n12 < 0) n12 = 0;
    if (n12 > 4095) n12 = 4095;
    if (n13 < 0) n13 = 0;
    if (n13 > 4095) n13 = 4095;
    if (n12 === p12 && n13 === p13) {
      say("encoder seam — free and turn through 0/4095, then hold");
      return;
    }
    send(cmdRaiseTo(n12, n13));
    return;
  }
  const id = ids[0];
  const pos = livePos(id);
  if (pos == null) return;
  const w = wallsOf(id);
  const curD = liveDelta(id);
  if (w && w.delta && curD != null) {
    let nextD = curD + d;
    if (nextD < w.lo) d = w.lo - curD;
    if (nextD > w.hi) d = w.hi - curD;
    if (!d) {
      say("at wall " + fmtDeg(curD));
      return;
    }
  }
  let next = pos + d;
  if (next < 0) next = 0;
  if (next > 4095) next = 4095;
  if (next === pos) {
    say("encoder seam — free and turn through 0/4095, then hold. servo mode cannot wrap");
    return;
  }
  const step = next - pos;
  ids.forEach(i => {
    const pi = livePos(i);
    if (pi == null) return;
    let ni = pi + step;
    if (ni < 0) ni = 0;
    if (ni > 4095) ni = 4095;
    send(cmdTo(i, ni));
  });
}

function paintTq() {
  calJoints().forEach(j => {
    const id = j.ids[0];
    const b = $("tq-" + id);
    const card = $("cal-" + id);
    if (!b) return;
    const on = j.ids.some(tqOn);
    b.textContent = on ? "free this" : "hold this";
    b.className = "lock" + (on ? " bad" : "");
    if (!card) return;
    card.classList.toggle("loose", j.ids.some(i => !tqOn(i) && G.rows[i] && !G.rows[i].miss));
    card.classList.toggle("testing", !!(test && test.ids.some(i => j.ids.indexOf(i) >= 0)));
  });
}

function paintSense() {
  calJoints().forEach(j => {
    j.ids.forEach(i => {
      const load = $("load-" + i);
      const cur = $("cur-" + i);
      const r = G.rows[i];
      const miss = !r || r.miss;
      if (load) {
        load.textContent = miss ? "—" : r.load;
        load.className = !miss && Math.abs(r.load) > 400 ? "hot" : "";
      }
      if (cur) {
        cur.textContent = miss ? "—" : (r.cur || 0);
        cur.className = !miss && Math.abs(r.cur) >= 5 ? "hot" : "";
      }
    });
  });
}

function paintTestBtns() {
  calJoints().forEach(j => {
    const id = j.ids[0];
    const b = $("test-" + id);
    if (!b || b.classList.contains("wait") || b.classList.contains("ok")) return;
    const running = !!(test && test.ids.some(i => j.ids.indexOf(i) >= 0));
    const ready = canTest(id);
    b.disabled = !running && !ready;
    b.textContent = running ? "stop" : "test";
    b.className = running ? "stop" : "";
    if (isWrap(id)) b.title = "yaw and wrist roll cannot be swept through 0/4095 yet";
    else b.title = ready || running ? "" : "mark here, then − and + first";
  });
}

function resetMarkBtns() {
  calJoints().forEach(j => {
    const id = j.ids[0];
    Object.keys(KEYS).forEach(k => {
      const b = document.querySelector(`#cal-${id} button[data-k="${k}"]:not([data-act="set"])`);
      if (b && !b.classList.contains("ok")) {
        b.textContent = keyLabel(id, k);
        b.classList.remove("wait");
      }
    });
    const t = $("test-" + id);
    if (t && t.classList.contains("wait")) {
      t.classList.remove("wait");
      t.textContent = "test";
    }
  });
}

function pairLabel(s) {
  if (s === -1) return "mirror −  (13 opposite 12)";
  if (s === 1) return "same +  (13 with 12)";
  return "unset — do not jog";
}

function paintPair() {
  const el = $("pairState");
  if (!el) return;
  const s = pairSign();
  el.textContent = pairLabel(s);
  el.className = s == null ? "bad" : "ok";
  const m = $("btnPairMirror");
  const a = $("btnPairSame");
  if (m) m.classList.toggle("on", s === -1);
  if (a) a.classList.toggle("on", s === 1);
}

function watchPair() {
  const p12 = livePos(12);
  const p13 = livePos(13);
  if (p12 == null || p13 == null) return;
  if (pairVote.last12 == null) {
    pairVote.last12 = p12;
    pairVote.last13 = p13;
    return;
  }
  const d12 = p12 - pairVote.last12;
  const d13 = p13 - pairVote.last13;
  pairVote.last12 = p12;
  pairVote.last13 = p13;
  if (Math.abs(d12) < 18 || Math.abs(d13) < 18) return;
  if (d12 * d13 < 0) pairVote.mirror++;
  else pairVote.same++;
  if (pairSign() != null) return;
  if (pairVote.mirror >= 3 && pairVote.mirror > pairVote.same + 1) {
    savePairSign(-1);
    say("learned pair: mirror — 12 and 13 run opposite. click mirror if that matches what you felt");
    paintPair();
  } else if (pairVote.same >= 3 && pairVote.same > pairVote.mirror + 1) {
    savePairSign(1);
    say("learned pair: same direction. click same if that matches what you felt");
    paintPair();
  }
}

function build() {
  const body = $("calBody");
  if (!body) return;
  if (CAL_INDEX) {
    body.innerHTML = calJoints().map(j => {
      const id = j.ids[0];
      const pair = j.paired ? ' <span class="axis">both</span>' : "";
      return `<tr id="cal-${id}" data-slug="${j.slug}">
        <td class="goName"><a class="name" href="/calibrate/${j.slug}/">${j.name}</a>${pair}</td>
        <td>${j.ids.join("+")}</td>
        <td class="calPos" id="live-${id}">—</td>
        <td class="calDelta" id="delta-${id}">—</td>
        <td id="load-${id}">—</td>
        <td id="mark-home-${id}">—</td>
        <td id="mark-min-${id}">—</td>
        <td id="mark-max-${id}">—</td>
      </tr>`;
    }).join("");
    built = 1;
    fillMarks();
    return;
  }
  body.innerHTML = calJoints().map(j => wrapPage(j)).join("");
  if (calJoints().some(j => isWrap(j.ids[0]))) document.body.classList.add("wrapCal");
  built = 1;
  fillMarks();
  paintTq();
  paintTestBtns();
  paintPair();
  calJoints().forEach(j => {
    bindSlide(j.ids[0]);
    paintSlide(j.ids[0]);
  });
}

function wrapPage(j) {
  const id = j.ids[0];
  if (isWrap(id)) {
    const typed = (key, ph) => `<label class="calIn"><input id="in-${key}-${id}" type="number" step="1" placeholder="${ph}"></label>`;
    return `<div id="cal-${id}" class="calWrap">
      <div class="calTrio">
        <div>
          <span class="k">−</span>
          <b id="wallLo-${id}">—</b>
          <i id="wallLoCnt-${id}"></i>
        </div>
        <div class="now">
          <span class="k">now</span>
          <b id="deg-${id}">—</b>
          <i id="delta-${id}">—</i>
        </div>
        <div>
          <span class="k">+</span>
          <b id="wallHi-${id}">—</b>
          <i id="wallHiCnt-${id}"></i>
        </div>
      </div>
      <div class="calJog">
        <label>range <span id="slideLab-${id}"></span></label>
        <input id="slide-${id}" type="range" min="-2048" max="2048" value="0" data-id="${id}">
        <div class="calTrackLabs">
          <span id="slideLo-${id}">−</span>
          <span>0°</span>
          <span id="slideHi-${id}">+</span>
        </div>
      </div>
      <div class="calQuiet">
        <span>pos <i id="live-${id}">—</i> <em>ignore the wrap</em></span>
        <span>un <i id="un-${id}">—</i></span>
        <span>turns <i id="turns-${id}">—</i></span>
        <span class="dim">load <i id="load-${id}">—</i></span>
        <span class="dim">mA <i id="cur-${id}">—</i></span>
      </div>
      <p class="calNote" id="wrapMeta-${id}"></p>
      <div class="calWrapRow calMarks">
        <div class="calMark">
          <span class="k">home · slack</span>
          <span class="val" id="mark-home-${id}">—</span>
          ${typed("home", "pos")}
          <div class="rowBtns">
            <button type="button" data-id="${id}" data-k="home">here</button>
            <button type="button" data-id="${id}" data-act="set" data-k="home">set</button>
          </div>
        </div>
        <div class="calMark">
          <span class="k">− wall</span>
          <span class="val" id="mark-min-${id}">—</span>
          ${typed("amin", "Δ")}
          <div class="rowBtns">
            <button type="button" data-id="${id}" data-k="amin">−</button>
            <button type="button" data-id="${id}" data-act="set" data-k="amin">set</button>
          </div>
        </div>
        <div class="calMark">
          <span class="k">+ wall</span>
          <span class="val" id="mark-max-${id}">—</span>
          ${typed("amax", "Δ")}
          <div class="rowBtns">
            <button type="button" data-id="${id}" data-k="amax">+</button>
            <button type="button" data-id="${id}" data-act="set" data-k="amax">set</button>
          </div>
        </div>
      </div>
      <div class="calActs">
        <button type="button" class="go" data-id="${id}" data-act="learn" id="learn-${id}">calibrate</button>
        <button type="button" class="go" data-id="${id}" data-act="keep" id="keep-${id}" hidden>done</button>
        <button type="button" class="lock" id="tq-${id}" data-act="tq" data-id="${id}">hold this</button>
      </div>
    </div>`;
  }
  const motors = j.ids.map(i => `<article class="calMotor">
    <h2>${motorName(i)} <i>${i}</i></h2>
    <div class="calNums">
      <span><b>pos</b><i id="live-${i}">—</i></span>
      <span><b>Δ</b><i id="delta-${i}">—</i></span>
      <span class="dim"><b>load</b><i id="load-${i}">—</i></span>
      <span class="dim"><b>mA</b><i id="cur-${i}">—</i></span>
    </div>
  </article>`).join("");
  const typed = (key, ph) => j.ids.map(i => `<label class="calIn">${j.ids.length > 1 ? i : ""}<input id="in-${key}-${i}" type="number" step="1" placeholder="${ph}"></label>`).join("");
  return `<div id="cal-${id}">
    <div class="calLive">${motors}</div>
    <p class="calNote" id="wrapMeta-${id}"></p>
    <div class="calMarksGrid calMarks">
      <div class="calMark">
        <span class="k">home · raw pos</span>
        <span class="val" id="mark-home-${id}">—</span>
        ${typed("home", "0–4095")}
        <div class="rowBtns">
          <button type="button" data-id="${id}" data-k="home">here</button>
          <button type="button" data-id="${id}" data-act="set" data-k="home">set</button>
        </div>
      </div>
      <div class="calMark">
        <span class="k">− · Δ counts fall</span>
        <span class="val" id="mark-min-${id}">—</span>
        ${typed("amin", "Δ")}
        <div class="rowBtns">
          <button type="button" data-id="${id}" data-k="amin">−</button>
          <button type="button" data-id="${id}" data-act="set" data-k="amin">set</button>
        </div>
      </div>
      <div class="calMark">
        <span class="k">+ · Δ counts rise</span>
        <span class="val" id="mark-max-${id}">—</span>
        ${typed("amax", "Δ")}
        <div class="rowBtns">
          <button type="button" data-id="${id}" data-k="amax">+</button>
          <button type="button" data-id="${id}" data-act="set" data-k="amax">set</button>
        </div>
      </div>
    </div>
    <div class="calJog">
      <label>jog Δ</label>
      <input id="slide-${id}" type="range" min="0" max="4095" value="0" data-id="${id}">
    </div>
    <div class="calActs">
      <button type="button" class="go" data-id="${id}" data-act="learn" id="learn-${id}">calibrate</button>
      <button type="button" class="go" data-id="${id}" data-act="keep" id="keep-${id}" hidden>done</button>
      <button type="button" class="lock" id="tq-${id}" data-act="tq" data-id="${id}">hold this</button>
      <button type="button" data-id="${id}" data-act="test" id="test-${id}" disabled title="mark here, then − and + first">test</button>
    </div>
  </div>`;
}

function paintLive() {
  if (!built) build();
  syncHomeUn();
  calJoints().forEach(j => {
    const id = j.ids[0];
    if (CAL_INDEX) {
      const el = $("live-" + id);
      if (el) {
        el.textContent = j.ids.map(i => {
          const p = livePos(i);
          return p == null ? "—" : p;
        }).join(" · ");
      }
      const dEl = $("delta-" + id);
      if (dEl) dEl.textContent = j.ids.map(i => fmtDelta(liveDelta(i))).join(" · ");
      const load = $("load-" + id);
      if (load) {
        load.textContent = j.ids.map(i => {
          const r = G.rows[i];
          return !r || r.miss ? "—" : r.load;
        }).join(" · ");
        load.className = j.ids.some(i => G.rows[i] && Math.abs(G.rows[i].load) > 400) ? "hot" : "";
      }
    } else {
      j.ids.forEach(i => {
        tickWrapSess(i);
        const el = $("live-" + i);
        if (el) {
          const p = livePos(i);
          el.textContent = p == null ? "—" : p;
        }
        const dEl = $("delta-" + i);
        if (dEl) dEl.textContent = fmtDelta(liveDelta(i));
        const deg = $("deg-" + i);
        if (deg) deg.textContent = fmtDeg(liveDelta(i));
        const uEl = $("un-" + i);
        if (uEl) uEl.textContent = dash(liveUn(i));
        const tEl = $("turns-" + i);
        if (tEl) {
          const tn = liveTurns(i);
          tEl.textContent = tn == null ? "—" : tn;
        }
      });
    }
    const meta = $("wrapMeta-" + id);
    if (meta) {
      if (learn && learn.id === id) {
        meta.textContent = "calibrating  " + fmtDeg(learn.minD) + " … " + fmtDeg(learn.maxD);
      } else if (isWrap(id)) {
        const L = limits[id] || {};
        if (homeUn[id] == null) {
          meta.textContent = "slack the loom, then here. after that, one way now only falls (−), the other only rises (+).";
        } else if (isLegacyRaw(L)) {
          meta.textContent = "old raw min/max cleared — calibrate for ±° walls";
        } else if (L.amin != null && L.amax != null) {
          meta.textContent = "range " + fmtDeg(+L.amin) + " … " + fmtDeg(+L.amax)
            + "  (" + fmtDelta(L.amin) + " … " + fmtDelta(L.amax) + "). motion stays inside this.";
        } else {
          meta.textContent = "calibrate — move both ways by hand. through 0/4095 is fine while free.";
        }
      } else {
        const stale = j.ids.some(i => isLegacyRaw(limits[i] || {}));
        meta.textContent = homeUn[id] == null
          ? "press here, then − / +"
          : (stale ? "old min/max were raw encoder counts — re-mark − / + as Δ from here" : "");
      }
    }
    if (!CAL_INDEX) {
      if (learn && (learn.id === id || j.ids.indexOf(learn.id) >= 0)) tickLearn();
      paintLearnBtn(id);
      if (isWrap(id)) paintRange(id);
      paintSlide(id);
    }
  });
  if (!CAL_INDEX) watchPair();
  paintPair();
  fillMarks();
  paintTq();
  paintSense();
  paintTestBtns();
  if (learn) {
    const n = $("mark-min-" + learn.id);
    const x = $("mark-max-" + learn.id);
    if (n) n.textContent = fmtDeg(learn.minD) + "  " + fmtDelta(learn.minD);
    if (x) x.textContent = fmtDeg(learn.maxD) + "  " + fmtDelta(learn.maxD);
  }
}

function clearPending() {
  pending = null;
  pendingT = 0;
  resetMarkBtns();
  const all = $("btnFreeAll");
  if (all) {
    all.textContent = "free all";
    all.classList.remove("wait");
  }
  paintTestBtns();
}

function snapLive(ids) {
  const snap = {};
  for (const i of ids) {
    const p = livePos(i);
    if (p == null) return null;
    snap[i] = p;
  }
  return snap;
}

function snapDelta(ids) {
  const snap = {};
  for (const i of ids) {
    const d = liveDelta(i);
    if (d == null) return null;
    snap[i] = d;
  }
  return snap;
}

async function commitMark(id, key, snap, typed) {
  const ids = mates(id);
  try {
    if (key === "home") {
      for (const i of ids) {
        const pos = snap[i];
        if (pos == null) throw new Error("no home on id " + i);
        const live = livePos(i);
        if (!typed) {
          if (live == null) throw new Error("no live pos on id " + i);
          await send(cmdHere(i));
          if (G.rows[i] && !G.rows[i].miss) {
            G.rows[i].un = live;
            G.rows[i].turns = 0;
          }
          homeUn[i] = live;
          resetWrapSess(i);
        } else {
          if (live != null && Math.abs(live - pos) < 8) {
            await send(cmdHere(i));
            if (G.rows[i] && !G.rows[i].miss) {
              G.rows[i].un = live;
              G.rows[i].turns = 0;
            }
          }
          homeUn[i] = pos;
          resetWrapSess(i);
        }
        await saveLimit(i, "home", pos);
        await fetch("/api/limits", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ servo: i, amin: null, amax: null }),
        });
        const cur = limits[i] || {};
        cur.home = snap[i];
        cur.amin = null;
        cur.amax = null;
        limits[i] = cur;
      }
    } else {
      for (const i of ids) {
        const d = snap[i];
        if (d == null) throw new Error("no Δ on id " + i);
        await saveLimit(i, key, d, { chip: false });
        const L = limits[i] || {};
        L[key] = d;
        limits[i] = L;
        const origin = L.home != null ? +L.home : homeUn[i];
        if (!isWrap(i) && origin != null) {
          const raw = Math.max(0, Math.min(4095, origin + d));
          await send("lim " + i + " " + (key === "amin" ? "min" : "max") + " " + raw);
        }
      }
    }
    limits = await fetchLimits();
    fillMarks();
    paintSlide(id);
    paintTestBtns();
    const b = document.querySelector(`#cal-${id} button[data-k="${key}"]:not([data-act="set"])`);
    const sb = document.querySelector(`#cal-${id} button[data-act="set"][data-k="${key}"]`);
    const label = keyLabel(id, key);
    const shown = ids.map(i => (key === "home" ? snap[i] : fmtDelta(snap[i]))).join(" · ");
    if (b) {
      b.textContent = "saved " + shown;
      b.classList.remove("wait");
      b.classList.add("ok");
      setTimeout(() => {
        b.textContent = label;
        b.classList.remove("ok");
      }, 1400);
    }
    if (sb) {
      sb.textContent = "saved";
      sb.classList.add("ok");
      setTimeout(() => {
        sb.textContent = "set";
        sb.classList.remove("ok");
      }, 1400);
    }
    if (key === "home") {
      say((typed ? "home " : "here ") + ids.map(i => i + "=" + snap[i]).join(" · ") + "  Δ=0"
        + (typed ? "" : ". loom must look slack. pack-off forgets turns"));
    } else {
      say("saved " + label + " " + fmtDeg(snap[id]) + "  " + shown
        + (isWrap(id) ? " — not pushed as chip min/max" : " — chip wall is here+Δ"));
    }
  } catch (e) {
    say(String(e && e.message ? e.message : e));
  }
}

function armMark(id, key) {
  const ids = mates(id);
  if (key !== "home" && !hasHere(ids)) {
    say("press here first at a comfortable pose, then spin toward the wall and mark − / +");
    return;
  }
  const snap = key === "home" ? snapLive(ids) : snapDelta(ids);
  if (!snap) {
    say("no live pos on " + ids.join("+") + " — pack on, hold pose or wait for a number");
    return;
  }
  if (key === "amin" && ids.some(i => snap[i] >= -40)) {
    say("− is counts falling from here. this Δ is " + ids.map(i => fmtDelta(snap[i])).join(" · ")
      + ". spin the other way, or here is wrong");
    return;
  }
  if (key === "amax" && ids.some(i => snap[i] <= 40)) {
    say("+ is counts rising from here. this Δ is " + ids.map(i => fmtDelta(snap[i])).join(" · ")
      + ". spin the other way, or here is wrong");
    return;
  }
  const now = Date.now();
  const near = key === "home" ? 8 : 12;
  if (pending && pending.id === id && pending.key === key && pending.snap && now < pendingT) {
    const still = ids.every(i => {
      const cur = key === "home" ? livePos(i) : liveDelta(i);
      return cur != null && Math.abs(pending.snap[i] - cur) < near;
    });
    if (still) {
      const p = pending;
      clearPending();
      commitMark(p.id, p.key, p.snap);
      return;
    }
  }
  const L = limits[id] || {};
  const old = key === "home" ? L.home : key === "amin" ? L.amin : L.amax;
  const label = keyLabel(id, key);
  const shown = ids.map(i => (key === "home" ? snap[i] : fmtDelta(snap[i]))).join(" · ");
  clearPending();
  pending = { id, key, snap };
  pendingT = now + CONFIRM_MS;
  const b = document.querySelector(`#cal-${id} button[data-k="${key}"]:not([data-act="set"])`);
  if (b) {
    const oldS = key === "home" ? dash(old) : fmtDelta(old);
    b.textContent = old != null && old !== snap[id] ? (oldS + "→" + shown + "?") : ("set " + shown + "?");
    b.classList.add("wait");
  }
  if (key === "home") {
    say("click here again to zero Δ at " + ids.map(i => i + "=" + snap[i]).join(" · ") + ". loom must look slack");
  } else {
    say("click " + label + " again to save " + fmtDeg(snap[id]) + "  " + shown);
  }
}

function readTyped(ids, key) {
  const snap = {};
  for (const i of ids) {
    const el = $("in-" + key + "-" + i);
    if (!el || el.value === "" || el.value == null) return null;
    const v = Math.round(+el.value);
    if (!isFinite(v)) return null;
    snap[i] = v;
  }
  return snap;
}

function armTyped(id, key) {
  const ids = mates(id);
  const snap = readTyped(ids, key);
  if (!snap) {
    say("type a number" + (ids.length > 1 ? " for each motor" : "") + ", then set");
    return;
  }
  if (key === "home" && ids.some(i => snap[i] < 0 || snap[i] > 4095)) {
    say("home is raw encoder pos 0–4095");
    return;
  }
  if (key !== "home" && !hasHere(ids)) {
    say("set home first (here, or type home and set), then − / +");
    return;
  }
  if (key === "amin" && ids.some(i => snap[i] >= 0)) {
    say("− is a negative Δ (counts falling from home). this is " + ids.map(i => fmtDelta(snap[i])).join(" · "));
    return;
  }
  if (key === "amax" && ids.some(i => snap[i] <= 0)) {
    say("+ is a positive Δ (counts rising from home). this is " + ids.map(i => fmtDelta(snap[i])).join(" · "));
    return;
  }
  commitMark(id, key, snap, true);
}

function setTq(ids, on) {
  if (ids[0] === 12 || ids.indexOf(12) >= 0 || ids.indexOf(13) >= 0) {
    send(on ? "id 12 hold" : "id 12 free");
    send(on ? "id 13 hold" : "id 13 free");
    return;
  }
  ids.forEach(id => send(on ? cmdHold(id) : cmdFree(id)));
}

function hopMax(id) {
  if (id === 16) return 900;
  if (id === 17) return 0;
  return wallsOf(id) ? 0 : 80;
}

function toward(id, target) {
  const pos = livePos(id);
  if (pos == null) return null;
  const hop = hopMax(id);
  if (!hop || Math.abs(target - pos) <= hop) return target;
  return pos + (target > pos ? hop : -hop);
}

function rawGoal(id, deltaOrPos, delta) {
  if (!delta) return deltaOrPos;
  const origin = homeUn[id] != null ? homeUn[id] : (limits[id] && limits[id].home);
  if (origin == null) return null;
  return Math.max(0, Math.min(4095, +origin + +deltaOrPos));
}

function buildPhases(ids) {
  const phases = [
    { name: "−", goals: {} },
    { name: "+", goals: {} },
  ];
  let home = false;
  ids.forEach(id => {
    const w = wallsOf(id);
    phases[0].goals[id] = rawGoal(id, w.lo, w.delta);
    phases[1].goals[id] = rawGoal(id, w.hi, w.delta);
    if (!w.delta && w.home != null) home = true;
    if (w.delta && homeUn[id] != null) home = true;
  });
  if (home) {
    const ph = { name: "here", goals: {} };
    ids.forEach(id => {
      const w = wallsOf(id);
      if (w.delta) ph.goals[id] = homeUn[id];
      else ph.goals[id] = w.home != null ? w.home : livePos(id);
    });
    phases.push(ph);
  }
  return phases;
}

function stopTest(why) {
  const ids = test && test.ids;
  const estop = /^ESTOP/.test(why || "");
  test = null;
  hopBusy = 0;
  if (ids && !estop) ids.forEach(id => send(cmdHold(id)));
  if (why) say(why);
  paintTq();
  paintTestBtns();
}

async function issueHop() {
  if (!test || hopBusy) return;
  const ph = test.phases[test.i];
  if (!ph) return;
  hopBusy = 1;
  try {
    applySpeed();
    for (const id of test.ids) {
      const g = toward(id, ph.goals[id]);
      if (g == null) {
        hopBusy = 0;
        stopTest("no live pos on id " + id);
        return;
      }
      test.via[id] = g;
    }
    if (test.ids[0] === 12) {
      await send(cmdRaiseTo(test.via[12], test.via[13]));
    } else {
      for (const id of test.ids) {
        await send(cmdTo(id, test.via[id]));
      }
    }
    test.t0 = Date.now();
    const bits = test.ids.map(id => test.via[id]).join("/");
    say(namesOf(test.ids) + " → " + ph.name + " " + bits + " @ " + spdNow() + " " + spdWord(spdNow()));
  } finally {
    hopBusy = 0;
  }
}

function tickTest() {
  if (!test || hopBusy || !test.phases.length) return;
  const ph = test.phases[test.i];
  if (!ph || test.via == null) return;
  if (Date.now() - test.t0 > HOP_MS) {
    stopTest("timeout on " + ph.name + " — jammed, or walls not on chip. STOP if it fights");
    return;
  }
  const atVia = test.ids.every(id => {
    const pos = livePos(id);
    const via = test.via[id];
    return pos != null && via != null && Math.abs(pos - via) <= NEAR;
  });
  if (!atVia) return;
  const atGoal = test.ids.every(id => Math.abs(test.via[id] - ph.goals[id]) <= NEAR);
  if (!atGoal) {
    issueHop();
    return;
  }
  test.i++;
  if (test.i >= test.phases.length) {
    const ids = test.ids;
    test = null;
    ids.forEach(id => send(cmdHold(id)));
    say("test ok — " + namesOf(ids) + " held. watch load if a wall was short");
    paintTq();
    paintTestBtns();
    return;
  }
  issueHop();
}

async function startTest(id) {
  if (isWrap(id)) {
    say("yaw / wrist roll cannot be driven through 0/4095 — free, turn through, hold");
    return;
  }
  const ids = mates(id);
  if (!ids.every(wallsOf)) {
    say("mark here, then − and + first on " + namesOf(ids));
    return;
  }
  if (ids[0] === 12 && pairSign() == null) {
    say("set pair (mirror or same) before a commanded sweep");
    return;
  }
  if (test) stopTest("stopped");
  test = { ids, phases: buildPhases(ids), i: 0, via: {}, t0: Date.now() };
  if (ids.some(i => test.phases.some(ph => ph.goals[i] == null))) {
    test = null;
    say("need here on " + namesOf(ids) + " before a Δ test can convert to pos");
    paintTestBtns();
    return;
  }
  paintTestBtns();
  paintTq();
  MOTORS.forEach(m => {
    if (m.id === 12 || m.id === 13) return;
    if (ids.indexOf(m.id) < 0) send(cmdHold(m.id));
  });
  for (const i of ids) {
    const lo = chipWall(i, "amin");
    const hi = chipWall(i, "amax");
    if (lo != null) await send("lim " + i + " min " + lo);
    if (hi != null) await send("lim " + i + " max " + hi);
  }
  lastSpd = null;
  applySpeed();
  const w0 = wallsOf(ids[0]);
  say("test " + namesOf(ids) + " " + fmtDelta(w0.delta ? w0.lo : w0.lo) + " → "
    + (w0.delta ? fmtDelta(w0.hi) : w0.hi) + ". STOP cuts it");
  await issueHop();
}

function armTest(id) {
  if (test && test.ids.indexOf(id) >= 0) {
    stopTest("stopped — holding " + namesOf(test.ids) + " here");
    return;
  }
  if (!canTest(id)) {
    say(isWrap(id)
      ? "yaw / wrist roll cannot be swept through 0/4095 yet"
      : ("mark here, then − and + on " + namesOf(mates(id)) + " before test"));
    return;
  }
  const now = Date.now();
  if (pending && pending.key === "test" && pending.id === id && now < pendingT) {
    clearPending();
    startTest(id);
    return;
  }
  const w = wallsOf(id);
  clearPending();
  pending = { id, key: "test" };
  pendingT = now + CONFIRM_MS;
  const b = $("test-" + id);
  const loS = w.delta ? fmtDelta(w.lo) : w.lo;
  const hiS = w.delta ? fmtDelta(w.hi) : w.hi;
  if (b) {
    b.disabled = false;
    b.textContent = loS + "↔" + hiS + "?";
    b.classList.add("wait");
  }
  say("click test again to sweep " + namesOf(mates(id)) + " " + loS + " → " + hiS
    + " @ " + spdNow() + " " + spdWord(spdNow()));
}

$("calBody").onclick = e => {
  if (CAL_INDEX) {
    const tr = e.target.closest("tr[data-slug]");
    if (tr) location.href = "/calibrate/" + tr.getAttribute("data-slug") + "/";
    return;
  }
  const b = e.target.closest("button");
  if (!b) return;
  const id = +b.getAttribute("data-id");
  if (b.getAttribute("data-act") === "tq") {
    if (test && test.ids.indexOf(id) >= 0) stopTest("stopped");
    const ids = mates(id);
    const on = ids.some(tqOn);
    setTq(ids, !on);
    say(on
      ? ("free " + ids.join("+") + " — others stay held. find the wall, then mark twice")
      : ("hold " + ids.join("+") + " at live pos"));
    return;
  }
  if (b.getAttribute("data-act") === "learn") {
    startLearn(id);
    return;
  }
  if (b.getAttribute("data-act") === "keep") {
    keepLearn();
    return;
  }
  if (b.getAttribute("data-act") === "test") {
    armTest(id);
    return;
  }
  if (b.getAttribute("data-act") === "set") {
    armTyped(id, b.getAttribute("data-k"));
    return;
  }
  const k = b.getAttribute("data-k");
  if (k) armMark(id, k);
};
$("btnHoldAll").onclick = () => {
  if (test) stopTest("stopped");
  MOTORS.forEach(m => {
    if (m.id === 12 || m.id === 13) return;
    send(cmdHold(m.id));
  });
  send("id 12 hold");
  send("id 13 hold");
  say("hold pose — arm stays. free this on one joint to find a wall");
};
$("btnFreeAll").onclick = () => {
  const b = $("btnFreeAll");
  const now = Date.now();
  if (pending && pending.id === "all" && now < pendingT) {
    if (test) stopTest("stopped");
    clearPending();
    MOTORS.forEach(m => {
      if (m.id === 12 || m.id === 13) return;
      send(cmdFree(m.id));
    });
    send("id 12 free");
    send("id 13 free");
    say("free all — gravity will drop unsupported joints");
    return;
  }
  clearPending();
  pending = { id: "all", key: "free" };
  pendingT = now + CONFIRM_MS;
  b.textContent = "drop all?";
  b.classList.add("wait");
  say("click free all again if you really want the arm to drop (" + (CONFIRM_MS / 1000) + "s)");
};
$("btnPush").onclick = async () => {
  let n = 0;
  for (const m of MOTORS) {
    const home = chipWall(m.id, "home");
    const lo = chipWall(m.id, "amin");
    const hi = chipWall(m.id, "amax");
    if (lo != null) { await send("lim " + m.id + " min " + lo); n++; }
    if (hi != null) { await send("lim " + m.id + " max " + hi); n++; }
    if (home != null) { await send("lim " + m.id + " home " + home); n++; }
  }
  say(n ? ("pushed " + n + " field(s) to chip — wrap −/+ stay host-only") : "nothing marked yet");
};
if ($("calSpd")) {
  $("calSpd").oninput = () => {
    paintSpd();
    if (!test || !test.via) return;
    lastSpd = null;
    applySpeed();
    if (test.ids[0] === 12 && test.via[12] != null && test.via[13] != null) {
      send(cmdRaiseTo(test.via[12], test.via[13]));
      return;
    }
    test.ids.forEach(id => {
      if (test.via[id] != null) send(cmdTo(id, test.via[id]));
    });
  };
}
if ($("btnPairMirror")) {
  $("btnPairMirror").onclick = () => {
    savePairSign(-1).then(() => {
      paintPair();
      say("pair mirror — 13 moves opposite 12. one raise SyncWrite");
    });
  };
}
if ($("btnPairSame")) {
  $("btnPairSame").onclick = () => {
    savePairSign(1).then(() => {
      paintPair();
      say("pair same — 13 moves with 12. one raise SyncWrite");
    });
  };
}
setInterval(() => {
  if (pending && Date.now() >= pendingT) {
    say("cancelled");
    clearPending();
  }
}, 250);

G.hooks.push((kind, rec) => {
  if (kind === "lim" && rec && rec.id) {
    const cur = limits[rec.id] || {};
    if (rec.home != null) cur.home = rec.home;
    if (rec.amin != null) cur.amin = rec.amin;
    if (rec.amax != null) cur.amax = rec.amax;
    limits[rec.id] = cur;
  }
  if (kind === "guard" && rec && String(rec).startsWith("ESTOP")) {
    if (test) {
      test = null;
      hopBusy = 0;
      say("test cut — " + rec);
      paintTestBtns();
    }
  }
  paintLive();
  if (test) tickTest();
});
fetchLimits().then(async m => {
  limits = m;
  syncPairFromLimits(m);
  for (const j of calJoints()) {
    const id = j.ids[0];
    if (!isWrap(id) || !isLegacyRaw(m[id] || {})) continue;
    await fetch("/api/limits", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ servo: id, amin: null, amax: null }),
    });
    m[id].amin = null;
    m[id].amax = null;
    limits[id] = m[id];
    say("cleared leftover raw " + id + " min/max — calibrate for −° / +°");
  }
  built = 0;
  paintLive();
});
setInterval(() => {
  if (!G.alive || !G.transport) return;
  if (MOTORS.some(m => drag[m.id])) return;
  if (test) {
    test.ids.forEach(id => send("get " + id, true));
    tickTest();
  }
}, 150);
window.addEventListener("pagehide", () => {
  if (learn) setCalStream(false);
});
paintSpd();
paintLive();
