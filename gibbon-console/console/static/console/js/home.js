const HAND = (window.CATALOG && window.CATALOG.hand) || { open: 689, to_max: 2075 };
let limits = {};
let built = 0;
let lastSpd = null;
const drag = {};
const gripUntil = {};
const gripTimer = {};

function livePos(id) {
  const r = G.rows[id];
  return r && !r.miss ? r.pos : null;
}

function cmdFree(id) {
  if (id === 17) return "hand free";
  if (id === 16) return "twist free";
  return "id " + id + " free";
}

function cmdHold(id) {
  return "id " + id + " hold";
}

function cmdTo(id, goal) {
  if (id === 17) return "hand to " + goal;
  if (id === 16) return "twist to " + goal;
  return "id " + id + " to " + goal;
}

function tqOn(ids) {
  return ids.some(id => G.rows[id] && G.rows[id].tq === "1");
}

function prepSpeed() {
  if (lastSpd === "300") return;
  lastSpd = "300";
  send("move speed 300");
  send("move acc 20");
}

function limOf(id) {
  const chip = (G.lim && G.lim[id]) || {};
  const L = limits[id] || {};
  return {
    amin: chip.amin != null ? chip.amin : L.amin,
    amax: chip.amax != null ? chip.amax : L.amax,
  };
}

function band(j) {
  const id = j.ids[0];
  if (id === 17) return { lo: HAND.open, hi: HAND.to_max };
  const L = limOf(id);
  let lo = L.amin, hi = L.amax;
  if (lo != null && +lo < 0) lo = hi = null;
  if (hi != null && +hi < 0) lo = hi = null;
  if (lo != null && hi != null && +lo > 0 && +hi > 0 && (id === 11 || id === 16)) lo = hi = null;
  if (lo != null && hi != null && Math.abs(hi - lo) >= 40) {
    if (+lo > +hi) { const t = lo; lo = hi; hi = t; }
    return { lo: +lo, hi: +hi };
  }
  const pos = livePos(id);
  if (pos == null) return { lo: 0, hi: 4095 };
  const hop = id === 16 ? 900 : 80;
  return { lo: pos - hop, hi: pos + hop };
}

function applyBand(j) {
  const el = $("slide-" + j.slug);
  if (!el || drag[j.slug]) return;
  const { lo, hi } = band(j);
  el.min = lo;
  el.max = hi;
}

function paintRow(j) {
  const miss = j.ids.some(id => !G.rows[id] || G.rows[id].miss);
  const pos = livePos(j.ids[0]);
  const load = j.ids.map(id => (G.rows[id] && !G.rows[id].miss ? G.rows[id].load : "—")).join(" · ");
  const cur = j.ids.map(id => (G.rows[id] && !G.rows[id].miss ? (G.rows[id].cur || 0) : "—")).join(" · ");
  const hot = j.ids.some(id => G.rows[id] && Math.abs(G.rows[id].load) > 400);
  const hotC = j.ids.some(id => G.rows[id] && Math.abs(G.rows[id].cur) >= 5);
  const on = tqOn(j.ids);
  const tr = $("row-" + j.slug);
  if (tr) tr.className = miss || !G.alive ? "no" : "go";
  const p = $("pos-" + j.slug);
  if (p) p.textContent = pos == null ? "—" : pos;
  const l = $("load-" + j.slug);
  if (l) {
    l.textContent = load;
    l.className = hot ? "hot" : "";
  }
  const c = $("cur-" + j.slug);
  if (c) {
    c.textContent = cur;
    c.className = hotC ? "hot" : "";
  }
  const b = $("lock-" + j.slug);
  if (b) {
    b.textContent = on ? "free" : "hold";
    b.className = "lock" + (on ? " bad" : "");
  }
  const sl = $("slide-" + j.slug);
  const hold = drag[j.slug] || (gripUntil[j.slug] && Date.now() < gripUntil[j.slug]);
  if (sl) {
    const blocked = j.paired && !raiseReady();
    sl.disabled = blocked;
    sl.title = blocked ? "raise blocked until header is cyan (info raise=1)" : "";
    if (!hold && pos != null && !blocked) {
      applyBand(j);
      sl.value = Math.max(+sl.min, Math.min(+sl.max, pos));
    }
  }
}

function build() {
  $("homeBody").innerHTML = JOINTS.map(j => `<tr id="row-${j.slug}" class="no">
    <td class="goName"><a class="name" href="/j/${j.slug}/">${j.name}</a>${j.paired ? ' <span class="axis">both</span>' : ""}</td>
    <td>${j.ids.join("+")}</td>
    <td id="pos-${j.slug}">—</td>
    <td id="load-${j.slug}">—</td>
    <td id="cur-${j.slug}">—</td>
    <td class="grip"><input id="slide-${j.slug}" type="range" min="0" max="4095" value="2000" data-slug="${j.slug}"></td>
    <td><button type="button" class="lock" id="lock-${j.slug}" data-slug="${j.slug}">hold</button></td>
    <td class="mut"><a href="/j/${j.slug}/">sensors</a></td>
  </tr>`).join("");
  JOINTS.forEach(j => bindSlide(j));
  built = 1;
}

function bindSlide(j) {
  const g = $("slide-" + j.slug);
  const end = () => {
    if (!drag[j.slug]) return;
    clearTimeout(gripTimer[j.slug]);
    sendGrip(j);
    drag[j.slug] = 0;
  };
  g.addEventListener("pointerdown", e => {
    drag[j.slug] = 1;
    try { g.setPointerCapture(e.pointerId); } catch {}
  });
  g.addEventListener("input", () => {
    drag[j.slug] = 1;
    clearTimeout(gripTimer[j.slug]);
    gripTimer[j.slug] = setTimeout(() => sendGrip(j), 60);
  });
  g.addEventListener("pointerup", end);
  g.addEventListener("pointercancel", () => { drag[j.slug] = 0; clearTimeout(gripTimer[j.slug]); });
}

function sendGrip(j) {
  const g = $("slide-" + j.slug);
  if (!g || !G.alive) return;
  const goal = Math.round(+g.value);
  const pos = livePos(j.ids[0]);
  gripUntil[j.slug] = Date.now() + 2000;
  prepSpeed();
  if (j.ids[0] === 17) {
    send(cmdTo(17, goal));
    return;
  }
  if (pos == null) return;
  const hop = j.ids[0] === 16 ? 900 : 80;
  let d = Math.max(-hop, Math.min(hop, goal - pos));
  if (!d) return;
  if (j.ids[0] === 16) {
    const L = limits[16] || {};
    const r = G.rows[16];
    if (r && !r.miss && L.home != null && +L.amin < 0 && +L.amax > 0) {
      const un = r.un != null ? r.un : r.pos;
      const lo = +L.home + +L.amin;
      const hi = +L.home + +L.amax;
      let goalUn = un + d;
      if (goalUn < lo) goalUn = lo;
      if (goalUn > hi) goalUn = hi;
      d = goalUn - un;
      if (!d) return;
    }
  }
  if (j.paired) {
    if (!raiseReady()) {
      log("raise blocked — flash first. header fw old");
      return;
    }
    const sign = pairSign();
    const p12 = livePos(12);
    const p13 = livePos(13);
    if (sign == null || p12 == null || p13 == null) {
      log("raise needs pair (mirror/same) on /calibrate/shoulder-pitch/ — not jogging 12+13");
      return;
    }
    send(cmdRaiseTo(p12 + d, p13 + sign * d));
    return;
  }
  j.ids.forEach(id => send("id " + id + " nudge " + d));
}

function toggleHold(slug) {
  const j = JOINTS.find(x => x.slug === slug);
  if (!j || !G.alive) return;
  const on = tqOn(j.ids);
  if (j.paired) {
    sendRaiseTq(!on);
    return;
  }
  j.ids.forEach(id => send(on ? cmdFree(id) : cmdHold(id)));
}

function paintPower() {
  const bus = $("pwrBus"), maEl = $("pwrMa"), wEl = $("pwrW"), note = $("pwrNote");
  if (!bus) return;
  let n = 0, ma = 0;
  MOTORS.forEach(m => {
    const r = G.rows[m.id];
    if (!r || r.miss) return;
    n++;
    ma += Math.abs(+r.cur || 0);
  });
  const v = G.batt && !G.batt.miss && G.batt.pack != null ? +G.batt.pack : null;
  const w = v != null ? (v * ma / 1000) : null;
  bus.textContent = G.alive ? n + "/7" : "—";
  maEl.textContent = G.alive ? String(ma) : "—";
  maEl.parentElement.className = ma >= 80 ? "hot" : "";
  wEl.textContent = w != null ? w.toFixed(1) : "—";
  if (note) note.textContent = (v != null ? "@ " + v + " V  ·  " : "") + "winding estimates, not a pack shunt. light moves often read 0";
}

function paint() {
  G.paintQ = 0;
  if (!built) build();
  JOINTS.forEach(paintRow);
  paintPower();
}

function requestPaint() {
  if (!G.paintQ) G.paintQ = requestAnimationFrame(paint);
}

G.hooks.push((kind, rec) => {
  if (kind === "lim" && rec && rec.id) {
    const cur = limits[rec.id] || {};
    if (rec.home != null) cur.home = rec.home;
    if (rec.amin != null) cur.amin = rec.amin;
    if (rec.amax != null) cur.amax = rec.amax;
    limits[rec.id] = cur;
  }
  requestPaint();
});
$("homeBody").onclick = e => {
  if (e.target.closest("input, button.lock, a")) {
    const b = e.target.closest("button.lock");
    if (b) toggleHold(b.getAttribute("data-slug"));
    return;
  }
  const tr = e.target.closest("tr[id^='row-']");
  if (!tr) return;
  location.href = "/j/" + tr.id.slice(4) + "/";
};
fetchLimits().then(m => { limits = m; syncPairFromLimits(m); requestPaint(); });
paint();
