const slug = window.JOINT_SLUG;
const joint = JOINTS.find(j => j.slug === slug);
const OPS = {
  yaw: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  "shoulder-pitch": [["free", "free both"], ["hold", "hold both"], ["nudge", "nudge both Δ"]],
  elbow: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  "wrist-pitch": [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  "wrist-roll": [["free", "free"], ["hold", "hold"], ["to", "to pos"], ["servo", "servo"], ["wheel", "wheel"], ["speed", "wheel speed"], ["here", "here"], ["estop", "twist estop"]],
  hand: [["open", "open"], ["close", "close"], ["to", "to pos"], ["nudge", "nudge Δ"], ["free", "free"], ["hold", "hold"]],
};
const NEED = new Set(["nudge", "to", "speed"]);
const hist = {};
const VARS = [
  { key: "pos", label: "pos", color: "#00f0c8" },
  { key: "load", label: "load", color: "#ff2bd6" },
];
joint.ids.forEach(id => {
  hist[id] = { t: [], pos: [], load: [] };
});

function fillOps() {
  $("op").innerHTML = (OPS[slug] || []).map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  $("actHint").textContent = joint.paired
    ? "same command to 12 and 13. if they fight: STOP."
    : (joint.note || joint.say);
  showVal();
}

function showVal() {
  const op = $("op").value;
  $("valWrap").hidden = !NEED.has(op);
  if (op === "nudge") { $("val").value = 40; $("val").min = -80; $("val").max = 80; }
  if (op === "to") { $("val").value = 2000; $("val").min = 0; $("val").max = 4095; }
  if (op === "speed") { $("val").value = 0; $("val").min = -3073; $("val").max = 3073; }
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

function paintTable() {
  $("armBody").innerHTML = joint.ids.map(id => {
    const r = G.rows[id];
    const miss = !r || r.miss;
    const m = MOTORS.find(x => x.id === id);
    const cells = miss
      ? [id, "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"]
      : [id, r.pos, r.un, r.turns, r.speed, r.load, r.v, r.temp, r.cur, r.moving, r.tq];
    const hot = r && Math.abs(r.load) > 400;
    return `<tr class="${miss ? "no" : "go"}"><td>${m ? m.name : id}</td>${
      cells.map((c, i) => `<td${i === 5 && hot ? ' class="hot"' : ""}>${dash(c)}</td>`).join("")
    }</tr>`;
  }).join("");
}

function spark(canvas, ts, ys, color) {
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

function drawCharts() {
  const wrap = $("charts");
  wrap.innerHTML = joint.ids.map(id => {
    const m = MOTORS.find(x => x.id === id);
    return VARS.map(v =>
      `<div class="chart"><h3>${m ? m.name : id} ${v.label}</h3><canvas data-id="${id}" data-k="${v.key}"></canvas></div>`
    ).join("");
  }).join("");
  wrap.querySelectorAll("canvas").forEach(c => {
    const id = +c.getAttribute("data-id");
    const k = c.getAttribute("data-k");
    const spec = VARS.find(v => v.key === k);
    spark(c, hist[id].t, hist[id][k], spec.color);
  });
}

function push(rec) {
  const h = hist[rec.id];
  if (!h) return;
  h.t.push(rec.t);
  h.pos.push(rec.pos);
  h.load.push(rec.load);
  if (h.t.length > 400) {
    h.t.shift(); h.pos.shift(); h.load.shift();
  }
}

G.hooks.push((kind, rec) => {
  paintTable();
  if (kind === "sample" && rec && !rec.miss && hist[rec.id]) {
    push(rec);
    drawCharts();
  }
});
fillOps();
$("op").onchange = showVal;
$("actForm").onsubmit = async e => {
  e.preventDefault();
  const op = $("op").value;
  const n = $("val").value;
  for (const id of joint.ids) {
    await send(cmdFor(id, op, n));
  }
};
$("btnMoveSet").onclick = async () => {
  await send("move speed " + $("mvSpeed").value);
  await send("move acc " + $("mvAcc").value);
};
paintTable();
drawCharts();
