const VARS = [
  { key: "pos", label: "pos @56", color: "#00f0c8" },
  { key: "un", label: "un", color: "#7dffc8" },
  { key: "speed", label: "speed @58", color: "#4ad4ff" },
  { key: "load", label: "load @60", color: "#ff2bd6" },
  { key: "cur", label: "current @69 mA", color: "#ff6bd6" },
  { key: "v", label: "vin @62 V", color: "#ffb000" },
  { key: "temp", label: "temp @63 °C", color: "#c8f060" },
];
const MAXN = 500;
const hist = {};
MOTORS.forEach(j => { hist[j.id] = { t: [] }; VARS.forEach(v => { hist[j.id][v.key] = []; }); });

const sel = $("gJoint");
MOTORS.forEach(j => {
  const o = document.createElement("option");
  o.value = j.id; o.textContent = j.name + "  " + j.id;
  sel.appendChild(o);
});
sel.value = "17";
const box = $("gVars");
VARS.forEach((v, i) => {
  const lab = document.createElement("label");
  lab.innerHTML = `<input type="checkbox" data-k="${v.key}" ${i < 4 ? "checked" : ""}> ${v.label}`;
  box.appendChild(lab);
});

function keysOn() {
  return [...box.querySelectorAll("input:checked")].map(x => x.getAttribute("data-k"));
}

function push(rec) {
  const h = hist[rec.id];
  if (!h) return;
  h.t.push(rec.t);
  VARS.forEach(v => h[v.key].push(rec[v.key]));
  if (h.t.length > MAXN) {
    h.t.shift();
    VARS.forEach(v => h[v.key].shift());
  }
}

function draw() {
  const id = +sel.value;
  const h = hist[id];
  const wrap = $("charts");
  const on = keysOn();
  wrap.innerHTML = on.map(k => {
    const spec = VARS.find(v => v.key === k);
    return `<div class="chart"><h3>${spec.label}</h3><canvas data-k="${k}"></canvas></div>`;
  }).join("");
  on.forEach(k => {
    const c = wrap.querySelector(`canvas[data-k="${k}"]`);
    const spec = VARS.find(v => v.key === k);
    spark(c, h.t, h[k], spec.color);
  });
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

if (window.TRIAL_ID) {
  fetch("/api/trials/" + TRIAL_ID + "/samples?servo=" + sel.value)
    .then(r => r.json())
    .then(d => {
      (d.samples || []).forEach(s => push({ ...s, id: s.id, t: s.t }));
      draw();
    });
  sel.onchange = () => {
    MOTORS.forEach(j => { hist[j.id] = { t: [] }; VARS.forEach(v => { hist[j.id][v.key] = []; }); });
    fetch("/api/trials/" + TRIAL_ID + "/samples?servo=" + sel.value)
      .then(r => r.json())
      .then(d => {
        (d.samples || []).forEach(s => push({ ...s, id: s.id, t: s.t }));
        draw();
      });
  };
} else {
  G.hooks.push((kind, rec) => {
    if (kind === "sample" && rec && !rec.miss) { push(rec); if (rec.id === +sel.value) draw(); }
  });
  sel.onchange = draw;
}
box.onchange = draw;
draw();
