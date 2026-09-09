function blk(title, pairs) {
  return `<div class="blk"><h2>${title}</h2>${pairs.map(([k, val, cls]) =>
    `<div><b>${k}</b><i class="${cls || ""}">${dash(val)}</i></div>`).join("")}</div>`;
}

function paint() {
  G.paintQ = 0;
  const n = foundN();
  const bleOn = G.chip.via === "ble" || G.chip.bleConn === 1 || G.chip.bleConn === "1";
  $("meta").innerHTML =
    blk("chip", [
      ["soc", G.chip.soc || "ESP32"], ["rev", G.chip.rev], ["cores", G.chip.cores],
      ["flash", G.chip.flash ? G.chip.flash + "KB" : null], ["heap", G.chip.heap],
      ["uptime", G.chip.uptime != null ? Math.floor(G.chip.uptime / 1000) + "s" : null],
      ["reset", G.chip.reset], ["tasks", G.chip.tasks], ["idf", G.chip.idf], ["mac", G.chip.mac],
    ]) +
    blk("link", [
      ["usb", G.chip.via === "usb" ? "in" : (G.alive ? "out" : null), G.chip.via === "usb" ? "ok" : ""],
      ["ble", bleOn ? "link " + (G.chip.ble || "") : (G.chip.ble ? "ready " + G.chip.ble : null), bleOn ? "ok" : ""],
      ["stream", G.chip.stream != null ? G.chip.stream + "ms" : null],
      ["dump", G.chip.dump], ["bus_rx", G.chip.bus_rx], ["bus", G.chip.bus], ["gpio", G.chip.gpio], ["via", G.chip.via],
    ]) +
    blk("pack", [
      ["pack", G.batt.miss ? "off" : (G.batt.pack != null ? G.batt.pack + "V" : null), G.batt.miss ? "bad" : "ok"],
      ["est", G.batt.est != null ? G.batt.est + "%" : null], ["cells", G.batt.cells], ["servos", G.batt.n],
      ["level", G.batt.lvl], ["found", G.alive ? n + "/7" : null, n === 7 ? "ok" : "bad"], ["guess", G.chip.guess],
    ]);
  $("hint").innerHTML = !G.alive
    ? "waiting for chip."
    : G.batt.miss && n === 0
      ? `<span class="next">no motors — plug XT60. usb only powers the chip.</span>`
      : n === 7 ? `<span class="ok">bus live — present @56 every ${dash(G.chip.stream)}ms.</span>`
        : n === 0 ? `<span class="next">pack on, motors still silent.</span>`
          : `<span class="next">${n}/7 on the bus — others miss.</span>`;
  $("armBody").innerHTML = JOINTS.map(j => {
    const r = G.rows[j.id];
    const miss = !r || r.miss;
    const hot = r && Math.abs(r.load) > 400;
    const on = G.motorOpen === j.id ? " on" : "";
    const cells = miss
      ? [j.id, "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"]
      : [j.id, r.pos, r.un, r.turns, r.speed, r.load, r.v, r.temp, r.cur, r.moving, r.tq, modeOf(r), r.st, r.fw, r.motor];
    const tds = cells.map((c, i) => `<td${i === 5 && hot ? ' class="hot"' : ""}>${dash(c)}</td>`).join("");
    return `<tr class="${miss ? "no" : "go"}${on}" data-id="${j.id}"><td>${j.name}</td>${tds}</tr>`;
  }).join("");
  const j = JOINTS.find(x => x.id === G.motorOpen);
  const r = j && G.rows[j.id];
  const box = $("acts");
  if (!j || !G.alive) { box.innerHTML = ""; return; }
  if (!r || r.miss) { box.innerHTML = `<span class="bad">${j.name} id ${j.id} — no present @56</span>`; return; }
  let extra = "";
  if (j.id === 17) extra = `<button data-cmd="hand open">open</button><button data-cmd="hand close">close</button>`;
  else if (j.id === 16) extra = `<button data-cmd="twist speed 0">stop</button>`;
  else extra = `<button data-act="nudge" data-dir="-1">−</button><button data-act="nudge" data-dir="1">+</button>`;
  box.innerHTML = `<span class="ok">${j.name} id ${j.id}</span>${extra}<button data-act="free">free</button><button data-act="hold">hold</button>` +
    (r.kp != null ? `<span>kp ${dash(r.kp)} kd ${dash(r.kd)} ki ${dash(r.ki)} amin ${dash(r.amin)} amax ${dash(r.amax)} goal ${dash(r.goal)} tqlim ${dash(r.tqlim)}</span>` : "") +
    (r.raw ? `<span class="raw">${r.raw}</span>` : "");
}

function requestPaint() {
  if (!G.paintQ) G.paintQ = requestAnimationFrame(paint);
}

G.hooks.push(() => requestPaint());
$("armBody").onclick = e => {
  const row = e.target.closest("tr[data-id]");
  if (!row) return;
  const id = +row.getAttribute("data-id");
  G.motorOpen = G.motorOpen === id ? 0 : id;
  paint();
};
$("acts").onclick = e => {
  const b = e.target.closest("button");
  if (!b || !G.motorOpen) return;
  const cmd = b.getAttribute("data-cmd");
  if (cmd) { send(cmd); return; }
  const act = b.getAttribute("data-act");
  if (act === "nudge") send("id " + G.motorOpen + " nudge " + (b.getAttribute("data-dir") === "-1" ? -40 : 40));
  if (act === "free" || act === "hold") send("id " + G.motorOpen + " " + act);
};
paint();
