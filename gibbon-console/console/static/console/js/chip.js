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
      ["ble", bleOn ? "link " + (G.chip.ble || "") : (G.chip.ble ? "ready " + (G.chip.ble) : null), bleOn ? "ok" : ""],
      ["stream", G.chip.stream != null ? G.chip.stream + "ms" : null],
      ["dump", G.chip.dump], ["bus_rx", G.chip.bus_rx], ["bus", G.chip.bus], ["gpio", G.chip.gpio], ["via", G.chip.via],
    ]) +
    blk("pack", [
      ["pack", G.batt.miss ? "off" : (G.batt.pack != null ? G.batt.pack + "V" : null), G.batt.miss ? "bad" : "ok"],
      ["est", G.batt.est != null ? G.batt.est + "%" : null], ["cells", G.batt.cells], ["servos", G.batt.n],
      ["level", G.batt.lvl], ["found", G.alive ? n + "/7" : null, n === 7 ? "ok" : "bad"], ["guess", G.chip.guess],
    ]);
  $("whoBody").innerHTML = MOTORS.map(m => {
    const r = G.rows[m.id];
    const miss = !r || r.miss;
    const cells = !r
      ? [m.id, "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"]
      : [m.id, dash(r.fw), miss ? "—" : modeOf(r), dash(r.st), dash(r.motor),
         dash(r.kp), dash(r.kd), dash(r.ki), dash(r.amin), dash(r.amax), dash(r.goal)];
    return `<tr class="${miss ? "no" : "go"}"><td>${m.name}</td>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
  }).join("");
}

function requestPaint() {
  if (!G.paintQ) G.paintQ = requestAnimationFrame(paint);
}

G.hooks.push(() => requestPaint());
paint();
