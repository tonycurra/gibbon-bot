function motorName(id) {
  const m = MOTORS.find(x => x.id === id);
  return m ? m.name : ("id " + id);
}

function rowState(ids) {
  const rows = ids.map(id => G.rows[id]);
  const miss = rows.some(r => !r || r.miss);
  const tq = rows.some(r => r && r.tq === "1");
  const pos = rows.map(r => (r && !r.miss ? r.pos : "—")).join(" · ");
  const spd = rows.map(r => (r && !r.miss ? r.speed : "—")).join(" · ");
  const load = rows.map(r => (r && !r.miss ? r.load : "—")).join(" · ");
  const hot = rows.some(r => r && Math.abs(r.load) > 400);
  return { miss, tq, pos, spd, load, hot };
}

function paint() {
  G.paintQ = 0;
  $("homeBody").innerHTML = JOINTS.map(j => {
    const s = rowState(j.ids);
    const bus = !G.alive ? "—" : s.miss ? "miss" : "ok";
    return `<tr class="${s.miss || !G.alive ? "no" : "go"}" data-slug="${j.slug}">
      <td>${j.name}</td>
      <td><span class="axis">${j.axis}</span></td>
      <td>${j.ids.join("+")}</td>
      <td class="${bus === "ok" ? "ok" : "bad"}">${bus}</td>
      <td class="${s.tq ? "bad" : "mut"}">${s.tq ? "on" : "off"}</td>
      <td>${s.pos}</td>
      <td>${s.spd}</td>
      <td class="${s.hot ? "hot" : ""}">${s.load}</td>
    </tr>`;
  }).join("");
}

function requestPaint() {
  if (!G.paintQ) G.paintQ = requestAnimationFrame(paint);
}

G.hooks.push(() => requestPaint());
$("homeBody").onclick = e => {
  const tr = e.target.closest("tr[data-slug]");
  if (tr) location.href = "/j/" + tr.getAttribute("data-slug") + "/";
};
paint();
