const sel = $("calJoint");
let limits = {};

function currentJoint() {
  return JOINTS.find(j => j.slug === sel.value);
}

async function loadLimits() {
  const d = await fetch("/api/limits").then(r => r.json());
  limits = {};
  (d.limits || []).forEach(x => { limits[x.servo] = x; });
}

function livePos(id) {
  const r = G.rows[id];
  return r && !r.miss ? r.pos : null;
}

function paint() {
  const j = currentJoint();
  $("calNote").textContent = j.note || j.say;
  $("calBody").innerHTML = j.ids.map(id => {
    const L = limits[id] || {};
    const pos = livePos(id);
    return `<tr>
      <td>${id}</td>
      <td>${pos == null ? "—" : pos}</td>
      <td>${dash(L.home)}</td>
      <td>${dash(L.amin)}</td>
      <td>${dash(L.amax)}</td>
      <td>
        <button type="button" data-id="${id}" data-k="home">home</button>
        <button type="button" data-id="${id}" data-k="amin">min</button>
        <button type="button" data-id="${id}" data-k="amax">max</button>
      </td>
    </tr>`;
  }).join("");
}

async function mark(id, key) {
  const pos = livePos(id);
  if (pos == null) return;
  await fetch("/api/limits", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ servo: id, [key]: pos }),
  });
  await loadLimits();
  paint();
}

sel.onchange = () => { location.href = "/calibrate/?j=" + sel.value; };
$("calBody").onclick = e => {
  const b = e.target.closest("button[data-k]");
  if (!b) return;
  mark(+b.getAttribute("data-id"), b.getAttribute("data-k"));
};
G.hooks.push(() => paint());
loadLimits().then(paint);
