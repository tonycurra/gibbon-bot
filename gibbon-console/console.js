const NAMES = {11:"turn",12:"raise-A",13:"raise-B",14:"bend",15:"wrist",16:"twist",17:"hand"};
const GET_RE = /get id=(\d+) pos=(-?\d+) speed=(-?\d+) load=(-?\d+) v_raw=(\d+) v=([\d.]+) temp=(\d+) cur=(-?\d+) moving=(\d+) tq=(\S+)(?: un=(-?\d+) turns=(-?\d+))?/;
const BATT_RE = /batt pack=([\d.]+)V/;

let port, reader, writer, alive = false, polling = false;
const rows = {};
const logEl = document.getElementById("log");
const tb = document.getElementById("tb");
const live = document.getElementById("live");

function log(s) {
  logEl.textContent += s + "\n";
  if (logEl.textContent.length > 12000) logEl.textContent = logEl.textContent.slice(-8000);
  logEl.scrollTop = logEl.scrollHeight;
}

function setLink(ok, msg) {
  const el = document.getElementById("link");
  el.textContent = msg;
  el.className = ok ? "ok" : "bad";
  document.querySelectorAll("button[data-cmd], #btnStop, #btnDisc, #handTo, #twistTo, #spdGo, #rawGo").forEach(b => {
    if (b.id === "btnConnect") return;
    b.disabled = !ok;
  });
}

function paint() {
  let html = "";
  let pack = "";
  for (let id = 11; id <= 17; id++) {
    const r = rows[id];
    if (!r) {
      html += `<tr><td>${id}</td><td>${NAMES[id]}</td><td colspan="10" class="mut">—</td></tr>`;
      continue;
    }
    pack = r.v;
    const cls = [r.tq === "1" ? "on" : "", r.moving ? "mv" : "", Math.abs(r.load) >= 400 ? "ld" : ""].filter(Boolean).join(" ");
    html += `<tr class="${cls}"><td>${id}</td><td>${NAMES[id]}</td><td>${r.pos}</td><td>${r.un}</td><td>${r.turns}</td><td>${r.speed}</td><td class="load">${r.load}</td><td>${r.cur}</td><td>${r.moving}</td><td>${r.tq}</td><td>${r.v}</td><td>${r.temp}</td></tr>`;
  }
  tb.innerHTML = html;
  document.getElementById("pack").textContent = pack ? ("pack " + pack + "V") : "";
}

function onLine(line) {
  if (!line) return;
  const m = GET_RE.exec(line);
  if (m) {
    const id = +m[1];
    rows[id] = {
      pos: +m[2], speed: +m[3], load: +m[4], v: m[6], temp: +m[7],
      cur: +m[8], moving: +m[9], tq: m[10],
      un: m[11] != null ? +m[11] : +m[2],
      turns: m[12] != null ? +m[12] : 0,
    };
    paint();
    return;
  }
  const b = BATT_RE.exec(line);
  if (b) document.getElementById("pack").textContent = "pack " + b[1] + "V";
  if (line.startsWith("get id=")) return;
  log(line);
}

async function send(cmd) {
  if (!writer) return;
  log(">> " + cmd);
  await writer.write(cmd + "\n");
}

async function readLoop() {
  const dec = new TextDecoder();
  let buf = "";
  try {
    while (port.readable && alive) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        onLine(buf.slice(0, i).replace(/\r$/, "").trim());
        buf = buf.slice(i + 1);
      }
    }
  } catch (e) {
    if (alive) { setLink(false, "read fail"); log(String(e)); }
  }
}

async function poll() {
  if (polling) return;
  polling = true;
  while (alive) {
    if (live.checked) {
      await send("get");
      await new Promise(r => setTimeout(r, 1100));
    } else {
      await new Promise(r => setTimeout(r, 200));
    }
  }
  polling = false;
}

async function connect() {
  if (!navigator.serial) { alert("Chrome/Edge + http://localhost (Web Serial)"); return; }
  port = await navigator.serial.requestPort();
  await port.open({ baudRate: 115200 });
  writer = port.writable.getWriter();
  reader = port.readable.getReader();
  alive = true;
  setLink(true, "ACM connected");
  log("connected 115200");
  readLoop();
  poll();
}

async function disconnect() {
  alive = false;
  try { reader && reader.cancel(); } catch {}
  try { writer && writer.releaseLock(); } catch {}
  try { port && await port.close(); } catch {}
  writer = reader = port = null;
  setLink(false, "disconnected");
}

document.getElementById("btnConnect").onclick = () => connect().catch(e => { log(String(e)); alert(e); });
document.getElementById("btnDisc").onclick = disconnect;
document.getElementById("btnStop").onclick = async () => {
  await send("twist speed 0");
  await send("estop");
  await send("twist servo");
  await send("hand free");
  await send("twist free");
};
document.getElementById("handTo").onclick = () => send("hand to " + document.getElementById("handN").value);
document.getElementById("twistTo").onclick = () => send("twist to " + document.getElementById("twistN").value);
document.getElementById("spdGo").onclick = () => send("twist speed " + document.getElementById("spdN").value);
document.getElementById("rawGo").onclick = () => {
  const v = document.getElementById("raw").value.trim();
  if (v) send(v);
};
document.getElementById("raw").addEventListener("keydown", e => { if (e.key === "Enter") document.getElementById("rawGo").click(); });
document.querySelectorAll("button[data-cmd]").forEach(b => {
  b.onclick = () => send(b.getAttribute("data-cmd"));
});
setLink(false, "disconnected");
paint();
