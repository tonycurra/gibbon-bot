const HAND = (window.CATALOG && window.CATALOG.hand) || { open: 689, close: 2008, to_max: 2075 };
const ID = 17;
const LO = HAND.open;
const HI = HAND.to_max;
let lastSpd = null;
let lastAcc = null;
let tripped = 0;
let poll = 0;

function rec() {
  return G.rows[ID];
}

function clamp(n) {
  n = +n;
  if (n < LO) return LO;
  if (n > HI) return HI;
  return n;
}

function paint() {
  const r = rec();
  const miss = !r || r.miss;
  $("lvPos").textContent = miss ? "—" : r.pos;
  $("lvUn").textContent = miss ? "un —" : ("un " + r.un + "  turns " + r.turns);
  $("lvLoad").textContent = miss ? "—" : r.load;
  $("lvTq").textContent = miss ? "tq —" : ("tq " + r.tq + (r.moving ? "  mv" : ""));
  $("lvCur").textContent = miss ? "—" : r.cur;
  $("lvVT").textContent = miss ? "—" : ((r.v != null ? r.v + " V" : "V —") + "  " + (r.temp != null ? r.temp + " °C" : ""));
  $("lvSpd").textContent = miss ? "—" : r.speed;
  $("lvMode").textContent = miss ? "—" : modeOf(r);
  $("lvPid").textContent = (r && r.kp != null) ? (r.kp + " / " + r.kd + " / " + r.ki) : "—";
  $("lvHex").textContent = (r && r.hex) ? r.hex : "—";
  if (!miss && r.raw) $("rawGets").textContent = r.raw;
  const tqOn = r && (r.tq === "1" || r.tq === 1);
  const sense = +$("sense").value;
  $("senseVal").textContent = sense === 0 ? "off" : String(sense);
  const el = $("tripEl");
  if (tripped) {
    el.className = "guard bad";
    el.textContent = "tripped |load| — STOP + free. lower load or free, then move again";
  } else if (tqOn && sense > 0) {
    el.className = "guard on";
    el.textContent = "watching |load| ≥ " + sense + " → estop + hand free";
  } else {
    el.className = "guard";
    el.textContent = "sensitivity idle" + (sense === 0 ? " (page trip off)" : "");
  }
}

async function applyMove() {
  const spd = +$("mvSpeed").value;
  const acc = +$("mvAcc").value;
  if (spd !== lastSpd) { lastSpd = spd; await send("move speed " + spd); }
  if (acc !== lastAcc) { lastAcc = acc; await send("move acc " + acc); }
}

async function goTo(n) {
  tripped = 0;
  await applyMove();
  await send("hand to " + clamp(n));
}

$("goal").oninput = () => { $("goalVal").textContent = $("goal").value; };
$("sense").oninput = paint;
$("sense").onchange = () => {
  const n = +$("sense").value;
  if (G.transport) send("guard hand " + n);
};

$("btnOpen").onclick = () => goTo(LO);
$("btnClose").onclick = () => goTo(HAND.close);
$("btnGo").onclick = () => goTo($("goal").value);
$("btnHold").onclick = () => send("id 17 hold");
$("btnFree").onclick = () => { tripped = 0; send("hand free"); };
$("btnWho").onclick = () => send("who 17");
$("btnRaw").onclick = () => send("raw 17");

G.hooks.push((ev, sample) => {
  if (ev === "sample" && sample && sample.id === ID) {
    const sense = +$("sense").value;
    const tqOn = sample.tq === "1" || sample.tq === 1;
    if (!tripped && tqOn && sense > 0 && Math.abs(+sample.load) >= sense) {
      tripped = 1;
      send("estop");
      send("hand free");
    }
    if (tripped && Math.abs(+sample.load || 0) < sense * 0.4) tripped = 0;
  }
  paint();
});

function tick() {
  if (!G.transport || poll) return;
  poll = 1;
  send("get 17", true).finally(() => { poll = 0; });
}

setInterval(tick, 220);
paint();
if (G.transport) {
  send("who 17", true);
  send("guard", true);
}
