const OPS = {
  11: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  12: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  13: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  14: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  15: [["free", "free"], ["hold", "hold"], ["nudge", "nudge Δ"], ["to", "to pos"]],
  16: [["free", "free"], ["hold", "hold"], ["to", "to pos"], ["mid", "mid"], ["speed", "wheel speed"], ["servo", "servo"], ["wheel", "wheel"], ["here", "here"], ["estop", "twist estop"]],
  17: [["open", "open"], ["close", "close"], ["to", "to pos"], ["nudge", "nudge Δ"], ["free", "free"], ["hold", "hold"]],
};
const HINT = {
  11: "base yaw. hop ≤80.",
  12: "shoulder dual-drive with 13. one motor fights the other.",
  13: "shoulder dual-drive with 12. one motor fights the other.",
  14: "elbow. hop ≤80.",
  15: "wrist nod. hop ≤80.",
  16: "twist roll. hop ≤900. wheel speed needs twist wheel first.",
  17: "hand 689–2075. open 689, close 2008.",
};
const NEED = new Set(["nudge", "to", "speed"]);

const joint = $("joint");
JOINTS.forEach(j => {
  const o = document.createElement("option");
  o.value = j.id; o.textContent = j.name + "  id " + j.id;
  joint.appendChild(o);
});

function fillOps() {
  const id = +joint.value;
  $("op").innerHTML = OPS[id].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  $("actHint").textContent = HINT[id];
  showVal();
}

function showVal() {
  const need = NEED.has($("op").value);
  $("valWrap").hidden = !need;
  if ($("op").value === "nudge") { $("val").value = 40; $("val").min = -80; $("val").max = 80; }
  if ($("op").value === "to") { $("val").value = 2000; $("val").min = 0; $("val").max = 4095; }
  if ($("op").value === "speed") { $("val").value = 0; $("val").min = -3073; $("val").max = 3073; }
}

joint.onchange = fillOps;
$("op").onchange = showVal;
fillOps();

$("actForm").onsubmit = e => {
  e.preventDefault();
  const id = +joint.value;
  const op = $("op").value;
  const n = $("val").value;
  let cmd;
  if (id === 17 && (op === "open" || op === "close" || op === "to" || op === "free")) {
    cmd = op === "to" ? "hand to " + n : "hand " + op;
  } else if (id === 16 && op !== "nudge" && op !== "hold") {
    if (op === "to") cmd = "twist to " + n;
    else if (op === "speed") cmd = "twist speed " + n;
    else if (op === "estop") cmd = "twist estop";
    else cmd = "twist " + op;
  } else if (op === "to") cmd = "id " + id + " to " + n;
  else if (op === "nudge") cmd = "id " + id + " nudge " + n;
  else cmd = "id " + id + " " + op;
  send(cmd);
};

$("btnMoveSet").onclick = async () => {
  await send("move speed " + $("mvSpeed").value);
  await send("move acc " + $("mvAcc").value);
};
