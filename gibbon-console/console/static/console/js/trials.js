$("startForm").onsubmit = async e => {
  e.preventDefault();
  await fetch("/api/trial", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: $("title").value, note: $("note").value }),
  });
  location.reload();
};
$("btnStopTrial").onclick = async () => {
  await fetch("/api/trial/stop", { method: "POST" });
  location.reload();
};
$("noteForm").onsubmit = async e => {
  e.preventDefault();
  const r = await fetch("/api/trial/note", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: $("mark").value }),
  });
  if (r.ok) { $("mark").value = ""; location.reload(); }
  else log(await r.text());
};
