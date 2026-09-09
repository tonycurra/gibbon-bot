import json
import queue
import time

from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import bridge
from .models import Note, Sample, Trial

PAGES = [
    ("live", "live"),
    ("graphs", "graphs"),
    ("act", "act"),
    ("trials", "trials"),
]


def ctx(page, crumb, extra=None):
    d = {"page": page, "pages": PAGES, "crumb": crumb}
    if extra:
        d.update(extra)
    return d


def live(request):
    return render(request, "console/live.html", ctx("live", "live"))


def graphs(request):
    return render(request, "console/graphs.html", ctx("graphs", "graphs"))


def act(request):
    return render(request, "console/act.html", ctx("act", "act"))


def trials(request):
    rows = Trial.objects.all()[:80]
    return render(request, "console/trials.html", ctx("trials", "trials", {"trials": rows}))


def trial_detail(request, pk):
    t = get_object_or_404(Trial, pk=pk)
    cmds = t.commands.order_by("-t_ms")[:80]
    return render(request, "console/trial.html", ctx("trials", "trials / %s" % t, {"trial": t, "cmds": cmds}))


def api_status(request):
    st = bridge.status()
    cur = Trial.objects.filter(ended__isnull=True).first()
    st["trial"] = {"id": cur.id, "title": cur.title or ("trial %d" % cur.id)} if cur else None
    return JsonResponse(st)


def api_rx(request):
    q = bridge.subscribe()

    def gen():
        yield "data: bridge\n\n"
        try:
            while True:
                try:
                    line = q.get(timeout=15)
                    yield "data: " + line + "\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        except GeneratorExit:
            pass
        finally:
            bridge.unsubscribe(q)

    resp = StreamingHttpResponse(gen(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


@csrf_exempt
@require_POST
def api_tx(request):
    body = request.body.decode("utf-8", "replace")
    try:
        bridge.write_serial(body)
        return HttpResponse(status=204)
    except OSError as e:
        return JsonResponse({"err": str(e)}, status=503)


@csrf_exempt
def api_trial(request):
    cur = Trial.objects.filter(ended__isnull=True).first()
    if request.method == "GET":
        if not cur:
            return JsonResponse({"trial": None})
        return JsonResponse({"trial": {"id": cur.id, "title": cur.title, "note": cur.note}})
    if request.method == "POST":
        data = json.loads(request.body or b"{}")
        if cur:
            cur.ended = timezone.now()
            cur.save(update_fields=["ended"])
        t = Trial.objects.create(
            title=(data.get("title") or "").strip()[:200],
            note=(data.get("note") or "").strip(),
            via=bridge.via(),
        )
        return JsonResponse({"trial": {"id": t.id, "title": t.title}})
    return JsonResponse({"err": "method"}, status=405)


@csrf_exempt
@require_POST
def api_trial_stop(request):
    cur = Trial.objects.filter(ended__isnull=True).first()
    if cur:
        cur.ended = timezone.now()
        cur.save(update_fields=["ended"])
    return JsonResponse({"ok": True})


@csrf_exempt
@require_POST
def api_trial_note(request):
    cur = Trial.objects.filter(ended__isnull=True).first()
    if not cur:
        return JsonResponse({"err": "no trial"}, status=400)
    data = json.loads(request.body or b"{}")
    text = (data.get("text") or "").strip()
    if not text:
        return JsonResponse({"err": "empty"}, status=400)
    Note.objects.create(trial=cur, t_ms=int(time.time() * 1000), text=text)
    if not cur.note:
        cur.note = text
        cur.save(update_fields=["note"])
    else:
        cur.note = cur.note + "\n" + text
        cur.save(update_fields=["note"])
    return JsonResponse({"ok": True})


def api_samples(request, pk):
    qs = Sample.objects.filter(trial_id=pk).order_by("t_ms")
    servo = request.GET.get("servo")
    if servo:
        qs = qs.filter(servo=int(servo))
    qs = qs[:8000]
    out = [{
        "t": s.t_ms, "id": s.servo, "pos": s.pos, "un": s.un, "turns": s.turns,
        "speed": s.speed, "load": s.load, "v": s.v, "temp": s.temp, "cur": s.cur,
        "moving": s.moving, "tq": s.tq, "mode": s.mode,
    } for s in qs]
    return JsonResponse({"samples": out})
