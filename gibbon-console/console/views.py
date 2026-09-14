import json
import queue
import time

from django.db.models import Count
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import bridge, hat as hat_gold
from .catalog import CATALOG, JOINTS, SLUGS, WIZARD
from .models import Calibration, Limit, Note, Sample, Trial

PAGES = [
    ("home", "home"),
    ("calibration", "calibration"),
]
MORE = [
    ("trials", "trials"),
    ("hat", "hat"),
    ("raw", "hand"),
    ("info", "info"),
    ("flash", "flash"),
    ("advanced", "advanced"),
]


def ctx(page, crumb, extra=None):
    d = {
        "page": page,
        "pages": PAGES,
        "more": MORE,
        "crumb": crumb,
        "joints": JOINTS,
        "joint_slug": "",
        "motor_nav": False,
        "nav_prefix": "/j/",
        "catalog_json": mark_safe(json.dumps(CATALOG)),
    }
    if extra:
        d.update(extra)
    return d


def home(request):
    return render(request, "console/home.html", ctx("home", "home"))


def joint_page(request, slug):
    j = SLUGS.get(slug)
    if not j:
        return HttpResponse("unknown joint", status=404)
    return render(
        request,
        "console/joint.html",
        ctx("home", j["name"], {"joint": j, "joint_slug": j["slug"], "motor_nav": True}),
    )


def calibrate(request, slug=None):
    if slug:
        return redirect("/calibration/" + slug + "/")
    return redirect("/calibration/")


def calibration(request, slug=None):
    order = [SLUGS[w["slug"]] for w in WIZARD if w["slug"] in SLUGS]
    if not slug:
        return render(request, "console/calibrate.html", ctx("calibration", "calibration", {
            "cal_rows": order,
        }))
    j = SLUGS.get(slug)
    if not j:
        return HttpResponse("unknown joint", status=404)
    slugs = [x["slug"] for x in order]
    i = slugs.index(j["slug"]) if j["slug"] in slugs else -1
    nxt = order[i + 1] if i >= 0 and i + 1 < len(order) else None
    return render(
        request,
        "console/calibrate_joint.html",
        ctx("calibration", "calibration / " + j["name"], {
            "joint": j,
            "joint_slug": j["slug"],
            "motor_nav": True,
            "nav_prefix": "/calibration/",
            "next_joint": nxt,
            "cal_rows": order,
        }),
    )


def info(request):
    return render(request, "console/info.html", ctx("info", "info"))


def hat(request):
    motors = hat_gold.MOTORS
    by_id = {m["id"]: m for m in motors}
    get_gold = [{**by_id[r["id"]], **r} for r in hat_gold.GET_GOLD]
    return render(request, "console/hat.html", ctx("hat", "hat", {
        "chip": hat_gold.CHIP,
        "motors": motors,
        "fields": hat_gold.FIELDS,
        "get_gold": get_gold,
        "batt_gold": hat_gold.BATT_GOLD,
        "hand_rest": hat_gold.HAND_REST,
        "hand_net": hat_gold.HAND_NET,
        "bench_off": hat_gold.BENCH_OFF,
        "bench_on": hat_gold.BENCH_ON,
        "get_on": hat_gold.GET_ON,
        "batt_on": hat_gold.BATT_ON,
        "batt_off": hat_gold.BATT_OFF,
        "info_off": hat_gold.INFO_OFF,
        "state_off": hat_gold.STATE_OFF,
        "unplug": hat_gold.UNPLUG,
        "timing": hat_gold.TIMING,
        "memory": hat_gold.MEMORY,
        "modes": hat_gold.MODES,
        "links": hat_gold.LINKS,
    }))


def raw(request):
    return render(request, "console/raw.html", ctx("raw", "raw"))


def flash(request):
    return render(request, "console/flash.html", ctx("flash", "flash"))


def advanced(request):
    return render(request, "console/advanced.html", ctx("advanced", "advanced"))


def graphs(request):
    return render(request, "console/graphs.html", ctx("home", "graphs"))


def act(request):
    return render(request, "console/act.html", ctx("home", "act"))


def trials(request):
    rows = Trial.objects.annotate(
        n_cmd=Count("commands", distinct=True),
        n_samp=Count("samples", distinct=True),
        n_note=Count("notes", distinct=True),
    )[:80]
    return render(request, "console/trials.html", ctx("trials", "trials", {"trials": rows}))


def trial_detail(request, pk):
    t = get_object_or_404(Trial, pk=pk)
    cmds = list(t.commands.order_by("-t_ms")[:200])
    notes = list(t.notes.order_by("-t_ms")[:200])
    events = sorted(
        [{"kind": "cmd", "t_ms": c.t_ms, "text": c.text} for c in cmds]
        + [{"kind": "note", "t_ms": n.t_ms, "text": n.text} for n in notes],
        key=lambda x: -x["t_ms"],
    )[:200]
    return render(request, "console/trial.html", ctx("trials", "trials / %s" % t, {
        "trial": t,
        "events": events,
        "n_samp": t.samples.count(),
        "n_cmd": t.commands.count(),
        "n_note": t.notes.count(),
    }))


def api_status(request):
    st = bridge.status()
    try:
        cur = Trial.objects.filter(ended__isnull=True).first()
        st["trial"] = {"id": cur.id, "title": cur.title or ("trial %d" % cur.id)} if cur else None
    except Exception:
        st["trial"] = None
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


def _limit_row(row):
    return {
        "servo": row.servo, "home": row.home, "amin": row.amin, "amax": row.amax,
        "note": row.note, "updated": row.updated.isoformat() if row.updated else None,
    }


@csrf_exempt
def api_limits(request):
    if request.method == "GET":
        return JsonResponse({"limits": [_limit_row(x) for x in Limit.objects.order_by("servo")]})
    if request.method != "POST":
        return JsonResponse({"err": "method"}, status=405)
    data = json.loads(request.body or b"{}")
    servo = int(data.get("servo") or 0)
    if servo < 11 or servo > 17:
        return JsonResponse({"err": "servo 11-17"}, status=400)
    row, _ = Limit.objects.get_or_create(servo=servo)
    for k in ("home", "amin", "amax"):
        if k not in data:
            continue
        if data[k] is None or data[k] == "":
            setattr(row, k, None)
        else:
            setattr(row, k, int(data[k]))
    if "note" in data:
        row.note = str(data.get("note") or "")[:240]
    row.save()
    return JsonResponse({"limit": _limit_row(row)})


def _limits_blob():
    out = {}
    for row in Limit.objects.order_by("servo"):
        out[str(row.servo)] = {
            "home": row.home, "amin": row.amin, "amax": row.amax, "note": row.note,
        }
    return out


def _cal_row(row):
    return {
        "id": row.id,
        "created": row.created.isoformat() if row.created else None,
        "board": row.board,
        "fw": row.fw,
        "via": row.via,
        "joints": row.joints,
    }


@csrf_exempt
def api_calibrations(request):
    if request.method == "GET":
        qs = Calibration.objects.all()
        board = (request.GET.get("board") or "").strip()
        if board:
            qs = qs.filter(board=board)
        return JsonResponse({"calibrations": [_cal_row(x) for x in qs[:40]]})
    if request.method != "POST":
        return JsonResponse({"err": "method"}, status=405)
    data = json.loads(request.body or b"{}")
    board = str(data.get("board") or "unknown")[:32]
    fw = str(data.get("fw") or "")[:16]
    via = str(data.get("via") or "")[:8]
    joints = data.get("joints")
    if not isinstance(joints, dict):
        joints = _limits_blob()
    row = Calibration.objects.create(board=board, fw=fw, via=via, joints=joints)
    return JsonResponse({"ok": True, "calibration": _cal_row(row)})
