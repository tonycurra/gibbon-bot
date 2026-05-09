#!/usr/bin/env python3
"""Autopilot helpers — used by Claude on wakeup loops to advance the
walk-to-tree training plan (see PLAN.md).

Not meant to run unattended on its own — this is a toolkit:
  python -m utils.autopilot status     — read current state, summarize
  python -m utils.autopilot phase A    — switch params to phase A
  python -m utils.autopilot phase B    — switch params to phase B
  python -m utils.autopilot phase C    — switch params to phase C
  python -m utils.autopilot phase D    — switch params to phase D
  python -m utils.autopilot fallback   — apply current phase's stall fallback
  python -m utils.autopilot log "msg"  — append a line to autopilot.log
"""
import argparse
import json
import pathlib
import re
import sys
import time

ROOT       = pathlib.Path(__file__).parent.parent
PARAMS     = ROOT / "envs" / "params_walk_mjx.py"
RUN_DIR    = ROOT / "runs" / "ape_walk_mjx"
STATE_PATH = RUN_DIR / "autopilot_state.json"
LOG_PATH   = RUN_DIR / "autopilot.log"
TRAIN_LOG  = RUN_DIR / "train.log"

# (phase, knobs dict, success criterion description)
PHASES = {
    "A": {
        "desc": "Stand & survive — don't flip, hold torso near TARGET_Z",
        "knobs": {
            # explicit values — phase A keeps the current defaults
            "FLIP_PENALTY": 200.0, "W_GROUND": 50.0, "ENT_COEF": 0.05,
            "W_PROGRESS": 80.0, "W_HEIGHT": 4.0, "LEARNING_RATE": 0.00037,
            "SUCCESS_DIST": 0.6, "W_SUCCESS": 50.0, "CLIP_EPS": 0.15,
            "W_SMOOTH": 0.1, "W_JVEL": 0.005,
        },
        "fallback": {
            "FLIP_PENALTY": 80.0, "W_GROUND": 25.0, "ENT_COEF": 0.10,
        },
        "success": "mean_rew > 3.0 sustained 10 updates",
    },
    "B": {
        "desc": "Walk toward tree — progress reward dominant",
        "knobs": {"W_PROGRESS": 100.0, "W_HEIGHT": 6.0},
        "fallback": {
            "W_PROGRESS": 150.0, "LEARNING_RATE": 0.00020, "ENT_COEF": 0.03,
        },
        "success": "mean_dist < 1.8 sustained 10 updates",
    },
    "C": {
        "desc": "Reach base — tighten success trigger",
        "knobs": {"SUCCESS_DIST": 0.5, "W_SUCCESS": 100.0},
        "fallback": {"CLIP_EPS": 0.10},
        "success": "mean_dist < 0.7 sustained 10 updates",
    },
    "D": {
        "desc": "Polish — smoother gait",
        "knobs": {"W_SMOOTH": 0.5, "W_JVEL": 0.02},
        "fallback": {},
        "success": "success rate > 70%, no flips for 50 episodes",
    },
}


METRIC_RE = re.compile(r"^\s*(\d+)\s+([\d,]+)\s+([\d,]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "phase": "A",
        "phase_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase_started_steps": 0,
        "fallback_applied": False,
        "best_mean_rew": float("-inf"),
        "best_mean_dist": float("inf"),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_state(s: dict):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    s["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATE_PATH.write_text(json.dumps(s, indent=2))


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def parse_metrics(n: int = 50) -> list[dict]:
    """Return last N parsed metric rows from train.log."""
    if not TRAIN_LOG.exists():
        return []
    out = []
    for line in TRAIN_LOG.read_text(errors="replace").splitlines():
        m = METRIC_RE.match(line)
        if m:
            try:
                out.append({
                    "update":   int(m.group(1)),
                    "steps":    int(m.group(2).replace(",", "")),
                    "sps":      int(m.group(3).replace(",", "")),
                    "mean_rew": float(m.group(4)),
                    "mean_dist": float(m.group(5)),
                })
            except ValueError:
                pass
    return out[-n:]


def update_params_file(knobs: dict):
    """Edit envs/params_walk_mjx.py in place — replace `KEY = old` with `KEY = new`."""
    text = PARAMS.read_text()
    for k, v in knobs.items():
        # Match "KEY = <number>" possibly with float/int and optional inline comment
        pattern = re.compile(rf"^({re.escape(k)}\s*=\s*)[-\d.eE+]+", re.MULTILINE)
        new_text, n_subs = pattern.subn(rf"\g<1>{v}", text)
        if n_subs == 0:
            log(f"[autopilot] WARNING: param {k} not found in {PARAMS}")
        text = new_text
    PARAMS.write_text(text)


def apply_phase(phase: str):
    if phase not in PHASES:
        print(f"unknown phase {phase!r}; pick from {list(PHASES)}", file=sys.stderr)
        sys.exit(2)
    cfg = PHASES[phase]
    update_params_file(cfg["knobs"])
    s = load_state()
    s["phase"] = phase
    s["phase_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    metrics = parse_metrics(1)
    s["phase_started_steps"] = metrics[-1]["steps"] if metrics else 0
    s["fallback_applied"] = False
    save_state(s)
    log(f"[autopilot] phase → {phase}: {cfg['desc']}")
    log(f"[autopilot] knobs: {cfg['knobs']}")
    log(f"[autopilot] success criterion: {cfg['success']}")


def apply_fallback():
    s = load_state()
    cfg = PHASES[s["phase"]]
    if not cfg["fallback"]:
        log(f"[autopilot] phase {s['phase']} has no fallback defined")
        return
    update_params_file(cfg["fallback"])
    s["fallback_applied"] = True
    save_state(s)
    log(f"[autopilot] fallback applied for phase {s['phase']}: {cfg['fallback']}")


def status():
    s = load_state()
    metrics = parse_metrics(20)
    print(f"phase: {s['phase']}  ({PHASES[s['phase']]['desc']})")
    print(f"phase started: {s['phase_started_at']} at step {s['phase_started_steps']:,}")
    print(f"fallback applied: {s['fallback_applied']}")
    print(f"best mean_rew so far: {s['best_mean_rew']}")
    print(f"best mean_dist so far: {s['best_mean_dist']}")
    print(f"success criterion: {PHASES[s['phase']]['success']}")
    if metrics:
        print()
        print("recent metrics:")
        for m in metrics[-5:]:
            print(f"  update={m['update']} steps={m['steps']:,} sps={m['sps']:,} "
                  f"mean_rew={m['mean_rew']:.3f} mean_dist={m['mean_dist']:.3f}")
    else:
        print("(no metric lines parsed from train.log yet)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p_phase = sub.add_parser("phase"); p_phase.add_argument("name")
    sub.add_parser("fallback")
    p_log = sub.add_parser("log"); p_log.add_argument("msg")
    args = ap.parse_args()

    if args.cmd == "status": status()
    elif args.cmd == "phase": apply_phase(args.name)
    elif args.cmd == "fallback": apply_fallback()
    elif args.cmd == "log": log(args.msg)


if __name__ == "__main__":
    main()
