#!/usr/bin/env python3
"""Watchdog for the H100 training loop.

Runs on the same box as training (in its own tmux pane). Polls the train
log every 60s and:
  - writes runs/<task>/health.json with current status
  - detects stalls (no new line in N min, sps drop, NaN losses)
  - on stall: kills the train tmux session and restarts it (auto-resume picks
    up the latest checkpoint, so at most ~1 min of compute is lost)

You should NOT run this locally — it operates on the local filesystem of the
training box. Launch it from `remote.sh start` or as a separate tmux pane.

Usage on remote:
  python3 monitor.py                              # default settings
  STALL_MIN=2 RESTART=0 python3 monitor.py        # detect-only, no auto-restart
"""
import json
import os
import pathlib
import re
import subprocess
import time

RUN_DIR     = pathlib.Path("runs/ape_walk_mjx")
LOG_PATH    = RUN_DIR / "train.log"
HEALTH_PATH = RUN_DIR / "health.json"
SELF_LOG    = RUN_DIR / "monitor.log"
TMUX_NAME   = os.environ.get("TMUX_NAME", "train")
POLL_SEC    = int(os.environ.get("POLL_SEC", 60))
STALL_MIN   = float(os.environ.get("STALL_MIN", 8))   # JIT compile can be 3-5min; need headroom
SPS_DROP    = float(os.environ.get("SPS_DROP", 0.5))   # alarm if sps drops below 50% of best
RESTART     = int(os.environ.get("RESTART", 1))
ENVS        = int(os.environ.get("ENVS", 16384))

# matches a per-update line:    1   16,384       45,210      0.123       2.901
LINE_RE = re.compile(r"^\s*(\d+)\s+([\d,]+)\s+([\d,]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


def _last_metric_lines(log_path: pathlib.Path, n: int = 200) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    with log_path.open("rb") as f:
        try:
            f.seek(-200_000, 2)
        except OSError:
            f.seek(0)
        tail = f.read().decode("utf-8", errors="replace").splitlines()
    mtime = log_path.stat().st_mtime
    for line in tail[-n*4:]:
        m = LINE_RE.match(line)
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
    return out


def _tmux_alive(name: str) -> bool:
    r = subprocess.run(["tmux", "has-session", "-t", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def _restart_training():
    _log(f"[monitor] restarting training in tmux session '{TMUX_NAME}'")
    subprocess.run(["tmux", "kill-session", "-t", TMUX_NAME],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    train_cmd = (
        f"cd {os.getcwd()} && export JAX_COMPILATION_CACHE_DIR=~/.jax_cache PYTHONUNBUFFERED=1 && "
        f"mkdir -p {RUN_DIR} && : > {LOG_PATH} && "
        f"( stdbuf -oL -eL python3 -u train_walk_mjx.py --no-watch --envs {ENVS} "
        f"  --save-rollouts-every 50 --rollout-envs 256; "
        f"  ec=$?; echo \"[remote.sh] python exited with code $ec\" ) "
        f"2>&1 | tee {LOG_PATH}"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_NAME, train_cmd])


def _write_health(status: dict):
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_PATH.write_text(json.dumps(status, indent=2))


def _log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        SELF_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SELF_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    _log(f"[monitor] watching {LOG_PATH}; poll={POLL_SEC}s stall={STALL_MIN}min restart={RESTART}")
    best_sps = 0
    last_metric_at = time.time()
    last_seen_update = -1
    while True:
        now = time.time()
        metrics = _last_metric_lines(LOG_PATH)
        latest = metrics[-1] if metrics else None
        log_mtime = LOG_PATH.stat().st_mtime if LOG_PATH.exists() else 0
        idle_s = now - log_mtime if log_mtime else 0
        alive = _tmux_alive(TMUX_NAME)

        if latest:
            if latest["update"] != last_seen_update:
                last_seen_update = latest["update"]
                last_metric_at = now
                best_sps = max(best_sps, latest["sps"])

        stall = (now - last_metric_at) > STALL_MIN * 60 and last_seen_update >= 0
        sps_low = latest is not None and best_sps > 0 and latest["sps"] < SPS_DROP * best_sps
        nan_loss = any(
            (m["mean_rew"] != m["mean_rew"]) or (m["mean_dist"] != m["mean_dist"])  # NaN check
            for m in metrics[-3:]
        )
        problem = (not alive) or stall or sps_low or nan_loss

        status = {
            "ts":               time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tmux_alive":       alive,
            "log_idle_seconds": int(idle_s),
            "last_update":      latest["update"] if latest else None,
            "last_steps":       latest["steps"] if latest else None,
            "last_sps":         latest["sps"] if latest else None,
            "best_sps":         best_sps,
            "last_mean_rew":    latest["mean_rew"] if latest else None,
            "last_mean_dist":   latest["mean_dist"] if latest else None,
            "stall":            bool(stall),
            "sps_low":          bool(sps_low),
            "nan_loss":         bool(nan_loss),
            "ok":               not problem,
        }
        _write_health(status)
        _log(f"[monitor] alive={alive} update={status['last_update']} sps={status['last_sps']} "
             f"mean_rew={status['last_mean_rew']} idle={int(idle_s)}s "
             f"{'PROBLEM' if problem else 'ok'}")

        if problem and RESTART:
            _log(f"[monitor] problem detected — alive={alive} stall={stall} sps_low={sps_low} nan={nan_loss}")
            _restart_training()
            last_metric_at = time.time()
            best_sps = 0
            last_seen_update = -1
            time.sleep(60)  # let the new run get past JIT before checking again
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
