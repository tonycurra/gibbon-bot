"""Task registry — single source of truth for what each task means.

Each entry defines:
  - label:           human-friendly name shown in the UI
  - criterion_text:  one-line description of the success criterion
  - rewards:         dict of reward-weight overrides applied on top of env.py
  - progress(m):     function returning a 0..1 score from a metrics dict m
                     (the same dict logged to MLflow by train_mjx.py)

Add a new task by appending to TASKS. The dashboard's task dropdown is
populated from this dict via /api/tasks/registry.
"""

from typing import Callable

_HEIGHT_TARGET = 0.45     # torso z to qualify as "tall stand"
_HEIGHT_FLOOR  = 0.18     # below this z, basically lying down
_UPRIGHT_FLOOR = 0.6      # quat_w below this = on its side / flipped
_JVEL_CAP      = 4.0      # joint-velocity norm where the energy term saturates


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _height_score(mean_z: float, target: float = _HEIGHT_TARGET,
                  floor: float = _HEIGHT_FLOOR) -> float:
    """0 at or below floor, 1 at or above target, linear between."""
    if target <= floor:
        return 0.0
    return _clamp01((mean_z - floor) / (target - floor))


def _upright_score(mean_upright: float, target: float = 0.95,
                   floor: float = _UPRIGHT_FLOOR) -> float:
    """quat_w → 0..1. quat_w of 1.0 = perfectly upright."""
    if target <= floor:
        return 0.0
    return _clamp01((mean_upright - floor) / (target - floor))


def _energy_penalty(mean_jvel: float, cap: float = _JVEL_CAP) -> float:
    """High joint velocity = high energy = lower score. Saturates at cap."""
    return _clamp01(mean_jvel / cap)


def _stand_progress(m: dict) -> float:
    """Stand = height + uprightness − energy. Weighted 0.5 / 0.4 / 0.1."""
    z   = float(m.get("mean_z", 0.0))
    upr = float(m.get("mean_upright", 0.0))
    jv  = float(m.get("mean_jvel", 0.0))
    return _clamp01(0.5 * _height_score(z)
                  + 0.4 * _upright_score(upr)
                  + 0.1 * (1.0 - _energy_penalty(jv)))


def _walk_progress(m: dict) -> float:
    """Walk = blend of 'reached tree' success rate and 'how close' distance.
    success_frac dominates once any envs reach the tree."""
    sf = float(m.get("success_frac", 0.0))
    md = float(m.get("mean_dist", 3.0))
    # 3m = spawn-to-tree distance — 1.0 means right on top of it.
    distance_score = _clamp01((3.0 - md) / 3.0)
    # success_frac counts twice as much as distance progress.
    return _clamp01(0.66 * sf + 0.34 * distance_score)


TASKS = {
    "stand": {
        "label": "Stand tall",
        "criterion_text": f"torso ≥ {_HEIGHT_TARGET:.2f} m and upright > 0.95, low motor energy",
        "rewards": {
            "W_PROGRESS": 0.0, "W_FWD_VEL": 0.0, "W_HEADING": 0.0,
            "W_PROXIMITY_TREE": 0.0, "W_SUCCESS": 0.0,
            "W_UPRIGHT": 12.0, "W_HEIGHT": 5.0,
        },
        "progress": _stand_progress,
    },
    "walk": {
        "label": "Walk to olive tree",
        "criterion_text": "reach within 0.6 m of the tree at (3, 0)",
        "rewards": {},  # env.py defaults are already walk-tuned
        "progress": _walk_progress,
    },
}


def list_tasks() -> list[dict]:
    """Public registry view for the dashboard."""
    return [
        {"id": tid, "label": t["label"], "criterion": t["criterion_text"]}
        for tid, t in TASKS.items()
    ]


def task_progress(metrics: dict, task: str) -> float:
    """0..1 progress toward the named task's success criterion.
    Unknown task → 0."""
    cfg = TASKS.get(task)
    if cfg is None:
        return 0.0
    fn: Callable[[dict], float] = cfg["progress"]
    try:
        return float(fn(metrics))
    except Exception:
        return 0.0


def task_info(task: str) -> dict:
    """Label + criterion text for the named task. Unknown → minimal record."""
    cfg = TASKS.get(task)
    if cfg is None:
        return {"id": task, "label": task, "criterion": "(unknown task)"}
    return {"id": task, "label": cfg["label"], "criterion": cfg["criterion_text"]}


def task_reward_overrides(task: str) -> dict:
    """Per-task reward weight overrides (applied on top of env.py defaults)."""
    cfg = TASKS.get(task)
    return dict(cfg["rewards"]) if cfg else {}
