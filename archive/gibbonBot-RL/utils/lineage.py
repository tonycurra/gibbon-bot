"""
Run lineage tracking — records hardware params and inheritance chain for each training run.

Every training run appends one entry to runs/<task>/lineage.json.
This lets you trace exactly which physics settings each generation used
and which checkpoint it inherited its weights from.
"""

import json
import pathlib
import datetime
import re
import xml.etree.ElementTree as ET


def _scene_params(scene_path: pathlib.Path) -> dict:
    """Extract key physics params from scene.xml."""
    try:
        tree = ET.parse(scene_path)
        root = tree.getroot()
        actuators = root.findall(".//position")
        kp_values = list({a.get("kp") for a in actuators if a.get("kp")})
        force_ranges = list({a.get("forcerange") for a in actuators if a.get("forcerange")})
        return {
            "kp": kp_values[0] if len(kp_values) == 1 else kp_values,
            "forcerange": force_ranges[0] if len(force_ranges) == 1 else force_ranges,
        }
    except Exception:
        return {}


def _env_params(env_file: pathlib.Path) -> dict:
    """Extract key constants from a stand_env.py / ape_env.py file."""
    params = {}
    try:
        text = env_file.read_text()
        for name in ("PHYSICS_STEPS_PER_ACTION", "TARGET_Z", "MIN_HEIGHT",
                     "SPAWN_CLEARANCE", "MAX_STEPS"):
            m = re.search(rf"^{name}\s*=\s*([^\s#\n]+)", text, re.MULTILINE)
            if m:
                try:
                    params[name] = float(m.group(1))
                except ValueError:
                    params[name] = m.group(1)
    except Exception:
        pass
    return params


def record(
    runs_dir: pathlib.Path,
    task: str,
    n_envs: int,
    inherited_from: str | None,
    scene_path: pathlib.Path,
    env_file: pathlib.Path,
) -> None:
    """Append one lineage entry to runs_dir/lineage.json."""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "n_envs": n_envs,
        "inherited_from": inherited_from,
        "physics": _scene_params(scene_path),
        "env": _env_params(env_file),
    }

    log_path = runs_dir / "lineage.json"
    history = []
    if log_path.exists():
        try:
            history = json.loads(log_path.read_text())
        except Exception:
            pass

    history.append(entry)
    log_path.write_text(json.dumps(history, indent=2))
    print(f"[lineage] recorded run #{len(history)} — inherited from: {inherited_from or 'scratch'}")


def show(runs_dir: pathlib.Path) -> None:
    """Print the lineage table for a task's runs directory."""
    log_path = runs_dir / "lineage.json"
    if not log_path.exists():
        print("No lineage recorded yet.")
        return
    history = json.loads(log_path.read_text())
    print(f"\n{'#':<4} {'timestamp':<22} {'envs':<6} {'inherited from':<40} {'kp':<6} {'steps/act'}")
    print("-" * 100)
    for i, e in enumerate(history, 1):
        kp  = e.get("physics", {}).get("kp", "?")
        spa = e.get("env", {}).get("PHYSICS_STEPS_PER_ACTION", "?")
        inh = e.get("inherited_from") or "scratch"
        print(f"{i:<4} {e['timestamp']:<22} {e['n_envs']:<6} {inh:<40} {str(kp):<6} {spa}")
    print()
