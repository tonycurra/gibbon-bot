#!/usr/bin/env python3
"""CPU MuJoCo: one RoArm-M3 on a table. python look.py [--check]"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene.xml"

JOINTS = ("turn", "raise", "bend", "wrist", "twist", "hand")


def load(key: str = "zero"):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    return model, data


def report(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    names = [model.joint(i).name for i in range(model.njnt)]
    tcp = data.site("tcp").xpos
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  ngeom={model.ngeom}")
    print("joints:", " ".join(names))
    print("qpos  :", " ".join(f"{q:.4f}" for q in data.qpos))
    print(f"tcp   : [{tcp[0]:.4f} {tcp[1]:.4f} {tcp[2]:.4f}] m")


def check() -> int:
    model, data = load("zero")
    names = [model.joint(i).name for i in range(model.njnt)]
    if tuple(names) != JOINTS:
        print("unexpected joints:", names)
        return 1
    z0 = float(data.site("tcp").xpos[2])
    report(model, data)
    if z0 < 0.4:
        print(f"zero pose tcp too low ({z0:.3f} m) — kinematics?")
        return 1
    for _ in range(200):
        mujoco.mj_step(model, data)
    z = float(data.site("tcp").xpos[2])
    print("after 200 steps tcp z", round(z, 4), "qpos", " ".join(f"{q:.4f}" for q in data.qpos))
    if z < 0.3:
        print("arm collapsed onto table")
        return 1
    print("ok")
    return 0


def watch(key: str) -> None:
    import mujoco.viewer

    model, data = load(key)
    report(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="load, step, no window")
    p.add_argument("--home", action="store_true", help="stock elbow-bent pose instead of URDF zeros")
    args = p.parse_args()
    if args.check:
        return check()
    watch("home" if args.home else "zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
