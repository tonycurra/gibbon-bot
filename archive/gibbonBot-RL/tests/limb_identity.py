#!/usr/bin/env python3
"""Limb identity test — validates that the 4 morphologically identical
limbs really are identical under load (URDF sanity check).

Sends an identical canned trajectory to each limb in turn and measures
the response: torque sum, position error, settling time. If FL/FR/BL/BR
disagree by more than a tolerance, the URDF is asymmetric.

Run:
    python -m tests.limb_identity
"""
import pathlib
import sys
import numpy as np
import mujoco

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.limb_dynamics import LIMB_RANGES, per_limb

LIMBS = ("FL", "FR", "BL", "BR")
N_STEPS = 200
SCENE = ROOT / "data" / "scene.xml"


def run_one_limb(model, data, limb_name: str, command_amplitude: float = 0.5):
    """Send a single sinusoidal cycle to the given limb's actuators, hold others
    at zero. Return time-series of torque-sum + position-error for that limb."""
    sl = LIMB_RANGES[limb_name]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    torques = np.zeros(N_STEPS)
    pos_err = np.zeros(N_STEPS)
    for t in range(N_STEPS):
        ctrl = np.zeros(model.nu)
        # sinusoidal command on the named limb's actuators only
        for i, j in enumerate(range(sl.start, sl.stop)):
            ctrl[j] = command_amplitude * np.sin(2 * np.pi * t / 50.0 + i * 0.5)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        torques[t] = np.sum(np.abs(data.qfrc_actuator[6 + sl.start : 6 + sl.stop]))
        pos_err[t] = np.sum(np.abs(data.qpos[7 + sl.start : 7 + sl.stop] - ctrl[sl]))
    return {"torque_total": float(torques.sum()),
            "torque_peak":  float(torques.max()),
            "pos_err_mean": float(pos_err.mean())}


def main():
    print(f"Loading {SCENE}")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    print(f"\nRunning identical trajectory on each of 4 limbs ({N_STEPS} steps each)...")
    results = {limb: run_one_limb(model, data, limb) for limb in LIMBS}

    print(f"\n{'limb':<6}  {'torque_total':>14}  {'torque_peak':>14}  {'pos_err_mean':>14}")
    for limb in LIMBS:
        r = results[limb]
        print(f"{limb:<6}  {r['torque_total']:>14.4f}  {r['torque_peak']:>14.4f}  {r['pos_err_mean']:>14.6f}")

    # Pairwise asymmetry — should be near-zero for symmetric URDF
    print(f"\n{'pairwise asymmetry':<22}  {'torque_total_diff':>20}  {'pos_err_diff':>20}")
    pairs = [("FL", "FR"), ("BL", "BR"), ("FL", "BL"), ("FR", "BR")]
    max_asym = 0.0
    for a, b in pairs:
        d_t = abs(results[a]["torque_total"] - results[b]["torque_total"])
        d_p = abs(results[a]["pos_err_mean"] - results[b]["pos_err_mean"])
        rel = d_t / max(results[a]["torque_total"], 1e-6)
        max_asym = max(max_asym, rel)
        flag = "≠" if rel > 0.05 else " "
        print(f"{flag} {a}-{b:<19}  {d_t:>20.4f}  {d_p:>20.6f}")

    print(f"\nworst relative asymmetry: {max_asym*100:.2f}%")
    if max_asym > 0.05:
        print("FAIL — URDF is morphologically asymmetric beyond 5% tolerance")
        sys.exit(1)
    else:
        print("PASS — limbs are symmetric within tolerance ✓")


if __name__ == "__main__":
    main()
