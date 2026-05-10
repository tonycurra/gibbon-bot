#!/usr/bin/env python3
"""Probe each joint with +0.5 then -0.5 and report which world direction the
limb tip moves. Used to determine per-limb sign corrections for the CPG."""
import pathlib, sys
import numpy as np
import mujoco

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "scene.xml"

def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    ACT_DIM = model.nu
    DT = float(model.opt.timestep)
    jnt_lo = np.array([model.jnt_range[i][0] for i in range(1, 1 + ACT_DIM)])
    jnt_hi = np.array([model.jnt_range[i][1] for i in range(1, 1 + ACT_DIM)])

    DEG = np.pi / 180.0
    rest_pose = np.tile(np.array([0.0, 0.0, 30.0*DEG, 0.0, 0.0, 0.0]), 4).astype(np.float32)

    LIMBS = ["FL", "FR", "BL", "BR"]
    JOINTS = ["yaw", "pitch", "elbow/knee", "roll", "wrist/ankle", "gripper/foot"]
    TIPS = {"FL": "fl_gripper_link", "FR": "fr_gripper_link",
            "BL": "bl_gripper_link", "BR": "br_gripper_link"}

    def settle_at(target_action):
        # Reset.
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qpos[2] = 0.50      # spawn high so limb can move freely
        data.qpos[3] = 1.0
        # Map action → ctrl with rest_pose mapping.
        mp = target_action
        ctrl = np.where(mp >= 0,
                        rest_pose + mp * (jnt_hi - rest_pose),
                        rest_pose + mp * (rest_pose - jnt_lo))
        data.ctrl[:] = ctrl
        for _ in range(200):
            mujoco.mj_step(model, data)

    print(f"{'joint':<25} {'tip':<5} {'+0.5 → Δxyz':<25} {'-0.5 → Δxyz':<25}")
    print("-" * 80)
    for limb_idx, limb in enumerate(LIMBS):
        tip_id = model.body(TIPS[limb]).id
        for j, jname in enumerate(JOINTS):
            # Baseline: all zero action.
            settle_at(np.zeros(ACT_DIM, dtype=np.float32))
            base = data.xpos[tip_id].copy()
            # +0.5
            a = np.zeros(ACT_DIM, dtype=np.float32)
            a[limb_idx * 6 + j] = 0.5
            settle_at(a)
            d_pos = data.xpos[tip_id] - base
            # -0.5
            a[limb_idx * 6 + j] = -0.5
            settle_at(a)
            d_neg = data.xpos[tip_id] - base
            print(f"{limb}_{jname:<22} {limb} "
                  f"({d_pos[0]:+.2f},{d_pos[1]:+.2f},{d_pos[2]:+.2f})  "
                  f"({d_neg[0]:+.2f},{d_neg[1]:+.2f},{d_neg[2]:+.2f})")

if __name__ == "__main__":
    main()
