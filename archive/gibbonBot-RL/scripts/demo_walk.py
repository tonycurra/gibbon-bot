#!/usr/bin/env python3
"""Hand-coded sinusoidal walking gait — NO policy, NO RL, just sine waves.

Used as a sanity check / reference: 'this is what walking SHOULD look
like in this sim'. Trained policies are then judged against it.

Writes data.qpos to runs/.trial_qpos.npy at action rate so the
dashboard viewer picks it up — same pipe as scripts/play_trial.py.

Usage:
    python scripts/demo_walk.py [--max-steps N] [--freq HZ] [--amp RAD]
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "scene.xml"
QPOS_PATH = ROOT / "runs" / ".trial_qpos.npy"
INFO_PATH = ROOT / "runs" / ".trial_info.json"

# 24 actuators in order: FL, FR, BL, BR × (yaw, pitch, elbow/knee, roll, wrist/ankle, gripper/foot).
# We drive only the "pitch" and "elbow/knee" of each limb — the rest stay at 0.
# Cross-trot gait: FL+BR in phase, FR+BL in opposite phase.
LIMB_PHASE = np.array([0.0, np.pi, np.pi, 0.0])  # FL, FR, BL, BR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--freq", type=float, default=1.2, help="gait cycles per second")
    ap.add_argument("--amp", type=float, default=0.40, help="leg-swing amplitude (rad)")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    physics_steps_per_action = 16
    action_dt = physics_steps_per_action * float(model.opt.timestep)  # ~32ms

    INFO_PATH.parent.mkdir(parents=True, exist_ok=True)

    def write_info(payload):
        try: INFO_PATH.write_text(json.dumps(payload))
        except Exception: pass

    episode = 0
    while True:
        episode += 1

        # Reset to a standing-ish pose.
        data.qpos[:] = model.qpos0
        data.qpos[2] = 0.30  # torso z
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        write_info({
            "episode": episode,
            "checkpoint": f"scripted_walk(freq={args.freq}Hz, amp={args.amp})",
            "policy_kind": "scripted",
            "started_at": time.time(),
        })

        t_start = time.time()
        for step_i in range(args.max_steps):
            t = step_i * action_dt
            ctrl = np.zeros(model.nu, dtype=np.float32)
            # For each limb (FL, FR, BL, BR), drive pitch (idx 1) and elbow/knee (idx 2).
            for limb in range(4):
                base = limb * 6
                phase = 2 * np.pi * args.freq * t + LIMB_PHASE[limb]
                # Pitch swings forward/back; elbow/knee flexes anti-phase for lift.
                ctrl[base + 1] = args.amp * np.sin(phase)
                ctrl[base + 2] = 0.25 + 0.35 * np.maximum(0.0, np.sin(phase + np.pi / 2))
            # Clip to ctrlrange.
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            data.ctrl[:] = np.clip(ctrl, lo, hi)

            for _ in range(physics_steps_per_action):
                mujoco.mj_step(model, data)

            np.save(str(QPOS_PATH), np.asarray(data.qpos))

            # Pace to real time.
            elapsed = time.time() - t_start
            target = (step_i + 1) * action_dt
            if target > elapsed:
                time.sleep(target - elapsed)

            if data.qpos[3] < 0.2:
                print(f"[demo] flip at step {step_i}", flush=True)
                break

        print(f"[demo] ep{episode} done in {time.time() - t_start:.1f}s", flush=True)
        if not args.loop:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()
