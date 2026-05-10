#!/usr/bin/env python3
"""Render a few frames from the latest checkpoint, save as PNGs.

Lets Claude (or any tool without screen access) inspect the robot's pose
and motion qualitatively. Uses the same env dynamics as view.py.

Usage:
    python scripts/snapshot.py                          # default 6 frames @ 1s spacing
    python scripts/snapshot.py --frames 12 --gap 0.5   # 12 frames every 0.5s
    python scripts/snapshot.py --policy zero           # see neutral pose only
"""
import argparse
import pathlib
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

# Reuse view.py's net loader. view.py has a module-level argparse that grabs
# sys.argv at import — pass --env walk explicitly so OBS_DIM=88 (not 83 stand).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
_saved_argv = sys.argv
sys.argv = [sys.argv[0], "--env", "walk"]
from view import load_predict, _build_net  # noqa: E402
sys.argv = _saved_argv

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "scene.xml"
RUNS = ROOT / "runs" / "ape_walk_mjx"
OUT = ROOT / "_snapshots"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--gap", type=float, default=1.0, help="seconds of sim between frames")
    p.add_argument("--settle", type=float, default=2.0, help="settle seconds before policy")
    p.add_argument("--policy", choices=["latest", "zero"], default="latest")
    p.add_argument("--width",  type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    # Clear old frames so I never read stale ones.
    for f in OUT.glob("snap_*.png"):
        f.unlink()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # Same mesh-collision trick view.py uses to keep cost down.
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_contype[i]     = 2
            model.geom_conaffinity[i] = 1
    data = mujoco.MjData(model)
    ACT_DIM = model.nu
    DT = float(model.opt.timestep)
    PHYSICS_STEPS = 16
    jnt_lo = np.array([model.jnt_range[i][0] for i in range(1, 1 + ACT_DIM)])
    jnt_hi = np.array([model.jnt_range[i][1] for i in range(1, 1 + ACT_DIM)])

    # Motor limits — same per-actuator computation as view.py.
    PER_LIMB_RAD_PER_SEC = [4.71, 4.71, 4.71, 4.71, 4.71, 11.10]
    motor_max_radps = np.array(PER_LIMB_RAD_PER_SEC * 4, dtype=np.float32)
    action_dt = DT * PHYSICS_STEPS
    DEG = np.pi / 180.0
    rest_pose = np.tile(np.array([0.0, 0.0, 30.0*DEG, 0.0, 0.0, 0.0]), 4).astype(np.float32)
    half_range = np.maximum(np.abs(jnt_hi - rest_pose), np.abs(rest_pose - jnt_lo))
    motor_max_speed = motor_max_radps * action_dt / half_range
    motor_max_accel = motor_max_speed * 5.0 * action_dt
    substep_speed = motor_max_speed / PHYSICS_STEPS
    substep_accel = motor_max_accel / PHYSICS_STEPS

    # Find latest checkpoint, build predict.
    predict = None
    if args.policy == "latest":
        ckpts = sorted(RUNS.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
        for ckpt in reversed(ckpts):
            try:
                predict = load_predict(ckpt)
                print(f"[snap] loaded {ckpt.name}")
                break
            except Exception as e:
                print(f"[snap] skip {ckpt.name}: {e}")
        else:
            print("[snap] no compatible checkpoint found, using zero policy")

    # Spawn pose.
    data.qpos[2] = 0.30
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    mujoco.mj_forward(model, data)

    # Settle phase — drive ctrl to rest_pose so robot stands during settle.
    for _ in range(int(args.settle / DT)):
        data.ctrl[:] = rest_pose
        mujoco.mj_step(model, data)

    # Three cameras — side, front, top — for gait analysis.
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    def _make_cam(azimuth, elevation, distance, lookat):
        c = mujoco.MjvCamera()
        c.lookat[:] = lookat
        c.distance = distance
        c.azimuth = azimuth
        c.elevation = elevation
        return c
    cams = {
        "side_L":  _make_cam(azimuth=90.0,  elevation=-15.0, distance=2.5, lookat=[0.5, 0.0, 0.25]),
        "side_R":  _make_cam(azimuth=270.0, elevation=-15.0, distance=2.5, lookat=[0.5, 0.0, 0.25]),
        "front":   _make_cam(azimuth=180.0, elevation=-15.0, distance=2.5, lookat=[0.5, 0.0, 0.25]),
        "back":    _make_cam(azimuth=0.0,   elevation=-15.0, distance=2.5, lookat=[0.5, 0.0, 0.25]),
        "top":     _make_cam(azimuth=90.0,  elevation=-89.0, distance=3.0, lookat=[0.5, 0.0, 0.25]),
    }

    motor_pos = np.zeros(ACT_DIM, dtype=np.float32)
    motor_vel = np.zeros(ACT_DIM, dtype=np.float32)
    filt_action = np.zeros(ACT_DIM, dtype=np.float32)

    # CPG amplitudes — read from params (mirrors env).
    try:
        import envs.params_walk_mjx as _p_snap
        W_CPG_PITCH = float(getattr(_p_snap, "W_CPG_PITCH", 0.0))
        W_CPG_ELBOW = float(getattr(_p_snap, "W_CPG_ELBOW", 0.0))
    except Exception:
        W_CPG_PITCH = W_CPG_ELBOW = 0.0
    LIMB_PHASE = np.array([0.0, np.pi, 0.0, np.pi])  # pace gait

    # Tag is just for context — also save tree pos.
    tree_xy = None
    try:
        gid = model.geom("t0_trunk").id
        tree_xy = model.geom_pos[gid][:2].copy()
    except (KeyError, ValueError):
        pass

    print(f"[snap] start: torso z={data.qpos[2]:.3f} tree={tree_xy}")

    # Identify body indices for each limb's distal tip (hand/foot equivalents).
    # These are the "keypoints" we can compare to ape body landmarks.
    LANDMARK_BODIES = {
        "torso":       "torso",
        "FL_shoulder": "fl_link1", "FL_elbow": "fl_link3", "FL_hand": "fl_gripper_link",
        "FR_shoulder": "fr_link1", "FR_elbow": "fr_link3", "FR_hand": "fr_gripper_link",
        "BL_hip":      "bl_link1", "BL_knee":  "bl_link3", "BL_foot": "bl_gripper_link",
        "BR_hip":      "br_link1", "BR_knee":  "br_link3", "BR_foot": "br_gripper_link",
    }
    landmark_ids = {}
    for label, body_name in LANDMARK_BODIES.items():
        try:
            landmark_ids[label] = model.body(body_name).id
        except (KeyError, ValueError):
            pass
    print(f"[snap] landmarks resolved: {len(landmark_ids)}/{len(LANDMARK_BODIES)}")
    landmarks_log = []  # list of {t, label: [x,y,z]}

    steps_per_frame = int(args.gap / action_dt)
    for f in range(args.frames):
        # Render this frame BEFORE stepping so frame 0 = start.
        for view_name, cam in cams.items():
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            out = OUT / f"snap_{f:02d}_{view_name}.png"
            imageio.imwrite(out, img)
        print(f"[snap] frame {f:02d}  t={f*args.gap:.1f}s  z={data.qpos[2]:.3f}  "
              f"upright={data.qpos[3]:.3f}  xy=({data.qpos[0]:.2f},{data.qpos[1]:.2f})")
        # Snapshot 3D landmark positions (ape-pose-like keypoints).
        frame_landmarks = {"t": f * args.gap}
        for label, bid in landmark_ids.items():
            frame_landmarks[label] = data.xpos[bid].tolist()
        landmarks_log.append(frame_landmarks)
        # Advance steps_per_frame action-steps.
        for _ in range(steps_per_frame):
            if predict is not None:
                obs_parts = [
                    data.qpos[2:3], data.qpos[3:7],
                    data.qvel[0:3], data.qvel[3:6],
                    data.qpos[7:7 + ACT_DIM], data.qvel[6:6 + ACT_DIM],
                    data.qfrc_actuator[6:6 + ACT_DIM],
                ]
                if tree_xy is not None:
                    rel = tree_xy - data.qpos[0:2].astype(np.float32)
                    obs_parts.append(rel)
                    obs_parts.append(np.array([np.linalg.norm(rel)], dtype=np.float32))
                    ph = (2.0 * np.pi / 25) * f
                    obs_parts.append(np.array([np.sin(ph), np.cos(ph)], dtype=np.float32))
                obs = np.concatenate(obs_parts).astype(np.float32)
                action = np.array(predict(obs))
            else:
                action = np.zeros(ACT_DIM, dtype=np.float32)
            filt_action = 0.15 * action + 0.85 * filt_action
            # CPG calibrated trot — mirrors env.
            phase_step = f * steps_per_frame
            phase = 2.0 * np.pi * phase_step / 40.0 + LIMB_PHASE
            sin_p = np.sin(phase)
            cpg = np.zeros(ACT_DIM, dtype=np.float32)
            cpg[0]  = -W_CPG_PITCH * sin_p[0]
            cpg[1]  = -W_CPG_ELBOW * max(sin_p[0], 0.0)
            cpg[6]  = +W_CPG_PITCH * sin_p[1]
            cpg[7]  = -W_CPG_ELBOW * max(sin_p[1], 0.0)
            cpg[13] = +W_CPG_PITCH * sin_p[2]
            cpg[14] = +W_CPG_ELBOW * max(sin_p[2], 0.0)
            cpg[19] = +W_CPG_PITCH * sin_p[3]
            cpg[20] = +W_CPG_ELBOW * max(sin_p[3], 0.0)
            filt_action_cpg = filt_action + cpg
            for _ in range(PHYSICS_STEPS):
                err = filt_action_cpg - motor_pos
                desired_vel = np.clip(err, -substep_speed, substep_speed)
                motor_vel = np.clip(desired_vel,
                                    motor_vel - substep_accel,
                                    motor_vel + substep_accel)
                motor_vel = np.clip(motor_vel, -substep_speed, substep_speed)
                motor_pos += motor_vel
                data.ctrl[:] = np.where(
                    motor_pos >= 0,
                    rest_pose + motor_pos * (jnt_hi - rest_pose),
                    rest_pose + motor_pos * (rest_pose - jnt_lo),
                )
                mujoco.mj_step(model, data)

    # Dump 3D landmarks JSON for analysis.
    import json as _json
    (OUT / "landmarks.json").write_text(_json.dumps(landmarks_log, indent=2))
    print(f"[snap] {args.frames} frames + landmarks.json written to {OUT}")


if __name__ == "__main__":
    main()
