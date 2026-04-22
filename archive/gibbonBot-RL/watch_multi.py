"""
Split-screen live viewer: N parallel envs rendered in a tiled grid.

Run in a second terminal while train.py is running.
Policy hot-swaps automatically when a newer checkpoint is saved.

Usage:
    python watch_multi.py          # 4 envs in a 2x2 grid
    python watch_multi.py --n 8    # 8 envs in a 2x4 grid
    python watch_multi.py --n 9 --width 320 --height 240
"""

import argparse
import math
import pathlib
import time
import numpy as np
import cv2
import mujoco
import terrain as terrain_mod
from stable_baselines3 import PPO

RUNS          = pathlib.Path("runs/ape_reach")
TREE_POS      = np.array([5.0, 0.0, 0.0])
SUCCESS_DIST  = 1.5
MIN_HEIGHT    = 0.15
MAX_STEPS     = 2000
PHYSICS_STEPS = 5
POLL_INTERVAL = 5   # seconds between checkpoint checks


def latest_checkpoint():
    best = RUNS / "best" / "best_model.zip"
    if best.exists():
        return best
    ckpts = sorted(RUNS.glob("ape_reach_*.zip"))
    return ckpts[-1] if ckpts else None


def tile(frames, rows, cols):
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = f
    return grid


def label(frame, text, success, fallen):
    color = (0, 220, 0) if success else (0, 0, 220) if fallen else (200, 200, 200)
    cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,   1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=4,   help="number of envs")
    parser.add_argument("--width",  type=int, default=480, help="render width per env")
    parser.add_argument("--height", type=int, default=360, help="render height per env")
    args = parser.parse_args()

    N    = args.n
    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    scene = pathlib.Path(__file__).parent / "scene.xml"

    print(f"Waiting for first checkpoint ({N} envs, {cols}×{rows} grid)...")
    while True:
        ckpt = latest_checkpoint()
        if ckpt:
            break
        time.sleep(POLL_INTERVAL)

    print(f"Loading {ckpt.name} ...")
    policy     = PPO.load(str(ckpt))
    last_mtime = ckpt.stat().st_mtime

    # shared model (read-only), separate data per env
    mjmodel = mujoco.MjModel.from_xml_path(str(scene))
    terrain_mod.apply(mjmodel)

    datas     = [mujoco.MjData(mjmodel) for _ in range(N)]
    renderers = [mujoco.Renderer(mjmodel, height=args.height, width=args.width) for _ in range(N)]

    jnt_lo = np.array([mjmodel.jnt_range[i][0] for i in range(1, mjmodel.njnt)])
    jnt_hi = np.array([mjmodel.jnt_range[i][1] for i in range(1, mjmodel.njnt)])

    # per-env state
    steps    = [0]  * N
    dists    = [0.0] * N
    min_dist = [float("inf")] * N
    success  = [False] * N
    fallen   = [False] * N
    done     = [True]  * N   # start all as done so they reset immediately

    ep_count = [0] * N
    last_poll = time.time()

    cv2.namedWindow("Training — live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Training — live", cols * args.width, rows * args.height)

    while True:
        # ── poll for new checkpoint ──────────────────────────────────────────
        if time.time() - last_poll > POLL_INTERVAL:
            last_poll = time.time()
            new_ckpt = latest_checkpoint()
            if new_ckpt and new_ckpt.stat().st_mtime != last_mtime:
                print(f"\nNew checkpoint: {new_ckpt.name}")
                policy     = PPO.load(str(new_ckpt))
                last_mtime = new_ckpt.stat().st_mtime

        # ── reset finished envs ──────────────────────────────────────────────
        for i in range(N):
            if done[i]:
                if ep_count[i] > 0:
                    st = "OK " if success[i] else ("X  " if fallen[i] else "T  ")
                    print(f"  env{i} ep{ep_count[i]:3d}: {st}  closest={min_dist[i]:.2f}m")
                mujoco.mj_resetDataKeyframe(mjmodel, datas[i], 0)
                rng = np.random.default_rng(ep_count[i] * N + i)
                datas[i].qpos[7:] += rng.uniform(-0.05, 0.05, size=24)
                mujoco.mj_forward(mjmodel, datas[i])
                steps[i]    = 0
                min_dist[i] = float(np.linalg.norm(TREE_POS - datas[i].qpos[0:3]))
                success[i]  = False
                fallen[i]   = False
                done[i]     = False
                ep_count[i] += 1

        # ── build observations ───────────────────────────────────────────────
        obs_batch = []
        for i in range(N):
            d = datas[i]
            to_tree   = TREE_POS - d.qpos[0:3]
            dist      = float(np.linalg.norm(to_tree))
            direction = to_tree / (dist + 1e-8)
            obs = np.concatenate([
                d.qpos[2:3], d.qpos[3:7], d.qvel[0:3], d.qvel[3:6],
                d.qpos[7:],  d.qvel[6:],
                direction, [dist],
            ]).astype(np.float32)
            obs_batch.append(obs)

        actions, _ = policy.predict(np.array(obs_batch), deterministic=False)

        # ── step all envs ────────────────────────────────────────────────────
        frames = []
        for i in range(N):
            if done[i]:
                frames.append(np.zeros((args.height, args.width, 3), dtype=np.uint8))
                continue

            ctrl = 0.5 * (actions[i] + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
            datas[i].ctrl[:] = ctrl
            for _ in range(PHYSICS_STEPS):
                mujoco.mj_step(mjmodel, datas[i])

            steps[i]  += 1
            dist       = float(np.linalg.norm(TREE_POS - datas[i].qpos[0:3]))
            dists[i]   = dist
            min_dist[i] = min(min_dist[i], dist)
            torso_z    = float(datas[i].qpos[2])
            success[i] = dist  < SUCCESS_DIST
            fallen[i]  = torso_z < MIN_HEIGHT
            done[i]    = success[i] or fallen[i] or steps[i] >= MAX_STEPS

            # render
            renderers[i].update_scene(datas[i])
            px = renderers[i].render()                   # RGB uint8
            px = cv2.cvtColor(px, cv2.COLOR_RGB2BGR)
            txt = f"env{i}  {dist:.1f}m  step{steps[i]}"
            label(px, txt, success[i], fallen[i])
            frames.append(px)

        # ── display ──────────────────────────────────────────────────────────
        grid = tile(frames, rows, cols)
        cv2.imshow("Training — live", grid)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
