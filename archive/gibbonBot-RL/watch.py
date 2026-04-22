"""
Persistent live viewer — run in a second terminal while train.py runs.

One MuJoCo window stays open the whole time. Episodes run back-to-back.
When a newer checkpoint is saved, the policy is hot-swapped automatically.

Usage:
    python watch.py
"""

import time
import pathlib
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from ape_env import ApeEnv
import terrain as terrain_mod

RUNS          = pathlib.Path("runs/ape_reach")
POLL_INTERVAL = 5    # seconds between checkpoint checks


def latest_checkpoint():
    best = RUNS / "best" / "best_model.zip"
    if best.exists():
        return best
    checkpoints = sorted(RUNS.glob("ape_reach_*.zip"))
    return checkpoints[-1] if checkpoints else None


def main():
    # ── wait for first checkpoint ────────────────────────────────────────────
    print("Waiting for first checkpoint in runs/ape_reach/ ...")
    while True:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            break
        time.sleep(POLL_INTERVAL)

    # ── build env and persistent viewer once ────────────────────────────────
    # We manage MuJoCo directly so the viewer never closes.
    import pathlib as _pl
    scene = _pl.Path(__file__).parent / "scene.xml"
    mjmodel = mujoco.MjModel.from_xml_path(str(scene))
    terrain_mod.apply(mjmodel)
    mjdata  = mujoco.MjData(mjmodel)

    # joint ranges for action rescaling (same logic as ApeEnv)
    jnt_lo = np.array([mjmodel.jnt_range[i][0] for i in range(1, mjmodel.njnt)])
    jnt_hi = np.array([mjmodel.jnt_range[i][1] for i in range(1, mjmodel.njnt)])
    TREE_POS    = np.array([5.0, 0.0, 0.0])
    SUCCESS_DIST = 1.5
    MIN_HEIGHT   = 0.15
    MAX_STEPS    = 2000
    PHYSICS_STEPS = 5

    viewer = mujoco.viewer.launch_passive(mjmodel, mjdata)
    print(f"Viewer open. Loading {ckpt.name} ...")

    policy      = PPO.load(str(ckpt))
    last_mtime  = ckpt.stat().st_mtime
    ep = 0

    while viewer.is_running():
        # ── hot-swap policy if newer checkpoint exists ───────────────────────
        new_ckpt = latest_checkpoint()
        if new_ckpt is not None:
            mtime = new_ckpt.stat().st_mtime
            if mtime != last_mtime:
                print(f"\nNew checkpoint: {new_ckpt.name} — swapping policy")
                policy     = PPO.load(str(new_ckpt))
                last_mtime = mtime

        # ── reset episode ────────────────────────────────────────────────────
        mujoco.mj_resetDataKeyframe(mjmodel, mjdata, 0)
        rng = np.random.default_rng(ep)
        mjdata.qpos[7:] += rng.uniform(-0.05, 0.05, size=24)
        mujoco.mj_forward(mjmodel, mjdata)

        prev_dist   = float(np.linalg.norm(TREE_POS - mjdata.qpos[0:3]))
        total_reward = 0.0
        min_dist     = prev_dist
        step = 0
        done = False

        while not done and viewer.is_running():
            # build obs (same as ApeEnv._get_obs)
            qpos = mjdata.qpos
            qvel = mjdata.qvel
            torso_xyz = qpos[0:3]
            to_tree   = TREE_POS - torso_xyz
            dist      = float(np.linalg.norm(to_tree))
            direction = to_tree / (dist + 1e-8)
            obs = np.concatenate([
                qpos[2:3], qpos[3:7], qvel[0:3], qvel[3:6],
                qpos[7:], qvel[6:],
                direction, [dist],
            ]).astype(np.float32)

            action, _ = policy.predict(obs, deterministic=False)
            ctrl = 0.5 * (action + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
            mjdata.ctrl[:] = ctrl

            for _ in range(PHYSICS_STEPS):
                mujoco.mj_step(mjmodel, mjdata)
            viewer.sync()

            step += 1
            dist      = float(np.linalg.norm(TREE_POS - mjdata.qpos[0:3]))
            min_dist  = min(min_dist, dist)
            torso_z   = float(mjdata.qpos[2])
            fallen    = torso_z < MIN_HEIGHT
            success   = dist < SUCCESS_DIST
            done      = fallen or success or step >= MAX_STEPS
            total_reward += 0.1  # rough alive proxy for display

        status = "SUCCESS" if success else ("FALLEN" if fallen else "TIMEOUT")
        ep += 1
        print(f"Ep {ep:3d}: {status:<8}  closest={min_dist:.2f}m  "
              f"final={dist:.2f}m  steps={step}  [{new_ckpt.name if new_ckpt else '?'}]")

    print("Viewer closed.")


if __name__ == "__main__":
    main()
