"""
Persistent live viewer — run in a second terminal while train.py runs.

One MuJoCo window stays open. Episodes run back-to-back.
When a newer checkpoint is saved, the policy is hot-swapped automatically.

Usage:
    python watch.py                  # watch reach task
    python watch.py --task stand     # watch stand task
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import time
import numpy as np
from stable_baselines3 import PPO

TASKS = {
    "reach": ("runs/ape_reach", "ape_reach"),
    "stand": ("runs/ape_stand", "ape_stand"),
}
POLL_INTERVAL = 5   # seconds between checkpoint checks


def latest_checkpoint(runs_dir: pathlib.Path, prefix: str):
    best = runs_dir / "best" / "best_model.zip"
    if best.exists():
        return best
    checkpoints = sorted(runs_dir.glob(f"{prefix}_*.zip"))
    return checkpoints[-1] if checkpoints else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=list(TASKS))
    args = parser.parse_args()

    runs_path, prefix = TASKS[args.task]
    runs_dir = pathlib.Path(runs_path)

    if args.task == "reach":
        from envs import ApeEnv as EnvClass
    else:
        from envs import StandEnv as EnvClass

    print(f"[watch] task={args.task}  waiting for first checkpoint in {runs_dir} ...")
    while True:
        ckpt = latest_checkpoint(runs_dir, prefix)
        if ckpt is not None:
            break
        time.sleep(POLL_INTERVAL)

    print(f"[watch] loading {ckpt.name} ...")
    policy     = PPO.load(str(ckpt))
    last_mtime = ckpt.stat().st_mtime

    env = EnvClass(render_mode="human")
    ep  = 0

    def viewer_alive():
        v = getattr(env, "_viewer", None)
        return v is None or v.is_running()

    try:
        while viewer_alive():
            # hot-swap policy if a newer checkpoint exists
            new_ckpt = latest_checkpoint(runs_dir, prefix)
            if new_ckpt is not None:
                mtime = new_ckpt.stat().st_mtime
                if mtime != last_mtime:
                    print(f"\n[watch] new checkpoint: {new_ckpt.name} — swapping policy")
                    policy     = PPO.load(str(new_ckpt))
                    last_mtime = mtime

            obs, _ = env.reset(seed=ep)
            done   = False
            total_reward = 0.0
            step   = 0

            while not done and viewer_alive():
                action, _ = policy.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                done = terminated or truncated

            if not viewer_alive():
                break

            ep += 1
            ckpt_name = new_ckpt.name if new_ckpt else "?"
            if args.task == "reach":
                status = "SUCCESS" if info["success"] else ("FALLEN" if info["fallen"] else "TIMEOUT")
                print(f"Ep {ep:3d}: {status:<8}  "
                      f"closest={info.get('dist_to_tree', 0):.2f}m  "
                      f"reward={total_reward:.1f}  steps={step}  [{ckpt_name}]")
            else:
                status = "FALLEN" if info["fallen"] else "ALIVE"
                above  = info.get("above_ground", info["torso_z"])
                print(f"Ep {ep:3d}: {status:<6}  "
                      f"height={above:.3f}m  "
                      f"h_rew={info['height_reward']:.3f}  "
                      f"reward={total_reward:.1f}  steps={step}  [{ckpt_name}]")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[watch] error: {e}")
    finally:
        env.close()
        print("\n[watch] closed.")


if __name__ == "__main__":
    main()
