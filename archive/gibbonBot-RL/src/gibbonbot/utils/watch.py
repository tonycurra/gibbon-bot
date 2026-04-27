"""
Persistent live viewer — run in a second terminal while train.py runs.

One MuJoCo window stays open. Episodes run back-to-back.
When a newer checkpoint is saved, the policy is hot-swapped automatically.

Usage:
    python watch.py                  # watch reach task
    python watch.py --task stand     # watch stand task
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent / "src"))

import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from gibbonbot.envs.stand_env import StandEnv as _StandEnv

# Live controls — adjust with keyboard in terminal while viewer runs:
#   +/-  speed up / slow down playback
#   p    pause / unpause
_speed  = 0.5    # seconds per action step (matches PHYSICS_STEPS_PER_ACTION × 0.002)
_paused = False

def _print_controls():
    print("\n[watch] keyboard controls (type in this terminal):")
    print("  +  speed up    -  slow down    p  pause/unpause    q  quit\n")

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
        from gibbonbot.envs.ape_env import ApeEnv as EnvClass
    else:
        from gibbonbot.envs.stand_env import StandEnv as EnvClass

    env = EnvClass(render_mode="human")

    # Start immediately with random actions; swap in policy once first checkpoint appears
    ckpt = latest_checkpoint(runs_dir, prefix)
    if ckpt is None:
        print(f"[watch] no checkpoint yet — showing random policy, waiting for {runs_dir} ...")
        policy = None
        last_mtime = 0.0
    else:
        print(f"[watch] loading {ckpt.name} ...")
        policy = PPO.load(str(ckpt))
        last_mtime = ckpt.stat().st_mtime
    ep  = 0
    _print_controls()

    import threading, select, tty, termios, os

    def _key_listener():
        global _speed, _paused
        if not os.isatty(0):
            return
        old = termios.tcgetattr(0)
        try:
            tty.setcbreak(0)
            while viewer_alive():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '+':
                        _speed = max(0.05, _speed * 0.7)
                        print(f"\r[speed] {_speed:.2f}s/step", flush=True)
                    elif ch == '-':
                        _speed = min(5.0, _speed * 1.4)
                        print(f"\r[speed] {_speed:.2f}s/step", flush=True)
                    elif ch == 'p':
                        _paused = not _paused
                        print(f"\r[{'PAUSED' if _paused else 'RUNNING'}]", flush=True)
                    elif ch == 'q':
                        break
        except Exception:
            pass
        finally:
            try: termios.tcsetattr(0, termios.TCSADRAIN, old)
            except Exception: pass

    threading.Thread(target=_key_listener, daemon=True).start()

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
                    print(f"\n[watch] {'first' if policy is None else 'new'} checkpoint: {new_ckpt.name} — loading policy")
                    policy     = PPO.load(str(new_ckpt))
                    last_mtime = mtime

            obs, _ = env.reset(seed=ep)
            done   = False
            total_reward = 0.0
            step   = 0

            while not done and viewer_alive():
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                done = terminated or truncated
                while _paused and viewer_alive():
                    time.sleep(0.1)
                time.sleep(_speed)

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
                if info.get("upside_down"):
                    status = "FLIPPED"
                elif info["fallen"]:
                    status = "FALLEN"
                else:
                    status = "ALIVE"
                above  = info.get("above_ground", info["torso_z"])
                print(f"Ep {ep:3d}: {status:<7}  "
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
