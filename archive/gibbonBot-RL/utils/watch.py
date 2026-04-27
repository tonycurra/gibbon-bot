"""
Persistent live viewer — run in a second terminal while train.py runs.

One MuJoCo window stays open. Episodes run back-to-back.
When a newer checkpoint is saved, the policy is hot-swapped automatically.

Usage:
    python utils/watch.py --task stand
    python utils/watch.py --task stand --no-ui
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import subprocess
import threading
import time
import numpy as np
from stable_baselines3 import PPO

_paused = False

PARAMS_FILE = pathlib.Path(__file__).parent.parent / "envs" / "params.py"


def _set_realtime(factor):
    try:
        lines = PARAMS_FILE.read_text().splitlines()
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("REALTIME_FACTOR"):
                lines[i] = f"REALTIME_FACTOR = {factor:.3f}"
                found = True
                break
        if not found:
            lines.append(f"REALTIME_FACTOR = {factor:.3f}")
        PARAMS_FILE.write_text("\n".join(lines) + "\n")
        print(f"\r[speed] REALTIME_FACTOR={factor:.2f}x", flush=True)
    except Exception as e:
        print(f"\r[speed] error: {e}", flush=True)


def _get_realtime():
    try:
        for line in PARAMS_FILE.read_text().splitlines():
            if line.strip().startswith("REALTIME_FACTOR"):
                return float(line.split("=")[1].split("#")[0].strip())
    except Exception:
        pass
    return 1.0


def _print_controls():
    print("\n[watch] keyboard controls (type in this terminal):")
    print("  +  speed up    -  slow down    p  pause/unpause    q  quit\n")


TASKS = {
    "stand": ("runs/ape_stand", "ape_stand"),
}


def latest_checkpoint(runs_dir: pathlib.Path, prefix: str, best_only: bool = False):
    best = runs_dir / "best" / "best_model.zip"
    if best.exists():
        return best
    if best_only:
        return None
    checkpoints = sorted(runs_dir.glob(f"{prefix}_*.zip"))
    return checkpoints[-1] if checkpoints else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stand", choices=list(TASKS))
    parser.add_argument("--no-ui", action="store_true", help="disable params slider UI")
    parser.add_argument("--best-only", action="store_true", help="only show best/best_model.zip, skip numbered checkpoints")
    args = parser.parse_args()

    if not args.no_ui:
        params_script = pathlib.Path(__file__).parent.parent / "envs" / "params.py"
        subprocess.Popen([sys.executable, str(params_script)])

    runs_path, prefix = TASKS[args.task]
    runs_dir = pathlib.Path(runs_path)

    from envs.stand_env import StandEnv
    env = StandEnv(render_mode="human")

    ckpt = latest_checkpoint(runs_dir, prefix, args.best_only)
    if ckpt is None:
        print(f"[watch] no checkpoint yet — showing random policy, waiting for {runs_dir} ...")
        policy = None
        last_mtime = 0.0
    else:
        print(f"[watch] loading {ckpt.name} ...")
        policy = PPO.load(str(ckpt))
        last_mtime = ckpt.stat().st_mtime
    ep = 0
    _print_controls()

    import select, tty, termios, os

    def _key_listener():
        global _paused
        if not os.isatty(0):
            return
        old = termios.tcgetattr(0)
        try:
            tty.setcbreak(0)
            while viewer_alive():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '+':
                        _set_realtime(min(100.0, _get_realtime() * 1.5))
                    elif ch == '-':
                        _set_realtime(max(1.0, _get_realtime() / 1.5))
                    elif ch == 'p':
                        _paused = not _paused
                        print(f"\r[{'PAUSED' if _paused else 'RUNNING'}]", flush=True)
                    elif ch == 'q':
                        break
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(0, termios.TCSADRAIN, old)
            except Exception:
                pass

    threading.Thread(target=_key_listener, daemon=True).start()

    def viewer_alive():
        v = getattr(env, "_viewer", None)
        return v is None or v.is_running()

    try:
        while viewer_alive():
            new_ckpt = latest_checkpoint(runs_dir, prefix, args.best_only)
            if new_ckpt is not None:
                mtime = new_ckpt.stat().st_mtime
                if mtime != last_mtime:
                    print(f"\n[watch] {'first' if policy is None else 'new'} checkpoint: {new_ckpt.name} — loading policy")
                    policy = PPO.load(str(new_ckpt))
                    last_mtime = mtime

            obs, _ = env.reset(seed=ep)
            done = False
            total_reward = 0.0
            step = 0

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

            if not viewer_alive():
                break

            ep += 1
            ckpt_name = new_ckpt.name if new_ckpt else "?"
            status = "FLIPPED" if info.get("upside_down") else ("COLLISION" if info.get("arms_colliding") else "ALIVE")
            print(f"Ep {ep:3d}: {status:<10}  "
                  f"torso={info['torso_z']:.3f}m  "
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
