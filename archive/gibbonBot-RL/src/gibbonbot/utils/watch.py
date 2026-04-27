"""
Persistent live viewer — run in a second terminal while train.py runs.

One MuJoCo window stays open. Episodes run back-to-back.
When a newer checkpoint is saved, the policy is hot-swapped automatically.

Usage:
    python watch.py                  # watch reach task
    python watch.py --task stand     # watch stand task
    python watch.py --no-ui          # skip params panel
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent / "src"))

import argparse
import subprocess
import threading
import time
import numpy as np
from stable_baselines3 import PPO

_paused = False

PARAMS_FILE = pathlib.Path(__file__).parent.parent / "envs" / "params.py"

def _set_realtime(factor):
    """Write REALTIME_FACTOR into params.py so the env picks it up."""
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
    "reach": ("runs/ape_reach", "ape_reach"),
    "stand": ("runs/ape_stand", "ape_stand"),
}


def latest_checkpoint(runs_dir: pathlib.Path, prefix: str):
    best = runs_dir / "best" / "best_model.zip"
    if best.exists():
        return best
    checkpoints = sorted(runs_dir.glob(f"{prefix}_*.zip"))
    return checkpoints[-1] if checkpoints else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=list(TASKS))
    parser.add_argument("--no-ui", action="store_true", help="disable params slider UI")
    args = parser.parse_args()

    if not args.no_ui:
        ui_script = pathlib.Path(__file__).parent / "params_ui.py"
        subprocess.Popen([sys.executable, str(ui_script)])

    runs_path, prefix = TASKS[args.task]
    runs_dir = pathlib.Path(runs_path)

    if args.task == "reach":
        from gibbonbot.envs.ape_env import ApeEnv as EnvClass
    else:
        from gibbonbot.envs.stand_env import StandEnv as EnvClass

    env = EnvClass(render_mode="human")

    ckpt = latest_checkpoint(runs_dir, prefix)
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
                        _set_realtime(max(0.1, _get_realtime() * 0.7))
                    elif ch == '-':
                        _set_realtime(min(5.0, _get_realtime() * 1.4))
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
                sh = info.get("shoulder_z", info.get("above_ground", info["torso_z"]))
                stable_mark = " STABLE" if info.get("stable") else ""
                print(f"Ep {ep:3d}: {status:<7}{stable_mark:<8}  "
                      f"shoulder={sh:.3f}m  torso={info['torso_z']:.3f}m  "
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
