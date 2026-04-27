#!/usr/bin/env python3
"""
Train the ape robot to stand. CPU, SB3 PPO, live viewer.

Usage:
    python train.py               # train + live viewer
    python train.py --no-watch    # train only
    python train.py --inherit     # warm-start from best checkpoint
    python train.py --resume      # continue exact run from last checkpoint
    python train.py --envs 4      # more parallel envs
"""

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from gibbonbot.envs.stand_env import StandEnv
from gibbonbot.utils import lineage as lineage_mod

RUNS = pathlib.Path("runs/ape_stand")
TOTAL_STEPS = 10_000_000
N_STEPS = 2048


def best_checkpoint():
    best = RUNS / "best" / "best_model.zip"
    if best.exists():
        return best
    ckpts = sorted(RUNS.glob("ape_stand_*.zip"))
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--inherit", action="store_true", help="warm-start weights from best checkpoint")
    parser.add_argument("--resume", action="store_true", help="continue exact run from last checkpoint")
    parser.add_argument("--no-watch", dest="watch", action="store_false")
    parser.set_defaults(watch=True)
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(StandEnv, n_envs=args.envs)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.envs, 1),
        save_path=str(RUNS),
        name_prefix="ape_stand",
    )
    eval_env = make_vec_env(StandEnv, n_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RUNS / "best"),
        log_path=str(RUNS / "eval"),
        eval_freq=max(100_000 // args.envs, 1),
        n_eval_episodes=5,
        verbose=1,
    )

    if args.resume:
        ckpt = best_checkpoint()
        if ckpt is None:
            print("No checkpoint found, starting fresh.")
        else:
            print(f"Resuming from {ckpt.name}")
            model = PPO.load(str(ckpt), env=env)
            lineage_mod.record(RUNS, "stand", args.envs, str(ckpt), pathlib.Path("data/scene_fixed.xml"), pathlib.Path("src/gibbonbot/envs/stand_env.py"))
            _run(model, checkpoint_cb, eval_cb, resume=True, watch=args.watch)
            return

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        n_steps=N_STEPS,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.005,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(RUNS),
    )

    if args.inherit:
        ckpt = best_checkpoint()
        if ckpt is None:
            print("No checkpoint found, starting fresh.")
        else:
            print(f"Inheriting weights from {ckpt.name}")
            old = PPO.load(str(ckpt), env=env)
            model = PPO(**ppo_kwargs)
            model.policy.load_state_dict(old.policy.state_dict())
            lineage_mod.record(RUNS, "stand", args.envs, str(ckpt), pathlib.Path("data/scene_fixed.xml"), pathlib.Path("src/gibbonbot/envs/stand_env.py"))
            _run(model, checkpoint_cb, eval_cb, resume=False, watch=args.watch)
            return

    model = PPO(**ppo_kwargs)
    lineage_mod.record(RUNS, "stand", args.envs, None, pathlib.Path("data/scene_fixed.xml"), pathlib.Path("src/gibbonbot/envs/stand_env.py"))
    _run(model, checkpoint_cb, eval_cb, resume=False, watch=args.watch)


def _run(model, checkpoint_cb, eval_cb, *, resume, watch):
    watcher = None
    if watch:
        watcher = subprocess.Popen([sys.executable, "src/gibbonbot/utils/watch.py", "--task", "stand"])

    print(f"Training for {TOTAL_STEPS:,} steps...")
    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=not resume,
        )
    finally:
        if watcher is not None:
            watcher.terminate()

    final = RUNS / "final.zip"
    model.save(str(final))
    print(f"Saved: {final}")


if __name__ == "__main__":
    main()
