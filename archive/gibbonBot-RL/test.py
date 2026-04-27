#!/usr/bin/env python3
"""
Quick test run — CPU, single env, viewer embedded (you see the actual training robot).

Usage:
    python test.py               # fresh run with viewer
    python test.py --inherit     # warm-start from best checkpoint in runs/ape_stand/
    python test.py --resume      # continue from last test checkpoint
    python test.py --no-watch    # headless (no viewer)
"""

import argparse
import pathlib

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from envs.stand_env import StandEnv

RUNS = pathlib.Path("runs/ape_stand_test")
TRAIN_RUNS = pathlib.Path("runs/ape_stand")
TOTAL_STEPS = 1_000_000
N_STEPS = 2048


class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep = 0
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._max_height = 0.0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self._ep_reward += reward
        self._ep_steps += 1
        self._max_height = max(self._max_height, info.get("torso_z", 0.0))

        if done:
            self._ep += 1
            if info.get("upside_down"):
                status = "FLIPPED   "
            elif info.get("arms_colliding"):
                status = "COLLISION "
            else:
                status = "TIMEOUT   "

            h_rew = info.get("height_reward", 0.0)
            prox = info.get("proximity_penalty", 0.0)

            print(
                f"Ep {self._ep:4d} | {status} | "
                f"steps={self._ep_steps:4d} | "
                f"height={self._max_height:.3f}m | "
                f"h_rew={h_rew:.2f} | "
                f"prox={prox:.2f} | "
                f"reward={self._ep_reward:7.1f}"
            )

            self._ep_reward = 0.0
            self._ep_steps = 0
            self._max_height = 0.0

        return True


def best_checkpoint(runs_dir: pathlib.Path):
    best = runs_dir / "best" / "best_model.zip"
    if best.exists():
        return best
    ckpts = sorted(runs_dir.glob("ape_stand_*.zip"))
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inherit", action="store_true", help="warm-start weights from runs/ape_stand/"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue from last test checkpoint"
    )
    parser.add_argument("--no-watch", dest="watch", action="store_false")
    parser.set_defaults(watch=True)
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)

    render = "human" if args.watch else None
    env = make_vec_env(StandEnv, n_envs=1, env_kwargs={"render_mode": render})

    callbacks = [
        CheckpointCallback(
            save_freq=10_000, save_path=str(RUNS), name_prefix="ape_stand"
        ),
        EpisodeLogger(),
    ]

    ppo_kwargs = dict(
        device="cpu",
        policy="MlpPolicy",
        env=env,
        n_steps=N_STEPS,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.005,
        clip_range=0.2,
        verbose=0,  # suppress SB3 noise — EpisodeLogger handles output
    )

    if args.resume:
        ckpt = best_checkpoint(RUNS)
        if ckpt is None:
            print("No test checkpoint found, starting fresh.")
            model = PPO(**ppo_kwargs)
        else:
            print(f"Resuming from {ckpt.name}")
            model = PPO.load(str(ckpt), env=env, device="cpu")
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=callbacks,
                reset_num_timesteps=False,
            )
            return

    elif args.inherit:
        ckpt = best_checkpoint(TRAIN_RUNS)
        if ckpt is None:
            print("No checkpoint found in runs/ape_stand/, starting fresh.")
            model = PPO(**ppo_kwargs)
        else:
            print(f"Inheriting weights from {ckpt.name}")
            old = PPO.load(str(ckpt), env=env, device="cpu")
            model = PPO(**ppo_kwargs)
            model.policy.load_state_dict(old.policy.state_dict())

    else:
        model = PPO(**ppo_kwargs)

    print(
        f"{'Ep':>4} | {'Status':<10} | {'steps':>5} | {'height':>8} | {'h_rew':>6} | {'prox':>6} | {'reward':>8}"
    )
    print("-" * 70)
    print(f"Training for {TOTAL_STEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks)
    model.save(str(RUNS / "final.zip"))
    print(f"Saved: {RUNS / 'final.zip'}")


if __name__ == "__main__":
    main()
