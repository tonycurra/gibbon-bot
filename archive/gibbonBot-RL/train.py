"""
Train the ape robot using PPO.

Usage:
    python train.py --task stand           # train, no viewer
    python train.py --task stand --watch   # train + open live viewer
    python train.py --task stand --resume  # continue from last checkpoint

Checkpoints saved to: runs/<task>/
TensorBoard logs:      runs/<task>/

Monitor training:
    tensorboard --logdir runs/
"""

import argparse
import pathlib
import subprocess
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from ape_env import ApeEnv
from stand_env import StandEnv

TASKS = {
    "reach": (ApeEnv,   "ape_reach"),
    "stand": (StandEnv, "ape_stand"),
}

STEPS   = 2_048    # rollout steps per env before update
TOTAL   = 10_000_000  # total environment steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=list(TASKS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--watch", action="store_true", help="open live viewer alongside training")
    parser.add_argument("--envs", type=int, default=1, help="number of parallel training envs")
    args = parser.parse_args()

    n_envs = args.envs
    EnvClass, run_name = TASKS[args.task]
    RUNS = pathlib.Path(f"runs/{run_name}")
    RUNS.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(EnvClass, n_envs=n_envs)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(RUNS),
        name_prefix=run_name,
    )
    eval_env = make_vec_env(EnvClass, n_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RUNS / "best"),
        log_path=str(RUNS / "eval"),
        eval_freq=max(100_000 // n_envs, 1),
        n_eval_episodes=5,
        verbose=1,
    )

    if args.resume:
        checkpoints = sorted(RUNS.glob(f"{run_name}_*.zip"))
        if not checkpoints:
            print("No checkpoint found — starting from scratch.")
            args.resume = False
        else:
            ckpt = checkpoints[-1]
            print(f"Resuming from {ckpt}")
            model = PPO.load(ckpt, env=env)

    if not args.resume:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=STEPS,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.005,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=str(RUNS),
        )

    if args.watch:
        watcher = subprocess.Popen(
            [sys.executable, "watch.py", "--task", args.task],
            cwd=pathlib.Path(__file__).parent,
        )
    else:
        watcher = None

    print(f"Training for {TOTAL:,} steps across {n_envs} env(s)...")
    try:
        model.learn(
            total_timesteps=TOTAL,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=not args.resume,
        )
    finally:
        if watcher is not None:
            watcher.wait()

    final = RUNS / f"{run_name}_final.zip"
    model.save(str(final))
    print(f"Saved: {final}")

if __name__ == "__main__":
    main()
