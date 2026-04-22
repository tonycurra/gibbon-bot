"""
Train the ape robot to reach the tree using PPO.

Usage:
    python train.py                  # train from scratch
    python train.py --resume         # continue from last checkpoint

Checkpoints saved to: runs/ape_reach/
TensorBoard logs:      runs/ape_reach/

Monitor training:
    tensorboard --logdir runs/
"""

import argparse
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from ape_env import ApeEnv

RUNS    = pathlib.Path("runs/ape_reach")
N_ENVS  = 8        # parallel envs (CPU); raise to 64+ on GPU with MJX later
STEPS   = 2_048    # rollout steps per env before update
TOTAL   = 10_000_000  # total environment steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(ApeEnv, n_envs=N_ENVS)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=str(RUNS),
        name_prefix="ape_reach",
    )
    eval_env = make_vec_env(ApeEnv, n_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RUNS / "best"),
        log_path=str(RUNS / "eval"),
        eval_freq=max(100_000 // N_ENVS, 1),
        n_eval_episodes=5,
        verbose=1,
    )

    if args.resume:
        checkpoints = sorted(RUNS.glob("ape_reach_*.zip"))
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
            ent_coef=0.005,        # small entropy bonus to keep exploring
            clip_range=0.2,
            verbose=1,
            tensorboard_log=str(RUNS),
        )

    print(f"Training for {TOTAL:,} steps across {N_ENVS} envs...")
    model.learn(
        total_timesteps=TOTAL,
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=not args.resume,
    )

    final = RUNS / "ape_reach_final.zip"
    model.save(str(final))
    print(f"Saved: {final}")

if __name__ == "__main__":
    main()
