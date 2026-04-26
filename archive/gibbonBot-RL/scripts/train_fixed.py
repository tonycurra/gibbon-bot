"""
Train the ape robot using PPO.

Usage:
    python train.py --task stand            # fresh run (random init)
    python train.py --task stand --inherit  # warm-start from best checkpoint
    python train.py --task stand --resume   # continue exact run (preserves step count)
    python train.py --task stand --watch    # open live viewer alongside training
    python train.py --task stand --inherit --watch --envs 8

--inherit vs --resume
  --resume  : continues the same run — step counter keeps going, same TensorBoard curve.
              Use when you paused and want to continue.
  --inherit : starts a NEW run but copies the network weights from the best checkpoint.
              Use when you changed physics params (kp, torque, etc.) and want the agent
              to start with existing knowledge instead of from scratch.

Checkpoints:  runs/<task>/
TensorBoard:  tensorboard --logdir runs/
Lineage:      python train.py --lineage --task stand
"""

import argparse
import os
import pathlib
import subprocess
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)
print(f"Added to Python path: {src_path}")

from gibbonbot.envs import ApeEnv, StandEnv
from gibbonbot.utils import lineage as lineage_mod

TASKS = {
    "reach": (ApeEnv, "ape_reach", pathlib.Path("envs/ape_env.py")),
    "stand": (StandEnv, "ape_stand", pathlib.Path("envs/stand_env.py")),
}

SCENE = pathlib.Path("scene.xml")
STEPS = 2_048
TOTAL = 10_000_000


def best_checkpoint(runs_dir: pathlib.Path, run_name: str) -> pathlib.Path | None:
    best = runs_dir / "best" / "best_model.zip"
    if best.exists():
        return best
    checkpoints = sorted(runs_dir.glob(f"{run_name}_*.zip"))
    return checkpoints[-1] if checkpoints else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=list(TASKS))
    parser.add_argument(
        "--resume", action="store_true", help="continue exact run from last checkpoint"
    )
    parser.add_argument(
        "--inherit",
        action="store_true",
        help="warm-start weights from best checkpoint, fresh step counter",
    )
    parser.add_argument(
        "--watch", action="store_true", help="open live viewer alongside training"
    )
    parser.add_argument(
        "--envs", type=int, default=1, help="number of parallel training envs"
    )
    parser.add_argument(
        "--lineage", action="store_true", help="print lineage table and exit"
    )
    args = parser.parse_args()

    EnvClass, run_name, env_file = TASKS[args.task]
    RUNS = pathlib.Path(f"runs/{run_name}")
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.lineage:
        lineage_mod.show(RUNS)
        return

    n_envs = args.envs
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

    inherited_from = None

    if args.resume:
        ckpt = best_checkpoint(RUNS, run_name)
        if ckpt is None:
            print("No checkpoint found — starting from scratch.")
        else:
            print(f"[resume] continuing from {ckpt.name}")
            model = PPO.load(str(ckpt), env=env)
            lineage_mod.record(RUNS, args.task, n_envs, str(ckpt), SCENE, env_file)
            _run(
                model,
                TOTAL,
                checkpoint_cb,
                eval_cb,
                resume=True,
                task=args.task,
                watch=args.watch,
            )
            return

    if args.inherit:
        ckpt = best_checkpoint(RUNS, run_name)
        if ckpt is None:
            print("[inherit] no checkpoint found — starting from scratch.")
        else:
            print(f"[inherit] warm-starting weights from {ckpt.name}")
            inherited_from = str(ckpt)
            # Load weights only — fresh env, fresh step counter, fresh TensorBoard run
            old_model = PPO.load(str(ckpt), env=env)
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
            model.policy.load_state_dict(old_model.policy.state_dict())
            print("[inherit] weights transferred.")
            lineage_mod.record(RUNS, args.task, n_envs, inherited_from, SCENE, env_file)
            _run(
                model,
                TOTAL,
                checkpoint_cb,
                eval_cb,
                resume=False,
                task=args.task,
                watch=args.watch,
            )
            return

    # fresh run
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
    lineage_mod.record(RUNS, args.task, n_envs, None, SCENE, env_file)
    _run(
        model,
        TOTAL,
        checkpoint_cb,
        eval_cb,
        resume=False,
        task=args.task,
        watch=args.watch,
    )


def _run(model, total, checkpoint_cb, eval_cb, *, resume, task, watch):
    watcher = None
    if watch:
        watcher = subprocess.Popen(
            [sys.executable, "src/gibbonbot/utils/watch.py", "--task", task],
            cwd=pathlib.Path(__file__).parent,
        )
    print(f"Training for {total:,} steps ...")
    try:
        model.learn(
            total_timesteps=total,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=not resume,
        )
    finally:
        if watcher is not None:
            watcher.wait()

    run_name = (
        model.tensorboard_log and pathlib.Path(model.tensorboard_log).name or "model"
    )
    final = pathlib.Path(f"runs/{run_name}_final.zip")
    model.save(str(final))
    print(f"Saved: {final}")


if __name__ == "__main__":
    main()
