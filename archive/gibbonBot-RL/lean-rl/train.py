#!/usr/bin/env python3
"""
Unified training and visualization script for robot learning.
Supports training, visualization, and analysis in one tool.

Usage:
    # Train only (fast GPU training)
    python scripts/train.py

    # Train with live visualization
    python scripts/train.py --watch

    # Just visualize existing checkpoints
    python scripts/train.py --visualize

    # Visualize specific checkpoint
    python scripts/train.py --visualize mjx_00010000000.npz

    # Frame-by-frame analysis
    python scripts/train.py --visualize mjx_00010000000.npz --step

    # Train with fewer environments
    python scripts/train.py --envs 256

    # Continue from best checkpoint
    python scripts/train.py --inherit
"""

import argparse
import os
import pathlib
import sys
import time

# Set up the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "..", "src")
sys.path.insert(0, src_path)

import jax
import jax.numpy as jnp
import numpy as np

from gibbonbot.config.train_mjx import (
    ACT_DIM,
    OBS_DIM,
    ActorCritic,
    _save,
    gae,
    load_model,
    make_train_state,
    reset,
    sample_action,
    _save,
)
from gibbonbot.envs.stand_env import StandEnv
from gibbonbot.utils import lineage as lineage_mod


def get_latest_checkpoint():
    """Get latest checkpoint from runs directory."""
    runs_dir = pathlib.Path("runs/ape_stand_mjx")
    ckpts = sorted(runs_dir.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def load_params(path):
    """Load parameters from checkpoint."""
    data = np.load(str(path))
    flat = [
        jnp.array(data[k])
        for k in sorted(data.files, key=lambda s: int(s.replace("arr_", "")))
    ]
    net = ActorCritic()
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    _, treedef = jax.tree_util.tree_flatten(template)
    return jax.tree_util.tree_unflatten(treedef, flat)


def visualize_policy(params, checkpoint_name="current", step_mode=False, speed=0.05):
    """Visualize a policy (from checkpoint or current training)."""
    try:
        net = ActorCritic()

        @jax.jit
        def act(params, obs):
            mean, log_std, _ = net.apply(params, obs)
            return jnp.tanh(mean)

        env = StandEnv(render_mode="human")
        obs, _ = env.reset()

        # Pad observation to match MJX format
        _TREE_PAD = np.array([5.0, 0.0, 1.5], dtype=np.float32)
        obs_padded = np.concatenate([obs, _TREE_PAD])

        total_reward = 0.0
        step_count = 0
        done = False

        print(f"\n{'=' * 60}")
        print(f"🎬 VISUALIZING: {checkpoint_name}")
        if step_mode:
            print("   Mode: STEP-BY-STEP (Press Enter for each frame)")
        else:
            print(f"   Mode: CONTINUOUS (Speed: {speed}s per frame)")
        print("   Close MuJoCo window to exit")
        print(f"{'=' * 60}")

        start_time = time.time()

        while not done and step_count < 1000:
            action = np.array(act(params, jnp.array(obs_padded)))
            obs, reward, terminated, truncated, info = env.step(action)

            obs_padded = np.concatenate([obs, _TREE_PAD])

            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Display real-time info
            sys.stdout.write(
                f"\r📊 Step: {step_count:4d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Height: {info['above_ground']:.3f}m | "
                f"Status: {'FALLEN' if info['fallen'] else 'ALIVE'}"
            )
            sys.stdout.flush()

            # Check if viewer is still open
            viewer_alive = (
                getattr(env, "_viewer", None) is None or env._viewer.is_running()
            )
            if not viewer_alive:
                break

            # Control speed
            if step_mode:
                input(" <Press Enter>")
            else:
                time.sleep(speed)

        env.close()

        if viewer_alive:  # Only print if not interrupted
            elapsed = time.time() - start_time
            status = "FALLEN" if info.get("fallen", False) else "ALIVE"
            print(f"\n{'=' * 60}")
            print(f"📈 EPISODE SUMMARY:")
            print(f"   Outcome: {status}")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Steps: {step_count}")
            print(f"   Height: {info['above_ground']:.3f}m")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"⚠️ Visualization error: {e}")


def train_with_visualization(args):
    """Main training loop with optional visualization."""
    RUNS = pathlib.Path("runs/ape_stand_mjx")
    SCENE = pathlib.Path("scene.xml")
    N_STEPS = 128
    EPOCHS = 4
    MINIBATCHES = 8
    TOTAL_STEPS = 100_000_000
    SAVE_EVERY = 500_000
    LOG_EVERY = 2
    VIS_EVERY = args.vis_freq

    RUNS.mkdir(parents=True, exist_ok=True)
    N_ENVS = args.envs if args.envs != 1024 else _auto_envs()

    print("🚀 Starting unified training script")
    print(f"   Mode: {'TRAINING + VISUALIZATION' if args.watch else 'TRAINING ONLY'}")
    print(f"   Environments: {N_ENVS}")
    if args.watch:
        print(f"   Visualization: Every {VIS_EVERY} updates")
    print(f"   Target: {TOTAL_STEPS:,} steps")
    print(f"   JAX devices: {jax.devices()}")

    # Load model
    print("🤖 Loading model...")
    mx, jnt_lo, jnt_hi, hints = load_model()

    # Vmapped env functions
    v_reset = jax.jit(jax.vmap(lambda rng: reset(mx, jnt_lo, jnt_hi, hints, rng)))
    v_step = jax.jit(
        jax.vmap(lambda dx, a, pa: step(mx, jnt_lo, jnt_hi, hints, dx, a, pa))
    )

    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)
    train_state = make_train_state(init_rng, 5e-4)

    if args.inherit:
        try:
            ckpt = get_latest_checkpoint()
            if ckpt:
                params_flat = load_params(ckpt)
                train_state = train_state.replace(params=params_flat)
                print(f"🔄 Inherited weights from {ckpt.name}")
        except Exception as e:
            print(f"⚠️ Inheritance failed: {e}")

    # Reset environments
    env_rngs = jax.random.split(env_rng, N_ENVS)
    dx_batch, obs_batch, prev_action_batch = v_reset(env_rngs)
    step_counts = jnp.zeros(N_ENVS, dtype=jnp.int32)

    lineage_mod.record(
        RUNS,
        "stand_mjx",
        N_ENVS,
        "inherited" if args.inherit else None,
        SCENE,
        pathlib.Path("envs/stand_env_mjx.py"),
    )

    total_env_steps = 0
    update_count = 0
    last_save = 0
    t0 = time.time()

    print(f"🏋️ Training with {N_ENVS} parallel envs...")

    while total_env_steps < TOTAL_STEPS:
        # Collect rollout
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []

        for _ in range(N_STEPS):
            rng, act_rng = jax.random.split(rng)
            act_rngs = jax.random.split(act_rng, N_ENVS)

            actions, log_probs, values = jax.vmap(
                lambda o, r: sample_action(
                    train_state.params, train_state.apply_fn, o, r
                )
            )(obs_batch, act_rngs)

            buf_obs.append(obs_batch)
            buf_act.append(actions)
            buf_logp.append(log_probs)
            buf_val.append(values)

            dx_batch, obs_batch, rewards, dones, prev_action_batch = v_step(
                dx_batch, actions, prev_action_batch
            )
            step_counts = step_counts + 1

            # Reset done envs
            done_mask = dones | (step_counts >= 2000)
            if jnp.any(done_mask):
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, N_ENVS)
                new_dx, new_obs, new_pa = v_reset(reset_rngs)
                dx_batch = jax.tree.map(
                    lambda n, o: (
                        jnp.where(
                            done_mask.reshape(done_mask.shape + (1,) * (n.ndim - 1)),
                            n,
                            o,
                        )
                        if n.ndim > 0
                        else n
                    ),
                    new_dx,
                    dx_batch,
                )
                obs_batch = jnp.where(done_mask[:, None], new_obs, obs_batch)
                prev_action_batch = jnp.where(
                    done_mask[:, None], new_pa, prev_action_batch
                )
                step_counts = jnp.where(done_mask, 0, step_counts)

            buf_rew.append(rewards)
            buf_done.append(done_mask.astype(jnp.float32))

        total_env_steps += N_ENVS * N_STEPS

        # Stack rollout
        obs_t = jnp.stack(buf_obs)
        act_t = jnp.stack(buf_act)
        logp_t = jnp.stack(buf_logp)
        rew_t = jnp.stack(buf_rew)
        done_t = jnp.stack(buf_done)
        val_t = jnp.stack(buf_val)

        _, _, last_val = jax.vmap(
            lambda o, r: sample_action(train_state.params, train_state.apply_fn, o, r)
        )(obs_batch, jax.random.split(rng, N_ENVS))

        # GAE
        adv_t, ret_t = jax.vmap(
            lambda r, v, d, lv: gae(r, v, d, lv),
            in_axes=(1, 1, 1, 0),
            out_axes=1,
        )(rew_t, val_t, done_t, last_val)

        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        rollout = (flat(obs_t), flat(act_t), flat(logp_t), flat(adv_t), flat(ret_t))

        # PPO update
        train_state, metrics = update_step(train_state, rollout)
        update_count += 1

        # Logging
        if update_count % LOG_EVERY == 0:
            elapsed = time.time() - t0
            sps = total_env_steps / elapsed
            mean_rew = float(jnp.mean(jnp.stack(buf_rew)))
            print(
                f"📈 Step {total_env_steps:>10,} | SPS {sps:>8,.0f} | Reward {mean_rew:.3f} | Update {update_count}"
            )

        # Visualization
        if args.watch and update_count % VIS_EVERY == 0:
            visualize_policy(
                train_state.params, f"Update {update_count} ({total_env_steps:,} steps)"
            )

        # Saving
        if total_env_steps - last_save >= SAVE_EVERY:
            _save(train_state, total_env_steps)
            last_save = total_env_steps

    _save(train_state, total_env_steps)
    print("✅ Training completed!")


def _auto_envs():
    """Pick N_ENVS based on available GPU VRAM."""
    total_bytes = 0
    try:
        stats = jax.devices()[0].memory_stats()
        total_bytes = stats.get("bytes_limit", 0)
    except Exception:
        pass

    if total_bytes == 0:
        try:
            import subprocess

            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            total_bytes = int(out.strip()) * 1024 * 1024
        except Exception:
            pass

    gb = total_bytes / (1024**3)
    if gb >= 20:
        n = 2048
    elif gb >= 10:
        n = 1024
    elif gb >= 5:
        n = 512
    else:
        n = 256

    label = f"{gb:.1f} GB" if gb > 0 else "unknown"
    print(f"[auto] GPU VRAM: {label} → {n} parallel envs (override with --envs N)")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Unified training and visualization script"
    )

    # Training options
    parser.add_argument(
        "--envs", type=int, default=1024, help="Number of parallel environments"
    )
    parser.add_argument(
        "--inherit", action="store_true", help="Continue from latest checkpoint"
    )

    # Visualization options
    parser.add_argument(
        "--watch", action="store_true", help="Show live training progress"
    )
    parser.add_argument(
        "--vis-freq", type=int, default=10, help="Visualization frequency (updates)"
    )

    # Standalone visualization mode
    parser.add_argument(
        "--visualize",
        nargs="?",
        const="latest",
        help="Visualize checkpoint (latest or specific file)",
    )
    parser.add_argument(
        "--step", action="store_true", help="Step-by-step visualization"
    )
    parser.add_argument(
        "--speed", type=float, default=0.05, help="Visualization speed (seconds)"
    )

    args = parser.parse_args()

    # Handle visualization-only mode
    if args.visualize:
        if args.visualize == "latest":
            ckpt = get_latest_checkpoint()
            if not ckpt:
                print("❌ No checkpoints found!")
                sys.exit(1)
            print(f"🔍 Visualizing latest checkpoint: {ckpt.name}")
            visualize_policy(load_params(ckpt), ckpt.name, args.step, args.speed)
        else:
            ckpt_path = pathlib.Path("runs/ape_stand_mjx") / args.visualize
            if not ckpt_path.exists():
                print(f"❌ Checkpoint not found: {args.visualize}")
                sys.exit(1)
            print(f"🔍 Visualizing checkpoint: {args.visualize}")
            visualize_policy(
                load_params(ckpt_path), args.visualize, args.step, args.speed
            )
        return

    # Training mode
    train_with_visualization(args)

    train_with_visualization(args)

if __name__ == "__main__":
    main()
