#!/usr/bin/env python3
"""
Training script with live visualization of current policy.
Watches the robot learn in real-time during training.

Usage:
    python scripts/train_with_visualization.py  # Train and visualize every 10 updates
    python scripts/train_with_visualization.py --freq 5  # Visualize every 5 updates
    python scripts/train_with_visualization.py --envs 512  # Use fewer envs for stability
"""

import argparse
import os
import pathlib
import sys
import time

# Set up the Python path to include the src directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "..", "src")
sys.path.insert(0, src_path)

import subprocess

import jax
import jax.numpy as jnp
import numpy as np

# Import the training components
    ActorCritic,
    OBS_DIM,
    ACT_DIM,
    load_model,
    reset,
    step,
    make_train_state,
    sample_action,
    update_step,
    gae,
    _save,
)
from gibbonbot.envs.stand_env import StandEnv
from gibbonbot.utils import lineage as lineage_mod


def visualize_current_policy(train_state, step_count, total_env_steps):
    """Visualize the current policy using regular MuJoCo environment."""
    try:
        # Create deterministic action function
        net = ActorCritic()

        @jax.jit
        def act(params, obs):
            mean, log_std, _ = net.apply(params, obs)
            return jnp.tanh(mean)  # deterministic: use mean action

        env = StandEnv(render_mode="human")
        obs, _ = env.reset()

        # Pad observation to match MJX format
        _TREE_PAD = np.array([5.0, 0.0, 1.5], dtype=np.float32)
        obs_padded = np.concatenate([obs, _TREE_PAD])

        total_reward = 0.0
        step = 0
        done = False

        print(f"\n{'=' * 60}")
        print(f"🎬 VISUALIZING CURRENT POLICY")
        print(f"   Training Step: {total_env_steps:,}")
        print(f"   Update: {step_count}")
        print(f"   Press 'Q' in MuJoCo window to skip visualization")
        print(f"{'=' * 60}")

        start_time = time.time()

        while not done and step < 300:  # Limit to 300 steps
            action = np.array(act(train_state.params, jnp.array(obs_padded)))
            obs, reward, terminated, truncated, info = env.step(action)

            # Re-pad observation
            obs_padded = np.concatenate([obs, _TREE_PAD])

            total_reward += reward
            step += 1
            done = terminated or truncated

            # Check if user wants to skip
            viewer_alive = (
                getattr(env, "_viewer", None) is None or env._viewer.is_running()
            )
            if not viewer_alive:
                break

            time.sleep(0.03)  # Smooth visualization

        env.close()

        elapsed = time.time() - start_time
        status = "FALLEN" if info.get("fallen", False) else "ALIVE"
        print(f"📊 VISUALIZATION RESULTS:")
        print(f"   Outcome: {status}")
        print(f"   Reward: {total_reward:.2f}")
        print(f"   Steps: {step}")
        print(f"   Height: {info['above_ground']:.3f}m")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Checkpoint: Not saved yet (current policy)")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Train with live visualization")
    parser.add_argument(
        "--envs",
        type=int,
        default=512,
        help="Number of parallel environments (default: 512)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=10,
        help="Visualization frequency (updates between visualizations)",
    )
    parser.add_argument(
        "--inherit", action="store_true", help="Warm-start from best SB3 checkpoint"
    )
    args = parser.parse_args()

    # Training parameters
    RUNS = pathlib.Path("runs/ape_stand_mjx")
    SCENE = pathlib.Path("scene.xml")
    N_STEPS = 128
    EPOCHS = 4
    MINIBATCHES = 8
    TOTAL_STEPS = 100_000_000
    SAVE_EVERY = 500_000
    LOG_EVERY = 2

    RUNS.mkdir(parents=True, exist_ok=True)
    N_ENVS = args.envs

    print("🚀 Starting training with live visualization...")
    print(f"   Environments: {N_ENVS}")
    print(f"   Visualization: Every {args.freq} updates")
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
            from stable_baselines3 import PPO
            from envs import StandEnv

            env = StandEnv()
            sb3 = PPO.load("runs/ape_stand_mjx/best/best_model")
            print("🔄 Inherited weights from SB3 checkpoint")
            env.close()
        except Exception as e:
            print(f"⚠️  Inheritance failed: {e}")

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
                f"📈 Step {total_env_steps:>10,} | "
                f"SPS {sps:>8,.0f} | "
                f"Reward {mean_rew:.3f} | "
                f"Update {update_count}"
            )

        # Visualization
        if update_count % args.freq == 0:
            visualize_current_policy(train_state, update_count, total_env_steps)

        # Saving
        if total_env_steps - last_save >= SAVE_EVERY:
            _save(train_state, total_env_steps)
            last_save = total_env_steps

    _save(train_state, total_env_steps)
    print("✅ Training completed!")

        if update_count % args.freq == 0:
            visualize_current_policy(train_state, update_count, total_env_steps)

        # Saving
        if total_env_steps - last_save >= SAVE_EVERY:
            _save(train_state, total_env_steps)
            last_save = total_env_steps

    _save(train_state, total_env_steps)
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
