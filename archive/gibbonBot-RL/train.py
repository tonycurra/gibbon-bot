#!/usr/bin/env python3
"""
Train the ape robot to stand. CPU mode with live visualization.

Usage:
    python train.py                  # train + watch (default)
    python train.py --no-watch       # train only
    python train.py --envs 32        # fewer parallel envs (faster on slow CPU)
    python train.py --inherit        # continue from latest checkpoint
    python train.py --visualize      # replay latest checkpoint
    python train.py --visualize mjx_00010000000.npz
"""

import argparse
import os
import pathlib
import sys
import time

# Force CPU before JAX is imported
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

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
)
from gibbonbot.envs.stand_env import StandEnv
from gibbonbot.utils import lineage as lineage_mod

RUNS = pathlib.Path("runs/ape_stand_mjx")
SCENE = pathlib.Path("data/scene.xml")


def get_latest_checkpoint():
    ckpts = sorted(RUNS.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def load_params(path):
    data = np.load(str(path))
    flat = [
        jnp.array(data[k])
        for k in sorted(data.files, key=lambda s: int(s.replace("arr_", "")))
    ]
    net = ActorCritic()
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    _, treedef = jax.tree_util.tree_flatten(template)
    return jax.tree_util.tree_unflatten(treedef, flat)


def visualize_policy(params, label="current", step_mode=False, speed=0.05):
    try:
        net = ActorCritic()

        @jax.jit
        def act(params, obs):
            mean, log_std, _ = net.apply(params, obs)
            return jnp.tanh(mean)

        env = StandEnv(render_mode="human")
        obs, _ = env.reset()
        _TREE_PAD = np.array([5.0, 0.0, 1.5], dtype=np.float32)
        obs_padded = np.concatenate([obs, _TREE_PAD])

        total_reward = 0.0
        step_count = 0
        done = False
        viewer_alive = True

        print(f"\n{'=' * 60}\nVISUALIZING: {label}\nClose MuJoCo window to exit\n{'=' * 60}")

        while not done and step_count < 1000:
            action = np.array(act(params, jnp.array(obs_padded)))
            obs, reward, terminated, truncated, info = env.step(action)
            obs_padded = np.concatenate([obs, _TREE_PAD])
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            sys.stdout.write(
                f"\rStep: {step_count:4d} | Reward: {total_reward:7.2f} | "
                f"Height: {info['above_ground']:.3f}m | "
                f"{'FALLEN' if info['fallen'] else 'ALIVE'}"
            )
            sys.stdout.flush()

            viewer_alive = getattr(env, "_viewer", None) is None or env._viewer.is_running()
            if not viewer_alive:
                break

            if step_mode:
                input(" <Enter>")
            else:
                time.sleep(speed)

        env.close()
        print()
    except Exception as e:
        print(f"\nVisualization error: {e}")


def train(args):
    N_STEPS = 128
    EPOCHS = 4
    MINIBATCHES = 8
    TOTAL_STEPS = 100_000_000
    SAVE_EVERY = 500_000
    LOG_EVERY = 2
    VIS_EVERY = args.vis_freq
    N_ENVS = args.envs

    RUNS.mkdir(parents=True, exist_ok=True)

    print(f"Training on: {jax.devices()}")
    print(f"Envs: {N_ENVS} | Watch: {args.watch} | Vis every {VIS_EVERY} updates")

    mx, jnt_lo, jnt_hi, hints = load_model()

    v_reset = jax.jit(jax.vmap(lambda rng: reset(mx, jnt_lo, jnt_hi, hints, rng)))
    v_step = jax.jit(
        jax.vmap(lambda dx, a, pa: step(mx, jnt_lo, jnt_hi, hints, dx, a, pa))
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)
    train_state = make_train_state(init_rng, 5e-4)

    if args.inherit:
        ckpt = get_latest_checkpoint()
        if ckpt:
            train_state = train_state.replace(params=load_params(ckpt))
            print(f"Inherited weights from {ckpt.name}")
        else:
            print("No checkpoint found, starting fresh")

    env_rngs = jax.random.split(env_rng, N_ENVS)
    dx_batch, obs_batch, prev_action_batch = v_reset(env_rngs)
    step_counts = jnp.zeros(N_ENVS, dtype=jnp.int32)

    lineage_mod.record(
        RUNS, "stand_mjx", N_ENVS,
        "inherited" if args.inherit else None,
        SCENE,
        pathlib.Path("src/gibbonbot/envs/stand_env_mjx.py"),
    )

    total_env_steps = 0
    update_count = 0
    last_save = 0
    t0 = time.time()

    print(f"Training with {N_ENVS} parallel envs...")

    while total_env_steps < TOTAL_STEPS:
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []

        for _ in range(N_STEPS):
            rng, act_rng = jax.random.split(rng)
            act_rngs = jax.random.split(act_rng, N_ENVS)

            actions, log_probs, values = jax.vmap(
                lambda o, r: sample_action(train_state.params, train_state.apply_fn, o, r)
            )(obs_batch, act_rngs)

            buf_obs.append(obs_batch)
            buf_act.append(actions)
            buf_logp.append(log_probs)
            buf_val.append(values)

            dx_batch, obs_batch, rewards, dones, prev_action_batch = v_step(
                dx_batch, actions, prev_action_batch
            )
            step_counts = step_counts + 1

            done_mask = dones | (step_counts >= 2000)
            if jnp.any(done_mask):
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, N_ENVS)
                new_dx, new_obs, new_pa = v_reset(reset_rngs)
                dx_batch = jax.tree.map(
                    lambda n, o: (
                        jnp.where(
                            done_mask.reshape(done_mask.shape + (1,) * (n.ndim - 1)), n, o
                        ) if n.ndim > 0 else n
                    ),
                    new_dx, dx_batch,
                )
                obs_batch = jnp.where(done_mask[:, None], new_obs, obs_batch)
                prev_action_batch = jnp.where(done_mask[:, None], new_pa, prev_action_batch)
                step_counts = jnp.where(done_mask, 0, step_counts)

            buf_rew.append(rewards)
            buf_done.append(done_mask.astype(jnp.float32))

        total_env_steps += N_ENVS * N_STEPS

        obs_t = jnp.stack(buf_obs)
        act_t = jnp.stack(buf_act)
        logp_t = jnp.stack(buf_logp)
        rew_t = jnp.stack(buf_rew)
        done_t = jnp.stack(buf_done)
        val_t = jnp.stack(buf_val)

        _, _, last_val = jax.vmap(
            lambda o, r: sample_action(train_state.params, train_state.apply_fn, o, r)
        )(obs_batch, jax.random.split(rng, N_ENVS))

        adv_t, ret_t = jax.vmap(
            lambda r, v, d, lv: gae(r, v, d, lv),
            in_axes=(1, 1, 1, 0), out_axes=1,
        )(rew_t, val_t, done_t, last_val)

        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        rollout = (flat(obs_t), flat(act_t), flat(logp_t), flat(adv_t), flat(ret_t))
        train_state, metrics = update_step(train_state, rollout)
        update_count += 1

        if update_count % LOG_EVERY == 0:
            elapsed = time.time() - t0
            sps = total_env_steps / elapsed
            mean_rew = float(jnp.mean(jnp.stack(buf_rew)))
            print(f"Step {total_env_steps:>10,} | SPS {sps:>8,.0f} | Reward {mean_rew:.3f} | Update {update_count}")

        if args.watch and update_count % VIS_EVERY == 0:
            visualize_policy(train_state.params, f"Update {update_count} ({total_env_steps:,} steps)")

        if total_env_steps - last_save >= SAVE_EVERY:
            _save(train_state, total_env_steps)
            last_save = total_env_steps

    _save(train_state, total_env_steps)
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64, help="Parallel environments (default 64 for CPU)")
    parser.add_argument("--inherit", action="store_true", help="Continue from latest checkpoint")
    parser.add_argument("--no-watch", dest="watch", action="store_false", help="Disable live visualization")
    parser.add_argument("--vis-freq", type=int, default=10, help="Visualize every N updates")
    parser.add_argument("--visualize", nargs="?", const="latest", help="Replay checkpoint only")
    parser.add_argument("--step", action="store_true", help="Step-by-step visualization")
    parser.add_argument("--speed", type=float, default=0.05, help="Visualization speed")
    parser.set_defaults(watch=True)
    args = parser.parse_args()

    if args.visualize:
        if args.visualize == "latest":
            ckpt = get_latest_checkpoint()
            if not ckpt:
                print("No checkpoints found.")
                sys.exit(1)
        else:
            ckpt = RUNS / args.visualize
            if not ckpt.exists():
                print(f"Checkpoint not found: {args.visualize}")
                sys.exit(1)
        visualize_policy(load_params(ckpt), ckpt.name, args.step, args.speed)
        return

    train(args)


if __name__ == "__main__":
    main()
