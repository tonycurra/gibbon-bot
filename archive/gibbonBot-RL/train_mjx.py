#!/usr/bin/env python3
"""
GPU-accelerated PPO training using MuJoCo MJX + JAX.

Runs 1024 parallel environments entirely on the GPU.
Expected speedup: ~50x over train.py on an RTX 4070.

Usage:
    python train_mjx.py                  # 1024 envs + live viewer
    python train_mjx.py --no-watch       # headless
    python train_mjx.py --envs 2048      # more envs (needs more VRAM)
    python train_mjx.py --resume         # continue from latest mjx checkpoint
"""

import argparse
import pathlib
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import mujoco

from flax.training.train_state import TrainState

import envs.params as params_mod
import importlib
from envs.stand_env_mjx import (
    load_model, reset, step,
    OBS_DIM, ACT_DIM, MAX_STEPS,
)
import utils.lineage as lineage_mod

RUNS = pathlib.Path("runs/ape_stand_mjx")
PARAMS_FILE = pathlib.Path("envs/params.py")

# PPO hyperparameters
N_STEPS     = 128
EPOCHS      = 4
MINIBATCHES = 8
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
CLIP_EPS    = 0.2
ENT_COEF    = 0.005
VF_COEF     = 0.5
MAX_GRAD    = 0.5
LR          = 3e-4
TOTAL_STEPS = 100_000_000
SAVE_EVERY  = 1_000_000
LOG_EVERY   = 10


# ── network ──────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.tanh(nn.Dense(256)(obs))
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(128)(x))
        mean    = nn.Dense(ACT_DIM)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (ACT_DIM,))
        value   = nn.Dense(1)(x).squeeze(-1)
        return mean, log_std, value


def make_train_state(rng):
    net = ActorCritic()
    dummy = jnp.zeros((1, OBS_DIM))
    params = net.init(rng, dummy)
    tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD), optax.adam(LR))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


# ── PPO helpers ───────────────────────────────────────────────────────────────

def sample_action(params, apply_fn, obs, rng):
    mean, log_std, value = apply_fn(params, obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(rng, mean.shape)
    action = jnp.tanh(mean + std * eps)
    log_prob = _gaussian_log_prob(mean + std * eps, mean, log_std)
    log_prob -= jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
    return action, log_prob, value


def _gaussian_log_prob(x, mean, log_std):
    return jnp.sum(
        -0.5 * ((x - mean) / jnp.exp(log_std)) ** 2
        - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


def gae(rewards, values, dones, last_value):
    T = rewards.shape[0]

    def body(carry, t):
        gae_val = carry
        next_val = jnp.where(t == T - 1, last_value, values[t + 1])
        delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
        gae_val = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae_val
        return gae_val, gae_val

    _, advantages = jax.lax.scan(body, 0.0, jnp.arange(T - 1, -1, -1))
    advantages = jnp.flip(advantages, axis=0)
    return advantages, advantages + values


def ppo_loss(params, apply_fn, obs, actions, old_log_probs, advantages, returns):
    mean, log_std, values = jax.vmap(lambda o: apply_fn(params, o))(obs)
    actions_pre = jnp.arctanh(jnp.clip(actions, -0.999, 0.999))
    log_probs = _gaussian_log_prob(actions_pre, mean, log_std)
    log_probs -= jnp.sum(jnp.log(1 - actions ** 2 + 1e-6), axis=-1)

    ratio = jnp.exp(log_probs - old_log_probs)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = -jnp.minimum(
        ratio * adv_norm,
        jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_norm,
    ).mean()
    vf_loss = 0.5 * ((values - returns) ** 2).mean()
    entropy = (0.5 * (1 + jnp.log(2 * jnp.pi)) + log_std).sum(-1).mean()
    loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy
    return loss, (pg_loss, vf_loss, entropy)


@jax.jit
def update_step(train_state, rollout):
    obs, actions, old_log_probs, advantages, returns = rollout
    mb_size = obs.shape[0] // MINIBATCHES
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def epoch_fn(ts, _):
        def mb_fn(ts, idx):
            sl  = jax.lax.dynamic_slice_in_dim(obs,           idx * mb_size, mb_size)
            sa  = jax.lax.dynamic_slice_in_dim(actions,       idx * mb_size, mb_size)
            sp  = jax.lax.dynamic_slice_in_dim(old_log_probs, idx * mb_size, mb_size)
            sv  = jax.lax.dynamic_slice_in_dim(advantages,    idx * mb_size, mb_size)
            sr  = jax.lax.dynamic_slice_in_dim(returns,       idx * mb_size, mb_size)
            (loss, aux), grads = grad_fn(ts.params, ts.apply_fn, sl, sa, sp, sv, sr)
            return ts.apply_gradients(grads=grads), (loss, *aux)
        return jax.lax.scan(mb_fn, ts, jnp.arange(MINIBATCHES))

    train_state, metrics = jax.lax.scan(epoch_fn, train_state, None, length=EPOCHS)
    return train_state, metrics


# ── params ────────────────────────────────────────────────────────────────────

_params_mtime = 0.0

def read_rparams():
    global _params_mtime
    try:
        mtime = PARAMS_FILE.stat().st_mtime
        if mtime != _params_mtime:
            importlib.reload(params_mod)
            _params_mtime = mtime
    except Exception as e:
        print(f"[params] reload error: {e}")
    p = params_mod
    return (
        jnp.float32(p.TARGET_Z),
        jnp.float32(p.TARGET_TOL),
        jnp.float32(p.W_HEIGHT),
        jnp.float32(p.W_UPRIGHT),
        jnp.float32(p.ARM_SAFETY_MARGIN),
        jnp.float32(p.W_PROXIMITY),
        jnp.float32(p.FLIP_PENALTY),
    )


# ── checkpointing ─────────────────────────────────────────────────────────────

def save(train_state, steps):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"mjx_{steps:012d}.npz"
    flat, _ = jax.tree_util.tree_flatten(jax.device_get(train_state.params))
    np.savez(str(path), *flat)
    print(f"[save] {path.name}")


def load_latest(train_state):
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
    if not ckpts:
        print("[resume] no checkpoint found — starting fresh.")
        return train_state, 0
    latest = ckpts[-1]
    data = np.load(str(latest))
    flat = [jnp.asarray(data[k]) for k in data.files]
    _, treedef = jax.tree_util.tree_flatten(train_state.params)
    new_params = jax.tree_util.tree_unflatten(treedef, flat)
    steps = int(latest.stem.split("_")[1])
    print(f"[resume] loaded {latest.name} ({steps:,} steps)")
    return train_state.replace(params=new_params), steps


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",     type=int, default=1024)
    parser.add_argument("--resume",   action="store_true", help="continue from latest mjx checkpoint")
    parser.add_argument("--no-watch", dest="watch", action="store_false")
    parser.set_defaults(watch=True)
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    N_ENVS = args.envs

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading model and compiling...")
    mx, jnt_lo, jnt_hi = load_model()

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    train_state = make_train_state(init_rng)
    start_steps = 0
    if args.resume:
        train_state, start_steps = load_latest(train_state)

    rparams = read_rparams()

    v_reset = jax.jit(jax.vmap(lambda rng: reset(mx, jnt_lo, jnt_hi, rng)))
    v_step  = jax.jit(jax.vmap(
        lambda dx, a, pa: step(mx, jnt_lo, jnt_hi, dx, a, pa, rparams)
    ))

    env_rngs = jax.random.split(env_rng, N_ENVS)
    dx_batch, obs_batch, prev_action_batch = v_reset(env_rngs)
    step_counts = jnp.zeros(N_ENVS, dtype=jnp.int32)

    lineage_mod.record(
        RUNS, "stand_mjx", N_ENVS,
        str(sorted(RUNS.glob("mjx_*.npz"))[-1]) if args.resume and list(RUNS.glob("mjx_*.npz")) else None,
        pathlib.Path("data/scene.xml"),
        pathlib.Path("envs/stand_env_mjx.py"),
    )

    total_steps = start_steps
    update_count = 0
    last_save = start_steps
    t0 = time.time()

    watcher = None
    if args.watch:
        watcher = subprocess.Popen([sys.executable, "utils/watch_mjx.py"])

    print(f"Training with {N_ENVS} parallel envs — {TOTAL_STEPS:,} steps total.")
    print(f"{'updates':>8}  {'steps':>12}  {'sps':>8}  {'mean_rew':>10}")
    print("-" * 48)

    while total_steps < TOTAL_STEPS:
        # re-read params each rollout (hot-reload support)
        new_rparams = read_rparams()
        if any(float(a) != float(b) for a, b in zip(new_rparams, rparams)):
            rparams = new_rparams
            v_step = jax.jit(jax.vmap(
                lambda dx, a, pa: step(mx, jnt_lo, jnt_hi, dx, a, pa, rparams)
            ))

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
            done_mask = dones | (step_counts >= MAX_STEPS)

            if jnp.any(done_mask):
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, N_ENVS)
                new_dx, new_obs, new_pa = v_reset(reset_rngs)
                dx_batch = jax.tree.map(
                    lambda n, o: jnp.where(done_mask.reshape(done_mask.shape + (1,) * (n.ndim - 1)), n, o) if n.ndim > 0 else n,
                    new_dx, dx_batch,
                )
                obs_batch = jnp.where(done_mask[:, None], new_obs, obs_batch)
                prev_action_batch = jnp.where(done_mask[:, None], new_pa, prev_action_batch)
                step_counts = jnp.where(done_mask, 0, step_counts)

            buf_rew.append(rewards)
            buf_done.append(done_mask.astype(jnp.float32))

        total_steps += N_ENVS * N_STEPS

        obs_t  = jnp.stack(buf_obs)
        act_t  = jnp.stack(buf_act)
        logp_t = jnp.stack(buf_logp)
        rew_t  = jnp.stack(buf_rew)
        done_t = jnp.stack(buf_done)
        val_t  = jnp.stack(buf_val)

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
            sps = (total_steps - start_steps) / (time.time() - t0)
            mean_rew = float(jnp.mean(jnp.stack(buf_rew)))
            print(f"{update_count:>8}  {total_steps:>12,}  {sps:>8,.0f}  {mean_rew:>10.3f}")

        if total_steps - last_save >= SAVE_EVERY:
            save(train_state, total_steps)
            last_save = total_steps

    save(train_state, total_steps)
    if watcher is not None:
        watcher.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
