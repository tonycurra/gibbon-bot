#!/usr/bin/env python3
"""GPU-accelerated PPO for walking toward a tree (MJX). Mirrors train_mjx.py."""
import argparse
import functools
import os
import pathlib
import signal
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import mlflow

from flax.training.train_state import TrainState

from envs.walk_env_mjx import (
    load_model, reset, step, read_params,
    OBS_DIM, ACT_DIM, MAX_STEPS,
)
import envs.params_walk_mjx as P
import utils.lineage as lineage_mod

RUNS = pathlib.Path("runs/ape_walk_mjx")
ROLLOUTS = RUNS / "rollouts"

N_STEPS     = 128
EPOCHS      = 4
MINIBATCHES = 8
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
VF_COEF     = 0.5
MAX_GRAD    = 0.5

TOTAL_STEPS = 100_000_000
LOG_EVERY   = 1


def _auto_envs():
    try:
        backend = jax.default_backend()
        if backend == "cpu":
            return 64
        mem = jax.devices()[0].memory_stats()
        gb = mem.get("bytes_limit", 0) / 1e9
    except Exception:
        gb = 0
    if gb >= 20: return 2048
    if gb >= 10: return 1024
    if gb >= 5:  return 512
    return 256


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


def make_train_state(rng, lr):
    net = ActorCritic()
    dummy = jnp.zeros((1, OBS_DIM))
    params = net.init(rng, dummy)
    tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD), optax.adam(lr))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


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
        -0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi),
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


def ppo_loss(params, apply_fn, obs, actions, old_log_probs, advantages, returns, ent_coef, clip_eps):
    mean, log_std, values = jax.vmap(lambda o: apply_fn(params, o))(obs)
    actions_pre = jnp.arctanh(jnp.clip(actions, -0.999, 0.999))
    log_probs = _gaussian_log_prob(actions_pre, mean, log_std)
    log_probs -= jnp.sum(jnp.log(1 - actions ** 2 + 1e-6), axis=-1)
    ratio = jnp.exp(log_probs - old_log_probs)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = -jnp.minimum(
        ratio * adv_norm,
        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm,
    ).mean()
    vf_loss = 0.5 * ((values - returns) ** 2).mean()
    entropy = (0.5 * (1 + jnp.log(2 * jnp.pi)) + log_std).sum(-1).mean()
    loss = pg_loss + VF_COEF * vf_loss - ent_coef * entropy
    return loss, (pg_loss, vf_loss, entropy)


@functools.partial(jax.jit, static_argnums=())
def update_step(train_state, rollout, ent_coef, clip_eps):
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
            (loss, aux), grads = grad_fn(ts.params, ts.apply_fn, sl, sa, sp, sv, sr, ent_coef, clip_eps)
            return ts.apply_gradients(grads=grads), (loss, *aux)
        return jax.lax.scan(mb_fn, ts, jnp.arange(MINIBATCHES))
    train_state, metrics = jax.lax.scan(epoch_fn, train_state, None, length=EPOCHS)
    return train_state, metrics


def dump_rollout(update, steps, n_envs_keep, keep, **arrays):
    ROLLOUTS.mkdir(parents=True, exist_ok=True)
    path = ROLLOUTS / f"rollout_{update:08d}_{steps:012d}.npz"
    host = {k: np.asarray(jax.device_get(v))[:, :n_envs_keep] for k, v in arrays.items()}
    np.savez_compressed(str(path), **host)
    existing = sorted(ROLLOUTS.glob("rollout_*.npz"))
    for old in existing[:-keep]:
        old.unlink()
    print(f"[rollout] {path.name}  ({path.stat().st_size/1e6:.1f}MB, {len(existing[-keep:])} kept)")


def save(train_state, steps):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"mjx_{steps:012d}.npz"
    flat, _ = jax.tree_util.tree_flatten(jax.device_get(train_state.params))
    np.savez(str(path), *flat)
    print(f"[save] {path.name}")
    if mlflow.active_run() is not None:
        mlflow.set_tag("latest_checkpoint", path.name)
        mlflow.log_metric("checkpoint_steps", steps, step=steps)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",     type=int, default=None)
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--no-watch", dest="watch", action="store_false")
    parser.add_argument("--save-rollouts-every", type=int, default=0,
                        help="dump rollout buffer every N PPO updates (0=off)")
    parser.add_argument("--rollout-envs", type=int, default=256,
                        help="how many envs to keep per rollout dump (sub-sample for size)")
    parser.add_argument("--rollout-keep", type=int, default=50,
                        help="ring buffer: delete oldest rollouts beyond this count")
    parser.set_defaults(watch=True)
    args = parser.parse_args()

    N_ENVS = args.envs or _auto_envs()
    save_every = max(1_000, min(250_000, N_ENVS * 250))
    RUNS.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"{N_ENVS} envs  |  save every {save_every:,} steps")
    print(f"Loading model...")

    mx, jnt_lo, jnt_hi, tree_xy = load_model()

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    train_state = make_train_state(init_rng, float(P.LEARNING_RATE))
    start_steps = 0
    if args.resume:
        train_state, start_steps = load_latest(train_state)

    rparams = tuple(jnp.float32(v) for v in read_params())

    v_reset = jax.jit(jax.vmap(lambda rng: reset(mx, jnt_lo, jnt_hi, tree_xy, rng)))
    v_step  = jax.jit(jax.vmap(
        lambda dx, a, mp, pa, pd: step(mx, jnt_lo, jnt_hi, tree_xy, dx, a, mp, pa, pd, rparams)
    ))

    env_rngs = jax.random.split(env_rng, N_ENVS)
    dx_batch, obs_batch, motor_pos_batch, prev_action_batch, prev_dist_batch = v_reset(env_rngs)
    step_counts = jnp.zeros(N_ENVS, dtype=jnp.int32)

    inherited_from = (
        str(sorted(RUNS.glob("mjx_*.npz"))[-1])
        if args.resume and list(RUNS.glob("mjx_*.npz")) else None
    )
    lineage_mod.record(
        RUNS, "walk_mjx", N_ENVS, inherited_from,
        pathlib.Path("data/scene.xml"),
        pathlib.Path("envs/walk_env_mjx.py"),
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ape_walk_mjx")
    mlflow.start_run(run_name=time.strftime("%Y%m%d-%H%M%S"))
    p_params = {k: getattr(P, k) for k in dir(P) if k.isupper() and not k.startswith("_")}
    mlflow.log_params({
        **p_params,
        "N_ENVS": N_ENVS,
        "N_STEPS": N_STEPS,
        "EPOCHS": EPOCHS,
        "MINIBATCHES": MINIBATCHES,
        "GAMMA": GAMMA,
        "GAE_LAMBDA": GAE_LAMBDA,
        "VF_COEF": VF_COEF,
        "MAX_GRAD": MAX_GRAD,
        "TOTAL_STEPS": TOTAL_STEPS,
        "save_every": save_every,
        "MAX_STEPS_episode": int(MAX_STEPS),
        "OBS_DIM": OBS_DIM,
        "ACT_DIM": ACT_DIM,
    })
    mlflow.set_tags({
        "task": "walk_mjx",
        "inherited_from": inherited_from or "scratch",
        "resume": str(bool(args.resume)),
        "jax_backend": jax.default_backend(),
        "start_steps": str(start_steps),
    })

    total_steps  = start_steps
    update_count = 0
    last_save    = start_steps
    t0           = time.time()

    watcher = None
    if args.watch:
        watch_cmd = [sys.executable, "view.py", "--env", "walk", "--since", str(time.time())]
        print(f"[watch] spawning viewer: {' '.join(watch_cmd)}")
        watcher = subprocess.Popen(watch_cmd)

    print(f"Training — {TOTAL_STEPS:,} total steps.")
    print(f"{'updates':>8}  {'steps':>12}  {'sps':>8}  {'mean_rew':>10}  {'mean_dist':>10}")
    print("-" * 60)

    _stop = [False]
    def _on_sigint(signum, frame):
        if _stop[0]:
            print("\n[stop] force exit")
            sys.exit(130)
        _stop[0] = True
        print("\n[stop] Ctrl+C — finishing rollout then saving...")
    signal.signal(signal.SIGINT, _on_sigint)

    ent_coef = jnp.float32(P.ENT_COEF)
    clip_eps = jnp.float32(P.CLIP_EPS)

    while total_steps < TOTAL_STEPS and not _stop[0]:
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val, buf_dist = [], [], [], [], [], [], []

        for t in range(N_STEPS):
            steps_here = total_steps + t * N_ENVS
            if steps_here - last_save >= save_every and steps_here > start_steps:
                save(train_state, steps_here)
                last_save = steps_here
            if _stop[0]:
                save(train_state, steps_here)
                print("Done.")
                sys.exit(0)

            rng, act_rng = jax.random.split(rng)
            act_rngs = jax.random.split(act_rng, N_ENVS)

            actions, log_probs, values = jax.vmap(
                lambda o, r: sample_action(train_state.params, train_state.apply_fn, o, r)
            )(obs_batch, act_rngs)

            buf_obs.append(obs_batch)
            buf_act.append(actions)
            buf_logp.append(log_probs)
            buf_val.append(values)

            (dx_batch, obs_batch, rewards, dones,
             motor_pos_batch, prev_action_batch, prev_dist_batch) = v_step(
                dx_batch, actions, motor_pos_batch, prev_action_batch, prev_dist_batch
            )
            step_counts = step_counts + 1
            done_mask = dones | (step_counts >= MAX_STEPS)

            if jnp.any(done_mask):
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, N_ENVS)
                new_dx, new_obs, new_mp, new_pa, new_pd = v_reset(reset_rngs)
                dx_batch = jax.tree.map(
                    lambda n, o: jnp.where(done_mask.reshape(done_mask.shape + (1,) * (n.ndim - 1)), n, o) if n.ndim > 0 else n,
                    new_dx, dx_batch,
                )
                obs_batch         = jnp.where(done_mask[:, None], new_obs, obs_batch)
                motor_pos_batch   = jnp.where(done_mask[:, None], new_mp, motor_pos_batch)
                prev_action_batch = jnp.where(done_mask[:, None], new_pa, prev_action_batch)
                prev_dist_batch   = jnp.where(done_mask, new_pd, prev_dist_batch)
                step_counts       = jnp.where(done_mask, 0, step_counts)

            buf_rew.append(rewards)
            buf_done.append(done_mask.astype(jnp.float32))
            buf_dist.append(prev_dist_batch)

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
        train_state, _ = update_step(train_state, rollout, ent_coef, clip_eps)
        update_count += 1

        if args.save_rollouts_every and update_count % args.save_rollouts_every == 0:
            n_keep = min(args.rollout_envs, N_ENVS)
            dump_rollout(
                update_count, total_steps, n_keep, args.rollout_keep,
                obs=obs_t, act=act_t, logp=logp_t, rew=rew_t,
                done=done_t, val=val_t, adv=adv_t, ret=ret_t,
            )

        if update_count % LOG_EVERY == 0:
            sps = (total_steps - start_steps) / (time.time() - t0)
            mean_rew = float(jnp.mean(jnp.stack(buf_rew)))
            mean_dist = float(jnp.mean(jnp.stack(buf_dist)))
            print(f"{update_count:>8}  {total_steps:>12,}  {sps:>8,.0f}  {mean_rew:>10.3f}  {mean_dist:>10.3f}")
            mlflow.log_metrics(
                {"sps": sps, "mean_rew": mean_rew, "mean_dist": mean_dist,
                 "update": update_count},
                step=total_steps,
            )

        if total_steps - last_save >= save_every:
            save(train_state, total_steps)
            last_save = total_steps

    reason = "Ctrl+C" if _stop[0] else "completed"
    print(f"[stop] {reason} — saving at {total_steps:,} steps")
    save(train_state, total_steps)
    mlflow.set_tag("stop_reason", reason)
    mlflow.set_tag("final_steps", str(total_steps))
    mlflow.end_run()
    if watcher is not None:
        watcher.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
