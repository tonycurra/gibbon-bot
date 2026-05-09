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
from utils.gait_score import trot_coherence, stride_period  # fail fast at import, not mid-training

RUNS = pathlib.Path("runs/ape_walk_mjx")
ROLLOUTS = RUNS / "rollouts"

import hashlib, json
def _fingerprint():
    """Hash of files that, if changed, would invalidate a resume.
    Catches: reward shaping, env physics, scene geometry, network shape."""
    files = [
        pathlib.Path("envs/params_walk_mjx.py"),
        pathlib.Path("envs/walk_env_mjx.py"),
        pathlib.Path("data/scene.xml"),
    ]
    h = hashlib.sha256()
    parts = {}
    for f in files:
        if f.exists():
            b = f.read_bytes()
            digest = hashlib.sha256(b).hexdigest()[:16]
            parts[str(f)] = digest
            h.update(b)
    return {"hash": h.hexdigest()[:16], "parts": parts, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

N_STEPS     = 32   # smaller rollout = faster feedback per metric line
EPOCHS      = 4
MINIBATCHES = 8
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
VF_COEF     = 0.5
MAX_GRAD    = 0.5

TOTAL_STEPS = 100_000_000
LOG_EVERY   = 5    # metric block has ~10 host-device syncs; running every update was throttling GPU 5-10x


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
    fp = _fingerprint()
    (RUNS / "latest.fingerprint.json").write_text(json.dumps(fp, indent=2))
    print(f"[save] {path.name}", flush=True)
    if mlflow.active_run() is not None:
        mlflow.set_tag("latest_checkpoint", path.name)
        mlflow.log_metric("checkpoint_steps", steps, step=steps)


def load_latest(train_state, force=False):
    """Resume from the latest checkpoint.
    Returns (train_state, start_steps, status) where status is:
      'fresh'    — no checkpoint, starting from scratch
      'resumed'  — loaded checkpoint and fingerprint matches
      'mismatch' — checkpoint exists but training code/scene changed; only loaded if force=True
    """
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
    if not ckpts:
        print("[resume] no checkpoint found — starting fresh.", flush=True)
        return train_state, 0, "fresh"
    latest = ckpts[-1]
    fp_path = RUNS / "latest.fingerprint.json"
    cur = _fingerprint()
    if fp_path.exists():
        old = json.loads(fp_path.read_text())
        if old.get("hash") != cur.get("hash"):
            print(f"[resume] FINGERPRINT MISMATCH — training inputs changed since {latest.name}", flush=True)
            for f, d in cur["parts"].items():
                old_d = old.get("parts", {}).get(f, "(missing)")
                marker = "  " if old_d == d else "≠ "
                print(f"  {marker}{f}: was {old_d} now {d}", flush=True)
            if force:
                print("[resume] --resume given, loading anyway despite mismatch.", flush=True)
            else:
                # Auto-branch: archive the old run so nothing is lost, then start fresh.
                short = old.get("hash", "unknown")[:8]
                stamp = time.strftime("%Y%m%d-%H%M%S")
                archive_dir = RUNS / "_history" / f"{stamp}_{short}"
                archive_dir.mkdir(parents=True, exist_ok=True)
                moved = 0
                for f in list(RUNS.glob("mjx_*.npz")) + list(RUNS.glob("latest.fingerprint.json")) + list(RUNS.glob("train.log")):
                    if f.is_file():
                        f.rename(archive_dir / f.name)
                        moved += 1
                print(f"[resume] archived {moved} files → {archive_dir}", flush=True)
                print(f"[resume] starting fresh with new training inputs.", flush=True)
                return train_state, 0, "branched"
    data = np.load(str(latest))
    flat = [jnp.asarray(data[k]) for k in data.files]
    init_flat, treedef = jax.tree_util.tree_flatten(train_state.params)
    # Shape check — if obs/act dim changed, we can't load the weights even
    # with --resume. Auto-branch to _history/ and start fresh in that case.
    shape_mismatch = any(a.shape != b.shape for a, b in zip(flat, init_flat))
    if shape_mismatch:
        print("[resume] SHAPE MISMATCH — checkpoint weights don't fit current network (probably OBS_DIM/ACT_DIM changed)", flush=True)
        for i, (a, b) in enumerate(zip(flat, init_flat)):
            if a.shape != b.shape:
                print(f"  param {i}: ckpt {a.shape} vs net {b.shape}", flush=True)
        short = (json.loads(fp_path.read_text()).get("hash") if fp_path.exists() else "shape")[:8]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive_dir = RUNS / "_history" / f"{stamp}_{short}_shape"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in list(RUNS.glob("mjx_*.npz")) + list(RUNS.glob("latest.fingerprint.json")):
            if f.is_file():
                f.rename(archive_dir / f.name)
        print(f"[resume] archived old run → {archive_dir}, starting fresh", flush=True)
        return train_state, 0, "shape_branched"
    new_params = jax.tree_util.tree_unflatten(treedef, flat)
    steps = int(latest.stem.split("_")[1])
    print(f"[resume] loaded {latest.name} ({steps:,} steps)", flush=True)
    return train_state.replace(params=new_params), steps, "resumed"


def _install_log_tee():
    """Mirror stdout/stderr to runs/ape_walk_mjx/train.log line-buffered.
    The shell `| tee` was buffering at 4KB and leaving the file stale; doing
    this in-process guarantees flush-per-line."""
    RUNS.mkdir(parents=True, exist_ok=True)
    log_file = open(RUNS / "train.log", "a", buffering=1)
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams:
                try: st.write(s); st.flush()
                except Exception: pass
        def flush(self):
            for st in self.streams:
                try: st.flush()
                except Exception: pass
        def isatty(self): return False
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)


def main():
    _install_log_tee()
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",     type=int, default=None)
    parser.add_argument("--resume",   action="store_true",
                        help="force-resume even if training inputs changed")
    parser.add_argument("--no-resume", dest="auto_resume", action="store_false",
                        help="ignore existing checkpoints, start fresh")
    parser.add_argument("--no-watch", dest="watch", action="store_false")
    parser.set_defaults(auto_resume=True)
    parser.add_argument("--save-rollouts-every", type=int, default=0,
                        help="dump rollout buffer every N PPO updates (0=off)")
    parser.add_argument("--rollout-envs", type=int, default=256,
                        help="how many envs to keep per rollout dump (sub-sample for size)")
    parser.add_argument("--rollout-keep", type=int, default=50,
                        help="ring buffer: delete oldest rollouts beyond this count")
    parser.set_defaults(watch=True)
    args = parser.parse_args()

    N_ENVS = args.envs or _auto_envs()
    # Save aggressively: roughly every ~50 PPO updates so a crashed run loses
    # at most ~1 minute of compute, even on H100.
    save_every = max(1_000, min(100_000, N_ENVS * 50))
    RUNS.mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"{N_ENVS} envs  |  save every {save_every:,} steps")
    print(f"Loading model...")

    mx, jnt_lo, jnt_hi, tree_xy = load_model()

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    train_state = make_train_state(init_rng, float(P.LEARNING_RATE))
    start_steps = 0
    resume_status = "fresh"
    if args.auto_resume:
        train_state, start_steps, resume_status = load_latest(train_state, force=args.resume)

    rparams = tuple(jnp.float32(v) for v in read_params())

    v_reset = jax.jit(jax.vmap(lambda rng: reset(mx, jnt_lo, jnt_hi, tree_xy, rng)))
    v_step  = jax.jit(jax.vmap(
        lambda dx, a, mp, pa, pd, si: step(mx, jnt_lo, jnt_hi, tree_xy, dx, a, mp, pa, pd, si, rparams)
    ))

    env_rngs = jax.random.split(env_rng, N_ENVS)
    dx_batch, obs_batch, motor_pos_batch, prev_action_batch, prev_dist_batch, step_idx_batch = v_reset(env_rngs)
    step_counts = jnp.zeros(N_ENVS, dtype=jnp.int32)

    inherited_from = (
        str(sorted(RUNS.glob("mjx_*.npz"))[-1])
        if resume_status in ("resumed",) and list(RUNS.glob("mjx_*.npz")) else None
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
        "resume": resume_status,
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
            if _stop[0]:
                save(train_state, steps_here)
                print("Done.", flush=True)
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
             motor_pos_batch, prev_action_batch, prev_dist_batch, step_idx_batch) = v_step(
                dx_batch, actions, motor_pos_batch, prev_action_batch, prev_dist_batch, step_idx_batch
            )
            step_counts = step_counts + 1
            done_mask = dones | (step_counts >= MAX_STEPS)

            if jnp.any(done_mask):
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, N_ENVS)
                new_dx, new_obs, new_mp, new_pa, new_pd, new_si = v_reset(reset_rngs)
                dx_batch = jax.tree.map(
                    lambda n, o: jnp.where(done_mask.reshape(done_mask.shape + (1,) * (n.ndim - 1)), n, o) if n.ndim > 0 else n,
                    new_dx, dx_batch,
                )
                obs_batch         = jnp.where(done_mask[:, None], new_obs, obs_batch)
                motor_pos_batch   = jnp.where(done_mask[:, None], new_mp, motor_pos_batch)
                prev_action_batch = jnp.where(done_mask[:, None], new_pa, prev_action_batch)
                prev_dist_batch   = jnp.where(done_mask, new_pd, prev_dist_batch)
                step_idx_batch    = jnp.where(done_mask, new_si, step_idx_batch)
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
            stacked_rew = jnp.stack(buf_rew)
            stacked_dist = jnp.stack(buf_dist)
            stacked_done = jnp.stack(buf_done)
            stacked_obs = jnp.stack(buf_obs)
            mean_rew = float(jnp.mean(stacked_rew))
            mean_dist = float(jnp.mean(stacked_dist))
            min_dist = float(jnp.min(stacked_dist))
            max_dist = float(jnp.max(stacked_dist))
            best_rew = float(jnp.max(stacked_rew))
            done_frac = float(jnp.mean(stacked_done))
            success_frac = float(jnp.mean(stacked_dist < 0.6))
            # Body tracking — robot kinematics from obs (T, N_ENVS, OBS_DIM)
            # obs layout: [z(1), quat(4), linvel(3), angvel(3), jpos(24), jvel(24), torque(24), rel(2), dist(1), phase(2)]
            torso_z   = stacked_obs[..., 0]
            uprightness = stacked_obs[..., 1]           # quat w → 1 = upright, 0 = sideways, -1 = upside down
            speed_xy  = jnp.linalg.norm(stacked_obs[..., 5:7], axis=-1)
            jvel_norm = jnp.linalg.norm(stacked_obs[..., 35:59], axis=-1)  # joint velocity magnitude
            torque    = stacked_obs[..., 59:83]                            # 24 joint torques
            # bilateral symmetry (FL-FR + BL-BR torque magnitude diff)
            tFL, tFR = jnp.abs(torque[..., 0:6]),   jnp.abs(torque[..., 6:12])
            tBL, tBR = jnp.abs(torque[..., 12:18]), jnp.abs(torque[..., 18:24])
            asym = float(jnp.mean(jnp.sum((tFL - tFR) ** 2, axis=-1) + jnp.sum((tBL - tBR) ** 2, axis=-1)))
            mean_speed   = float(jnp.mean(speed_xy))
            max_speed    = float(jnp.max(speed_xy))
            mean_z       = float(jnp.mean(torso_z))
            mean_upright = float(jnp.mean(uprightness))
            mean_jvel    = float(jnp.mean(jvel_norm))
            # Gait-coherence score: how trot-like is the policy?
            # +ve = diagonal pairs synced, -ve = pacing/chaos. See utils/gait_score.
            jvel_avg = np.asarray(jnp.mean(stacked_obs[..., 35:59], axis=1))  # (T, 24) avg over envs
            gait_coh = float(trot_coherence(jvel_avg))
            stride_s = float(stride_period(jvel_avg, dt_action=16 * 0.002))
            print(f"{update_count:>8}  {total_steps:>12,}  {sps:>8,.0f}  rew={mean_rew:.2f}  d={mean_dist:.2f}  v={mean_speed:.2f}m/s  z={mean_z:.2f}  upr={mean_upright:.2f}  succ={success_frac*100:.1f}%  gait={gait_coh:+.2f}  stride={stride_s:.2f}s", flush=True)
            mlflow.log_metrics(
                {"sps": sps, "mean_rew": mean_rew, "mean_dist": mean_dist,
                 "min_dist": min_dist, "max_dist": max_dist,
                 "best_rew": best_rew, "done_frac": done_frac,
                 "success_frac": success_frac, "update": update_count,
                 "mean_speed": mean_speed, "max_speed": max_speed,
                 "mean_z": mean_z, "mean_upright": mean_upright,
                 "mean_jvel": mean_jvel, "asym_torque": asym,
                 "gait_coherence": gait_coh, "stride_period_s": stride_s},
                step=total_steps,
            )

            # Auto-promote: if this rollout achieved a meaningful new best,
            # snapshot the current weights into runs/<task>/keepers/ so they
            # survive even if next iteration regresses or git collects junk.
            if min_dist < 1.5 or success_frac > 0.05:
                kdir = RUNS / "keepers"
                kdir.mkdir(exist_ok=True)
                tag = f"min_d{min_dist:.2f}_succ{success_frac:.2f}_step{total_steps}"
                kpath = kdir / f"auto_{tag}.npz"
                if not kpath.exists():
                    flat, _ = jax.tree_util.tree_flatten(jax.device_get(train_state.params))
                    np.savez(str(kpath), *flat)
                    # Sidecar: full reproducibility unit (params + scene hash + metric snapshot).
                    sidecar = {
                        "tag": tag, "step": int(total_steps),
                        "metrics": {"min_dist": min_dist, "success_frac": success_frac,
                                    "mean_rew": mean_rew, "mean_speed": mean_speed,
                                    "mean_z": mean_z, "asym_torque": asym},
                        "params": p_params,
                        "fingerprint": _fingerprint(),
                        "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
                        "n_envs": N_ENVS, "n_steps": N_STEPS,
                    }
                    (kdir / f"auto_{tag}.json").write_text(json.dumps(sidecar, indent=2))
                    print(f"[keeper] saved {kpath.name} + sidecar", flush=True)

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
