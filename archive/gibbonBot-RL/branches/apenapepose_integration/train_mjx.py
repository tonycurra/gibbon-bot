#!/usr/bin/env python3
"""Generic GPU-accelerated PPO trainer for the gibbon (MJX).

Use --task to select which skill to train: walk, stand, walk_forward,
turn_left, turn_right, etc.

By convention each task has:
  envs/<task>_env_mjx.py          (env module — reset/step/reward)
  envs/params_<task>_mjx.py       (hyperparameters)
  runs/ape_<task>_mjx/            (checkpoints, train.log, keepers/)
  MLflow experiment "ape_<task>_mjx"
"""

import argparse
import functools
import importlib
import os
import pathlib
import signal
import subprocess
import sys
import time
from collections import deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import hashlib
import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
from flax.training.train_state import TrainState

import utils.lineage as lineage_mod
from utils.gait_score import stride_period, trot_coherence

# ── PPO hyperparameters (task-agnostic) ──────────────────────────────────────
N_STEPS = 32
EPOCHS = 4
MINIBATCHES = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
VF_COEF = 0.5
MAX_GRAD = 0.5
TOTAL_STEPS = 100_000_000
LOG_EVERY = 5


# ── Task plumbing ────────────────────────────────────────────────────────────
def task_module_paths(task: str):
    """Return (env_module_path, params_module_path) for the named task."""
    return f"envs.{task}_env_mjx", f"envs.params_{task}_mjx"


def task_runs_dir(task: str) -> pathlib.Path:
    return pathlib.Path(f"runs/ape_{task}_mjx")


def task_fingerprint_files(task: str):
    return [
        pathlib.Path(f"envs/params_{task}_mjx.py"),
        pathlib.Path(f"envs/{task}_env_mjx.py"),
        pathlib.Path("data/scene.xml"),
        pathlib.Path("envs/_shared_load_model.py"),
        pathlib.Path("envs/_shared_dynamics.py"),
        pathlib.Path("envs/_shared_obs.py"),
    ]


def fingerprint(task: str):
    h = hashlib.sha256()
    parts = {}
    for f in task_fingerprint_files(task):
        if f.exists():
            b = f.read_bytes()
            digest = hashlib.sha256(b).hexdigest()[:16]
            parts[str(f)] = digest
            h.update(b)
    return {
        "hash": h.hexdigest()[:16],
        "parts": parts,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _auto_envs():
    try:
        backend = jax.default_backend()
        if backend == "cpu":
            return 64
        mem = jax.devices()[0].memory_stats()
        gb = mem.get("bytes_limit", 0) / 1e9
    except Exception:
        gb = 0
    if gb >= 20:
        return 2048
    if gb >= 10:
        return 1024
    if gb >= 5:
        return 512
    return 256


# ── Network ──────────────────────────────────────────────────────────────────
def make_actor_critic(act_dim):
    class ActorCritic(nn.Module):
        @nn.compact
        def __call__(self, obs):
            x = nn.tanh(nn.Dense(256)(obs))
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(128)(x))
            mean = nn.Dense(act_dim)(x)
            log_std = self.param("log_std", nn.initializers.zeros, (act_dim,))
            value = nn.Dense(1)(x).squeeze(-1)
            return mean, log_std, value

    return ActorCritic()


def make_train_state(rng, lr, obs_dim, act_dim):
    net = make_actor_critic(act_dim)
    dummy = jnp.zeros((1, obs_dim))
    params = net.init(rng, dummy)
    tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD), optax.adam(lr))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


def sample_action(params, apply_fn, obs, rng):
    mean, log_std, value = apply_fn(params, obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(rng, mean.shape)
    action = jnp.tanh(mean + std * eps)
    log_prob = _gaussian_log_prob(mean + std * eps, mean, log_std)
    log_prob -= jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)
    return action, log_prob, value


def _gaussian_log_prob(x, mean, log_std):
    return jnp.sum(
        -0.5 * ((x - mean) / jnp.exp(log_std)) ** 2
        - log_std
        - 0.5 * jnp.log(2 * jnp.pi),
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


def ppo_loss(
    params,
    apply_fn,
    obs,
    actions,
    old_log_probs,
    advantages,
    returns,
    ent_coef,
    clip_eps,
):
    mean, log_std, values = jax.vmap(lambda o: apply_fn(params, o))(obs)
    actions_pre = jnp.arctanh(jnp.clip(actions, -0.999, 0.999))
    log_probs = _gaussian_log_prob(actions_pre, mean, log_std)
    log_probs -= jnp.sum(jnp.log(1 - actions**2 + 1e-6), axis=-1)
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
            sl = jax.lax.dynamic_slice_in_dim(obs, idx * mb_size, mb_size)
            sa = jax.lax.dynamic_slice_in_dim(actions, idx * mb_size, mb_size)
            sp = jax.lax.dynamic_slice_in_dim(old_log_probs, idx * mb_size, mb_size)
            sv = jax.lax.dynamic_slice_in_dim(advantages, idx * mb_size, mb_size)
            sr = jax.lax.dynamic_slice_in_dim(returns, idx * mb_size, mb_size)
            (loss, aux), grads = grad_fn(
                ts.params, ts.apply_fn, sl, sa, sp, sv, sr, ent_coef, clip_eps
            )
            return ts.apply_gradients(grads=grads), (loss, *aux)

        return jax.lax.scan(mb_fn, ts, jnp.arange(MINIBATCHES))

    train_state, metrics = jax.lax.scan(epoch_fn, train_state, None, length=EPOCHS)
    return train_state, metrics


# ── Checkpoint I/O ───────────────────────────────────────────────────────────
def save_checkpoint(train_state, RUNS, task, steps):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"mjx_{steps:012d}.npz"
    flat, _ = jax.tree_util.tree_flatten(jax.device_get(train_state.params))
    np.savez(str(path), *flat)
    try:
        from utils.checkpoint import save as _save_st

        _save_st(train_state.params, path.with_suffix(".safetensors"))
    except Exception as e:
        print(f"[save] safetensors sidecar skipped: {e}", flush=True)
    fp = fingerprint(task)
    (RUNS / "latest.fingerprint.json").write_text(json.dumps(fp, indent=2))
    print(f"[save] {path.name}", flush=True)
    if mlflow.active_run() is not None:
        mlflow.set_tag("latest_checkpoint", path.name)
        mlflow.log_metric("checkpoint_steps", steps, step=steps)
    if os.environ.get("AUTO_PUSH") == "1":
        try:
            subprocess.run(["scripts/sync.sh"], check=False, timeout=60)
        except Exception as e:
            print(f"[sync] failed: {e}", flush=True)


def _load_npz_into_params(npz_path, init_params):
    """Load .npz weights into a flax param tree. Return (new_params, ok)."""
    data = np.load(str(npz_path))
    # data.files alphabetic: arr_0, arr_1, arr_10, arr_2, ... — sort numerically.
    keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
    flat = [jnp.asarray(data[k]) for k in keys]
    init_flat, treedef = jax.tree_util.tree_flatten(init_params)
    if any(a.shape != b.shape for a, b in zip(flat, init_flat)):
        return None, False
    return jax.tree_util.tree_unflatten(treedef, flat), True


def load_explicit_checkpoint(train_state, path):
    """Load weights from an explicit .npz path (cross-task transfer)."""
    init_params = train_state.params
    new_params, ok = _load_npz_into_params(path, init_params)
    if not ok:
        print(
            f"[resume-from] SHAPE MISMATCH on {path} — cannot transfer weights",
            flush=True,
        )
        return train_state, False
    print(f"[resume-from] loaded weights from {path}", flush=True)
    return train_state.replace(params=new_params), True


def load_latest(train_state, RUNS, task, force=False):
    """Resume from the latest checkpoint in RUNS.
    Returns (train_state, start_steps, status)."""
    ckpts = sorted(RUNS.glob("mjx_*.npz"))
    if not ckpts:
        print("[resume] no checkpoint found — starting fresh.", flush=True)
        return train_state, 0, "fresh"
    latest = ckpts[-1]
    fp_path = RUNS / "latest.fingerprint.json"
    cur = fingerprint(task)
    if fp_path.exists():
        old = json.loads(fp_path.read_text())
        if old.get("hash") != cur.get("hash"):
            print(
                f"[resume] FINGERPRINT MISMATCH — training inputs changed since {latest.name}",
                flush=True,
            )
            for f, d in cur["parts"].items():
                old_d = old.get("parts", {}).get(f, "(missing)")
                marker = "  " if old_d == d else "≠ "
                print(f"  {marker}{f}: was {old_d} now {d}", flush=True)
            if force:
                print(
                    "[resume] --resume given, loading anyway despite mismatch.",
                    flush=True,
                )
            else:
                short = old.get("hash", "unknown")[:8]
                stamp = time.strftime("%Y%m%d-%H%M%S")
                archive_dir = RUNS / "_history" / f"{stamp}_{short}"
                archive_dir.mkdir(parents=True, exist_ok=True)
                moved = 0
                for f in (
                    list(RUNS.glob("mjx_*.npz"))
                    + list(RUNS.glob("latest.fingerprint.json"))
                    + list(RUNS.glob("train.log"))
                ):
                    if f.is_file():
                        f.rename(archive_dir / f.name)
                        moved += 1
                print(f"[resume] archived {moved} files → {archive_dir}", flush=True)
                print(f"[resume] starting fresh with new training inputs.", flush=True)
                return train_state, 0, "branched"

    new_params, ok = _load_npz_into_params(latest, train_state.params)
    if not ok:
        print(
            "[resume] SHAPE MISMATCH — checkpoint weights don't fit current network",
            flush=True,
        )
        short = (
            json.loads(fp_path.read_text()).get("hash") if fp_path.exists() else "shape"
        )[:8]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive_dir = RUNS / "_history" / f"{stamp}_{short}_shape"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in list(RUNS.glob("mjx_*.npz")) + list(
            RUNS.glob("latest.fingerprint.json")
        ):
            if f.is_file():
                f.rename(archive_dir / f.name)
        print(f"[resume] archived old run → {archive_dir}, starting fresh", flush=True)
        return train_state, 0, "shape_branched"

    steps = int(latest.stem.split("_")[1])
    print(f"[resume] loaded {latest.name} ({steps:,} steps)", flush=True)
    return train_state.replace(params=new_params), steps, "resumed"


def _install_log_tee(RUNS):
    """Mirror stdout/stderr to RUNS/train.log line-buffered."""
    RUNS.mkdir(parents=True, exist_ok=True)
    log_file = open(RUNS / "train.log", "a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass

        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

        def isatty(self):
            return False

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="stand",
        help="skill to train (stand, walk, walk_forward, turn_left, turn_right). "
        "Default is `stand` — the currently-active training focus.",
    )
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="force-resume even if training inputs changed",
    )
    parser.add_argument(
        "--no-resume",
        dest="auto_resume",
        action="store_false",
        help="ignore existing checkpoints, start fresh",
    )
    parser.add_argument("--no-watch", dest="watch", action="store_false")
    parser.set_defaults(auto_resume=True, watch=True)
    parser.add_argument("--save-rollouts-every", type=int, default=0)
    parser.add_argument("--rollout-envs", type=int, default=256)
    parser.add_argument("--rollout-keep", type=int, default=50)
    parser.add_argument("--label", default="")
    parser.add_argument("--auto-push", action="store_true")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="explicit .npz path for cross-task transfer (overrides --resume)",
    )
    parser.add_argument(
        "--early-stop-metric",
        default=None,
        help="metric key to monitor (e.g. mean_upright)",
    )
    parser.add_argument("--early-stop-op", choices=["gt", "lt"], default="gt")
    parser.add_argument("--early-stop-value", type=float, default=None)
    parser.add_argument(
        "--early-stop-window",
        type=int,
        default=20,
        help="how many consecutive metric blocks satisfying the criterion",
    )
    parser.add_argument("--mlflow-parent-run-id", default=None)
    parser.add_argument(
        "--brain-persist",
        default=None,
        help="If set, write ApeBrain episodic/semantic stores under this directory.",
    )
    parser.add_argument(
        "--brain-samples-per-rollout",
        type=int,
        default=32,
        help="Transitions to sample from each rollout into the episodic store.",
    )
    parser.add_argument(
        "--brain-sleep-every",
        type=int,
        default=20,
        help="Run consolidation replay every N updates (0 disables).",
    )
    parser.add_argument(
        "--brain-sleep-replays",
        type=int,
        default=64,
        help="Episodes to replay through BG per sleep tick.",
    )
    args = parser.parse_args()

    TASK = args.task
    RUNS = task_runs_dir(TASK)
    RUNS.mkdir(parents=True, exist_ok=True)
    _install_log_tee(RUNS)

    # Dynamic imports
    env_mod_path, params_mod_path = task_module_paths(TASK)
    try:
        env_mod = importlib.import_module(env_mod_path)
        P = importlib.import_module(params_mod_path)
    except ImportError as e:
        print(f"[task] '{TASK}' modules not found: {e}", flush=True)
        print(f"[task] expected: {env_mod_path}, {params_mod_path}", flush=True)
        sys.exit(1)

    OBS_DIM = env_mod.OBS_DIM
    ACT_DIM = env_mod.ACT_DIM
    MAX_STEPS = env_mod.MAX_STEPS

    if args.auto_push:
        os.environ["AUTO_PUSH"] = "1"

    N_ENVS = args.envs or _auto_envs()
    save_every = max(1_000, min(100_000, N_ENVS * 50))

    print(f"JAX devices: {jax.devices()}")
    print(f"Task: {TASK}  |  {N_ENVS} envs  |  save every {save_every:,} steps")
    print(f"Loading model...")

    # Each env's load_model returns 7 items. For tasks without a spatial target,
    # target_xy is jnp.zeros(2).
    mx, jnt_lo, jnt_hi, motor_max_speed, motor_max_accel, rest_pose, target_xy = (
        env_mod.load_model()
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    train_state = make_train_state(init_rng, float(P.LEARNING_RATE), OBS_DIM, ACT_DIM)
    start_steps = 0
    resume_status = "fresh"

    # Cross-task transfer takes precedence over auto-resume.
    if args.resume_from is not None:
        train_state, ok = load_explicit_checkpoint(train_state, args.resume_from)
        resume_status = "resume_from" if ok else "resume_from_failed"
    elif args.auto_resume:
        train_state, start_steps, resume_status = load_latest(
            train_state, RUNS, TASK, force=args.resume
        )

    rparams = tuple(jnp.float32(v) for v in env_mod.read_params())

    v_reset = jax.jit(
        jax.vmap(lambda rng: env_mod.reset(mx, jnt_lo, jnt_hi, target_xy, rng))
    )
    v_step = jax.jit(
        jax.vmap(
            lambda dx, a, mp, mv, pa, pd, si: env_mod.step(
                mx,
                jnt_lo,
                jnt_hi,
                motor_max_speed,
                motor_max_accel,
                rest_pose,
                target_xy,
                dx,
                a,
                mp,
                mv,
                pa,
                pd,
                si,
                rparams,
            )
        )
    )

    env_rngs = jax.random.split(env_rng, N_ENVS)
    (
        dx_batch,
        obs_batch,
        motor_pos_batch,
        motor_vel_batch,
        prev_action_batch,
        prev_dist_batch,
        step_idx_batch,
    ) = v_reset(env_rngs)
    step_counts = jnp.zeros(N_ENVS, dtype=jnp.int32)

    inherited_from = (
        str(sorted(RUNS.glob("mjx_*.npz"))[-1])
        if resume_status == "resumed" and list(RUNS.glob("mjx_*.npz"))
        else args.resume_from
        if args.resume_from
        else None
    )
    lineage_mod.record(
        RUNS,
        TASK,
        N_ENVS,
        inherited_from,
        pathlib.Path("data/scene.xml"),
        pathlib.Path(f"envs/{TASK}_env_mjx.py"),
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = f"ape_{TASK}_mjx"
    mlflow.set_experiment(experiment_name)
    parent_run_id = args.mlflow_parent_run_id
    mlflow.start_run(
        run_name=time.strftime("%Y%m%d-%H%M%S"),
        nested=parent_run_id is not None,
        parent_run_id=parent_run_id,
    )
    p_params = {
        k: getattr(P, k) for k in dir(P) if k.isupper() and not k.startswith("_")
    }
    mlflow.log_params(
        {
            **p_params,
            "TASK": TASK,
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
        }
    )
    mlflow.set_tags(
        {
            "task": TASK,
            "inherited_from": inherited_from or "scratch",
            "resume": resume_status,
            "jax_backend": jax.default_backend(),
            "start_steps": str(start_steps),
            "label": args.label or "(unlabeled)",
        }
    )
    if args.early_stop_metric is not None:
        mlflow.set_tag(
            "early_stop",
            f"{args.early_stop_metric} {args.early_stop_op} {args.early_stop_value} for {args.early_stop_window}",
        )

    total_steps = start_steps
    update_count = 0
    last_save = start_steps
    t0 = time.time()
    early_stop_hits = deque(maxlen=args.early_stop_window)

    watcher = None
    if args.watch:
        watch_cmd = [
            sys.executable,
            "view.py",
            "--env",
            TASK,
            "--since",
            str(time.time()),
        ]
        print(f"[watch] spawning viewer: {' '.join(watch_cmd)}")
        try:
            watcher = subprocess.Popen(watch_cmd)
        except Exception as e:
            print(f"[watch] failed to spawn: {e}", flush=True)

    print(f"Training — {TOTAL_STEPS:,} total steps.")
    print(f"{'updates':>8}  {'steps':>12}  {'sps':>8}  {'mean_rew':>10}  metrics")
    print("-" * 60)

    _stop = [False]
    _stop_reason = ["completed"]

    def _on_sigint(signum, frame):
        if _stop[0]:
            print("\n[stop] force exit")
            sys.exit(130)
        _stop[0] = True
        _stop_reason[0] = "sigint"
        print("\n[stop] Ctrl+C — finishing rollout then saving...")

    signal.signal(signal.SIGINT, _on_sigint)

    def _on_sigterm(signum, frame):
        # SIGTERM = process manager asked us to stop (train_bg.sh stop,
        # systemd, kill <pid>). Close MLflow run cleanly so it doesn't get
        # stuck in RUNNING state forever, then exit. We don't try to save a
        # checkpoint here — SIGTERM may be followed by SIGKILL on a short
        # grace period, and an unfinished save would corrupt the npz.
        try:
            if mlflow.active_run() is not None:
                mlflow.set_tag("stop_reason", "sigterm")
                mlflow.set_tag("final_steps", str(total_steps))
                mlflow.end_run(status="KILLED")
        except Exception as e:
            print(f"[sigterm] mlflow.end_run failed: {e}", flush=True)
        print("[sigterm] received — exiting cleanly", flush=True)
        sys.exit(143)

    signal.signal(signal.SIGTERM, _on_sigterm)

    # Final safety net: if the process dies any other way (uncaught exception
    # propagating, sys.exit, hardware fault that gives Python a chance to
    # unwind), close the MLflow run with status FAILED if it's still open.
    import atexit

    def _atexit_close_mlflow():
        try:
            if mlflow.active_run() is not None:
                mlflow.set_tag("stop_reason", _stop_reason[0] or "atexit_unknown")
                mlflow.end_run(status="FAILED")
        except Exception:
            pass

    atexit.register(_atexit_close_mlflow)

    ent_coef = jnp.float32(P.ENT_COEF)
    clip_eps = jnp.float32(P.CLIP_EPS)

    brain_hook = None
    if args.brain_persist is not None:
        from utils.brain_hook import BrainHook

        brain_hook = BrainHook(
            persist_dir=args.brain_persist,
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            samples_per_rollout=args.brain_samples_per_rollout,
            sleep_every_updates=args.brain_sleep_every,
            sleep_replays=args.brain_sleep_replays,
        )
        mlflow.set_tag("brain_persist", args.brain_persist)
        print(
            f"[brain] persisting episodic+semantic memory to {args.brain_persist}",
            flush=True,
        )

    while total_steps < TOTAL_STEPS and not _stop[0]:
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val, buf_dist = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for t in range(N_STEPS):
            steps_here = total_steps + t * N_ENVS
            if _stop[0]:
                save_checkpoint(train_state, RUNS, TASK, steps_here)
                print("Done.", flush=True)
                sys.exit(0)

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

            (
                dx_batch,
                obs_batch,
                rewards,
                dones,
                motor_pos_batch,
                motor_vel_batch,
                prev_action_batch,
                prev_dist_batch,
                step_idx_batch,
            ) = v_step(
                dx_batch,
                actions,
                motor_pos_batch,
                motor_vel_batch,
                prev_action_batch,
                prev_dist_batch,
                step_idx_batch,
            )
            step_counts = step_counts + 1
            done_mask = dones | (step_counts >= MAX_STEPS)

            if jnp.any(done_mask):
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, N_ENVS)
                new_dx, new_obs, new_mp, new_mv, new_pa, new_pd, new_si = v_reset(
                    reset_rngs
                )
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
                motor_pos_batch = jnp.where(done_mask[:, None], new_mp, motor_pos_batch)
                motor_vel_batch = jnp.where(done_mask[:, None], new_mv, motor_vel_batch)
                prev_action_batch = jnp.where(
                    done_mask[:, None], new_pa, prev_action_batch
                )
                prev_dist_batch = jnp.where(done_mask, new_pd, prev_dist_batch)
                step_idx_batch = jnp.where(done_mask, new_si, step_idx_batch)
                step_counts = jnp.where(done_mask, 0, step_counts)

            buf_rew.append(rewards)
            buf_done.append(done_mask.astype(jnp.float32))
            buf_dist.append(prev_dist_batch)

        total_steps += N_ENVS * N_STEPS

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
            in_axes=(1, 1, 1, 0),
            out_axes=1,
        )(rew_t, val_t, done_t, last_val)

        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        rollout = (flat(obs_t), flat(act_t), flat(logp_t), flat(adv_t), flat(ret_t))
        train_state, _ = update_step(train_state, rollout, ent_coef, clip_eps)
        update_count += 1

        if brain_hook is not None:
            # Build next_obs for each (t, env): obs_t shifted by 1, last step
            # uses the post-rollout obs_batch.
            next_obs_t = jnp.concatenate([obs_t[1:], obs_batch[None, ...]], axis=0)
            brain_hook.record_rollout(obs_t, act_t, rew_t, next_obs_t)
            slept = brain_hook.maybe_sleep(update_count)
            if slept:
                print(
                    f"[brain] consolidation: replayed {brain_hook._last_sleep_replays} episodes",
                    flush=True,
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
            success_frac = float(
                jnp.mean(stacked_dist < 0.6)
            )  # walk-specific but harmless for others
            # obs layout (frozen contract, see envs/_shared_obs.py):
            torso_z = stacked_obs[..., 0]
            uprightness = stacked_obs[..., 1]  # quat_w
            linvel_xyz = stacked_obs[..., 5:8]
            angvel_xyz = stacked_obs[..., 8:11]
            speed_xy = jnp.linalg.norm(stacked_obs[..., 5:7], axis=-1)
            jvel_norm = jnp.linalg.norm(stacked_obs[..., 35:59], axis=-1)
            torque = stacked_obs[..., 59:83]
            tFL, tFR = jnp.abs(torque[..., 0:6]), jnp.abs(torque[..., 6:12])
            tBL, tBR = jnp.abs(torque[..., 12:18]), jnp.abs(torque[..., 18:24])
            asym = float(
                jnp.mean(
                    jnp.sum((tFL - tFR) ** 2, axis=-1)
                    + jnp.sum((tBL - tBR) ** 2, axis=-1)
                )
            )
            mean_speed = float(jnp.mean(speed_xy))
            max_speed = float(jnp.max(speed_xy))
            mean_speed_x = float(jnp.mean(linvel_xyz[..., 0]))
            mean_angvel_z = float(jnp.mean(angvel_xyz[..., 2]))
            mean_z = float(jnp.mean(torso_z))
            mean_upright = float(jnp.mean(uprightness))
            mean_jvel = float(jnp.mean(jvel_norm))
            jvel_avg = np.asarray(jnp.mean(stacked_obs[..., 35:59], axis=1))
            gait_coh = float(trot_coherence(jvel_avg))
            stride_s = float(stride_period(jvel_avg, dt_action=16 * 0.002))

            metrics = {
                "sps": sps,
                "mean_rew": mean_rew,
                "mean_dist": mean_dist,
                "min_dist": min_dist,
                "max_dist": max_dist,
                "best_rew": best_rew,
                "done_frac": done_frac,
                "success_frac": success_frac,
                "update": update_count,
                "mean_speed": mean_speed,
                "max_speed": max_speed,
                "mean_speed_x": mean_speed_x,
                "mean_angvel_z": mean_angvel_z,
                "mean_z": mean_z,
                "mean_upright": mean_upright,
                "mean_jvel": mean_jvel,
                "asym_torque": asym,
                "gait_coherence": gait_coh,
                "stride_period_s": stride_s,
            }

            print(
                f"{update_count:>8}  {total_steps:>12,}  {sps:>8,.0f}  rew={mean_rew:+.2f}  d={mean_dist:.2f}  vx={mean_speed_x:+.2f}  z={mean_z:.2f}  upr={mean_upright:.2f}  ωz={mean_angvel_z:+.2f}",
                flush=True,
            )
            if brain_hook is not None:
                metrics.update(brain_hook.metrics())
            mlflow.log_metrics(metrics, step=total_steps)

            # Per-task keeper criterion (env may override).
            is_keeper_fn = getattr(env_mod, "is_keeper", None)
            if is_keeper_fn is not None:
                promote, tag_extra = is_keeper_fn(metrics)
            else:
                # Fallback: walk-style (min_dist or success_frac).
                promote = min_dist < 1.5 or success_frac > 0.05
                tag_extra = f"min_d{min_dist:.2f}_succ{success_frac:.2f}"

            if promote:
                kdir = RUNS / "keepers"
                kdir.mkdir(exist_ok=True)
                tag = f"{tag_extra}_step{total_steps}"
                kpath = kdir / f"auto_{tag}.npz"
                if not kpath.exists():
                    flat, _ = jax.tree_util.tree_flatten(
                        jax.device_get(train_state.params)
                    )
                    np.savez(str(kpath), *flat)
                    try:
                        from utils.checkpoint import save as _save_st

                        _save_st(train_state.params, kpath.with_suffix(".safetensors"))
                    except Exception as e:
                        print(f"[keeper] safetensors sidecar skipped: {e}", flush=True)
                    sidecar = {
                        "tag": tag,
                        "step": int(total_steps),
                        "task": TASK,
                        "metrics": metrics,
                        "params": p_params,
                        "fingerprint": fingerprint(TASK),
                        "obs_dim": OBS_DIM,
                        "act_dim": ACT_DIM,
                        "n_envs": N_ENVS,
                        "n_steps": N_STEPS,
                    }
                    (kdir / f"auto_{tag}.json").write_text(
                        json.dumps(sidecar, indent=2)
                    )
                    print(f"[keeper] saved {kpath.name} + sidecar", flush=True)

            # Early stop: if a sustained criterion is met, signal completion.
            if args.early_stop_metric is not None and args.early_stop_value is not None:
                m = metrics.get(args.early_stop_metric)
                if m is not None:
                    if args.early_stop_op == "gt":
                        ok = m > args.early_stop_value
                    else:
                        ok = m < args.early_stop_value
                    early_stop_hits.append(ok)
                    if len(early_stop_hits) == args.early_stop_window and all(
                        early_stop_hits
                    ):
                        print(
                            f"[early-stop] '{args.early_stop_metric}' {args.early_stop_op} {args.early_stop_value} for {args.early_stop_window} consecutive blocks — stopping cleanly",
                            flush=True,
                        )
                        _stop[0] = True
                        _stop_reason[0] = "early_stop_success"

        if total_steps - last_save >= save_every:
            save_checkpoint(train_state, RUNS, TASK, total_steps)
            last_save = total_steps

    reason = _stop_reason[0] if _stop[0] else "completed"
    print(f"[stop] {reason} — saving at {total_steps:,} steps")
    save_checkpoint(train_state, RUNS, TASK, total_steps)
    mlflow.set_tag("stop_reason", reason)
    mlflow.set_tag("final_steps", str(total_steps))
    mlflow.end_run()
    if brain_hook is not None:
        brain_hook.close()
    if watcher is not None:
        watcher.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
