"""
Live viewer for MJX training — loads .npz Flax checkpoints and runs them
on CPU MuJoCo with the same step semantics as MJX (direct ctrl, 16 physics steps).

Hot-swaps policy when a newer checkpoint appears.

Usage:
    python utils/watch_mjx.py
"""

import argparse
import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import time
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import mujoco
import mujoco.viewer

from envs.stand_env_mjx import OBS_DIM, ACT_DIM, MAX_STEPS, PHYSICS_STEPS_PER_ACTION, SCENE
import envs.params_mjx as params_mod
import importlib

RUNS = pathlib.Path("runs/ape_stand_mjx")
PARAMS_FILE = pathlib.Path(__file__).parent.parent / "envs" / "params_mjx.py"
LIVE_STATS = RUNS / "live_stats.json"

_parser = argparse.ArgumentParser()
_parser.add_argument("--since", type=float, default=0.0,
                     help="ignore checkpoints saved before this unix timestamp")
_args = _parser.parse_args()
SINCE = _args.since


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


def load_policy(path):
    net = ActorCritic()
    dummy = jnp.zeros((1, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), dummy)
    data = np.load(str(path))
    flat = [jnp.asarray(data[k]) for k in data.files]
    _, treedef = jax.tree_util.tree_flatten(params)
    loaded = jax.tree_util.tree_unflatten(treedef, flat)
    # Verify input layer shape matches current OBS_DIM
    # flat is alphabetical (bias before kernel), so find first 2-D tensor
    first_kernel = next((x for x in flat if x.ndim == 2), None)
    if first_kernel is None or first_kernel.shape[0] != OBS_DIM:
        got = first_kernel.shape[0] if first_kernel is not None else "?"
        raise ValueError(
            f"Checkpoint obs_dim={got} != current OBS_DIM={OBS_DIM}. "
            f"Delete old checkpoints and restart training."
        )
    return net, loaded


def predict(net, params, obs):
    obs_j = jnp.array(obs, dtype=jnp.float32)
    mean, _, _ = net.apply(params, obs_j)
    return np.array(jnp.tanh(mean))


def latest_checkpoint():
    ckpts = [c for c in sorted(RUNS.glob("mjx_*.npz")) if c.stat().st_mtime >= SINCE]
    return ckpts[-1] if ckpts else None


def get_obs(data):
    return np.concatenate([
        data.qpos[2:3],
        data.qpos[3:7],
        data.qvel[0:3],
        data.qvel[3:6],
        data.qpos[7:],
        data.qvel[6:],
    ]).astype(np.float32)


def write_live_stats(stats):
    """Atomically write the live stats JSON for the params UI to poll."""
    try:
        LIVE_STATS.parent.mkdir(parents=True, exist_ok=True)
        tmp = LIVE_STATS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stats))
        os.replace(tmp, LIVE_STATS)
    except Exception:
        pass


def reset(model, data, rng):
    mujoco.mj_resetData(model, data)
    yaw = rng.uniform(-np.pi, np.pi)
    data.qpos[0] = rng.uniform(-0.1, 0.1)
    data.qpos[1] = rng.uniform(-0.1, 0.1)
    data.qpos[2] = 0.45
    data.qpos[3] = np.cos(yaw / 2)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = np.sin(yaw / 2)
    data.qpos[7:] = rng.uniform(-0.15, 0.15, size=24)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)


def main():
    print(f"[watch_mjx] runs dir : {RUNS}")
    print(f"[watch_mjx] since    : {SINCE:.0f} (0 = load any checkpoint)")
    print(f"[watch_mjx] physics steps/action: {PHYSICS_STEPS_PER_ACTION}")
    print(f"[watch_mjx] params   : envs/params_mjx.py  (run it directly for slider UI)")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    jnt_lo = np.array([model.jnt_range[i][0] for i in range(1, model.njnt)])
    jnt_hi = np.array([model.jnt_range[i][1] for i in range(1, model.njnt)])

    viewer = mujoco.viewer.launch_passive(model, data)
    rng = np.random.default_rng(0)

    ckpt = latest_checkpoint()
    if ckpt is None:
        print(f"[watch_mjx] no checkpoint yet — random policy, waiting for training to save one...")
        net, params = None, None
        last_mtime = 0.0
    else:
        print(f"[watch_mjx] loading {ckpt.name} ...")
        net, params = load_policy(ckpt)
        last_mtime = ckpt.stat().st_mtime

    params_mtime = 0.0

    # Rolling per-episode stats for the UI panel
    recent_eps = deque(maxlen=20)  # each entry: {"flipped": bool, "ep_len": int}
    recent_scores = deque(maxlen=20)
    best_score = -float("inf")

    ep = 0
    try:
        while viewer.is_running():
            new_ckpt = latest_checkpoint()
            if new_ckpt is not None:
                mtime = new_ckpt.stat().st_mtime
                if mtime != last_mtime:
                    try:
                        net, params = load_policy(new_ckpt)
                        print(f"[watch_mjx] {'first' if net is None else 'new'} checkpoint: {new_ckpt.name}")
                        last_mtime = mtime
                    except ValueError as e:
                        print(f"[watch_mjx] skipping {new_ckpt.name}: {e}")
                        last_mtime = mtime  # don't retry this checkpoint

            reset(model, data, rng)
            motor_pos     = np.zeros(ACT_DIM, dtype=np.float32)
            prev_action   = np.zeros(ACT_DIM, dtype=np.float32)
            prev_raw_act  = np.zeros(ACT_DIM, dtype=np.float32)
            step = 0
            flipped = False
            ep_reward_sum = 0.0

            cur_speed = max(getattr(params_mod, "VIEWER_SPEED", 1.0), 0.01)
            ep_real_s = MAX_STEPS * PHYSICS_STEPS_PER_ACTION * float(model.opt.timestep) / cur_speed
            policy_label = "random" if net is None else (new_ckpt.name if new_ckpt else "?")
            print(f"[watch_mjx] ep {ep+1} start | policy={policy_label} | "
                  f"speed={cur_speed:.0f}x | ep≈{ep_real_s:.1f}s real-time"
                  + (" ← slow down in params.py to see motion" if cur_speed > 10 else ""))

            while step < MAX_STEPS and viewer.is_running():
                # hot-reload params
                try:
                    pm = PARAMS_FILE.stat().st_mtime
                    if pm != params_mtime:
                        importlib.reload(params_mod)
                        params_mtime = pm
                except Exception:
                    pass
                speed = max(getattr(params_mod, "VIEWER_SPEED", 1.0), 0.01)
                sleep_dt = float(model.opt.timestep) / speed

                obs = get_obs(data)
                if net is None:
                    action = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
                else:
                    action = predict(net, params, obs)

                # Apply same action low-pass filter as training env
                alpha = float(getattr(params_mod, "ACTION_SMOOTH_ALPHA", 1.0))
                filt_action = (alpha * action + (1.0 - alpha) * prev_action).astype(np.float32)

                # Per-action live stats (mostly pre-physics)
                smooth_delta  = float(np.mean(np.abs(filt_action - prev_action)))
                joints_moving = int(np.sum(np.abs(action - prev_raw_act) > 0.05))

                prev_action  = filt_action
                prev_raw_act = action.astype(np.float32).copy()

                motor_pos_before = motor_pos.copy()
                motor_max_speed  = float(getattr(params_mod, "MOTOR_MAX_SPEED", 0.2))
                substep_speed    = motor_max_speed / PHYSICS_STEPS_PER_ACTION
                for _ in range(PHYSICS_STEPS_PER_ACTION):
                    delta = np.clip(filt_action - motor_pos, -substep_speed, substep_speed)
                    motor_pos += delta
                    ctrl = 0.5 * (motor_pos + 1.0) * (jnt_hi - jnt_lo) + jnt_lo
                    data.ctrl[:] = ctrl
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)

                # Actual joint travel this action (bounded by MOTOR_MAX_SPEED)
                max_action_delta = float(np.max(np.abs(motor_pos - motor_pos_before)))

                # Post-physics measurements
                jvel    = data.qvel[6:]
                jvel_sq = float(np.mean(jvel ** 2))
                ctrl_sq = float(np.mean(data.ctrl ** 2))
                torso_z = float(data.qpos[2])
                upright = float(data.qpos[3])

                # Per-step reward estimate (matches the dominant terms of the
                # training reward — height + upright + alive — for trend tracking).
                tgt_z   = float(getattr(params_mod, "TARGET_Z", 0.3))
                tgt_tol = float(getattr(params_mod, "TARGET_TOL", 0.1))
                w_h     = float(getattr(params_mod, "W_HEIGHT", 10.0))
                w_u     = float(getattr(params_mod, "W_UPRIGHT", 5.0))
                w_a     = float(getattr(params_mod, "W_ALIVE", 0.05))
                h_err   = (torso_z - tgt_z) / max(tgt_tol, 1e-6)
                step_r  = (
                    w_h * float(np.exp(-0.5 * h_err * h_err))
                    + (w_u * upright * upright if upright > 0 else 0.0)
                    + (w_a if upright > 0 else 0.0)
                )
                ep_reward_sum += step_r

                flip_rate = (
                    sum(1 for e in recent_eps if e["flipped"]) / len(recent_eps)
                    if recent_eps else 0.0
                )
                avg_ep_len = (
                    sum(e["ep_len"] for e in recent_eps) / len(recent_eps)
                    if recent_eps else 0.0
                )

                write_live_stats({
                    "ts": time.time(),
                    "ckpt": (new_ckpt.name if new_ckpt else "(random)"),
                    "torso_z": torso_z,
                    "upright": upright,
                    "max_action_delta": max_action_delta,
                    "smooth_delta": smooth_delta,
                    "jvel_sq": jvel_sq,
                    "ctrl_sq": ctrl_sq,
                    "joints_moving": joints_moving,
                    "ep_len": step,
                    "flip_rate_20": flip_rate,
                    "avg_ep_len_20": avg_ep_len,
                    "n_eps": len(recent_eps),
                })

                if data.qpos[3] < 0:
                    flipped = True
                    break
                step += 1

            recent_eps.append({"flipped": flipped, "ep_len": step})

            if not viewer.is_running():
                break

            ep += 1
            ckpt_name = new_ckpt.name if new_ckpt else "?"
            status = "FLIPPED" if flipped else ("TIMEOUT" if step >= MAX_STEPS else "DONE")
            torso_z = float(data.qpos[2])

            ep_score = ep_reward_sum / max(step, 1)
            prev_mean = (sum(recent_scores) / len(recent_scores)) if recent_scores else ep_score
            arrow = "↑" if ep_score > prev_mean else ("↓" if ep_score < prev_mean else "·")
            new_best = ep_score > best_score
            best_score = max(best_score, ep_score)
            recent_scores.append(ep_score)

            tag = "  ← new best!" if new_best and ep > 1 else ""
            print(f"Ep {ep:3d} {status:<7} torso={torso_z:.3f}m  ep={step:>4}  "
                  f"reward≈{ep_score:.3f} {arrow}  (best {best_score:.3f}){tag}  [{ckpt_name}]")

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("\n[watch_mjx] closed.")


if __name__ == "__main__":
    main()
