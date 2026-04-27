"""
MJX environment — same reward/obs as stand_env.py, runs on GPU via JAX vmap.

Stateless functions (reset, step) operate on MJX data pytrees.
Call load_model() once, then vmap reset/step over N_ENVS.

Reward params are passed as a plain tuple of JAX scalars so they can be
updated between rollouts without recompiling the JIT.
"""

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from envs.common import SCENE, MAX_STEPS, ACT_DIM, ARM_GROUPS_NP, PAIR_A, PAIR_B

PHYSICS_STEPS_PER_ACTION = 16  # 16 × 0.002s = 0.032s per action
OBS_DIM = 59  # torso z(1) + quat(4) + lin_vel(3) + ang_vel(3) + jpos(24) + jvel(24)


def load_model():
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    mj_model.opt.timestep = 0.002

    # Mesh geoms cause combinatorial JIT explosion in MJX — disable their collision.
    # Box/capsule approximate geoms handle all physics contact.
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0

    mx = mjx.put_model(mj_model)
    jnt_lo = jnp.array([mj_model.jnt_range[i][0] for i in range(1, mj_model.njnt)])
    jnt_hi = jnp.array([mj_model.jnt_range[i][1] for i in range(1, mj_model.njnt)])
    return mx, jnt_lo, jnt_hi


def reset(mx, jnt_lo, jnt_hi, rng):
    rng, k1, k2 = jax.random.split(rng, 3)
    dx = mjx.make_data(mx)

    xy = jax.random.uniform(k1, (2,), minval=-0.1, maxval=0.1)
    yaw = jax.random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)

    qpos = dx.qpos.at[0].set(xy[0])
    qpos = qpos.at[1].set(xy[1])
    qpos = qpos.at[2].set(0.45)
    qpos = qpos.at[3].set(jnp.cos(yaw / 2))  # w
    qpos = qpos.at[4].set(0.0)
    qpos = qpos.at[5].set(0.0)
    qpos = qpos.at[6].set(jnp.sin(yaw / 2))

    rng, knoise = jax.random.split(rng)
    noise = jax.random.uniform(knoise, (24,), minval=-0.15, maxval=0.15)
    qpos = qpos.at[7:].set(noise)

    dx = dx.replace(qpos=qpos, qvel=jnp.zeros_like(dx.qvel), ctrl=jnp.zeros_like(dx.ctrl))
    dx = mjx.forward(mx, dx)

    obs = _get_obs(dx)
    prev_action = jnp.zeros(ACT_DIM)
    return dx, obs, prev_action


def step(mx, jnt_lo, jnt_hi, dx, action, prev_action, rparams):
    """
    rparams: (target_z, target_tol, w_height, w_upright,
               arm_safety_margin, w_proximity, flip_penalty)
    All are JAX scalars — change them between rollouts without recompiling.
    """
    target_z, target_tol, w_height, w_upright, arm_safety_margin, w_proximity, flip_penalty = rparams

    ctrl = 0.5 * (action + 1.0) * (jnt_hi - jnt_lo) + jnt_lo

    dx = dx.replace(ctrl=ctrl)
    dx, _ = jax.lax.scan(
        lambda d, _: (mjx.step(mx, d), None),
        dx, None, length=PHYSICS_STEPS_PER_ACTION,
    )

    obs = _get_obs(dx)
    torso_z = dx.qpos[2]
    w = dx.qpos[3]

    height_err = (torso_z - target_z) / target_tol
    height_reward = w_height * jnp.exp(-0.5 * height_err ** 2)
    upright_reward = jnp.where(w > 0, w_upright * w * w, 0.0)
    prox_penalty = w_proximity * _proximity_penalty(dx.xpos, arm_safety_margin)

    upside_down = w < 0
    reward = height_reward + upright_reward - prox_penalty - flip_penalty * upside_down
    done = upside_down

    return dx, obs, reward, done, action


def _proximity_penalty(xpos, arm_safety_margin):
    # xpos shape: (nbody, 3)
    groups = jnp.array(ARM_GROUPS_NP)  # (4, 6)
    pair_a = jnp.array(PAIR_A)         # (6,)
    pair_b = jnp.array(PAIR_B)         # (6,)

    def pair_penalty(a_idx, b_idx):
        pos_a = xpos[groups[a_idx]]   # (6, 3)
        pos_b = xpos[groups[b_idx]]   # (6, 3)
        # all pairwise distances between the two groups
        diffs = pos_a[:, None, :] - pos_b[None, :, :]   # (6, 6, 3)
        dists = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))  # (6, 6)
        violations = jnp.maximum(arm_safety_margin - dists, 0.0)
        return jnp.sum(violations ** 2)

    penalties = jax.vmap(pair_penalty)(pair_a, pair_b)
    return jnp.sum(penalties)


def _get_obs(dx):
    return jnp.concatenate([
        dx.qpos[2:3],   # torso z
        dx.qpos[3:7],   # quat (w x y z)
        dx.qvel[0:3],   # linear vel
        dx.qvel[3:6],   # angular vel
        dx.qpos[7:],    # 24 joint positions
        dx.qvel[6:],    # 24 joint velocities
    ]).astype(jnp.float32)
