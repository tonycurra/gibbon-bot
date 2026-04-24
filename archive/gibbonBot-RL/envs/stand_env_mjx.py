"""
MJX version of StandEnv — pure JAX, runs thousands of envs in parallel on GPU.

The environment is expressed as stateless functions (reset, step, get_obs)
that operate on an EnvState pytree. jax.vmap over N_ENVS gives free parallelism.
"""

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import pathlib

SCENE      = pathlib.Path(__file__).parent.parent / "scene.xml"
TARGET_Z   = 0.40
MIN_HEIGHT = 0.03
SPAWN_CLEARANCE = 0.07
MAX_STEPS  = 2000
PHYSICS_STEPS_PER_ACTION = 12

# arms-splayed init pose (24 joints)
_STAND_INIT = np.array([
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
     0.0, -0.4,  1.2, 0.0, 0.0, 0.0,
], dtype=np.float32)


def load_model():
    """Load MuJoCo model and put onto GPU via MJX."""
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    mj_model.opt.timestep = 0.002

    # Disable collision on STL mesh geoms — they're visual only.
    # MJX compiles a kernel per geometry pair; mesh geoms cause combinatorial explosion.
    # Collision is handled by the torso box + gripper approximate geoms.
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0

    mx = mjx.put_model(mj_model)
    jnt_lo = jnp.array([mj_model.jnt_range[i][0] for i in range(1, mj_model.njnt)])
    jnt_hi = jnp.array([mj_model.jnt_range[i][1] for i in range(1, mj_model.njnt)])
    return mx, jnt_lo, jnt_hi


def reset(mx, jnt_lo, jnt_hi, rng):
    """Reset one environment. Returns (data, obs, prev_action)."""
    rng, k1 = jax.random.split(rng)
    dx = mjx.make_data(mx)

    # set torso pose
    qpos = dx.qpos.at[0].set(0.0)
    qpos = qpos.at[1].set(0.0)
    qpos = qpos.at[2].set(SPAWN_CLEARANCE)
    qpos = qpos.at[3].set(1.0)   # quat w
    qpos = qpos.at[4:7].set(0.0)

    noise = jax.random.uniform(k1, (24,), minval=-0.1, maxval=0.1)
    qpos = qpos.at[7:].set(jnp.array(_STAND_INIT) + noise)

    dx = dx.replace(qpos=qpos)
    dx = mjx.forward(mx, dx)

    obs = _get_obs(dx)
    prev_action = jnp.zeros(24)
    return dx, obs, prev_action


def step(mx, jnt_lo, jnt_hi, dx, action, prev_action):
    """Step one environment. Returns (dx, obs, reward, done, prev_action)."""
    ctrl = 0.5 * (action + 1.0) * (jnt_hi - jnt_lo) + jnt_lo

    def physics_step(dx, _):
        return dx.replace(ctrl=ctrl), None

    dx = dx.replace(ctrl=ctrl)
    dx, _ = jax.lax.scan(
        lambda d, _: (mjx.step(mx, d), None),
        dx,
        None,
        length=PHYSICS_STEPS_PER_ACTION,
    )

    obs          = _get_obs(dx)
    torso_z      = dx.qpos[2]
    fallen       = torso_z < MIN_HEIGHT

    height_err    = (torso_z - TARGET_Z) / 0.08
    height_reward = jnp.exp(-0.5 * height_err ** 2)

    w              = dx.qpos[3]
    upright_reward = 0.3 * w * w

    jvel         = dx.qvel[6:]
    action_delta = jnp.sum((action - prev_action) ** 2)

    reward = (
          height_reward
        + upright_reward
        + 0.05
        - 5e-4 * jnp.sum(ctrl ** 2)
        - 1e-3 * jnp.sum(jvel ** 2)
        - 2e-2 * action_delta
        - 10.0 * fallen
    )

    return dx, obs, reward, fallen, action.copy()


def _get_obs(dx):
    return jnp.concatenate([
        dx.qpos[2:3],    # torso z
        dx.qpos[3:7],    # quat
        dx.qvel[0:3],    # linear vel
        dx.qvel[3:6],    # angular vel
        dx.qpos[7:],     # 24 joint pos
        dx.qvel[6:],     # 24 joint vel
    ]).astype(jnp.float32)


OBS_DIM = 59
ACT_DIM = 24
