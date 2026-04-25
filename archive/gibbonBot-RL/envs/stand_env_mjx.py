"""
MJX version of StandEnv — pure JAX, runs thousands of envs in parallel on GPU.

The environment is expressed as stateless functions (reset, step, get_obs)
that operate on an EnvState pytree. jax.vmap over N_ENVS gives free parallelism.
"""

import pathlib

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

SCENE = pathlib.Path(__file__).parent.parent / "scene.xml"
TARGET_Z = 0.40
MIN_HEIGHT = 0.03
SPAWN_CLEARANCE = 0.07
MAX_STEPS = 2000
PHYSICS_STEPS_PER_ACTION = 12

# Curriculum learning parameters
CURRICULUM_STEPS = 10_000_000  # Steps over which to increase difficulty
TARGET_Z_MIN = 0.25  # Starting target height
TARGET_Z_MAX = 0.40  # Final target height

# arms-splayed init pose (24 joints)
_STAND_INIT = np.array(
    [
        0.0,
        -0.4,
        1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
        1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
        1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
        1.2,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def load_model():
    """Load MuJoCo model and put onto GPU via MJX.

    Also reads envs/hints.py once at startup and baked its weights + gripper
    body IDs into a `hints` dict that step() adds to the reward.
    Changing hints.py requires restarting training (JIT won't pick up changes).
    """
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    mj_model.opt.timestep = 0.002

    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0

    mx = mjx.put_model(mj_model)
    jnt_lo = jnp.array([mj_model.jnt_range[i][0] for i in range(1, mj_model.njnt)])
    jnt_hi = jnp.array([mj_model.jnt_range[i][1] for i in range(1, mj_model.njnt)])

    # Read hints.py once and resolve gripper body IDs
    from . import hints as hints_mod

    h = hints_mod.HINTS

    def body_id(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    hints = {
        "upright": float(h.get("upright", 0.0)),
        "spread_legs": float(h.get("spread_legs", 0.0)),
        "front_arms_down": float(h.get("front_arms_down", 0.0)),
        "back_arms_down": float(h.get("back_arms_down", 0.0)),
        "stillness": float(h.get("stillness", 0.0)),
        "symmetry": float(h.get("symmetry", 0.0)),
        "arm_spacing": float(h.get("arm_spacing", 0.0)),
        "tree_seeking": float(h.get("tree_seeking", 1.0)),  # Weight for tree-seeking reward
        # static body indices
        "fl_id": body_id("fl_gripper_link"),
        "fr_id": body_id("fr_gripper_link"),
        "bl_id": body_id("bl_gripper_link"),
        "br_id": body_id("br_gripper_link"),
        "tree_id": body_id("tree"),
    }
    active = [k for k, v in hints.items() if isinstance(v, float) and v != 0.0]
    print(f"[mjx] hints baked in: {active or 'none'}")
    return mx, jnt_lo, jnt_hi, hints


def reset(mx, jnt_lo, jnt_hi, hints, rng):
    """Reset one environment. Returns (data, obs, prev_action)."""
    rng, k_xy, k_yaw, k_jnt, k_tree = jax.random.split(rng, 5)
    dx = mjx.make_data(mx)

    # randomise torso XY and yaw to break path-memorisation
    xy = jax.random.uniform(k_xy, (2,), minval=-0.05, maxval=0.05)
    yaw = jax.random.uniform(k_yaw, (), minval=-0.3, maxval=0.3)
    qw = jnp.cos(yaw / 2)
    qz = jnp.sin(yaw / 2)

    qpos = dx.qpos.at[0].set(xy[0])
    qpos = qpos.at[1].set(xy[1])
    qpos = qpos.at[2].set(SPAWN_CLEARANCE)
    qpos = qpos.at[3].set(qw)
    qpos = qpos.at[4].set(0.0)
    qpos = qpos.at[5].set(0.0)
    qpos = qpos.at[6].set(qz)

    noise = jax.random.uniform(k_jnt, (24,), minval=-0.4, maxval=0.4)
    qpos = qpos.at[7:].set(jnp.array(_STAND_INIT) + noise)

    # randomise tree position
    tree_xy = jax.random.uniform(k_tree, (2,), minval=3.0, maxval=7.0)
    tree_pos = jnp.array([tree_xy[0], tree_xy[1], 0.0])
    
    # Set tree position in the model
    tree_body_idx = hints["tree_id"]
    dx = dx.replace(xpos=dx.xpos.at[tree_body_idx].set(tree_pos))

    dx = dx.replace(qpos=qpos)
    dx = mjx.forward(mx, dx)

    # Calculate relative tree position for observation
    robot_pos = dx.qpos[0:2]  # robot XY position
    rel_tree_pos = tree_pos[0:2] - robot_pos
    # Add z component (tree height is fixed)
    rel_tree_pos = jnp.concatenate([rel_tree_pos, jnp.array([1.5])])

    obs = _get_obs(dx, rel_tree_pos)
    prev_action = jnp.zeros(24)
    return dx, obs, prev_action


def step(mx, jnt_lo, jnt_hi, hints, dx, action, prev_action, step_count=0):
    """Step one environment. Returns (dx, obs, reward, done, prev_action).

    `hints` is a dict of scalar weights baked in at load time (see load_model).
    Non-scalar entries (body IDs) are used to look up gripper positions.
    `step_count` is used for curriculum learning to gradually increase difficulty.
    """
    # Curriculum learning: gradually increase target height
    if step_count > 0:
        progress = jnp.clip(step_count / CURRICULUM_STEPS, 0.0, 1.0)
        current_target_z = TARGET_Z_MIN + progress * (TARGET_Z_MAX - TARGET_Z_MIN)
    else:
        current_target_z = TARGET_Z
    ctrl = 0.5 * (action + 1.0) * (jnt_hi - jnt_lo) + jnt_lo

    # Get tree position for reward calculation
    tree_pos = dx.xpos[hints["tree_id"]][0:2]  # XY position only
    robot_pos = dx.qpos[0:2]  # robot XY position
    rel_tree_pos = tree_pos - robot_pos

    # Adaptive stiffness based on joint positions
    jpos = dx.qpos[7:]  # 24 joint positions
    # Higher stiffness when joints are near center position, lower when extended
    joint_ranges = jnt_hi - jnt_lo
    joint_center = (jnt_hi + jnt_lo) / 2.0
    dist_from_center = jnp.abs(jpos - joint_center)
    # Normalize distance to [0, 1] range
    normalized_dist = dist_from_center / (joint_ranges / 2.0)
    # Adaptive kp: 8 * (1.2 - 0.4 * normalized_dist) gives range [4.8, 9.6]
    adaptive_kp = 8.0 * (1.2 - 0.4 * normalized_dist)

    # Apply adaptive stiffness by scaling control signal
    # Higher stiffness when joints are near center, lower when extended
    scaled_ctrl = ctrl * adaptive_kp

    dx = dx.replace(ctrl=scaled_ctrl)
    dx, _ = jax.lax.scan(
        lambda d, _: (mjx.step(mx, d), None),
        dx,
        None,
        length=PHYSICS_STEPS_PER_ACTION,
    )

    obs = _get_obs(dx, jnp.concatenate([rel_tree_pos, jnp.array([1.5])]))
    torso_z = dx.qpos[2]
    fallen = torso_z < MIN_HEIGHT

    height_err = (torso_z - current_target_z) / 0.08
    height_reward = jnp.exp(-0.5 * height_err**2)

    w = dx.qpos[3]
    upright_reward = 0.3 * w * w

    jvel = dx.qvel[6:]
    action_delta = jnp.sum((action - prev_action) ** 2)

    # Tree-seeking reward: encourage moving toward the tree
    tree_distance = jnp.linalg.norm(rel_tree_pos)
    tree_reward = hints["tree_seeking"] * (1.0 / (1.0 + tree_distance))  # Higher reward when closer
    
    reward = (
        height_reward
        + upright_reward
        + 0.05
        - 5e-4 * jnp.sum(ctrl**2)
        - 1e-3 * jnp.sum(jvel**2)
        - 2e-2 * action_delta
        - 10.0 * fallen
        + tree_reward
    )

    # ── hint rewards (weights baked in at load time) ─────────────────────────
    reward += hints["upright"] * w * w

    base_joints = dx.qpos[7:][::6]
    reward += hints["spread_legs"] * jnp.mean(jnp.abs(base_joints))

    fl_z = dx.xpos[hints["fl_id"]][2]
    fr_z = dx.xpos[hints["fr_id"]][2]
    reward -= hints["front_arms_down"] * (fl_z + fr_z)

    bl_z = dx.xpos[hints["bl_id"]][2]
    br_z = dx.xpos[hints["br_id"]][2]
    reward -= hints["back_arms_down"] * (bl_z + br_z)

    reward -= hints["stillness"] * jnp.sum(jvel**2)

    jpos = dx.qpos[7:]
    sym_diff = jnp.mean(jnp.abs(jpos[0:6] - jpos[6:12])) + jnp.mean(
        jnp.abs(jpos[12:18] - jpos[18:24])
    )
    reward -= hints["symmetry"] * sym_diff

    # penalise gripper pairs that get closer than MIN_GAP
    positions = jnp.stack(
        [
            dx.xpos[hints["fl_id"]],
            dx.xpos[hints["fr_id"]],
            dx.xpos[hints["bl_id"]],
            dx.xpos[hints["br_id"]],
        ]
    )
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    dists = jnp.stack([jnp.linalg.norm(positions[i] - positions[j]) for i, j in pairs])
    MIN_GAP = 0.08
    reward -= hints["arm_spacing"] * jnp.sum(jnp.maximum(0.0, MIN_GAP - dists) ** 2)

    return dx, obs, reward, fallen, action


def _get_obs(dx, tree_pos):
    return jnp.concatenate(
        [
            dx.qpos[2:3],  # torso z
            dx.qpos[3:7],  # quat
            dx.qvel[0:3],  # linear vel
            dx.qvel[3:6],  # angular vel
            dx.qpos[7:],  # 24 joint pos
            dx.qvel[6:],  # 24 joint vel
            tree_pos,     # 3D tree position relative to robot
        ]
    ).astype(jnp.float32)


OBS_DIM = 62  # 59 + 3 for tree position
ACT_DIM = 24
