"""
MJX environment — walk toward a tree.

Reward = progress toward tree position + upright + height-clearance,
penalised for flipping or torso touching ground. The tree position is
read from the model at load time (geom name `t0_trunk`, injected via
`gen_trees.py --inject ... --at X Y`).

Mirrors the structure of stand_env_mjx.py so train_walk_mjx.py is a
near-clone of train_mjx.py.
"""
import importlib
import pathlib

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

import envs.params_walk_mjx as _p

_ROOT        = pathlib.Path(__file__).parent.parent
SCENE        = _ROOT / "data" / "scene.xml"
_PARAMS_FILE = pathlib.Path(__file__).parent / "params_walk_mjx.py"

MAX_STEPS = 60    # ~1.9s sim time. Make-or-break: commit and reach tree, or episode ends.
ACT_DIM   = 24
# obs: torso z(1) + quat(4) + linvel(3) + angvel(3)
#      + jpos(24) + jvel(24) + torques(24)
#      + relative tree (dx, dy, dist) (3)
#      + gait phase (sin, cos) (2)  ← biological: spinal CPG-style clock
OBS_DIM   = 88
GAIT_PERIOD_STEPS = 25  # phase cycle length in env-action-steps; ~1Hz at PHYSICS_STEPS_PER_ACTION=16, dt=0.002 → period ≈ 0.8s

PHYSICS_STEPS_PER_ACTION = int(_p.PHYSICS_STEPS_PER_ACTION)

_params_mtime = 0.0


def read_params():
    global _params_mtime
    try:
        mtime = _PARAMS_FILE.stat().st_mtime
        if mtime != _params_mtime:
            importlib.reload(_p)
            _params_mtime = mtime
    except Exception as e:
        print(f"[params_walk_mjx] reload error: {e}")
    return (
        float(_p.TARGET_Z),
        float(_p.TARGET_TOL),
        float(_p.W_HEIGHT),
        float(_p.W_UPRIGHT),
        float(_p.W_PROGRESS),
        float(_p.W_PROXIMITY_TREE),
        float(_p.SUCCESS_DIST),
        float(_p.W_SUCCESS),
        float(_p.GROUND_CLEARANCE),
        float(_p.W_GROUND),
        float(_p.FLIP_PENALTY),
        float(_p.MOTOR_MAX_SPEED),
        float(_p.W_SMOOTH),
        float(_p.W_JVEL),
        float(_p.W_ENERGY),
        float(_p.W_ALIVE),
        float(getattr(_p, "ACTION_SMOOTH_ALPHA", 1.0)),
        float(getattr(_p, "W_FWD_VEL", 0.0)),
        float(getattr(_p, "W_SYMMETRY", 0.0)),
        float(getattr(_p, "W_HEADING", 0.0)),
    )


def load_model():
    mj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    mj_model.opt.timestep = 0.002

    # Mesh geoms cause combinatorial JIT explosion in MJX — disable their collision.
    for i in range(mj_model.ngeom):
        if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0

    kp = float(getattr(_p, "MOTOR_KP", 8.0))
    mj_model.actuator_gainprm[:, 0] = kp
    mj_model.actuator_biasprm[:, 1] = -kp

    # Look up tree position from the model. Expect `t0_trunk` injected by gen_trees.
    try:
        gid = mj_model.geom("t0_trunk").id
        tree_xy = np.array(mj_model.geom_pos[gid][:2], dtype=np.float32)
    except (KeyError, ValueError):
        raise RuntimeError(
            "walk_env_mjx: scene.xml must contain a tree (geom 't0_trunk'). "
            "Run: python gen_trees.py --inject data/scene.xml --at 3.0 0.0"
        )
    print(f"[walk_env] tree at xy={tree_xy.tolist()}")

    mx = mjx.put_model(mj_model)
    jnt_lo = jnp.array([mj_model.jnt_range[i][0] for i in range(1, mj_model.njnt)])
    jnt_hi = jnp.array([mj_model.jnt_range[i][1] for i in range(1, mj_model.njnt)])
    tree_xy_j = jnp.array(tree_xy)
    return mx, jnt_lo, jnt_hi, tree_xy_j


def reset(mx, jnt_lo, jnt_hi, tree_xy, rng):
    rng, k1, k2 = jax.random.split(rng, 3)
    dx = mjx.make_data(mx)

    # Spawn near origin facing roughly toward the tree. Yaw noise lets the
    # policy learn to orient — but bias toward the right direction.
    yaw_target = jnp.arctan2(tree_xy[1], tree_xy[0])
    yaw_noise  = jax.random.uniform(k1, (), minval=-0.5, maxval=0.5)
    yaw        = yaw_target + yaw_noise
    xy_noise   = jax.random.uniform(k2, (2,), minval=-0.1, maxval=0.1)

    qpos = dx.qpos.at[0].set(xy_noise[0])
    qpos = qpos.at[1].set(xy_noise[1])
    qpos = qpos.at[2].set(0.30)               # spawn already standing-ish
    qpos = qpos.at[3].set(jnp.cos(yaw / 2))
    qpos = qpos.at[4].set(0.0)
    qpos = qpos.at[5].set(0.0)
    qpos = qpos.at[6].set(jnp.sin(yaw / 2))

    rng, knoise = jax.random.split(rng)
    noise = jax.random.uniform(knoise, (24,), minval=-0.10, maxval=0.10)
    qpos = qpos.at[7:7 + ACT_DIM].set(noise)

    dx = dx.replace(qpos=qpos, qvel=jnp.zeros_like(dx.qvel), ctrl=jnp.zeros_like(dx.ctrl))
    dx = mjx.forward(mx, dx)

    motor_pos   = jnp.zeros(ACT_DIM)
    prev_action = jnp.zeros(ACT_DIM)
    step_idx    = jnp.int32(0)
    obs = _get_obs(dx, tree_xy, step_idx)
    prev_dist = jnp.linalg.norm(dx.qpos[0:2] - tree_xy)
    return dx, obs, motor_pos, prev_action, prev_dist, step_idx


def step(mx, jnt_lo, jnt_hi, tree_xy, dx, action, motor_pos, prev_action, prev_dist, step_idx, rparams):
    (target_z, target_tol, w_height, w_upright,
     w_progress, w_proximity_tree, success_dist, w_success,
     ground_clearance, w_ground, flip_penalty, motor_max_speed,
     w_smooth, w_jvel, w_energy, w_alive, action_alpha, w_fwd_vel, w_symmetry,
     w_heading) = rparams

    filt_action = action_alpha * action + (1.0 - action_alpha) * prev_action
    substep_speed = motor_max_speed / PHYSICS_STEPS_PER_ACTION

    def substep(carry, _):
        mpos, d = carry
        delta = jnp.clip(filt_action - mpos, -substep_speed, substep_speed)
        mpos  = mpos + delta
        # Re-centered mapping: action=0 → joint=0 (extended/neutral),
        # action=+1 → jnt_hi, action=-1 → jnt_lo. Asymmetric ranges no longer
        # bias the rest pose toward "folded".
        ctrl  = mpos * jnp.where(mpos >= 0, jnt_hi, -jnt_lo)
        d = d.replace(ctrl=ctrl)
        d = mjx.step(mx, d)
        return (mpos, d), None

    (motor_pos, dx), _ = jax.lax.scan(
        substep, (motor_pos, dx), None, length=PHYSICS_STEPS_PER_ACTION
    )

    step_idx = step_idx + 1
    obs = _get_obs(dx, tree_xy, step_idx)

    torso_xy = dx.qpos[0:2]
    torso_z  = dx.qpos[2]
    w        = dx.qpos[3]

    dist = jnp.linalg.norm(torso_xy - tree_xy)
    progress = prev_dist - dist                             # +ve when approaching

    # Forward velocity reward — directly rewards motion toward tree, even tiny
    # initial movements before progress accumulates. Critical for bootstrapping
    # exploration when standing still has good intrinsic reward.
    to_tree_xy = tree_xy - torso_xy
    to_tree_norm = jnp.maximum(jnp.linalg.norm(to_tree_xy), 1e-3)
    to_tree_unit = to_tree_xy / to_tree_norm
    fwd_vel = jnp.dot(dx.qvel[0:2], to_tree_unit)           # m/s toward tree
    fwd_vel_reward = w_fwd_vel * jnp.maximum(fwd_vel, 0.0)  # only positive (no penalty for retreat)

    # Heading reward: torso forward axis (body +x rotated by torso quat)
    # projected onto xy and aligned with to_tree_unit. Stops squirm/turn-in-place.
    qw, qx, qy, qz = dx.qpos[3], dx.qpos[4], dx.qpos[5], dx.qpos[6]
    fwd_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    fwd_y = 2.0 * (qx * qy + qw * qz)
    fwd_norm = jnp.maximum(jnp.sqrt(fwd_x * fwd_x + fwd_y * fwd_y), 1e-3)
    heading_dot = (fwd_x * to_tree_unit[0] + fwd_y * to_tree_unit[1]) / fwd_norm
    heading_reward = w_heading * heading_dot   # +w when facing tree, -w when facing away

    # Height shaping — torso must stay near target_z (off the ground).
    height_err    = (torso_z - target_z) / target_tol
    height_reward = w_height * jnp.exp(-0.5 * height_err ** 2)

    upright_reward = jnp.where(w > 0, w_upright * w * w, 0.0)
    progress_reward = w_progress * progress

    # Penalty grows when torso dips below ground_clearance.
    ground_violation = jnp.maximum(ground_clearance - torso_z, 0.0)
    ground_penalty   = w_ground * ground_violation ** 2

    # Smooth proximity reward — extra signal near goal so the policy learns
    # to stop, not just orbit.
    prox_tree_reward = w_proximity_tree * jnp.exp(-dist / jnp.maximum(success_dist, 1e-3))

    smooth_penalty = w_smooth * jnp.sum((filt_action - prev_action) ** 2)
    jvel_penalty   = w_jvel * jnp.sum(dx.qvel[6:6 + ACT_DIM] ** 2)
    energy_penalty = w_energy * jnp.sum(dx.ctrl ** 2)

    # Bilateral symmetry: penalize lopsided torques across left/right limbs.
    # Joints 0-5 = FL, 6-11 = FR, 12-17 = BL, 18-23 = BR. Mirror-pair the
    # torque magnitudes — well-behaved gaits keep front-left and front-right
    # similarly loaded over a stride. Biological prior: tetrapods evolved
    # bilaterally symmetric locomotion; asymmetry usually = lameness or chaos.
    torq = dx.qfrc_actuator[6:6 + ACT_DIM]
    fl, fr = jnp.abs(torq[0:6]), jnp.abs(torq[6:12])
    bl, br = jnp.abs(torq[12:18]), jnp.abs(torq[18:24])
    symmetry_penalty = w_symmetry * (jnp.sum((fl - fr) ** 2) + jnp.sum((bl - br) ** 2))

    upside_down = w < 0
    on_ground   = torso_z < (ground_clearance * 0.5)
    success     = dist < success_dist
    # Make-or-break: after a short grace period, if the robot isn't moving
    # toward the tree at a meaningful pace, kill the episode and reset. No
    # reward for standing still — get on with it.
    GRACE_STEPS = 15
    MIN_FWD_VEL = 0.20
    stalled = (step_idx > GRACE_STEPS) & (fwd_vel < MIN_FWD_VEL)

    alive_bonus = w_alive * (1.0 - upside_down.astype(jnp.float32))
    success_bonus = jnp.where(success, w_success, 0.0)

    reward = (height_reward + upright_reward + progress_reward + prox_tree_reward
              + fwd_vel_reward + heading_reward
              + alive_bonus + success_bonus
              - ground_penalty - smooth_penalty - jvel_penalty - energy_penalty
              - symmetry_penalty
              - flip_penalty * upside_down)

    done = upside_down | on_ground | success | stalled

    return dx, obs, reward, done, motor_pos, filt_action, dist, step_idx


def _get_obs(dx, tree_xy, step_idx):
    torso_xy = dx.qpos[0:2]
    rel = tree_xy - torso_xy
    dist = jnp.linalg.norm(rel)
    # Gait phase: a 2π/GAIT_PERIOD_STEPS clock. Biological CPG analog —
    # gives the policy a ready phase signal so it can produce rhythmic output
    # without having to internally track time. (Tetrapods inherit spinal CPGs
    # from fish locomotion ~400Mya; quadruped gaits are phase patterns over
    # this clock.)
    phase = (2.0 * jnp.pi / GAIT_PERIOD_STEPS) * step_idx.astype(jnp.float32)
    return jnp.concatenate([
        dx.qpos[2:3],            # torso z
        dx.qpos[3:7],            # quat
        dx.qvel[0:3],            # linvel
        dx.qvel[3:6],            # angvel
        dx.qpos[7:7 + ACT_DIM],  # joint pos
        dx.qvel[6:6 + ACT_DIM],  # joint vel
        dx.qfrc_actuator[6:30],  # joint torques
        rel,                     # relative tree xy
        dist[None],              # distance scalar
        jnp.sin(phase)[None],    # gait phase clock
        jnp.cos(phase)[None],
    ]).astype(jnp.float32)
