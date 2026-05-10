"""Shared 88-dim observation contract — frozen across all tasks so policy
weights transfer directly between skills.

Layout:
  [0]      torso_z                                          (1)
  [1:5]    torso quaternion (w, x, y, z)                    (4)
  [5:8]    torso linear velocity                            (3)
  [8:11]   torso angular velocity                           (3)
  [11:35]  joint positions (24 actuated joints)            (24)
  [35:59]  joint velocities                                (24)
  [59:83]  joint torques (qfrc_actuator on actuated dofs)  (24)
  [83]     target dx (target_xy[0] - torso_x), 0 if no target
  [84]     target dy
  [85]     target distance (Euclidean), 0 if no target
  [86]     gait phase sin
  [87]     gait phase cos
"""
import jax.numpy as jnp


ACT_DIM_DEFAULT = 24


def proprio_obs(dx, act_dim=ACT_DIM_DEFAULT):
    """83-dim proprioceptive subset of obs (everything except target + gait clock)."""
    return jnp.concatenate([
        dx.qpos[2:3],                # 1   torso z
        dx.qpos[3:7],                # 4   quat
        dx.qvel[0:3],                # 3   linvel
        dx.qvel[3:6],                # 3   angvel
        dx.qpos[7:7 + act_dim],      # 24  joint positions
        dx.qvel[6:6 + act_dim],      # 24  joint velocities
        dx.qfrc_actuator[6:6 + act_dim],  # 24 joint torques
    ])


def gait_clock(step_idx, gait_period_steps):
    phase = (2.0 * jnp.pi / gait_period_steps) * step_idx.astype(jnp.float32)
    return jnp.array([jnp.sin(phase), jnp.cos(phase)])


def assemble_obs(dx, target_xy, step_idx, gait_period_steps, act_dim=ACT_DIM_DEFAULT):
    """Build the 88-dim obs vector.

    target_xy: jnp.array of shape (2,) — pass jnp.zeros(2) for tasks without
               a spatial target. Distance computed from torso position.
    """
    proprio = proprio_obs(dx, act_dim)  # 83
    rel = target_xy - dx.qpos[0:2]
    dist = jnp.linalg.norm(rel)
    target_slot = jnp.concatenate([rel, dist[None]])  # 3
    clock = gait_clock(step_idx, gait_period_steps)   # 2
    return jnp.concatenate([proprio, target_slot, clock]).astype(jnp.float32)
