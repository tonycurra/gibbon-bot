# envs/

Gymnasium environments for the ape robot.

| File | Purpose |
|------|---------|
| `stand_env.py` | Main environment — robot falls, settles, then the policy tries to lift and hold the torso level |
| `params.py` | All tunable parameters. Hot-reloaded every episode — edit while training and changes take effect immediately. Run directly (`python envs/params.py`) to open the slider UI. |
| `hints.py` | Optional extra reward terms, also hot-reloaded. Set a weight to `0.0` to disable. |

## How the environment works

1. Robot spawns upright at a random height, falls, and settles onto the floor (up to 3000 physics steps).
2. Policy takes over for up to 2000 steps, outputting target joint positions for all 24 joints.
3. Each step runs 240 physics substeps — motors move gradually toward the target rather than snapping.
4. Reward is purely: height Gaussian + upright bonus − arm proximity penalty. Episode ends on flip or arm collision.
