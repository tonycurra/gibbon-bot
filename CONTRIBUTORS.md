# Contributors

Authorship below follows the commits. Dates and messages for branches that are not in this tree are copied from the lab atlas, [`archive/WORK_OVERVIEW.md`](archive/WORK_OVERVIEW.md).

## Tony

- Scene geometry: torso and four arms in [`archive/gibbonBot-RL/data/scene.xml`](archive/gibbonBot-RL/data/scene.xml), and the xacro/URDF in [`archive/gibbon-rl-mujoco/simulation_code_jade/`](archive/gibbon-rl-mujoco/simulation_code_jade/), added 26 April 2026, commit `62b599e`. The directory keeps the name `simulation_code_jade`.
- Simulation on `gibbonBot-RL` `main`, 23 April 2026 to 13 July 2026. The atlas for that repository records 389 commits by Tony and 10 by Jade Leurs.
- Firmware (`roarm-fw`), console (`gibbon-console`), and the one-arm model (`digital-twin`).

## Jade Leurs

Simulation, April through July 2026. Ten commits on `gibbonBot-RL`, off `main`, and one on the CPU trainer. The branches were not in the checkout this tree was built from.

Pose priors, branch `apenapepose_integration`:

| Date | Commit | Message |
| --- | --- | --- |
| 2026-05-21 | `7a2fe9e` | fixed formatting |
| 2026-05-21 | `7729580` | added openapepose clustering |
| 2026-05-21 | `fe244e7` | added GMM pose priors |
| 2026-05-21 | `6c6538b` | derive approximate joint angle limits from 2D OpenApePose keypoints |
| 2026-05-21 | `08d141a` | added docs for clustering and integration options |
| 2026-05-22 | `36ec54b` | integrated pose priors into training loop |
| 2026-06-17 | `21170cb` | some updates |

Quadruped environment, branch `quadruped_training`. The same two commits also sit on `unigree-go2`:

| Date | Commit | Message |
| --- | --- | --- |
| 2026-06-26 | `e4c87ef` | Initial commit |
| 2026-06-28 | `fc4a84a` | added custom (Ant-v5 based) environment |

Unitree Go2, branch `unigree-go2`:

| Date | Commit | Message |
| --- | --- | --- |
| 2026-07-13 | `0580087` | unitree_go2 training setup |

CPU trainer, branch `jade` on `olivabot-mujoco-cpu`. The `main` line in this snapshot is the 27 April commit by tony_olivabot. Her commit is the next one on that branch:

| Date | Commit | Author | Message |
| --- | --- | --- | --- |
| 2026-05-01 | `5160c10` | Jade Leurs | add telemetry logs |
| 2026-05-01 | `2da11dd` | jajayumei | update requirements.txt to include tensorboard |

The atlas names the tip of that branch as Jade Leurs. The tensorboard commit is recorded as `jajayumei`. Both names are left as written.

Firmware, the console, and the one-arm model have no commits under these names.

## Waveshare

Meshes and the RoArm-M3 kinematic chain are from [waveshareteam/roarm_ws](https://github.com/waveshareteam/roarm_ws), MIT. They stay under that license. See `NOTICE` in `digital-twin/` and in the archive trees that carry the same meshes.
