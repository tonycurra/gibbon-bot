# Credit

Tony designed most of this and wrote the commits in the checkouts this repository was built from. Jade Leurs was on the simulation from April through July 2026. Her credit is the commits recorded under her name.

The local history of `roarm-fw`, `gibbon-console`, `digital-twin`, and the archive checkouts has one human author: Tony. Her commits are written down in [`archive/WORK_OVERVIEW.md`](archive/WORK_OVERVIEW.md), from the Codeberg history of that season. Those branches were not in the checkout exported here, so the files from her commits are not in this snapshot. The dates and messages below are copied from that atlas.

## Tony

- The scene. Torso and four arms in [`archive/gibbonBot-RL/data/scene.xml`](archive/gibbonBot-RL/data/scene.xml), and the earlier xacro/URDF in [`archive/gibbon-rl-mujoco/simulation_code_jade/`](archive/gibbon-rl-mujoco/simulation_code_jade/). That folder was added 26 April 2026 by Tony, commit `62b599e`. The directory name is `simulation_code_jade`. The geometry is his.
- The simulation on the lines in this snapshot. On `gibbonBot-RL` the atlas counts 389 commits by Tony and 10 by Jade Leurs, across 23 April 2026 to 13 July 2026. The `main` line here is his.
- Sitting firmware (`roarm-fw`), the laptop console (`gibbon-console`), and the one-arm model (`digital-twin`).

## Jade Leurs

Ten commits on `gibbonBot-RL`, off `main`, and one on the CPU trainer.

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

No commits of hers are on the firmware, the console, or the one-arm twin.

## Waveshare

Meshes and the RoArm-M3 kinematic chain are from [waveshareteam/roarm_ws](https://github.com/waveshareteam/roarm_ws), MIT. They stay under that license. See `NOTICE` in `digital-twin/` and in the archive trees that carry the same meshes.
