# Contributors

Two people. Names on the commits vary because the same person used more than one address.

## Tony

Design, the simulation season, and the sitting stack: firmware, laptop console, and the one-arm model. 22 April to 28 September 2026.

| Author on the commit | Commits | Dates | Where |
| --- | --- | --- | --- |
| Tony \<tonycurra@proton.me\> | 386 | 22 Apr–15 May 2026 | [`archive/gibbonBot-RL`](archive/gibbonBot-RL/) |
| Tony \<tony@example.com\> | 3 | 26 Apr 2026 | [`archive/gibbon-rl-mujoco`](archive/gibbon-rl-mujoco/), including `Add gibbon-like robotic ape simulation (JADE design)` (`ee71a06`) |
| tony_olivabot \<tony_olivabot@codeberg.org\> | 1 | 27 Apr 2026 | [`archive/olivabot-mujoco-cpu`](archive/olivabot-mujoco-cpu/), `first commit` (`99fddc6`) |
| Tony \<tony@olivabot.com\> | 44 | 12 May–28 Sep 2026 | [`roarm-fw`](roarm-fw/), [`gibbon-console`](gibbon-console/), [`digital-twin`](digital-twin/) |
| tony \<mail@tonycurra.dev\> | 17 | 13 Oct 2026 | Packing this public repository |

## Jade Leurs

Simulation, 1 May to 13 July 2026. Twelve commits. Eleven are signed Jade Leurs \<jadeleurs@hotmail.com\>. The tensorboard commit on 1 May is signed jajayumei \<jajayumei@noreply.codeberg.org\>. Same person.

She added logging and tensorboard to the CPU trainer, built a pose prior from OpenApePose ape video, and set up a quadruped practice environment and a Unitree Go2 training scene.

| Date | Commit | Message | Tree |
| --- | --- | --- | --- |
| 2026-05-01 | `99b41fb` | add telemetry logs | [`archive/olivabot-mujoco-cpu`](archive/olivabot-mujoco-cpu/) |
| 2026-05-01 | `538a53a` | update requirements.txt to include tensorboard | same. Signed `jajayumei` |
| 2026-05-21 | `30ec788` | fixed formatting | [`apenapepose_integration`](archive/gibbonBot-RL/branches/apenapepose_integration/) |
| 2026-05-21 | `6dbd380` | added openapepose clustering | same |
| 2026-05-21 | `d6398ce` | added GMM pose priors | same |
| 2026-05-21 | `75d82e5` | derive approximate joint angle limits from 2D OpenApePose keypoints | same |
| 2026-05-21 | `b2e0623` | added docs for clustering and integration options | same |
| 2026-05-22 | `29d733a` | integrated pose priors into training loop | same |
| 2026-06-17 | `8dbe693` | some updates | same |
| 2026-06-26 | `3c790aa` | Initial commit | [`unigree-go2`](archive/gibbonBot-RL/branches/unigree-go2/) |
| 2026-06-28 | `7994f60` | added custom (Ant-v5 based) environment | same |
| 2026-07-13 | `d2e0578` | unitree_go2 training setup | same |

## Waveshare

RoArm-M3 meshes and the kinematic chain are from [waveshareteam/roarm_ws](https://github.com/waveshareteam/roarm_ws), MIT. See `NOTICE` in `digital-twin/` and in the archive trees that carry the same meshes.
