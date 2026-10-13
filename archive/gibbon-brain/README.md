# gibbon-brain

**Parked, July 2026.** Cartesian PD experiments on the four-limb model. Joint commands were a linear map from Cartesian error, with no inverse kinematics and no clamp to the joint limits. The workspace was not checked against the arm. This did not become a controller.

The season result is in [gibbon-archive](https://github.com/tonycurra/gibbon-archive). The one-arm model that replaced the four-limb scene is [gibbon-digital-twin](https://github.com/tonycurra/gibbon-digital-twin). The live laboratory is [gibbon-bot](https://github.com/tonycurra/gibbon-bot).

## What is here

| Path | What |
| --- | --- |
| `agent_control.py` | Viewer paths: circle, square, reach, stand |
| `control_robot.py` | Cartesian error mapped onto joint indexes |
| `evaluate.py` | Log reader |
| `gibbonBot-RL-brain-modeling_data/` | Scene, torso mesh, motor notes |

Waveshare RoArm meshes in that data folder stay MIT. Olivabot glue in this repository is Apache-2.0. See [NOTICE](NOTICE).

```bash
python3 agent_control.py
```

The viewer opens on the machine that has MuJoCo installed. Close it with Escape.
