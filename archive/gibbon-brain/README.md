# gibbon-brain

July 2026 experiments in [Gibbon Bot](../../README.md). They try to move a hand in Cartesian space on the four-limb model. Joint commands are a straight map from that error, with no inverse kinematics and no clamp to the joint limits. The reachable space was not checked against the arm. This did not become a controller.

Tony wrote it. Jade Leurs’s pose and Go2 work is in [gibbonBot-RL](../gibbonBot-RL/README.md). The season result is [EVALUATION.md](../EVALUATION.md). The one-arm table model is [digital-twin](../../digital-twin/readme.md). The arm in the lab is [roarm-fw](../../roarm-fw/readme.md).

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
