# Naming

**Folder: `roarm-fw`.** Firmware for one modified RoArm-M3 Pro (ESP32 + ST3215 bus).

Rejected or deferred:

| Name | Why not |
| --- | --- |
| `gibbon-limb` | Honest for four-limb product; this board is still a RoArm |
| `gibbon-brain` / `gibbon-control` | Too high — later coordinator / old logging stack |
| `gibbonBot-RL` | Sim, not metal |
| HAT / `busServoDriver` / migration-guide | Board revision and notes, not the software |

Same image on all four boards. Role (`arm_l`, `arm_r`, `leg_l`, `leg_r`) is config later, not four repos.

Later (not this folder): laptop UI is `gibbon-console` (USB/BLE text pipe, all four limbs later). Coordinator for four chips can still be `gibbon-body`.
