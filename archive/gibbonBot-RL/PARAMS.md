# Parameter Reference

All parameters live in `src/gibbonbot/envs/params.py` and are **hot-reloaded every physics step** — no restart needed. You can edit the file directly or use the slider panel that opens with `python train.py` or `watch.py`.

---

## Motor

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `MOTOR_MAX_SPEED` | `0.015` | How far each joint can move per physics step, in normalized [-1, 1] space. Low = slow deliberate motion. High = fast but jerky. At `0.015` a full joint sweep takes ~8 seconds real-time. **If this is too low the robot barely moves; if too high it shakes.** |
| `MAX_ACTIVE_JOINTS` | `24` | How many joints are allowed to move at once before a penalty kicks in. Set to 24 to disable. Lower values (e.g. 3) force the robot to move one limb at a time, encouraging animal-like sequential motion. |

---

## Timing

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `PHYSICS_STEPS_PER_ACTION` | `250` | Number of 0.002s physics steps per policy decision. 250 = the policy acts every 0.5s of sim time. Lower = more decisions per second, faster training, less realistic. Higher = slower training, smoother motion. |
| `REALTIME_FACTOR` | `1.0` | Viewer playback speed. `1.0` = real time. `0.5` = 2× faster. `2.0` = slow motion. `0.0` = no sleep (max speed, for training). Only affects the viewer — training always runs at full speed. Use `+`/`-` keys in the terminal to adjust while watching. |

---

## Goals

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `TARGET_Z` | `0.10` | Target shoulder height in metres. The reward peaks here. Change this to make the robot aim higher or lower. |
| `TARGET_TOL` | `0.02` | Width of the height reward peak (Gaussian sigma, in metres). Small = very precise goal, robot must hit exactly. Large = rewarded over a wider range. |
| `LIFTED_Z` | `0.07` | Threshold for the "got up" bonus. Once shoulder height clears this, the robot gets a one-time `W_GOT_UP` reward. Should be below `TARGET_Z`. |
| `MIN_HEIGHT` | `0.03` | Torso height below which the robot is considered fallen. Episode does not terminate on fall — it just stops getting height reward. |
| `STABLE_VEL` | `0.05` | Maximum joint velocity (rad/s) that counts as "still". Used for the stability bonus. |

---

## Reward Weights

Rewards are summed each step. Positive = incentive, negative = penalty.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `W_HEIGHT` | `2.0` | Scale of the main height reward. Gaussian shaped: peaks at `TARGET_Z`, falls off within `TARGET_TOL`. This is the primary objective. |
| `W_UPRIGHT` | `0.3` | Reward for keeping the torso upright. Proportional to `w²` from the torso quaternion — zero when horizontal, max when vertical. |
| `W_GOT_UP` | `0.5` | One-time bonus when shoulder height first exceeds `LIFTED_Z`. Encourages the robot to make an initial lifting effort. |
| `W_STABLE` | `1.0` | Bonus per step when the robot is both at target height (within `TARGET_TOL`) and still (joint velocity below `STABLE_VEL`). Rewards holding the position, not just reaching it. |
| `W_PLANT_LIFT` | `0.3` | Reward for pushing off the floor. Fires when grippers are on the floor and the torso is moving upward simultaneously — encourages the robot to use its arms as legs. |
| `W_ALIVE` | `0.05` | Small reward per step just for surviving. Prevents the robot from learning to fall immediately. |
| `W_ENERGY` | `5e-4` | Penalty proportional to `sum(ctrl²)`. Discourages running motors at full torque continuously. |
| `W_JVEL` | `1e-3` | Penalty proportional to `sum(joint_velocity²)`. Discourages flailing and fast uncontrolled movement. |
| `W_SMOOTH` | `5e-2` | Penalty proportional to actual motor movement this step (`sum(delta²)`). Discourages rapid back-and-forth oscillation. |
| `W_DRAG` | `0.5` | Penalty for dragging body parts along the floor. Fires when a link is near the floor and moving horizontally. Encourages lifting rather than sliding. |
| `W_PROXIMITY` | `50.0` | Continuous penalty when arms come within `ARM_SAFETY_MARGIN` of each other. Quadratic: gets much larger as arms get closer. |
| `W_JOINTS` | `0.0` | Penalty per joint beyond `MAX_ACTIVE_JOINTS` that moves in a step. Set to 0 to disable. |

---

## Arm Safety

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `ARM_SAFETY_MARGIN` | `0.06` | Distance in metres at which the proximity penalty starts ramping up. Arms farther apart than this = no penalty. |
| `ARM_TOUCH_DIST` | `0.01` | Hard collision distance. If any two cross-arm body parts get closer than 1 cm, the episode ends immediately. |
| `ARM_COLLISION_PENALTY` | `50.0` | One-time penalty applied when `ARM_TOUCH_DIST` is violated. Same magnitude as `FLIP_PENALTY`. |
| `FLIP_PENALTY` | `10.0` | One-time penalty when the torso flips upside down (quaternion w < 0). Episode also terminates. |

---

## How the reward adds up (typical step)

```
reward = W_HEIGHT × height_gaussian          # main goal
       + W_UPRIGHT × w²                      # stay upright
       + W_GOT_UP  (once, if newly lifted)   # initial lift bonus
       + W_STABLE  (if at goal and still)    # hold position
       + W_PLANT_LIFT × grippers × torso_vz  # push-off
       + W_ALIVE                              # survival
       - W_ENERGY × sum(ctrl²)
       - W_JVEL   × sum(jvel²)
       - W_SMOOTH × sum(delta²)
       - W_DRAG   × drag
       - W_PROXIMITY × proximity²
       - W_JOINTS × excess_joints
       [- ARM_COLLISION_PENALTY  if arms touch → terminate]
       [- FLIP_PENALTY           if upside down → terminate]
```

---

## Tips

- **Robot not moving**: check `MOTOR_MAX_SPEED` — if it's below `0.005` the joints barely move.
- **Robot shaking**: increase `W_SMOOTH` or decrease `MOTOR_MAX_SPEED`.
- **Arms crossing**: increase `W_PROXIMITY` or decrease `ARM_SAFETY_MARGIN`.
- **Training too slow**: decrease `PHYSICS_STEPS_PER_ACTION` (try 100).
- **Robot collapses immediately**: check `W_ALIVE` and `W_GOT_UP` are positive; increase `W_HEIGHT`.
- **Robot learned to stay flat**: increase `W_HEIGHT` or raise `TARGET_Z`.
