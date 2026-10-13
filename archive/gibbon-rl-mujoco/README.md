# Gibbon RL MuJoCo

**Parked, April 2026.** Early four-arm URDF / xacro and a Gymnasium simulation. This is not the live robot. The walk-to-tree attempt that followed is recorded in [gibbon-archive](https://github.com/tonycurra/gibbon-archive). The laboratory that replaced it is [gibbon-bot](https://github.com/tonycurra/gibbon-bot).

---

A reinforcement learning project for MuJoCo environments, featuring a four-RoArm ape structure.

## Setup

### Prerequisites
- Ubuntu 24.04
- Python 3.10 or higher

### Create Virtual Environment
```bash
python3 -m venv .venv
```

### Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Overview
- **MuJoCo**: Physics engine for robotics simulation.
- **NumPy**: Numerical computing library for handling arrays and mathematical operations.
- **PyOpenGL**: OpenGL bindings for rendering and visualization.
- **Gymnasium**: Updated version of the `gym` library for reinforcement learning environments.
- **PyTorch**: Deep learning framework for training models.
- **Stable Baselines3**: Reinforcement learning library for implementing algorithms.
- **Matplotlib**: Plotting library for visualization.
- **Pandas**: Data manipulation and analysis library.
- **CUDA-related packages**: For GPU acceleration (e.g., `cuda-toolkit`, `nvidia-cuda-runtime`).
- **Triton**: GPU-accelerated programming language for PyTorch.

### Quick Start
To test if MuJoCo is working correctly, run the following command:
```bash
python test/test_robot.py
```
This will open a visualizer window showing a red ball falling onto a plane. For more details, see the [test/README.md](test/README.md).

### Robot Description
The `/roarm_ws` folder contains the official content from the [Waveshare ROARM repository](https://github.com/waveshareteam/roarm_ws/tree/ros2-humble). This includes the robot description and characteristics used for simulation. While the original repository uses ROS and MoveIt, this project focuses on using the robot description for MuJoCo simulations without ROS dependencies. So we dont need all the files in this folder roarmws

### Gibbon-like Robotic Ape (JADE Design)
The `/simulation_code_jade` folder contains a unique robotic structure designed by Jade, combining four ROARM robotic arms into a mobile "gibbon-like" ape robot:

- **Structure**: Four ROARM M3 arms attached to a central torso
- **Configuration**: 
  - Front-left arm (fl_)
  - Front-right arm (fr_)
  - Back-left arm (bl_)
  - Back-right arm (br_)
- **Files**:
  - `ape_robot.urdf.xacro`: Main robot description combining four arms
  - `roarm_m3_macro.xacro`: Macro definition for individual ROARM M3 arms
  - `ape_simulation.py`: Simulation script to visualize the robot

To run the gibbon-like robot simulation:
```bash
cd simulation_code_jade
python ape_simulation.py
```

This will compile the Xacro files and launch a MuJoCo viewer showing the four-armed robotic ape structure. The robot will fall under gravity, demonstrating the physics simulation.
