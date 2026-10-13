# Gibbon-like robotic ape simulation

Four ROARM M3 arms on one torso. Tony added this on 26 April 2026, commit `62b599e`. The directory name is `simulation_code_jade`, from the commit message “JADE design”. Jade Leurs is the person who later committed the pose study and the Go2 setup, under [gibbonBot-RL/branches](../../gibbonBot-RL/README.md). The project overview is [Gibbon Bot](../../../README.md).

## Structure Overview

The robotic ape consists of:
- **Central Torso**: A rectangular base (0.30m × 0.20m × 0.08m) with mass of 1.5kg
- **Four ROARM M3 Arms**: 
  - Front-left arm (prefix: `fl_`)
  - Front-right arm (prefix: `fr_`)
  - Back-left arm (prefix: `bl_`)
  - Back-right arm (prefix: `br_`)

Each arm is attached to the torso with specific orientations to enable gibbon-like mobility.

## Files

### `ape_robot.urdf.xacro`
Main robot description file that:
- Defines the central torso with physical properties
- Includes and positions four ROARM M3 arms using the macro
- Configures MuJoCo simulation parameters (RK4 integrator, 0.001s timestep)
- Sets up collision exclusions between torso and arm base links

### `roarm_m3_macro.xacro`
Macro definition for individual ROARM M3 arms containing:
- Complete kinematic chain: base_link → link1 → link2 → link3 → link4 → link5 → gripper_link
- Physical properties (mass, inertia) for each link
- Joint definitions with limits and dynamics parameters
- Collision and visual geometries using STL meshes

### `ape_simulation.py`
Python script to:
1. Compile Xacro files to URDF format
2. Load the robot model into MuJoCo
3. Launch an interactive viewer

## Running the Simulation

### Prerequisites
- MuJoCo simulation engine installed
- STL mesh files in `meshes/roarm_m3/` directory (from roarm_ws)
- Python 3.10+

### Quick Start
```bash
# Navigate to the simulation directory
cd simulation_code_jade

# Run the simulation
python ape_simulation.py
```

### Alternative: Using Xacro (if available)
If you have ROS and xacro installed, you can use the original Xacro workflow:
```bash
# Install xacro if needed
sudo apt install ros-humble-xacro  # or your ROS distribution

# Compile and run
xacro ape_robot.urdf.xacro -o ape_robot_compiled.urdf
python ape_simulation.py  # Will use the pre-compiled URDF
```

The simulation will:
1. Load the pre-compiled `ape_robot.urdf` file
2. Initialize the MuJoCo model
3. Launch an interactive viewer showing the robotic ape
4. Simulate physics (the robot will fall under gravity)

## Troubleshooting

**Error: "No such file or directory: 'xacro'"**
- This means ROS/xacro is not installed
- The solution: Use the pre-compiled `ape_robot.urdf` file (default behavior)
- The Xacro compilation step has been removed from the main script

**Error: "File not found: meshes/roarm_m3/*.stl"**
- Copy the mesh files from `roarm_ws/src/roarm_else/roarm_m3/meshes/` to `simulation_code_jade/meshes/roarm_m3/`
- Or create a symlink: `ln -s ../../roarm_ws/src/roarm_else/roarm_m3/meshes meshes`

## Technical Details

### Arm Configuration
Each arm is attached to the torso with specific transformations:
- **Front-left**: `xyz="0.10 0.09 -0.04" rpy="3.1415 0.15 0.15"`
- **Front-right**: `xyz="0.10 -0.09 -0.04" rpy="3.1415 0.15 -0.15"`
- **Back-left**: `xyz="-0.10 0.09 -0.04" rpy="3.1415 -0.15 0.15"`
- **Back-right**: `xyz="-0.10 -0.09 -0.04" rpy="3.1415 -0.15 -0.15"`

### Physics Parameters
- **Integrator**: RK4 (4th-order Runge-Kutta)
- **Timestep**: 0.001 seconds
- **Joint Dynamics**: Damping=0.1, Friction=0.05
- **Collision Handling**: Specific exclusions between torso and arm bases

## Requirements

- MuJoCo simulation engine
- xacro compiler (for URDF compilation)
- Python 3.10+
- STL mesh files in `meshes/roarm_m3/` directory

## Future Development

This structure provides a foundation for:
- Reinforcement learning for gibbon-like locomotion
- Multi-arm coordination algorithms
- Dynamic balancing and mobility research
- Bio-inspired robotics applications