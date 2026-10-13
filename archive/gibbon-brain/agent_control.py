#!/usr/bin/env python3
"""
Agentic Control System for GibbonBot
====================================

Implements path following with real-time feedback and evaluation.
Run this script, observe the robot in the viewer, and describe what you see.

Modes:
  1. CIRCLE - FL gripper follows a circle in XY plane
  2. SQUARE - FL gripper follows a square path
  3. STAND - All limbs move to a standing pose
  4. REACH - FL gripper reaches for a fixed point

Controls:
  - Change MODE variable below to switch behaviors
  - Adjust KP, KD gains for responsiveness vs stability
  - Press ESC in viewer to quit

Data logged to logs/agent_run_*.csv for post-run analysis.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION - Edit these to change behavior
# ============================================================================

# Choose mode: 'CIRCLE', 'SQUARE', 'STAND', 'REACH'
MODE = 'CIRCLE'

# Controller gains
KP = 30.0   # Position gain (higher = faster response, may oscillate)
KD = 3.0    # Damping gain (higher = smoother, slower)
DT = 0.02   # Simulation timestep (seconds)

# Path parameters
PATH_RADIUS = 0.25     # Radius for circle/square
PATH_HEIGHT = 0.35     # Z height for paths
PATH_SPEED = 0.8      # Speed multiplier

# ============================================================================
# SETUP
# ============================================================================

SCENE_PATH = "gibbonBot-RL-brain-modeling_data/gibbonbot-rl/scene.xml"

# Create logs directory
Path("logs").mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path(f"logs/agent_run_{timestamp}_{MODE}.csv")

# Load model
print(f"Loading model from {SCENE_PATH}...")
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_body_id(name):
    """Get body ID by name, return -1 if not found."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

def get_joint_id(name):
    """Get joint ID by name, return -1 if not found."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

def get_actuator_id(name):
    """Get actuator ID by name."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

# ============================================================================
# BODY AND JOINT MAPPINGS
# ============================================================================

# End effectors (what we track)
end_effectors = {
    'fl': get_body_id("fl_gripper_link"),
    'fr': get_body_id("fr_gripper_link"),
    'bl': get_body_id("bl_gripper_link"),
    'br': get_body_id("br_gripper_link"),
}

# FL arm joints (what we control for path following)
fl_joints = {
    'shoulder_yaw': get_joint_id("fl_shoulder_yaw"),
    'shoulder_pitch': get_joint_id("fl_shoulder_pitch"),
    'elbow': get_joint_id("fl_elbow"),
    'forearm_roll': get_joint_id("fl_forearm_roll"),
    'wrist': get_joint_id("fl_wrist"),
    'gripper': get_joint_id("fl_gripper"),
}

# All limb prefix order in data.ctrl
limb_order = {
    'fl': range(0, 6),
    'fr': range(6, 12),
    'bl': range(12, 18),
    'br': range(18, 24),
}

# ============================================================================
# TARGET GENERATORS
# ============================================================================

def generate_circle(t):
    """Circle path in XY plane."""
    angle = t * PATH_SPEED
    x = PATH_RADIUS * np.cos(angle)
    y = PATH_RADIUS * np.sin(angle)
    z = PATH_HEIGHT
    return np.array([x, y, z])

def generate_square(t):
    """Square path in XY plane."""
    cycle_time = 4 / PATH_SPEED  # Time for one full square
    phase = (t * PATH_SPEED) % 4
    
    if phase < 1:
        # Right side
        x = PATH_RADIUS
        y = PATH_RADIUS * (2 * phase - 1)
    elif phase < 2:
        # Front side
        x = PATH_RADIUS * (3 - 2 * phase)
        y = -PATH_RADIUS
    elif phase < 3:
        # Left side
        x = -PATH_RADIUS
        y = PATH_RADIUS * (2 * phase - 5)
    else:
        # Back side
        x = PATH_RADIUS * (2 * phase - 7)
        y = PATH_RADIUS
    
    z = PATH_HEIGHT
    return np.array([x, y, z])

def generate_stand(t):
    """Target positions for standing pose."""
    # Return dict of target positions for each end effector
    return {
        'fl': np.array([0.15, 0.12, 0.2]),
        'fr': np.array([0.15, -0.12, 0.2]),
        'bl': np.array([-0.15, 0.12, 0.2]),
        'br': np.array([-0.15, -0.12, 0.2]),
    }

def generate_reach(t):
    """FL gripper reaches for a fixed point."""
    return np.array([0.3, 0.0, 0.4])

# Select target generator based on mode
target_generators = {
    'CIRCLE': generate_circle,
    'SQUARE': generate_square,
    'REACH': generate_reach,
}

# ============================================================================
# CONTROLLERS
# ============================================================================

def pd_control(current, target, prev_error, dt):
    """Simple PD controller."""
    error = target - current
    derivative = (error - prev_error) / dt if dt > 0 else np.zeros(3)
    control = KP * error + KD * derivative
    return control, error

# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================

def run_agent():
    print(f"\n{'='*60}")
    print(f"Starting Agent Control - MODE: {MODE}")
    print(f"KP={KP}, KD={KD}, DT={DT}")
    print(f"Logging to: {log_file}")
    print(f"{'='*60}\n")
    
    # Setup logging
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # CSV header
        header = ['time', 'mode']
        
        # Target columns
        header += ['target_x', 'target_y', 'target_z']
        
        # End effector positions
        for ee in ['fl', 'fr', 'bl', 'br']:
            header += [f'{ee}_x', f'{ee}_y', f'{ee}_z']
        
        # End effector errors (for tracked limb)
        header += ['error_x', 'error_y', 'error_z', 'error_norm']
        
        # Control signals
        header += ['ctrl_fl_yaw', 'ctrl_fl_pitch', 'ctrl_fl_elbow']
        
        # Torso position
        header += ['torso_x', 'torso_y', 'torso_z']
        
        writer.writerow(header)
        
        # State
        target_gen = target_generators.get(MODE)
        torso_id = get_body_id("torso")
        
        prev_error = np.zeros(3)
        start_time = time.time()
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t = time.time() - start_time
                
                # --- Generate targets ---
                if MODE == 'STAND':
                    targets = generate_stand(t)
                    target_pos = targets.get('fl', np.zeros(3))
                else:
                    if target_gen:
                        target_pos = target_gen(t)
                    else:
                        target_pos = np.zeros(3)
                
                # --- Get current positions ---
                current_positions = {}
                for name, bid in end_effectors.items():
                    if bid >= 0:
                        current_positions[name] = data.xpos[bid].copy()
                    else:
                        current_positions[name] = np.zeros(3)
                
                # --- Compute control for FL ---
                fl_pos = current_positions['fl']
                error = target_pos - fl_pos
                error_norm = np.linalg.norm(error)
                
                # PD control in Cartesian space
                # Map error to joint controls (simplified Jacobian-free approach)
                control_signal, prev_error = pd_control(
                    fl_pos, target_pos, prev_error, DT
                )
                
                # Reset all controls
                data.ctrl[:] = 0
                
                # Map Cartesian error to joint controls
                # This is a simplified mapping - y error -> yaw, z error -> pitch, x error -> elbow
                if fl_joints['shoulder_yaw'] >= 0:
                    data.ctrl[fl_joints['shoulder_yaw']] = -control_signal[1]  # Y direction
                if fl_joints['shoulder_pitch'] >= 0:
                    data.ctrl[fl_joints['shoulder_pitch']] = -control_signal[2]  # Z direction
                if fl_joints['elbow'] >= 0:
                    data.ctrl[fl_joints['elbow']] = control_signal[0]  # X direction (reach)
                
                # For STAND mode, also control other limbs
                if MODE == 'STAND':
                    for limb in ['fr', 'bl', 'br']:
                        for j_idx in limb_order[limb]:
                            # Just hold position with small stiffness
                            data.ctrl[j_idx] = -data.qpos[j_idx] * 2.0
                
                # --- Step simulation ---
                mujoco.mj_step(model, data)
                
                # --- Log data ---
                row = [t, MODE]
                row += list(target_pos)
                for ee in ['fl', 'fr', 'bl', 'br']:
                    row += list(current_positions[ee])
                row += list(error)
                row += [error_norm]
                
                # Get actual control values applied
                row += [
                    data.ctrl[fl_joints['shoulder_yaw']] if fl_joints['shoulder_yaw'] >= 0 else 0,
                    data.ctrl[fl_joints['shoulder_pitch']] if fl_joints['shoulder_pitch'] >= 0 else 0,
                    data.ctrl[fl_joints['elbow']] if fl_joints['elbow'] >= 0 else 0,
                ]
                
                # Torso position
                if torso_id >= 0:
                    row += list(data.xpos[torso_id])
                else:
                    row += [0, 0, 0]
                
                writer.writerow(row)
                
                # --- Sync viewer ---
                viewer.sync()
                time.sleep(DT)
    
    print(f"\nAgent stopped. Data saved to {log_file}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_run(log_path):
    """Analyze the logged data and print results."""
    import pandas as pd
    
    df = pd.read_csv(log_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print(f"\nRun duration: {df['time'].max():.2f} seconds")
    print(f"Data points: {len(df)}")
    print(f"Timestep: {df['time'].diff().mean():.4f}s (expected: {DT}s)")
    
    # Error analysis
    print(f"\n--- Error Metrics (FL Gripper) ---")
    print(f"Mean error: {df['error_norm'].mean():.4f}m")
    print(f"Max error: {df['error_norm'].max():.4f}m")
    print(f"Min error: {df['error_norm'].min():.4f}m")
    print(f"Std error: {df['error_norm'].std():.4f}m")
    
    # Path completion
    if MODE in ['CIRCLE', 'SQUARE']:
        print(f"\n--- Path Coverage ---")
        fl_path_length = 0
        for i in range(1, len(df)):
            dx = df[f'fl_x'].iloc[i] - df[f'fl_x'].iloc[i-1]
            dy = df[f'fl_y'].iloc[i] - df[f'fl_y'].iloc[i-1]
            dz = df[f'fl_z'].iloc[i] - df[f'fl_z'].iloc[i-1]
            fl_path_length += np.sqrt(dx**2 + dy**2 + dz**2)
        print(f"FL gripper path length: {fl_path_length:.4f}m")
    
    # Control effort
    print(f"\n--- Control Effort ---")
    for col in ['ctrl_fl_yaw', 'ctrl_fl_pitch', 'ctrl_fl_elbow']:
        if col in df.columns:
            print(f"{col}: mean={df[col].abs().mean():.2f}, max={df[col].abs().max():.2f}")
    
    print(f"\nData saved to: {log_path}")


# ============================================================================
# RUN AND ANALYZE
# ============================================================================

if __name__ == "__main__":
    run_agent()
    analyze_run(log_file)
