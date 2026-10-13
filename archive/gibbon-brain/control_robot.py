#!/usr/bin/env python3
"""GibbonBot Console Controller - Type commands, robot moves in viewer"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time

# Load scene
model = mujoco.MjModel.from_xml_path(
    "gibbonBot-RL-brain-modeling_data/gibbonbot-rl/scene.xml"
)
data = mujoco.MjData(model)

# Actuator name to index mapping
ACTUATOR_MAP = {
    # Front Left Arm
    'fl_yaw': 0,      'fl_pitch': 1,    'fl_elbow': 2,
    'fl_roll': 3,     'fl_wrist': 4,    'fl_grip': 5,
    # Front Right Arm
    'fr_yaw': 6,      'fr_pitch': 7,    'fr_elbow': 8,
    'fr_roll': 9,     'fr_wrist': 10,   'fr_grip': 11,
    # Back Left Leg
    'bl_yaw': 12,     'bl_pitch': 13,   'bl_knee': 14,
    'bl_roll': 15,    'bl_ankle': 16,   'bl_foot': 17,
    # Back Right Leg
    'br_yaw': 18,     'br_pitch': 19,   'br_knee': 20,
    'br_roll': 21,    'br_ankle': 22,   'br_foot': 23,
}

# Tree position from scene.xml (geom t0_trunk)
TREE_POS = np.array([3.0, 0.0, 0.8])

# Global state
running = True
n_actuators = len(model.actuator_ctrlrange)
controls = np.zeros(n_actuators)


def get_body_position(body_name):
    """Get world position of a body"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id >= 0:
        return data.xpos[body_id]
    return np.zeros(3)


def set_actuator(name, value):
    """Set target position for an actuator"""
    if name in ACTUATOR_MAP:
        idx = ACTUATOR_MAP[name]
        ctrl_min, ctrl_max = model.actuator_ctrlrange[idx]
        controls[idx] = np.clip(value, ctrl_min, ctrl_max)
        return True
    return False


def all_grippers(action):
    """Open or close all grippers/feet"""
    targets = {'open': 0.0, 'close': 1.5, 'grab': 1.5, 'release': 0.0}
    val = targets.get(action, 0.0)
    for name in ['fl_grip', 'fr_grip', 'bl_foot', 'br_foot']:
        set_actuator(name, val)


def stand():
    """Stand up, neutral pose"""
    controls[:] = 0.0
    all_grippers('open')
    # Lift torso slightly
    set_actuator('bl_pitch', -0.2)
    set_actuator('br_pitch', -0.2)
    set_actuator('bl_knee', 0.3)
    set_actuator('br_knee', 0.3)
    return controls.copy()


def walk():
    """Walking gait - alternating back legs"""
    t = time.time()
    cycle = (t * 3) % (2 * np.pi)  # 3 Hz walking cycle
    
    # Back legs: alternating stride
    bl_hip = 0.4 * np.sin(cycle)
    br_hip = 0.4 * np.sin(cycle + np.pi)
    bl_knee = 0.3 * np.sin(cycle + np.pi/2) + 0.2
    br_knee = 0.3 * np.sin(cycle + 3*np.pi/2) + 0.2
    
    set_actuator('bl_pitch', bl_hip)
    set_actuator('br_pitch', br_hip)
    set_actuator('bl_knee', bl_knee)
    set_actuator('br_knee', br_knee)
    
    # Keep arms ready
    set_actuator('fl_pitch', 0.0)
    set_actuator('fr_pitch', 0.0)
    
    return controls.copy()


def reach():
    """Reach front arms toward tree"""
    torso_pos = get_body_position('torso')
    to_tree = TREE_POS - torso_pos[:3]
    distance = np.linalg.norm(to_tree[:2])  # 2D distance in X-Y plane
    
    # Extend arms forward when close to tree
    if distance < 2.5:
        set_actuator('fl_pitch', -0.8)
        set_actuator('fr_pitch', -0.8)
        set_actuator('fl_elbow', 0.8)
        set_actuator('fr_elbow', 0.8)
        all_grippers('open')
    else:
        # Arms at rest when far
        set_actuator('fl_pitch', 0.0)
        set_actuator('fr_pitch', 0.0)
        set_actuator('fl_elbow', 0.2)
        set_actuator('fr_elbow', 0.2)
    
    return controls.copy()


def climb():
    """Climbing motion - pull with arms, push with legs"""
    t = time.time()
    cycle = (t * 1.5) % (2 * np.pi)
    
    # Arms: alternating pull
    fl_pull = 0.6 * np.sin(cycle) - 0.4
    fr_pull = 0.6 * np.sin(cycle + np.pi) - 0.4
    
    # Legs: alternating push against tree
    bl_push = -0.4 * np.sin(cycle) + 0.2
    br_push = -0.4 * np.sin(cycle + np.pi) + 0.2
    
    set_actuator('fl_pitch', fl_pull)
    set_actuator('fr_pitch', fr_pull)
    set_actuator('bl_pitch', bl_push)
    set_actuator('br_pitch', br_push)
    
    # Bend knees for climbing
    set_actuator('bl_knee', 0.5 + 0.2 * np.sin(cycle + np.pi/2))
    set_actuator('br_knee', 0.5 + 0.2 * np.sin(cycle + 3*np.pi/2))
    
    # Close grippers to grab tree
    all_grippers('grab')
    
    return controls.copy()


def go_to_tree():
    """Autonomous: walk forward, reach when close, climb when at tree"""
    torso_pos = get_body_position('torso')
    to_tree = TREE_POS - torso_pos[:3]
    distance = np.linalg.norm(to_tree[:2])
    
    if distance < 0.5:
        # At tree - climb
        climb()
    elif distance < 2.0:
        # Close to tree - reach and stop walking
        reach()
        walk()  # Continue walking but slower
    else:
        # Far from tree - walk toward it
        walk()
        reach()
    
    return controls.copy()


def stop():
    """Freeze all motion"""
    return np.zeros(n_actuators)


def print_actuator_list():
    """Print all available actuators"""
    print("\nAvailable actuators (24 total):")
    print("-" * 40)
    for name, idx in sorted(ACTUATOR_MAP.items(), key=lambda x: x[1]):
        joint_name = model.actuator_trnid[idx]
        jid = model.jnt_type[joint_name]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        ctrl_range = model.actuator_ctrlrange[idx]
        print(f"  [{idx:2d}] {name:12s} -> {jname:20s} range=[{ctrl_range[0]:.2f}, {ctrl_range[1]:.2f}]")
    print("-" * 40)


def print_help():
    """Print help menu"""
    print("\n" + "="*50)
    print("GibbonBot Console Controller - Commands")
    print("="*50)
    print("Movement:")
    print("  stand     - Stand up, neutral pose")
    print("  walk     - Walk forward (back legs)")
    print("  stop     - Freeze all motion")
    print("  ")
    print("Tree Actions:")
    print("  reach    - Extend arms toward tree")
    print("  climb   - Climb the tree")
    print("  gotree   - Auto: walk to tree, reach, climb")
    print("  ")
    print("Grippers:")
    print("  grab     - Close all grippers")
    print("  release  - Open all grippers")
    print("  ")
    print("Direct Control:")
    print("  fl_pitch=0.5   - Set specific actuator")
    print("  fr_elbow=-0.3  - Any actuator name from list")
    print("  ")
    print("Info:")
    print("  list     - List all 24 actuators")
    print("  help     - Show this menu")
    print("  quit     - Exit")
    print("="*50)


def process_command(cmd):
    """Parse and execute a command"""
    global controls
    
    cmd = cmd.strip().lower()
    
    if cmd in ['quit', 'exit', 'q']:
        return None
    
    elif cmd in ['help', '?', 'h']:
        print_help()
        return controls.copy()
    
    elif cmd in ['list', 'actuators', 'ls']:
        print_actuator_list()
        return controls.copy()
    
    elif cmd in ['stand', 'reset', 'home', 'neutral']:
        return stand()
    
    elif cmd in ['walk', 'go', 'forward', 'move']:
        return walk()
    
    elif cmd in ['reach', 'arm', 'arms']:
        return reach()
    
    elif cmd in ['climb', 'up', 'ascend']:
        return climb()
    
    elif cmd in ['gotree', 'goto', 'tree', 'approach']:
        return go_to_tree()
    
    elif cmd in ['stop', 'freeze', 'halt', 'pause']:
        return stop()
    
    elif cmd in ['grab', 'close', 'grip']:
        all_grippers('grab')
        return controls.copy()
    
    elif cmd in ['release', 'open']:
        all_grippers('release')
        return controls.copy()
    
    elif '=' in cmd:
        # Direct actuator control: name=value
        parts = cmd.split('=')
        if len(parts) == 2:
            name = parts[0].strip()
            try:
                value = float(parts[1].strip())
                if set_actuator(name, value):
                    print(f"  Set {name} = {value:.3f}")
                else:
                    print(f"  Unknown actuator: {name}. Type 'list' to see all.")
                return controls.copy()
            except ValueError:
                print(f"  Invalid number: {parts[1]}")
        return controls.copy()
    
    else:
        print(f"  Unknown command: '{cmd}'. Type 'help' for menu.")
        return controls.copy()


# ============================================================
# Main Execution
# ============================================================

def viewer_loop():
    """Viewer runs in a separate thread"""
    global running, controls, data, n_actuators
    
    # Initialize to stand pose
    controls[:] = stand()
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Apply initial controls
        data.ctrl[:] = controls
        viewer.sync()
        
        while running:
            # Apply current controls
            data.ctrl[:] = controls
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync with viewer
            viewer.sync()
            
            # Small delay for smoothness
            time.sleep(1/60)


def main():
    """Main function"""
    global running
    
    print("\n" + "="*60)
    print("  GibbonBot Console Controller")
    print("="*60)
    print("  Viewer will open in a separate window.")
    print("  Type commands in this terminal to control the robot.")
    print("  Type 'help' for command list, 'quit' to exit.")
    print("="*60)
    print()
    
    # Start viewer thread
    viewer_thread = threading.Thread(target=viewer_loop, daemon=True)
    viewer_thread.start()
    
    # Small delay to let viewer initialize
    time.sleep(1)
    
    # Control loop in main thread
    print_help()
    
    try:
        while running:
            cmd = input("\n[GibbonBot] > ").strip()
            result = process_command(cmd)
            
            if result is None:
                running = False
            elif result is not None:
                controls[:] = result
    
    except (EOFError, KeyboardInterrupt):
        print("\n  Exiting...")
        running = False
    
    # Cleanup
    time.sleep(0.5)
    running = False
    viewer_thread.join(timeout=2)
    print("  Done.")


if __name__ == "__main__":
    main()
