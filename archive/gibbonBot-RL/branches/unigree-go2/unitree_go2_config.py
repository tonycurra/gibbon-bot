"""
Configuration for Unitree Go2 environment.
Centralized configuration to avoid duplication between training and visualization.
"""

# Common environment parameters
UNITREE_GO2_PARAMS = {
    "xml_file": "unitree_go2/scene_mjx.xml",
    "default_pose_xml_file": "unitree_go2/go2_mjx.xml",
    "forward_reward_weight": 1,
    "ctrl_cost_weight": 0.1,
    "contact_cost_weight": 5e-4,
    "healthy_reward": 1,
    "pose_penalty_weight": 0.1,
    "main_body": 1,  # Changed from 0 to 1 - base body instead of world
    "healthy_z_range": (0.25, 0.9),
    "include_cfrc_ext_in_observation": True,
    "exclude_current_positions_from_observation": False,
    "reset_noise_scale": 0.1,
    "frame_skip": 15,
}

# UNITREE_GO2_PARAMS = {
#     "xml_file": "unitree_go2/scene_mjx.xml",
#     "forward_reward_weight": 1,
#     "ctrl_cost_weight": 0.05,
#     "contact_cost_weight": 5e-4,
#     "healthy_reward": 1,
#     "main_body": 1,
#     "healthy_z_range": (0.25, 0.9),
#     "include_cfrc_ext_in_observation": True,
#     "exclude_current_positions_from_observation": False,
#     "reset_noise_scale": 0.1,
#     "frame_skip": 15,
# }

# Training-specific parameters
TRAINING_PARAMS = {
    "render_mode": "rgb_array",
}

# Visualization-specific parameters
VISUALIZATION_PARAMS = {
    "render_mode": "human",
}

# Time limit (used in both)
MAX_EPISODE_STEPS = 1000
