# gibbonbot package
# Main package for gibbonBot-RL reinforcement learning framework

__version__ = "0.1.0"

# Import key modules for easy access
try:
    from .envs import StandEnv
except ImportError as e:
    print(f"Warning: Could not import environments: {e}")
