# gibbonbot package
# Main package for gibbonBot-RL reinforcement learning framework

__version__ = "0.1.0"

# Import key modules for easy access
try:
    from .envs import ApeEnv, StandEnv, UnifiedEnv
except ImportError as e:
    print(f"Warning: Could not import environments: {e}")

try:
    from .training import evaluate, train
except ImportError as e:
    print(f"Warning: Could not import training functions: {e}")

try:
    from .config import load_config
except ImportError as e:
    print(f"Warning: Could not import config functions: {e}")
