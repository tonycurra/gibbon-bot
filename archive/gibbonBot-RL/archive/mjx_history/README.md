# MJX Training History

This folder contains historical versions of MJX (MuJoCo JAX) training files extracted from key git commits.

## Files

### Environment Files
- `stand_env_mjx_67f032a.py` - Initial MJX implementation (46× speedup)
- `stand_env_mjx_a5af3b5.py` - Updated with tree seeking removal
- `stand_env_mjx_c5ca157.py` - Inheritance fix for MJX weights
- `stand_env_mjx_cf6614d.py` - Live viewer support

### Training Files
- `train_mjx_67f032a.py` - Initial GPU training script
- `train_mjx_c5ca157.py` - Inheritance and checkpointing improvements
- `train_mjx_cf6614d.py` - Watch tool integration
- `train_mjx_.py` - Current version (from HEAD)

## Key Commits

- **67f032a**: Initial MJX implementation (46× speedup over CPU)
- **a5af3b5**: Tree seeking removal and updates
- **c5ca157**: Fix inheritance to actually load MJX weights
- **cf6614d**: Add tools/watch_mjx.py for live viewing
- **076795f**: Robust path resolution

## Usage

These files are for reference and comparison. The current working versions are:
- `envs/stand_env_mjx.py`
- `train_mjx.py`

To see how the MJX implementation evolved, compare these historical versions with the current ones.