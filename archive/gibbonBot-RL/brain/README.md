# Brain Architecture

See docs/APEBRAIN.md for full documentation.

## New Components

- policy_database.py: SQLite storage for learned policies
- utils/policy_router.py: Intelligent policy selection
- utils/brain_viewer.py: View robot with brain memory

## Usage

python -m utils.policy_router --import-from-runs
python -m utils.brain_viewer --task stand --brain
