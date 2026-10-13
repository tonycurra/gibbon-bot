# MuJoCo Test

This folder contains a simple test to verify that MuJoCo is installed and working correctly.

## Description
The test simulates a red ball falling onto a plane under gravity. This is a basic physics simulation to confirm that MuJoCo is functioning as expected.

## Files
- `robot.xml`: Defines the MuJoCo model with a plane and a free-falling ball.
- `test_robot.py`: Python script to run the simulation and visualize it.

## How to Run
1. Ensure you have activated the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Run the test script:
   ```bash
   python test/test_robot.py
   ```

## Expected Output
A visualizer window will open, showing a red ball falling onto a gray plane. The ball should bounce and settle due to gravity and physics simulation.

## Troubleshooting
- If the window does not open, ensure all dependencies are installed (`pip install -r requirements.txt`).
- If the simulation appears frozen, check for errors in the terminal output.