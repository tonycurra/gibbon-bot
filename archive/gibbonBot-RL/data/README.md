# data/

| File/Folder | Purpose |
|-------------|---------|
| `scene.xml` | MuJoCo scene — robot definition, joints, actuators, physics parameters, floor. Joint ranges and torques match the real ST32xx motors; angles in degrees |
| `motors.md` | Per-joint motor assignment + datasheets for ST3235 / ST3215 / ST3215-HS |
| `controls.md` | Real-robot web UI (AngleCtrl, Torque, DEFA), safety, startup, sim-to-real differences |
| `meshes/` | STL mesh files for the robot's visual and collision geometry |
| `roarm_ws/` | Vendor ROS workspace (URDF source for the original arm) |
