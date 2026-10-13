# data/

| File/Folder | Purpose |
|-------------|---------|
| `scene.xml` | Canonical MuJoCo scene — 4-limb robot, joints, actuators, two-slab torso, floor. See `SCENE.md` |
| `SCENE.md` | High-level tour of `scene.xml`: body layout, mount positions, motors, ranges, conventions |
| `SCENE_GEOMETRY.md` | Dimensional reference for `scene.xml` (mesh sizes, base-to-base formula, mount positions) |
| `scenes_archive/` | Historical scene XMLs (pre-slabs, pre-promote, 30 mm torso, walk variants, etc.) — read-only |
| `motors.md` | Per-joint motor assignment + datasheets for ST3235 / ST3215 / ST3215-HS |
| `controls.md` | Real-robot web UI (AngleCtrl, Torque, DEFA), safety, startup, sim-to-real differences |
| `meshes/` | STL mesh files for the robot's visual and collision geometry |
| `roarm_ws/` | Vendor ROS workspace (URDF source for the original arm) |
