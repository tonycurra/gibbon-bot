import mujoco
import mujoco.viewer


# THIS CODE REQUIRES THE MESHES FOLDER TO BE IN THE CURRENT DIRECTORY
def main() -> None:
    # Load the pre-compiled URDF file directly
    urdf_file = "ape_robot.urdf"

    print(f"Loading {urdf_file}...")
    model = mujoco.MjModel.from_xml_path(urdf_file)
    data = mujoco.MjData(model)

    # Launch Viewer
    print("Launching MuJoCo Viewer. Robot should fall under gravity.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
