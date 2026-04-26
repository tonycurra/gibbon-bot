import subprocess

import mujoco
import mujoco.viewer

# THIS CODE REQUIRES THE MESHES FOLDER TO BE IN THE CURRENT DIRECTORY
def main() -> None:
    # 1. Compile Xacro to URDF
    input_file = "ape_robot.urdf.xacro"
    output_file = "ape_robot.urdf"

    print(f"Compiling {input_file}...")
    subprocess.run(["xacro", input_file, "-o", output_file], check=True)

    # 2. Load into MuJoCo
    model = mujoco.MjModel.from_xml_path(output_file)
    data = mujoco.MjData(model)

    # 3. Launch Viewer
    print("Launching MuJoCo Viewer. Robot should fall under gravity.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
