import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time


def main():
    print("Loading VLA Environment...")

    # 1. Load the Model
    xml_path = "assets/tabletop.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. Setup the Renderer (This extracts pixels for the AI's "Vision")
    renderer = mujoco.Renderer(model, height=480, width=640)

    saved_test_frame = False

    # 3. Launch Physics Engine
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation running. (OpenCV window disabled for macOS).")
        print("Click the MuJoCo window and press 'Esc' to exit.")

        while viewer.is_running():
            step_start = time.time()

            # Step physics
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- EXTRACT VISION (The "V" in VLA) ---
            # Update the renderer with the current physics state
            renderer.update_scene(data, camera="overhead_cam")

            # Extract the RGB pixel array
            pixels = renderer.render()

            # Convert RGB (MuJoCo) to BGR (OpenCV) for saving
            cv_image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Save the very first frame to prove the AI has "eyes"
            if not saved_test_frame:
                cv2.imwrite("ai_vision_test.png", cv_image)
                print(f"✅ Success! Saved what the AI sees to 'ai_vision_test.png'.")
                print(f"   Image Shape: {cv_image.shape} (Height, Width, Colors)")
                saved_test_frame = True

            # Keep loop at 60Hz
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()