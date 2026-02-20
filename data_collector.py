import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
import os
import json


def main():
    print("Initializing VLA Data Collector...")

    # 1. Load the Model
    xml_path = "assets/tabletop.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # We use a smaller image size (160x120) so the neural network trains faster on your Mac
    renderer = mujoco.Renderer(model, height=120, width=160)

    # 2. Dataset Storage Setup
    dataset_dir = "vla_dataset"
    img_dir = os.path.join(dataset_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    dataset = []
    recording = False
    frame_count = 0

    # The "Language" part of VLA
    instruction = "Push the red block"

    # Motor targets (Starting at 0 radians)
    target_angles = [0.0, 0.0]

    # 3. Keyboard Controller
    def key_callback(keycode):
        nonlocal recording
        try:
            char = chr(keycode).lower()
        except ValueError:
            char = None

        step_size = 0.05  # How fast the arm moves per key press

        # Teleoperation mapping
        if char == 'w':
            target_angles[0] += step_size
        elif char == 's':
            target_angles[0] -= step_size
        elif char == 'a':
            target_angles[1] += step_size
        elif char == 'd':
            target_angles[1] -= step_size

        # Clamp to the physical joint limits (-1.5 to 1.5 radians)
        target_angles[0] = np.clip(target_angles[0], -1.5, 1.5)
        target_angles[1] = np.clip(target_angles[1], -1.5, 1.5)

        # Toggle recording with 'R'
        if char == 'r':
            recording = not recording
            if recording:
                print("\n🔴 RECORDING STARTED! Push the block.")
            else:
                print(f"\n⏹ RECORDING STOPPED! Saved {len(dataset)} steps so far.")

    # 4. Launch Simulation
    print("\n=== VLA TELEOP CONTROLS ===")
    print("[W] / [S] : Move Shoulder Joint")
    print("[A] / [D] : Move Elbow Joint")
    print("[R]       : Toggle Recording (ON/OFF)")
    print("[ESC]     : Exit and Save Dataset")
    print("===========================\n")

    # Pass our custom key_callback to MuJoCo
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Apply your keyboard commands to the robot's motors
            data.ctrl[0] = target_angles[0]
            data.ctrl[1] = target_angles[1]

            mujoco.mj_step(model, data)
            viewer.sync()

            # We record at 10Hz (every 6th frame of a 60Hz loop) to save disk space
            if recording and frame_count % 6 == 0:
                # Extract AI Vision
                renderer.update_scene(data, camera="overhead_cam")
                pixels = renderer.render()
                cv_image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                # Save Image
                img_name = f"frame_{frame_count}.jpg"
                img_path = os.path.join(img_dir, img_name)
                cv2.imwrite(img_path, cv_image)

                # Save State-Action-Text Tuple
                dataset.append({
                    "image_file": img_name,
                    "instruction": instruction,
                    "state": data.qpos[:2].tolist(),
                    "action": target_angles.copy()
                })
                print(f"Recorded step {len(dataset)}...", end="\r")

            if recording:
                frame_count += 1

            # Maintain real-time 60Hz physics
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    # 5. Save the JSON labels safely on exit
    if len(dataset) > 0:
        json_path = os.path.join(dataset_dir, "labels.json")
        with open(json_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"\n✅ Dataset saved! {len(dataset)} VLA pairs written to {json_path}")
    else:
        print("\nNo data recorded. Press 'R' next time to record!")


if __name__ == "__main__":
    main()