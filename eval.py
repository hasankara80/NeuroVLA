import mujoco
import mujoco.viewer
import numpy as np
import cv2
import torch
import time

# Import the neural network architecture you just trained
from train import NeuroVLAPolicy


def main():
    print("Initializing Autonomous VLA Agent...")

    # 1. Setup Device & Load PyTorch Brain
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = NeuroVLAPolicy().to(device)

    # Load the weights you just saved
    policy.load_state_dict(torch.load("models/vla_policy.pt"))
    policy.eval()  # Freeze the network for inference
    print("✅ Brain Loaded.")

    # 2. Setup MuJoCo Environment
    xml_path = "assets/tabletop.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Must match the training image size (120x160)
    renderer = mujoco.Renderer(model, height=120, width=160)

    # 3. The Language Prompt
    # In a real 1X robot, this comes from a human speaking.
    instruction = "Push the red block"
    vocab = {"Push the red block": 0}

    # Convert text to PyTorch tensor
    text_idx = torch.tensor([vocab[instruction]]).to(device)

    print(f"🧠 AI Prompt: '{instruction}'")
    print("Launching Simulation in Autonomous Mode...")

    # 4. The Inference Loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- A. OBSERVE (Vision & State) ---

            # Extract Vision (Camera Image)
            renderer.update_scene(data, camera="overhead_cam")
            img = renderer.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Format image for PyTorch exactly like we did in training
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img).unsqueeze(0).to(device)  # Add batch dimension

            # Extract State (Joint Angles)
            state = np.array(data.qpos[:2], dtype=np.float32)
            state_tensor = torch.tensor(state).unsqueeze(0).to(device)

            # --- B. THINK (Forward Pass) ---
            with torch.no_grad():  # Disable gradients for fast inference
                action_pred = policy(img_tensor, text_idx, state_tensor)

            target_action = action_pred.cpu().numpy()[0]

            # --- C. ACT (Motor Control) ---
            data.ctrl[0] = target_action[0]
            data.ctrl[1] = target_action[1]

            # Step Physics
            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep at 60Hz
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()