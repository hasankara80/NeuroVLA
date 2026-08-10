# NeuroVLA: Multimodal Behaviour Cloning in MuJoCo

**NeuroVLA** is a compact, end-to-end robot-learning pipeline that collects human teleoperation demonstrations and trains a multimodal PyTorch policy to predict continuous robot actions from RGB observations, task instructions, and robot state.

The project covers the complete workflow from demonstration collection to autonomous evaluation in a MuJoCo simulation. It is designed to run locally on Apple Silicon using PyTorch's MPS backend.

> **Scope:** NeuroVLA is a simulation-based behaviour-cloning prototype.
> Its language component uses a learned instruction embedding rather than
> a pretrained language model.

![Adobe Express - Screen Recording 2026-02-20 at 16 52 55](https://github.com/user-attachments/assets/3ed34fe7-11e2-40da-886c-886f44882660)

---

## 🛠️ Tech Stack

* **Physics Simulation:** mujoco, mujoco.viewer

* **Deep Learning:** PyTorch (Optimized for macOS mps backend)

* **Vision Processing:** OpenCV, numpy

## 🚀 How to Run

**1. Environment Setup**

conda create -n neuro_vla python=3.11 -y

conda activate neuro_vla

pip install mujoco opencv-python torch numpy

**2. Data Collection (Teleoperation)**

Drive the robot using W/A/S/D to push the block. Press R to record a demonstration.

mjpython data_collector.py

Saves RGB frames, text labels, and state-action pairs to vla_dataset/.

**3. Model Training**

Trains the multi-modal PyTorch policy on the collected dataset.

python train.py

**4. Autonomous Evaluation**

Runs the MuJoCo simulation without human input, driven entirely by the trained VLA policy conditioned on the text prompt.

mjpython eval.py

## 🧠 System Architecture

The agent fuses three distinct modalities to predict continuous motor control actions:

1. **Vision (The Eyes):** A Convolutional Neural Network (CNN) processes a 120x160 RGB camera feed from the MuJoCo renderer to establish spatial awareness of the tabletop and objects.
2. **Language (The Ears):** An Embedding layer encodes the text instruction (e.g., *"Push the red block"*) into a dense vector.
3. **State (The Proprioception):** A linear layer processes the robot's current joint angles.

These features are concatenated and passed through an MLP Action Head to output target motor angles.

```mermaid
graph TD
    A[Overhead Camera RGB] -->|CNN| D(Feature Fusion)
    B[Text Instruction] -->|Embedding| D
    C[Robot Joint State] -->|Linear| D
    D -->|MLP| E[Predicted Motor Action]
    E -->|PD Control| F[MuJoCo Simulation]
