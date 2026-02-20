import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# --- 1. THE DATASET LOADER ---
class VLADataset(Dataset):
    def __init__(self, dataset_dir="vla_dataset"):
        self.img_dir = os.path.join(dataset_dir, "images")
        json_path = os.path.join(dataset_dir, "labels.json")

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Create a simple vocabulary for Language Conditioning
        self.vocab = {"Push the red block": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Load Vision (Image)
        img_path = os.path.join(self.img_dir, item["image_file"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to PyTorch format [Channels, Height, Width] and normalize 0-1
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0

        # 2. Load Language (Text)
        text_idx = self.vocab.get(item["instruction"], 0)

        # 3. Load State (Robot Joints)
        state = np.array(item["state"], dtype=np.float32)

        # 4. Load Target Action (What the human did)
        action = np.array(item["action"], dtype=np.float32)

        return torch.tensor(img), torch.tensor(text_idx), torch.tensor(state), torch.tensor(action)


# --- 2. THE VLA NEURAL NETWORK ---
class NeuroVLAPolicy(nn.Module):
    def __init__(self):
        super(NeuroVLAPolicy, self).__init__()

        # A. Vision Encoder (CNN to process the 120x160 image)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 10, 64)  # Compresses the image into 64 numbers
        )

        # B. Language Encoder (Embeds the text instruction)
        self.language_encoder = nn.Embedding(num_embeddings=10, embedding_dim=16)

        # C. Action Decoder (Fuses Vision + Language + State -> Action)
        # 64 (Vision) + 16 (Language) + 2 (State) = 82 inputs
        self.action_head = nn.Sequential(
            nn.Linear(82, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Outputs the 2 motor commands
        )

    def forward(self, img, text, state):
        v_feat = self.vision_encoder(img)
        l_feat = self.language_encoder(text)

        # Fuse the three modalities together
        fused_features = torch.cat([v_feat, l_feat, state], dim=1)

        # Predict action
        action_pred = self.action_head(fused_features)
        return action_pred


# --- 3. THE TRAINING LOOP ---
def main():
    print("Initializing M1-Optimized VLA Training...")

    # Use Apple Silicon GPU (MPS) if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load Data
    dataset = VLADataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = NeuroVLAPolicy().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for robotics regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    print(f"Starting training on {len(dataset)} frames...")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for imgs, texts, states, actions in dataloader:
            # Move data to GPU/CPU
            imgs, texts, states, actions = imgs.to(device), texts.to(device), states.to(device), actions.to(device)

            # 1. Forward pass (Ask the network what to do)
            pred_actions = model(imgs, texts, states)

            # 2. Calculate Loss (Compare network prediction vs human demonstration)
            loss = criterion(pred_actions, actions)

            # 3. Backward pass (Update the brain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")

    # Save the trained brain
    os.makedirs("models", exist_ok=True)
    model_path = "models/vla_policy.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Training Complete! VLA Model saved to {model_path}")


if __name__ == "__main__":
    main()