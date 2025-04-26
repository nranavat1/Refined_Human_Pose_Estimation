# trains and tests the model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np

# load poses
with open("pose_sequences.pkl", "rb") as f:
    all_sequences = pickle.load(f)

import matplotlib.pyplot as plt

# Visualize poses
COCO_SKELETON = [
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (5, 6), (11, 12),    # Shoulders & hips
    (5, 11), (6, 12)     # Torso
]

def visualize_pose(pose, ax, title="Pose", color='b'):
    pose = pose.reshape(-1, 2)
    ax.scatter(pose[:, 0], pose[:, 1], c=color, s=20)
    for (i, j) in COCO_SKELETON:
        if i < len(pose) and j < len(pose):
            ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], color=color, linewidth=2)
    ax.set_title(title)
    ax.invert_yaxis()  # Flip y-axis for display (origin at top-left)


# define dataset
class PoseSequenceDataset(Dataset):
    def __init__(self, sequences):
        # Normalize keypoints to [0, 1] range
        self.inputs = [torch.tensor(inp / 1280.0, dtype=torch.float32) for inp, _ in sequences]
        self.targets = [torch.tensor(tgt / 1280.0, dtype=torch.float32) for _, tgt in sequences]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# transformer def
class PosePredictor(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=120, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # shape: (seq_len, batch_size, hidden_dim)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # shape: (batch_size, seq_len, hidden_dim)
        x = self.decoder(x)
        return x

# training
def train_model():
    dataset = PoseSequenceDataset(all_sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosePredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []

    for epoch in range(10):
        model.train()
        total_loss = 0

        for inputs, targets in dataloader:
            # print("Sample input:", all_sequences[0][0])
            # print("Sample target:", all_sequences[0][1])

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # outputs = model(inputs)
            outputs = model(inputs)
            outputs = outputs[:, -5:, :]  # Keep only the last 5 frames
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "temporal_pose_transformer.pth")
    print("Model saved as temporal_pose_transformer.pth")

    # Plot loss history
    plt.title("Training Losses")
    plt.plot(loss_history, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gcf().set_size_inches(9, 4)
    plt.grid(True)
    plt.show()

    # Unnormalize
    SCALE = 1280.0

    model.eval()
    with torch.no_grad():
        sample_input, sample_target = next(iter(dataloader))
        sample_input = sample_input.to(device)
        predicted = model(sample_input).cpu().numpy() * SCALE
        ground_truth = sample_target.numpy() * SCALE
        input_pose = sample_input.cpu().numpy() * SCALE

        # Plot a few examples
        for i in range(3):
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            visualize_pose(ground_truth[i][-1], axs[0], title="Ground Truth")
            visualize_pose(predicted[i][-1], axs[1], title="Predicted")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    train_model()