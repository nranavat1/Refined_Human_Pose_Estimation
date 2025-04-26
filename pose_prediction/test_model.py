# issue: poses aren't aligned properly with the original images. working on that

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
from PIL import Image

# ---------------------- Load Data ----------------------

class PoseImageSequenceDataset(Dataset):
    def __init__(self, sequence_file, image_resize=(256, 256)):
        with open(sequence_file, "rb") as f:
            self.all_sequences = pickle.load(f)
        self.image_resize = image_resize

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        (input_keypoints, input_image_paths), (target_keypoints, target_image_paths) = self.all_sequences[idx]

        input_images = [self.load_image(p) for p in input_image_paths]
        target_images = [self.load_image(p) for p in target_image_paths]

        input_keypoints = torch.tensor(input_keypoints / 1280.0, dtype=torch.float32)
        target_keypoints = torch.tensor(target_keypoints / 1280.0, dtype=torch.float32)

        return input_images, input_keypoints, target_images, target_keypoints

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.image_resize:
            img = img.resize(self.image_resize)
        img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)  # (C, H, W)
        return img

# ---------------------- Visualization Helper ----------------------

COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

def overlay_pose(image_tensor, pose, ax, color='lime', title=""):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    H, W = image.shape[:2]

    pose = pose.reshape(-1, 2)

    # Step 1: undo old wrong normalization
    pose *= 1280.0

    # Step 2: scale down to real resized image size
    pose[:, 0] = pose[:, 0] * (W / 1280.0)
    pose[:, 1] = pose[:, 1] * (H / 1280.0)

    ax.imshow(image)
    ax.scatter(pose[:, 0], pose[:, 1], c=color, s=20)
    for (i, j) in COCO_SKELETON:
        if i < len(pose) and j < len(pose):
            ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], color=color, linewidth=2)
    ax.set_title(title)
    ax.axis('off')

# ---------------------- Pose Predictor Model ----------------------

class PosePredictor(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=120, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.decoder(x)
        return x

# ---------------------- Testing Code ----------------------

def test_model_with_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoseImageSequenceDataset(sequence_file="./pose_image_sequences.pkl")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = PosePredictor(input_dim=34, hidden_dim=120, nhead=6).to(device)
    model.load_state_dict(torch.load("temporal_pose_transformer.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        for batch_idx, (input_images, input_poses, target_images, target_poses) in enumerate(dataloader):
            input_poses = input_poses.to(device)
            predicted_poses = model(input_poses)
            predicted_poses = predicted_poses[:, -5:, :]

            input_images = input_images[0]  # remove batch dimension
            target_images = target_images[0]
            predicted_poses = predicted_poses[0].cpu().numpy()
            target_poses = target_poses[0].cpu().numpy()

            # Plot for first 3 future frames
            for i in range(min(3, len(target_images))):
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                overlay_pose(target_images[i], target_poses[i], axs[0], color='red', title="Ground Truth")
                overlay_pose(target_images[i], predicted_poses[i], axs[1], color='lime', title="Predicted")
                plt.tight_layout()
                plt.show()

            if batch_idx >= 4:  # Show for 5 batches only
                break

# ---------------------- Main ----------------------

if __name__ == "__main__":
    test_model_with_images()
