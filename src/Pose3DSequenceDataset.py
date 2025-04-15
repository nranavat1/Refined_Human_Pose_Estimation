import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class Pose3DSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = [os.path.join(root_dir, seq) for seq in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, seq))]

    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):
        sequence_folder = self.sequences[idx]
        frames = self._load_sequence(sequence_folder)

        # Pad if too short
        if len(frames) < self.sequence_length:
            padding = self.sequence_length - len(frames)
            blank = Image.new("RGB", frames[0].size, (0, 0, 0))
            frames.extend([blank] * padding)

        # Select contiguous sequence
        start_idx = np.random.randint(0, len(frames) - self.sequence_length + 1)
        frames = frames[start_idx:start_idx + self.sequence_length]

        # Transform and stack
        frames_tensor = [self.transform(f) if self.transform else f for f in frames]
        return torch.stack(frames_tensor)

    def _load_sequence(self, folder_path):
        frame_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        frames = []
        for file in frame_files:
            try:
                image = Image.open(os.path.join(folder_path, file)).convert("RGB")
                frames.append(image)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return frames

    def load_raw_sequence(self, idx, max_frames=None):
        sequence_folder = self.sequences[idx]
        frames = self._load_sequence(sequence_folder)
        if max_frames:
            frames = frames[:max_frames]
        return frames
