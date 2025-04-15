import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import os

def visualize_feature_maps(feature_maps):
    num_levels = len(feature_maps)
    fig, axes = plt.subplots(1, num_levels, figsize=(4*num_levels, 4))

    for i, fmap in enumerate(feature_maps):
        fmap = fmap[0]  # Take first image in batch
        avg_map = fmap.mean(dim=0).detach().cpu()  # Average over channels

        ax = axes[i] if num_levels > 1 else axes
        ax.imshow(avg_map, cmap='viridis')
        ax.set_title(f'FPN Level {i+1}: {fmap.shape[-2]}x{fmap.shape[-1]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
# Visualize a few sequences
def visualize_sequence(dataset, index, max_frames=10):
    frames = dataset.load_raw_sequence(index, max_frames=max_frames)

    if len(frames) == 0:
        print(f"No frames found at index {index}")
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")
    plt.tight_layout()
    plt.show()