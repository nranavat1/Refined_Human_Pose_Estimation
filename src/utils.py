import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import os

def patchify(x, downsampling_ratio = 16):
    """
        A helper function to convert input image(s) into a set of patches 
        for input to a transformer.

        Input:
        - x: Input data of shape (N, C, H, W)
        - downsampling_ratio: The ratio at which we want to reduce the image int0 patches
    
        This function requires H and W are multiples of patch_size

        Returns:
        - out: Output data of shape (N, H'*W', ph*pw, C) where H', W', ph, pw are given by
          H' = H // patch_size
          W' = W // patch_size
          ph = patch_size
          pw = patch_size
        """
    N, C, H, W = x.shape
    patch_size = (H//downsampling_ratio, W//downsampling_ratio)
    assert H % patch_size==0, "Height must be divisible by patch_size"
    assert W % patch_size==0, "Width must be divisible by patch_size"
    out = None
    ph, pw = patch_size #patch size dimensions H/d and W/d
    
    out = torch.zeros((N, (H*W)//(ph*pw),  (ph*pw)*C)).to(device=x.device)
    x = x.permute(0,2,3,1)

    
    
    patch_no = 0
    for i in range(0,H,ph):
        for j in range(0,W,pw):
            out[:,patch_no, :] = x[:, i:i+ph, j:j+pw,:].flatten(start_dim=1, end_dim=-1)
            patch_no+=1    
    
    return out


def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """

    cross_entropy = -torch.sum(labels * torch.log(scores + 1e-10), dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)

    return loss

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