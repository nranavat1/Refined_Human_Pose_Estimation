import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from coco_loader import TopDownCocoDataset

def visualize_all_heatmaps(dataset):
    sample = dataset[20]
    image = sample['image'].numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]) #un-normalize from pre-processing
    image = np.clip(image, 0, 1)

    joints = sample['meta']['joints_3d']
    joints_vis = sample['meta']['joints_3d_visible']
    heatmaps = sample['target'].numpy()

    print(f"Cropped image size: {image.shape[1]} x {image.shape[0]}")  # width x height

    # Plot keypoints over the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    for (x, y), v in zip(joints, joints_vis):
        if v > 0:
            plt.plot(x, y, 'ro', markersize=4)
    plt.title("COCO val2017 keypoints")
    plt.axis('off')
    plt.show()

    # Plot all heatmaps in a grid
    num_joints = heatmaps.shape[0]
    cols = 6
    rows = int(np.ceil(num_joints / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axs[r, c] if rows > 1 else axs[c]

        if i < num_joints:
            ax.imshow(heatmaps[i], cmap='hot')
            ax.set_title(f"Joint {i}")
        ax.axis('off')

    plt.suptitle("Heatmaps for All Joints", fontsize=16)
    plt.tight_layout()
    plt.show()






if __name__ == '__main__':
    val_dataset = TopDownCocoDataset(
        ann_file='/home/klingjac/DeepRob/final_scratch/data/annotations_trainval2017/annotations/person_keypoints_val2017.json',
        img_prefix='/home/klingjac/DeepRob/final_scratch/data/val2017/val2017/',
        is_train=False
    )
    visualize_all_heatmaps(val_dataset)
