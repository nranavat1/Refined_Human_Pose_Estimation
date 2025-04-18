import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
sequence_file = '/home/klingjac/DeepRob/final_project/3DPW/sequenceFiles/sequenceFiles/test/downtown_bus_00.pkl'
image_root = '/home/klingjac/DeepRob/final_project/3DPW/imageFiles/imageFiles'
actor_idx = 0  # actor to visualize

# === LOAD SEQUENCE ===
with open(sequence_file, 'rb') as f:
    seq = pickle.load(f, encoding='latin1')

seq_name = seq['sequence']
img_frame_ids = seq['img_frame_ids']
poses2d = seq['poses2d']

# === LOOP THROUGH FRAMES WHERE 2D KEYPOINTS EXIST ===
for frame_idx in range(min(100, len(poses2d[actor_idx]))):
    img_id = img_frame_ids[frame_idx]
    img_path = os.path.join(image_root, seq_name, f'image_{img_id:05d}.jpg')

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 10))
    plt.imshow(img)
    kp = np.array(poses2d[actor_idx][frame_idx])

    if kp.shape == (3, 18):
        kp = kp.T  # (18, 3)
    else:
        print(f"Unexpected shape for keypoints at frame {frame_idx}: {kp.shape}")
        continue

    for x, y, v in kp:
        if v > 0:
            plt.plot(x, y, 'ro', markersize=3)

    plt.title(f"Frame {frame_idx} | Image ID {img_id} | Sequence: {seq_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
