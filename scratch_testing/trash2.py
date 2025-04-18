import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# === CONFIGURATION ===
sequence_file = '/home/klingjac/DeepRob/final_project/3DPW/sequenceFiles/sequenceFiles/test/downtown_crossStreets_00.pkl'
image_root = '/home/klingjac/DeepRob/final_project/3DPW/imageFiles/imageFiles'
frame_idx = 100
# Try all actors and auto-pick match
actor_idx = None

# === LOAD SEQUENCE ===
with open(sequence_file, 'rb') as f:
    seq = pickle.load(f, encoding='latin1')

seq_name = seq['sequence']
img_id = seq['img_frame_ids'][frame_idx]
img_path = os.path.join(image_root, seq_name, f'image_{img_id:05d}.jpg')

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(img)

# === AUTO-MATCH ACTOR ===
poses2d = seq['poses2d']
num_actors = len(poses2d)
best_match = None
lowest_dist = float('inf')
kps2d_final = None
joints3d_final = None

for a_idx in range(num_actors):
    if frame_idx >= len(poses2d[a_idx]):
        continue
    kps2d = np.array(poses2d[a_idx][frame_idx])
    if kps2d.shape[0] == 3:
        kps2d = kps2d.T
    if kps2d.shape[0] > 17:
        kps2d = kps2d[:17]

    # project joints3d to image space
    joints3d = np.array(seq['jointPositions'][a_idx][frame_idx]).reshape(-1, 3)
    joints3d_h = np.concatenate([joints3d, np.ones((joints3d.shape[0], 1))], axis=1).T
    cam_pose = np.array(seq['cam_poses'][frame_idx])
    cam_T_world = np.linalg.inv(cam_pose)
    joints_cam = cam_T_world @ joints3d_h
    joints_cam = joints_cam[:3, :]
    joints_img = cam_K @ joints_cam
    joints_img = joints_img[:2, :] / joints_img[2:3, :]

    # distance between projected 3D and 2D keypoints
    dist = np.linalg.norm(joints_img[:, :17].T - kps2d[:, :2], axis=1).mean()
    if dist < lowest_dist:
        best_match = a_idx
        lowest_dist = dist
        kps2d_final = kps2d
        joints3d_final = joints_img

print(f"Best actor match: {best_match} with avg 2D–3D dist = {lowest_dist:.2f}")

# === PLOT FINAL MATCH ===
for x, y, v in kps2d_final:
    if v > 0:
        plt.plot(x, y, 'ro', markersize=4, label='2D keypoint')
if kps2d.shape[0] == 3:
    kps2d = kps2d.T
if kps2d.shape[0] > 17:
    kps2d = kps2d[:17]

for x, y, v in kps2d:
    if v > 0:
        plt.plot(x, y, 'ro', markersize=4, label='2D keypoint')

# === PROJECT 3D JOINTS TO 2D ===
joints3d = np.array(seq['jointPositions'][actor_idx][frame_idx]).reshape(-1, 3)
cam_K = np.array(seq['cam_intrinsics'])  # shape: (3, 3)
cam_pose = np.array(seq['cam_poses'][frame_idx])  # (4, 4)

# convert joints to homogeneous coords
joints3d_h = np.concatenate([joints3d, np.ones((joints3d.shape[0], 1))], axis=1).T
# transform to camera frame
cam_T_world = np.linalg.inv(cam_pose)
joints_cam = cam_T_world @ joints3d_h  # (4, N)
joints_cam = joints_cam[:3, :]

# project to image plane
joints_img = cam_K @ joints_cam
joints_img = joints_img[:2, :] / joints_img[2:3, :]

# plot projected 3D joints
for i in range(min(17, joints3d_final.shape[1])):
    x, y = joints3d_final[:, i]
    plt.plot(x, y, 'bs', markersize=4, label='Projected 3D' if i == 0 else "")

plt.title(f"Frame {frame_idx} | 2D vs Projected 3D Joints (Auto-matched Actor {best_match})")
plt.axis('off')
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()
