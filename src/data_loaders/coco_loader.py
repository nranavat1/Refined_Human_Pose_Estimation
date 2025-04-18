import os
import json
import math
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter

class TopDownCocoDataset(Dataset):
    def __init__(self, ann_file, img_prefix, image_size=(192, 256), heatmap_size=(48, 64), sigma=2,
                 use_udp=True, is_train=True, transform=None, normalize=True):
        """
        Top-down pose estimation dataset for COCO-style annotations.

        Args:
            ann_file (str): Path to annotation file.
            img_prefix (str): Directory with image files.
            image_size (tuple): Size to resize input images to.
            heatmap_size (tuple): Size of the output heatmaps.
            sigma (int): Standard deviation for the Gaussian target.
            use_udp (bool): Whether to apply unbiased data processing.
            is_train (bool): Flag for training mode (activates augmentations).
            transform (callable): Optional transform to apply to image.
            normalize (bool): Whether to normalize the image using ImageNet stats.
        """
        self.coco = COCO(ann_file)
        self.img_prefix = img_prefix
        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.sigma = sigma
        self.use_udp = use_udp
        self.is_train = is_train
        self.transform = transform
        self.normalize = normalize

        self.img_ids = self.coco.getImgIds(catIds=self.coco.getCatIds(catNms=['person']))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        anns = [a for a in anns if a['num_keypoints'] > 0]

        if len(anns) == 0:
            return self.__getitem__((idx + 1) % len(self))

        ann = max(anns, key=lambda a: a['num_keypoints'])
        img_info = self.coco.loadImgs(img_id)[0]

        image_path = os.path.join(self.img_prefix, img_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        bbox = ann['bbox']
        center, scale = self.bbox_to_center_scale(bbox, self.image_size)

        # Random scale/rotation augmentation (MMPose-style)
        if self.is_train:
            scale_factor = 0.25
            rot_factor = 30
            sf = scale_factor
            rf = rot_factor

            scale *= np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0
        else:
            rotation = 0

        trans = self.get_affine_transform(center, scale, self.image_size)
        input_img = cv2.warpAffine(image, trans, tuple(self.image_size.astype(int)))

        if self.transform:
            input_img = self.transform(input_img)
        else:
            input_img = input_img.astype(np.float32) / 255.0
            if self.normalize:
                input_img = (input_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            input_img = input_img.transpose(2, 0, 1)

        joints = keypoints[:, :2].copy()
        joints_vis = keypoints[:, 2].copy()
        for i in range(len(joints)):
            if joints_vis[i] > 0:
                joints[i] = self.affine_transform(joints[i], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        meta = {
            'image_file': image_path,
            'center': center,
            'scale': scale,
            'rotation': rotation,
            'joints_3d': joints,
            'joints_3d_visible': joints_vis,
            'bbox_score': 1.0,
        }

        return {
            'image': torch.from_numpy(input_img).float(),
            'target': torch.from_numpy(target).float(),
            'target_weight': torch.from_numpy(target_weight).float(),
            'meta': meta
        }

    # USAGE EXAMPLE:
    # To load training data with augmentations:
    # train_dataset = TopDownCocoDataset(
    #     ann_file='data/coco/annotations/person_keypoints_train2017.json',
    #     img_prefix='data/coco/train2017/',
    #     is_train=True
    # )

    # To load validation data:
    # val_dataset = TopDownCocoDataset(
    #     ann_file='data/coco/annotations/person_keypoints_val2017.json',
    #     img_prefix='data/coco/val2017/',
    #     is_train=False
    # )

    # Then wrap with DataLoader:
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)

    def bbox_to_center_scale(self, bbox, image_size):
        x, y, w, h = bbox
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        aspect_ratio = image_size[0] / image_size[1]
        pixel_std = 200
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32)
        return center, scale

    def get_affine_transform(self, center, scale, output_size):
        src_w = scale[0] * 200
        dst_w, dst_h = output_size

        src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])

        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def get_3rd_point(self, a, b):
        direction = a - b
        return b + np.array([-direction[1], direction[0]], dtype=np.float32)

    def affine_transform(self, pt, trans):
        new_pt = np.array([pt[0], pt[1], 1.])
        new_pt = np.dot(trans, new_pt)
        return new_pt[:2]

    def generate_target(self, joints, joints_vis):
        num_joints = joints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        for joint_id in range(num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            if joints_vis[joint_id] <= 0 or mu_x < 0 or mu_y < 0 \
                    or mu_x >= self.heatmap_size[0] or mu_y >= self.heatmap_size[1]:
                target_weight[joint_id] = 0
                continue

            tmp_size = self.sigma * 3
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]

            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
