import os

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.logging import logger

__all__ = ['SemanticLearnerKITTI']

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class SemanticLearnerKITTI(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        logger.info("SemanticKITTI  Learner")

        if submit_to_server:
            super().__init__({
                'train':
                    SemanticLearnerKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          submit=True),
                'test':
                    SemanticLearnerKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='test')
            })
        else:
            super().__init__({
                'train':
                    SemanticLearnerKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          google_mode=google_mode),
                'test':
                    SemanticLearnerKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=sample_stride,
                                          split='val')
            })


class SemanticLearnerKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1,
                 submit=False,
                 google_mode=True):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.seqs = []
        if split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]
            if self.google_mode or trainval:
                self.seqs.append('08')
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 1. 读取原始点云和标签
        with open(self.files[index], 'rb') as f:
            block_ = np.fromfile(f, dtype=np.float32).reshape(-1, 4)

        orig_indices = np.arange(block_.shape[0])

        # 标签处理
        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as f:
                all_labels = np.fromfile(f, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block_.shape[0], dtype=np.int32)
        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        # 转换到体素坐标 (尚未去重)
        pc_original = np.round(block_[:, :3] / self.voxel_size).astype(np.int32)

        # 2. 随机采样点 (点级别降采样)
        if 'train' in self.split and len(block_) > self.num_points:
            idxes = np.random.choice(len(block_), self.num_points, replace=False)

            pc_original = pc_original[idxes]
            block_ = block_[idxes]
            labels_ = labels_[idxes]
            orig_indices = orig_indices[idxes]

        # 归一化体素坐标
        pc_original -= pc_original.min(0, keepdims=1)

        _, inds_clean, _ = sparse_quantize(
            pc_original,
            return_index=True,
            return_inverse=True
        )

        ratio_clean = np.random.random() * 0.2 + 0.8
        if 'train' in self.split and len(inds_clean) > int(self.num_points * ratio_clean):
            # inds_clean 是体素索引，对其进行二次采样
            inds_clean = np.random.choice(inds_clean,
                                          int(self.num_points * ratio_clean),
                                          replace=False)

        inverse_map_clean_F = orig_indices[inds_clean]

        pc_voxel_clean = pc_original[inds_clean]
        feat_voxel_clean = block_[inds_clean]

        lidar_clean = SparseTensor(feat_voxel_clean, pc_voxel_clean)
        # 逆映射 F 字段长度现在等于 pc_voxel_clean 的长度 (N_voxel)
        inverse_map_clean_st = SparseTensor(inverse_map_clean_F[:, np.newaxis], pc_voxel_clean)

        # 4. 数据增强 (在采样的 block_ 上进行)
        block_aug = block_.copy()
        labels_aug = labels_.copy()
        orig_indices_aug = orig_indices.copy()

        # a. 随机 dropout
        if 'train' in self.split and np.random.rand() < 0.5:
            keep_ratio = np.random.uniform(0.8, 1.0)
            keep_idx = np.random.choice(np.arange(block_aug.shape[0]),
                                        int(block_aug.shape[0] * keep_ratio),
                                        replace=False)
            block_aug = block_aug[keep_idx]
            labels_aug = labels_aug[keep_idx]
            orig_indices_aug = orig_indices_aug[keep_idx]

            # b. 加噪声
        if 'train' in self.split and np.random.rand() < 0.5:
            xmin, xmax = block_aug[:, 0].min(), block_aug[:, 0].max()
            ymin, ymax = block_aug[:, 1].min(), block_aug[:, 1].max()
            zmin, zmax = block_aug[:, 2].min(), block_aug[:, 2].max()
            imin, imax = block_aug[:, 3].min(), block_aug[:, 3].max()
            noise_num = int(np.random.rand() * 2000)
            noise = np.stack([
                np.random.uniform(xmin, xmax, noise_num),
                np.random.uniform(ymin, ymax, noise_num),
                np.random.uniform(zmin, zmax, noise_num),
                np.random.normal((imin + imax) / 2, 0.5, noise_num)
            ], axis=1).astype(np.float32)
            noise_labels = np.ones(noise.shape[0], dtype=np.int64) * 255

            # 噪声点没有原始点索引，用 -1 占位
            noise_indices = np.ones(noise.shape[0], dtype=orig_indices_aug.dtype) * -1
            orig_indices_aug = np.concatenate([orig_indices_aug, noise_indices], axis=0)

            block_aug = np.concatenate([block_aug, noise], axis=0)
            labels_aug = np.concatenate([labels_aug, noise_labels], axis=0)

        # c. rotate + scale
        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
            block_aug[:, :3] = block_aug[:, :3].dot(rot_mat.T) * scale

        # d. flip X/Y
        if 'train' in self.split and np.random.rand() < 0.5:
            block_aug[:, 0] *= -1
        if 'train' in self.split and np.random.rand() < 0.5:
            block_aug[:, 1] *= -1

        # e. jitter
        if 'train' in self.split and np.random.rand() < 0.5:
            jitter = np.random.normal(0, 0.01, (block_aug.shape[0], 3))
            block_aug[:, :3] += np.clip(jitter, -0.05, 0.05)

        # 5. voxelization & sparse tensor (Augmented Data)
        pc_aug = np.round(block_aug[:, :3] / self.voxel_size).astype(np.int32)
        pc_aug -= pc_aug.min(0, keepdims=1)  # 归一化体素坐标

        _, inds_aug, _ = sparse_quantize(
            pc_aug,
            return_index=True,
            return_inverse=True
        )

        ratio_aug = np.random.random() * 0.2 + 0.8
        if 'train' in self.split and len(inds_aug) > int(self.num_points * ratio_aug):
            inds_aug = np.random.choice(inds_aug,
                                        int(self.num_points * ratio_aug),
                                        replace=False)

        inverse_map_aug_F = orig_indices_aug[inds_aug]

        pc_voxel_aug = pc_aug[inds_aug]
        feat_voxel_aug = block_aug[inds_aug]
        labels_voxel_aug = labels_aug[inds_aug]

        lidar_aug = SparseTensor(feat_voxel_aug, pc_voxel_aug)
        labels_aug_st = SparseTensor(labels_voxel_aug, pc_voxel_aug)

        # 逆映射 F 字段的长度现在等于 pc_voxel_aug 的长度 (N_voxel)
        inverse_map_aug_st = SparseTensor(inverse_map_aug_F[:, np.newaxis], pc_voxel_aug)

        # targets_mapped (使用所有点)
        targets_mapped = SparseTensor(labels_aug, pc_aug)

        return {
            'lidar': lidar_aug,
            'targets': labels_aug_st,
            'targets_mapped': targets_mapped,
            'bawa_clean': lidar_clean,
            'inverse_map': inverse_map_aug_st,
            'inverse_map_clean': inverse_map_clean_st,
            'file_name': self.files[index]
        }
    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
