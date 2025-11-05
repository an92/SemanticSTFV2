import os
from typing import Dict, Any, Tuple

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.logging import logger

__all__ = ['SingleAugSemanticKITTI']

from PointDR.core.datasets.transform_3d import apply_rotate_scale, apply_semantic_targeted_point_drop, \
    apply_controlled_structure_jittering, apply_random_jittering, apply_random_drop_out, apply_add_noise_points, \
    apply_flip_axis

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
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain', 'person', 'bicyclist', 'motorcyclist',
    'fence', 'pole', 'traffic-sign'
]

AUG_MAP = {
    'rotate_scale': apply_rotate_scale,
    'flip_axis': apply_flip_axis,
    'semantic_targeted_point_drop': apply_semantic_targeted_point_drop,
    'controlled_structure_jittering': apply_controlled_structure_jittering,
    'random_general_jittering': apply_random_jittering,
    'random_drop_out': apply_random_drop_out,
    'add_random_noise_points': apply_add_noise_points,
}

class AugmentationPipeline:
    def __init__(self, weak_steps: list, strong_steps: list, ignore_label: int = 255):
        self.weak_steps = weak_steps
        self.strong_steps = strong_steps
        self.ignore_label = ignore_label


    def run_pipeline(self, block: np.ndarray, labels: np.ndarray, ids: np.ndarray, steps: list) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        # 构造传递给增强函数的全局配置
        # 全局配置只包含 ignore_label
        pipeline_config = {'ignore_label': self.ignore_label}

        for step in steps:
            aug_name = None
            aug_prob = 1.0
            method_config = {}

            # 1. 解析配置格式 (处理 name: param 和 method: {param} 两种格式)
            if 'name' in step:
                # 格式: - name: rotate_scale, prob: 1.0, min_angle: 0.0, ...
                aug_name = step['name']
                aug_prob = step.get('prob', 1.0)
                method_config = step

            elif isinstance(step, dict) and len(step) == 1:
                # 格式: - add_random_noise_points: {prob: 1.0, min_num: 1000, ...}
                aug_name = list(step.keys())[0]
                method_config = step[aug_name]
                aug_prob = method_config.get('prob', 1.0)

            if not aug_name or aug_name not in AUG_MAP:
                continue

            # 2. 执行增强
            if np.random.rand() < aug_prob:
                # full_config 包含了全局配置和当前方法的参数
                full_config = pipeline_config.copy()
                # 增强函数需要以 'aug_name' 为键的参数字典 (如 full_config['rotate_scale'] = {...})
                full_config[aug_name] = method_config

                # 确保当前步骤是独立于其他步骤执行的，并传递最新的 block, labels, ids
                block, labels, ids = AUG_MAP[aug_name](block, labels, ids, full_config)

        return block, labels, ids


class SingleAugSemanticKITTI(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        sample_stride = kwargs.get('sample_stride', 1)
        weak_aug = kwargs.get('weak_aug', [])
        strong_aug = kwargs.get('strong_aug', [])


        super().__init__({
            'train': SingleAugSemanticKITTIInternal(root, voxel_size, num_points, weak_aug, strong_aug, sample_stride=1, split='train'),
            'test': SingleAugSemanticKITTIInternal(root, voxel_size, num_points,  weak_aug, strong_aug, sample_stride=sample_stride, split='val')
        })


class SingleAugSemanticKITTIInternal:

    def __init__(self, root, voxel_size, num_points, weak_aug, strong_aug, sample_stride=1, split='train'):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride

        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.pipeline = AugmentationPipeline(weak_aug, strong_aug, ignore_label=255)

        self.seqs = []
        if split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [os.path.join(self.root, seq, 'velodyne', x) for x in seq_files]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-', '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[label_name_mapping[label_id]] = cnt
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

    def return_double_views(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        # assign an id for each point for consistency
        ids = np.arange(block_.shape[0])
        # read labels
        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block_.shape[0]).astype(np.int32)
        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        # >>> Weak Augmented View (Anchor View) <<<
        block_1, labels_1_, ids_1_ = self.pipeline.run_pipeline(block_.copy(), labels_.copy(), ids.copy(), self.pipeline.weak_steps)

        # Voxelization for View 1
        pc_1_ = np.round(block_1[:, :3] / self.voxel_size).astype(np.int32)
        pc_1_ -= pc_1_.min(0, keepdims=1)

        feat_1_ = block_1
        _, inds_1, inverse_map = sparse_quantize(pc_1_, return_index=True, return_inverse=True)
        if len(inds_1) > self.num_points:
            inds_1 = np.random.choice(inds_1, self.num_points, replace=False)

        pc_1 = pc_1_[inds_1]
        feat_1 = feat_1_[inds_1]
        labels_1 = labels_1_[inds_1]
        ids_1 = ids_1_[inds_1]
        lidar_1 = SparseTensor(feat_1, pc_1)
        labels_1 = SparseTensor(labels_1, pc_1)
        ids_1 = SparseTensor(ids_1, pc_1)
        inverse_map = SparseTensor(inverse_map, pc_1_)

        # >>> Strong Augmented View (Positive View) <<<
        block_2, labels_2_, ids_2_ = self.pipeline.run_pipeline(block_.copy(), labels_.copy(), ids.copy(), self.pipeline.strong_steps)

        feat_2_ = block_2
        pc_2_ = np.round(block_2[:, :3] / self.voxel_size).astype(np.int32)
        pc_2_ -= pc_2_.min(0, keepdims=1)

        _, inds_2, _ = sparse_quantize(pc_2_, return_index=True, return_inverse=True)

        if len(inds_2) > self.num_points:
            inds_2 = np.random.choice(inds_2, self.num_points, replace=False)

        pc_2 = pc_2_[inds_2]
        labels_2 = labels_2_[inds_2]
        feat_2 = feat_2_[inds_2]
        ids_2 = ids_2_[inds_2]

        lidar_2 = SparseTensor(feat_2, pc_2)
        labels_2 = SparseTensor(labels_2, pc_2)
        ids_2 = SparseTensor(ids_2, pc_2)

        return {
            'lidar': lidar_1,
            'targets': labels_1,
            'inverse_map_dense': inverse_map,
            'file_name': self.files[index],
            'ids_1': ids_1,
            'lidar_2': lidar_2,
            'ids_2': ids_2,
            'targets_2': labels_2,
        }

    def return_single_view(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        # read labels
        pc_ = np.round(block_[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(pc_.shape[0]).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        _, inds, inverse_map = sparse_quantize(pc_, return_index=True, return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = block_[inds]
        labels = labels_[inds]

        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {'lidar': lidar, 'targets': labels, 'targets_mapped': labels_, 'inverse_map': inverse_map, 'file_name': self.files[index]}

    def __getitem__(self, index):
        # return double views for contrastive learning
        if self.split in ['val', 'test']:
            return self.return_single_view(index)
        else:
            return self.return_double_views(index)

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
