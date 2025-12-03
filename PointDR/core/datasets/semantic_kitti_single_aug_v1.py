import os
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

__all__ = ['SingleAugSemanticV1KITTI']

from PointDR.core.datasets.settings import kept_labels, label_name_mapping
from PointDR.core.datasets.transform_3d_v1 import apply_rotate_scale, \
    apply_random_jittering, apply_random_drop_out, apply_add_noise_points, \
    apply_flip_axis, apply_intensity_channel_distortion, apply_physical_attenuation_model, \
    apply_selective_range_jittering,  apply_geometry_selective_jitter, \
    apply_distance_biased_point_drop, apply_intensity_jitter, apply_occlusion_patch, apply_semantic_aware_point_drop, \
    _compute_structure_strength, apply_depth_adaptive_sparsity_augmentation, \
    apply_nonuniform_region_perturbation

AUG_MAP = {
    'rotate_scale': apply_rotate_scale,
    'flip_axis': apply_flip_axis,
    'random_general_jittering': apply_random_jittering,
    'random_drop_out': apply_random_drop_out,
    'add_random_noise_points': apply_add_noise_points,
    'intensity_channel_distortion': apply_intensity_channel_distortion,
    'physical_attenuation_model': apply_physical_attenuation_model,
    'selective_range_jittering': apply_selective_range_jittering,
    'geometry_selective_jitter': apply_geometry_selective_jitter,
    'distance_biased_point_drop': apply_distance_biased_point_drop,
    'intensity_jitter': apply_intensity_jitter,
    'occlusion_patch': apply_occlusion_patch,
    'semantic_aware_point_drop': apply_semantic_aware_point_drop,
    'depth_adaptive_sparsity_augmentation': apply_depth_adaptive_sparsity_augmentation,
    'nonuniform_region_perturbation': apply_nonuniform_region_perturbation,
}


class AugmentationPipeline:

    def __init__(self, strong_steps: list, ignore_label: int = 255):
        self.strong_steps = strong_steps
        self.ignore_label = ignore_label

    def run_pipeline(self, block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, steps: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 构造传递给增强函数的全局配置
        pipeline_config = {'ignore_label': self.ignore_label}

        for step in steps:
            aug_name = None
            aug_prob = 1.0
            method_config = {}

            # 1. 解析配置格式 (处理 name: param 和 method: {param} 两种格式)
            if 'name' in step:
                aug_name = step['name']
                aug_prob = step.get('prob', 1.0)
                method_config = step

            elif isinstance(step, dict) and len(step) == 1:
                aug_name = list(step.keys())[0]
                method_config = step[aug_name]
                aug_prob = method_config.get('prob', 1.0)

            if not aug_name or aug_name not in AUG_MAP:
                continue

            # 2. 执行增强
            if np.random.rand() < aug_prob:
                # full_config 包含了全局配置和当前方法的参数
                full_config = pipeline_config.copy()
                full_config[aug_name] = method_config

                # 传递并接收更新后的 strength 数组
                block, labels, ids, strength = AUG_MAP[aug_name](block, labels, ids, strength, full_config)

        return block, labels, ids, strength


class SingleAugSemanticV1KITTI(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        sample_stride = kwargs.get('sample_stride', 1)
        weak_aug = kwargs.get('weak_aug', [])
        strong_aug = kwargs.get('strong_aug', [])

        super().__init__({
            'train': SingleAugSemanticKITTIInternalV1(root, voxel_size, num_points, strong_aug, sample_stride=1, split='train'),
            'test': SingleAugSemanticKITTIInternalV1(root, voxel_size, num_points, strong_aug, sample_stride=sample_stride, split='val')
        })


class SingleAugSemanticKITTIInternalV1:

    def __init__(self, root, voxel_size, num_points, strong_aug, sample_stride=1, split='train'):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride

        self.strong_aug = strong_aug
        self.pipeline = AugmentationPipeline(strong_aug, ignore_label=255)

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

    def return_aug_single_views(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)

        ids = np.arange(block_.shape[0])

        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block_.shape[0]).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        # 1. 在增强前计算初始结构强度
        # 使用一个合理的 voxel size (2.0m) 来计算粗粒度结构
        strength_ = _compute_structure_strength(block_[:, :3], cluster_voxel=2.0)

        # 2. 运行增强管道，传入并接收 strength_
        block_, labels_, ids, strength_ = self.pipeline.run_pipeline(
            block_.copy(),
            labels_.copy(),
            ids.copy(),
            strength_.copy(), # 传入结构强度
            self.pipeline.strong_steps,
        )

        pc_ = np.round(block_[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=True)

        _, inds, inverse_map = sparse_quantize(pc_, return_index=True, return_inverse=True)

        if len(inds) > self.num_points:
            inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = block_[inds]
        labels = labels_[inds]
        strength_subsampled = strength_[inds].astype(np.float32)


        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        structure_strength = SparseTensor(strength_subsampled, pc)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index],
            'structure_strength':structure_strength
        }

    def __getitem__(self, index):
        return self.return_aug_single_views(index)

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)