import os

import numpy as np
import torch
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

        logger.info("SKT Learner")

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

        labels_raw = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        coords_float = block_[:, :3].copy()

        feats_float = block_.copy()
        labels_raw_float = labels_raw.copy()

        pc_quantized = np.round(coords_float / self.voxel_size).astype(np.int32)
        pc_quantized -= pc_quantized.min(0, keepdims=1)  # 最小平移

        pc, inds, inverse_map = sparse_quantize(pc_quantized, return_index=True, return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc_sampled = pc_quantized[inds]
        feat_sampled = feats_float[inds]
        labels_sampled = labels_raw_float[inds]

        lidar = SparseTensor(feat_sampled, pc_sampled)
        labels = SparseTensor(labels_sampled, pc_sampled)

        labels_full_quantized = labels_raw_float[inverse_map]
        labels_full_quantized[inverse_map] = labels_raw_float
        labels_ = SparseTensor(labels_full_quantized, pc_quantized)

        inverse_map = SparseTensor(inverse_map, pc_quantized)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,

            'raw_coords': feats_float[:, :3].copy(),
            'raw_feats': feats_float.copy(),
            'raw_labels': labels_raw_float.copy(),

            'file_name': self.files[index]
        }


    @staticmethod
    def collate_fn(inputs):
        """
                批处理函数，需处理新增的原始浮点数据。
                """
        # 提取 SparseTensor 列表
        sparse_inputs = []
        for key in ['lidar', 'targets', 'targets_mapped', 'inverse_map']:
            sparse_inputs.append([data[key] for data in inputs])

        # 提取原始浮点数据和文件名
        raw_coords = [data['raw_coords'] for data in inputs]
        raw_feats = [data['raw_feats'] for data in inputs]
        raw_labels = [data['raw_labels'] for data in inputs]
        file_names = [data['file_name'] for data in inputs]

        # 使用 TorchSparse 的 collate_fn 批处理 SparseTensor
        sparse_collated = sparse_collate_fn(sparse_inputs)

        # 构造最终的输出字典
        output_dict = {
            'lidar': sparse_collated[0],
            'targets': sparse_collated[1],
            'targets_mapped': sparse_collated[2],
            'inverse_map': sparse_collated[3],

            'raw_coords': [torch.from_numpy(c).float() for c in raw_coords],
            'raw_feats': [torch.from_numpy(f).float() for f in raw_feats],
            'raw_labels': [torch.from_numpy(l).long() for l in raw_labels],

            'file_name': file_names
        }

        return output_dict
