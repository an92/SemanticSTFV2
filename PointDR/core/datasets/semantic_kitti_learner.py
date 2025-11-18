import os

import numpy as np
from scipy import ndimage
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.logging import logger

__all__ = ['SemanticLearnerKITTI']

from PointDR.core.datasets.settings import kept_labels, label_name_mapping


class SemanticLearnerKITTI(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)
        self.augment = kwargs.get('args', {})

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

    def _occlude_components(self, block, pc_full_vox, comps_full, keep_prob=0.6, max_occlude_ratio=0.4):
        """
        完全向量化的组件遮挡 (component occlusion)
        block: 原始点云 (N, 4)
        pc_full_vox: voxelized 坐标 (N, 3)
        comps_full: voxel connected component id (N,)
        """
        unique_comps = np.unique(comps_full)
        # 随机选择要遮挡的 component
        choose_mask = np.random.rand(len(unique_comps)) < 0.2
        chosen_comps = unique_comps[choose_mask]
        if chosen_comps.size == 0:
            return block, None

        # 找到需要删除的点
        rm_mask_global = np.isin(comps_full, chosen_comps)
        if rm_mask_global.sum() == 0:
            return block, None

        # 按距离排序，保留近距离点
        pts_to_remove = np.where(rm_mask_global)[0]
        ranges = np.linalg.norm(block[pts_to_remove, :3], axis=1)
        order = np.argsort(-ranges)  # 远距离优先删除
        num_rm = int(np.ceil(max_occlude_ratio * len(pts_to_remove)))
        rm_idx = pts_to_remove[order[:num_rm]]

        # 根据 keep_prob 再随机丢弃
        final_mask = np.zeros(block.shape[0], dtype=bool)
        keep_random = np.random.rand(len(rm_idx)) > keep_prob
        final_mask[rm_idx[keep_random]] = True

        if final_mask.sum() > 0:
            block = block[~final_mask]

        return block, final_mask if final_mask.sum() > 0 else None

    def _attenuate_by_range(self, block, min_keep=0.6, max_keep=0.95):
        pts = block[:, :3]
        ranges = np.linalg.norm(pts, axis=1)
        rmin, rmax = ranges.min(), ranges.max()
        if rmax - rmin < 1e-6:
            return np.ones(block.shape[0], dtype=bool)
        norm_r = (ranges - rmin) / (rmax - rmin)
        keep_probs = max_keep - (max_keep - min_keep) * norm_r
        keep_mask = np.random.rand(block.shape[0]) < keep_probs
        return keep_mask

    def _scatter_noise(self, block, labels, ids, num_noise_factor=0.01):
        """
        向点云中添加噪声点（一次性向量化生成）
        """
        N = block.shape[0]
        num_noise = max(0, int(N * num_noise_factor))
        if num_noise == 0:
            return block, labels, ids

        # 随机选择点中心
        idx = np.random.choice(N, num_noise, replace=True)
        centers = block[idx, :3]

        # 随机扰动 jitter
        jitter = np.random.normal(scale=0.02, size=(num_noise, 3)).astype(np.float32)

        # 激光强度噪声
        noise_i = np.random.normal(loc=np.mean(block[:, 3]), scale=0.5, size=(num_noise, 1)).astype(np.float32)

        noise_pts = np.hstack([centers + jitter, noise_i])

        block = np.vstack([block, noise_pts])
        labels = np.concatenate([labels, np.ones(num_noise, dtype=np.int64) * 255])
        ids = np.concatenate([ids, -np.ones(num_noise, dtype=np.int64)])

        return block, labels, ids

    def _compute_components(self, pc_full):
        """
        输入:
            pc_full: (N,3) int32 voxelized point coordinates
        输出:
            comps: (N,) 每个 voxel 的 connected component id
        """
        # 构建稀疏占据体素 grid
        coords = pc_full - pc_full.min(0)  # 保证坐标从 0 开始
        shape = coords.max(0) + 1

        # 构建稀疏 voxel occupancy grid
        grid = np.zeros(shape, dtype=np.int32)
        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

        # 使用 6 邻域连通域标记
        structure = np.zeros((3, 3, 3), dtype=int)
        structure[1, 1, 0] = structure[1, 1, 2] = 1
        structure[1, 0, 1] = structure[1, 2, 1] = 1
        structure[0, 1, 1] = structure[2, 1, 1] = 1

        labeled, num_features = ndimage.label(grid, structure=structure)

        # 将 voxel label 映射回每个点
        comps = labeled[coords[:, 0], coords[:, 1], coords[:, 2]]

        return comps

    def __getitem__(self, index):
        # ---------------- Load raw point cloud ----------------
        with open(self.files[index], 'rb') as f:
            block = np.fromfile(f, dtype=np.float32).reshape(-1, 4)  # (N,4)
        ids = np.arange(block.shape[0])

        # Load labels
        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as f:
                all_labels = np.fromfile(f, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block.shape[0], dtype=np.int32)
        labels = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        # ---------------- Original view ----------------
        block_1 = block.copy()
        theta = np.random.uniform(0, 2 * np.pi)
        scale = np.random.uniform(0.95, 1.05)
        rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                            [-np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
        block_1[:, :3] = block_1[:, :3] @ rot_mat.T * scale

        pc_1_ = np.round(block_1[:, :3] / self.voxel_size).astype(np.int32)
        pc_1_ -= pc_1_.min(0, keepdims=1)
        feat_1_ = block_1.copy()
        labels_1_ = labels.copy()
        ids_1_ = ids.copy()
        _, inds_1, inverse_map = sparse_quantize(pc_1_, return_index=True, return_inverse=True)

        # 采样固定点数
        if len(inds_1) > self.num_points:
            inds_1 = np.random.choice(inds_1, self.num_points, replace=False)
        elif len(inds_1) < self.num_points:
            inds_1 = np.random.choice(inds_1, self.num_points, replace=True)

        pc_1 = pc_1_[inds_1]
        feat_1 = feat_1_[inds_1]
        labels_1 = labels_1_[inds_1]
        ids_1 = ids_1_[inds_1]
        lidar_1 = SparseTensor(feat_1, pc_1)
        labels_1 = SparseTensor(labels_1, pc_1)
        ids_1 = SparseTensor(ids_1, pc_1)
        inverse_map = SparseTensor(inverse_map, pc_1_)

        # ---------------- Augmented view ----------------
        block_2 = block.copy()
        labels_2 = labels.copy()
        ids_2 = ids.copy()

        # 1. compute voxel components
        pc_full_vox = np.round(block_2[:, :3] / self.voxel_size).astype(np.int32)
        pc_full_vox -= pc_full_vox.min(0, keepdims=True)
        comps_full = self._compute_components(pc_full_vox)

        # 2. component occlusion
        block_2, mask_occlude = self._occlude_components(block_2, pc_full_vox, comps_full)
        if mask_occlude is not None:
            labels_2 = labels_2[~mask_occlude]
            ids_2 = ids_2[~mask_occlude]

        # 3. attenuation by range
        if np.random.rand() < 0.7:
            keep_mask = self._attenuate_by_range(block_2)
            block_2 = block_2[keep_mask]
            labels_2 = labels_2[keep_mask]
            ids_2 = ids_2[keep_mask]

        # 4. scatter noise
        if np.random.rand() < 0.5:
            block_2, labels_2, ids_2 = self._scatter_noise(block_2, labels_2, ids_2)

        # 5. random dropout
        if np.random.rand() < 0.5:
            ratio = np.random.uniform(0.8, 1.0)
            idxes = np.random.choice(block_2.shape[0], int(ratio * block_2.shape[0]), replace=False)
            block_2 = block_2[idxes]
            labels_2 = labels_2[idxes]
            ids_2 = ids_2[idxes]

        # 6. rotation, scaling, flip, jitter
        theta = np.random.uniform(0, 2 * np.pi)
        scale = np.random.uniform(0.95, 1.05)
        rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                            [-np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
        block_2[:, :3] = block_2[:, :3] @ rot_mat.T * scale
        if np.random.rand() < 0.5: block_2[:, 0] *= -1
        if np.random.rand() < 0.5: block_2[:, 1] *= -1
        if np.random.rand() < 0.5:
            jitter = np.clip(np.random.normal(0, 0.01, size=(block_2.shape[0], 3)), -0.05, 0.05)
            block_2[:, :3] += jitter

        # 7. voxelization & 固定采样
        feat_2_ = block_2
        pc_2_ = np.round(block_2[:, :3] / self.voxel_size).astype(np.int32)
        pc_2_ -= pc_2_.min(0, keepdims=1)
        _, inds_2, _ = sparse_quantize(pc_2_, return_index=True, return_inverse=True)

        if len(inds_2) > self.num_points:
            inds_2 = np.random.choice(inds_2, self.num_points, replace=False)
        elif len(inds_2) < self.num_points:
            inds_2 = np.random.choice(inds_2, self.num_points, replace=True)

        pc_2 = pc_2_[inds_2]
        feat_2 = feat_2_[inds_2]
        labels_2 = labels_2[inds_2]
        ids_2 = ids_2[inds_2]

        lidar_2 = SparseTensor(feat_2, pc_2)
        labels_2 = SparseTensor(labels_2, pc_2)
        ids_2 = SparseTensor(ids_2, pc_2)

        # 8. compute correspond_idx using KDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(block_1[:, :3])
        _, correspond_idx = tree.query(block_2[:, :3], k=1)

        return {
            'lidar': lidar_1,
            'targets': labels_1,
            'inverse_map_dense': inverse_map,
            'file_name': self.files[index],
            'ids_1': ids_1,
            'lidar_2': lidar_2,
            'ids_2': ids_2,
            'targets_2': labels_2,
            # 'correspond_idx': correspond_idx.astype(np.int32)
        }
    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
