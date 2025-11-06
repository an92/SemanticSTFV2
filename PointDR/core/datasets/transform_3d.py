import numpy as np
from typing import Tuple, Dict, Any
from numba import njit
import faiss
import sys

@njit(cache=True)
def _numba_pca_curvature(points_xyz: np.ndarray, all_indices: np.ndarray, curvatures: np.ndarray):
    N = points_xyz.shape[0]
    k = all_indices.shape[1]

    for i in range(N):
        neighbor_indices = all_indices[i]

        # 1. 提取邻居点
        neighbor_pts = np.empty((k, 3), dtype=points_xyz.dtype)
        for j in range(k):
            neighbor_pts[j] = points_xyz[neighbor_indices[j]]

        if k < 3:
            curvatures[i] = 0.0
            continue

        # 2. 质心和中心化
        centroid = np.sum(neighbor_pts, axis=0) / k
        centered_pts = neighbor_pts - centroid

        # 3. 协方差矩阵 (其余逻辑保持不变，因为它们是 Numba 支持的矩阵运算)
        if k > 1:
            cov_matrix = centered_pts.T @ centered_pts / (k - 1)
        else:
            curvatures[i] = 0.0
            continue

        # 4. 特征值分解
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        # 5. 排序: 从大到小
        eigenvalues = np.sort(eigenvalues)[::-1]

        # 6. 计算曲率代理
        sum_eigenvalues = eigenvalues[0] + eigenvalues[1] + eigenvalues[2]
        if sum_eigenvalues > 1e-6:
            curvatures[i] = eigenvalues[2] / sum_eigenvalues
        else:
            curvatures[i] = 0.0

    return curvatures


def calculate_local_curvature(block: np.ndarray, k_neighbors: int = 15) -> np.ndarray:
    """使用 Faiss 进行批量 KNN 搜索，Numba 计算曲率。"""
    N = block.shape[0]
    points_xyz = block[:, :3].astype('float32')
    if N < k_neighbors or 'faiss' not in sys.modules:
        return np.zeros(N, dtype=np.float32)
    curvatures = np.zeros(N, dtype=np.float32)
    try:
        points_xyz_faiss = np.ascontiguousarray(points_xyz)
        index = faiss.IndexFlatL2(3)
        index.add(points_xyz_faiss)
        _, all_indices = index.search(points_xyz_faiss, k_neighbors)
        all_indices = all_indices.astype(np.int64)
    except Exception as e:
        print(f"Faiss KNN search failed. Proceeding with zeros: {e}", file=sys.stderr)
        return np.zeros(N, dtype=np.float32)
    curvatures = _numba_pca_curvature(points_xyz, all_indices, curvatures)
    return curvatures.astype(np.float32)


def apply_rotate_scale(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """通用旋转和尺度变换"""
    cfg = config['rotate_scale']
    theta = np.random.uniform(cfg['min_angle'], cfg['max_angle'])
    scale_factor = np.random.uniform(cfg['min_scale'], cfg['max_scale'])
    rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                        [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    block[:, :3] = np.dot(block[:, :3], rot_mat) * scale_factor
    return block, labels, ids


def apply_flip_axis(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """沿 X/Y 轴翻转"""
    cfg = config['flip_axis']
    if np.random.rand() < cfg['prob_x']: block[:, 0] *= -1
    if np.random.rand() < cfg['prob_y']: block[:, 1] *= -1
    return block, labels, ids


def apply_random_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """通用随机抖动 (旧 Aug6)"""
    cfg = config['random_general_jittering']
    jittering = np.random.normal(loc=0., scale=cfg['scale'], size=(block.shape[0], 3)).astype(np.float32)
    jittering = np.clip(jittering, a_min=-cfg['max_clip'], a_max=cfg['max_clip'])
    block[:, :3] += jittering
    return block, labels, ids


def apply_random_drop_out(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """随机均匀删除"""
    cfg = config['random_drop_out']
    idxes = np.arange(block.shape[0])
    ratio = np.random.random() * (cfg['max_ratio'] - cfg['min_ratio']) + cfg['min_ratio']
    keep_count = int(ratio * block.shape[0])
    if keep_count == 0 and block.shape[0] > 0: keep_count = 1

    idxes = np.random.choice(idxes, keep_count, replace=False)
    return block[idxes], labels[idxes], ids[idxes]


def apply_add_noise_points(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """添加随机噪声点 """
    cfg = config['add_random_noise_points']
    ignore_label = config.get('ignore_label', 255)

    if block.shape[0] == 0: return block, labels, ids

    xmin, xmax = block[:, 0].min(), block[:, 0].max()
    ymin, ymax = block[:, 1].min(), block[:, 1].max()
    zmin, zmax = block[:, 2].min(), block[:, 2].max()
    imin, imax = block[:, 3].min(), block[:, 3].max()

    noise_num = int(np.random.random() * (cfg['max_num'] - cfg['min_num']) + cfg['min_num'])

    if noise_num == 0: return block, labels, ids

    noise_x = np.random.uniform(xmin, xmax, noise_num).astype(np.float32)
    noise_y = np.random.uniform(ymin, ymax, noise_num).astype(np.float32)
    noise_z = np.random.uniform(zmin, zmax, noise_num).astype(np.float32)
    noise_i = np.random.normal(loc=(imin + imax) / 2, scale=cfg['intensity_scale'], size=noise_num).astype(np.float32)

    noise = np.stack((noise_x, noise_y, noise_z, noise_i), axis=1)

    noise_labels = np.ones(noise.shape[0], dtype=labels.dtype) * ignore_label
    noise_ids = np.ones(noise.shape[0], dtype=ids.dtype) * (-1)

    block = np.concatenate((block, noise), axis=0)
    labels = np.concatenate((labels, noise_labels), axis=0)
    ids = np.concatenate((ids, noise_ids), axis=0)

    return block, labels, ids


def apply_semantic_targeted_point_drop(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                       config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CL-RA LPD: 语义-目标保留删减（使用精简参数）。"""
    if block.shape[0] == 0: return block, labels, ids

    cfg = config['semantic_targeted_point_drop']
    thing_class_ids = np.array(cfg.get('thing_class_ids', []), dtype=labels.dtype)
    base_drop_prob = cfg.get('base_drop_prob', 0.4)  # Stuff 类基础概率 (调优为 0.4)
    thing_drop_ratio = cfg.get('thing_drop_ratio', 0.01)  # Thing 类相对概率 (调优为 0.01)

    is_thing = np.isin(labels, thing_class_ids)
    drop_probabilities = np.zeros(block.shape[0], dtype=np.float32)

    # Stuff 类 (背景) 应用基础删除概率
    drop_probabilities[~is_thing] = base_drop_prob

    # Thing 类 (目标) 应用低得多的删除概率
    drop_probabilities[is_thing] = base_drop_prob * thing_drop_ratio

    # 确保概率在 [0, 1] 范围内
    drop_probabilities = np.clip(drop_probabilities, 0.0, 1.0)

    points_to_keep_mask = np.random.rand(block.shape[0]) > drop_probabilities

    return block[points_to_keep_mask], labels[points_to_keep_mask], ids[points_to_keep_mask]

def apply_controlled_structure_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                         config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CL-RA SJ: 结构分解和异构抖动组合（使用优化后的曲率计算）。"""
    if block.shape[0] < 10: return block, labels, ids

    cfg = config['controlled_structure_jittering']
    # 核心：调用高性能的曲率计算
    curvatures = calculate_local_curvature(block, k_neighbors=cfg.get('k_neighbors', 15))

    N = block.shape[0]

    # 获取精简后的参数
    sigma_flat = cfg.get('sigma_flat', 0.001)
    edge_factor = cfg.get('edge_factor', 2.0)
    curvature_threshold = cfg.get('curvature_threshold', 0.1)

    # 1. 计算 sigma_final (不再依赖复杂的 r_gradient_factor)
    sigma_base = sigma_flat

    curvature_factor = np.ones_like(curvatures)
    high_curvature_mask = curvatures > curvature_threshold

    # 对高曲率点应用边缘放大因子
    curvature_factor[high_curvature_mask] = edge_factor * (
            curvatures[high_curvature_mask] / curvature_threshold  # 比例缩放，增强边缘效果
    )
    # 确保边缘因子不至于过大
    curvature_factor = np.clip(curvature_factor, 1.0, edge_factor * 2)

    sigma_final = sigma_base * curvature_factor

    # 2. 生成噪声 (使用相同的 sigma 简化 Z 轴差异)
    sigma_xyz = sigma_final  # 保持XYZ轴抖动一致

    jitter_x = np.random.normal(loc=0., scale=sigma_xyz).astype(np.float32)
    jitter_y = np.random.normal(loc=0., scale=sigma_xyz).astype(np.float32)
    jitter_z = np.random.normal(loc=0., scale=sigma_xyz).astype(np.float32)

    jitter_xyz = np.stack((jitter_x, jitter_y, jitter_z), axis=1)

    block[:, :3] += jitter_xyz
    return block, labels, ids