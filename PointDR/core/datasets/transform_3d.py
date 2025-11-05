import numpy as np
from typing import Tuple, Dict, Any
import open3d as o3d


def calculate_local_curvature(block: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
    """
    通过 Open3D 实现 K近邻搜索，然后使用标准的 NumPy 逐点计算局部曲率代理。
    """
    N = block.shape[0]
    points_xyz = block[:, :3].astype('float32')

    if N < k_neighbors:
        return np.zeros(N, dtype=np.float32)

    curvatures = np.zeros(N, dtype=np.float32)

    # 1. Open3D: 构建点云对象和 KDTree，加速 KNN 搜索
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 2. Open3D: 搜索所有点的邻居索引
    all_indices = []
    for i in range(N):
        [k, indices, _] = pcd_tree.search_knn_vector_3d(points_xyz[i], k_neighbors)
        # 将 Open3D 返回的索引列表转换为 NumPy 数组
        all_indices.append(np.asarray(indices, dtype=np.int64))

    # 转换为 Numba/FAISS 版本中使用的二维 NumPy 数组结构
    all_indices = np.array(all_indices, dtype=np.int64)

    # 3. Python 循环: 逐点计算 PCA 和曲率
    for i in range(N):
        neighbor_indices = all_indices[i]
        neighbor_pts = points_xyz[neighbor_indices, :]

        if neighbor_pts.shape[0] < 3:
            curvatures[i] = 0.0
            continue

        # 质心和中心化
        centroid = np.mean(neighbor_pts, axis=0)
        centered_pts = neighbor_pts - centroid

        # 协方差矩阵
        if neighbor_pts.shape[0] > 1:
            cov_matrix = np.cov(centered_pts, rowvar=False)
        else:
            curvatures[i] = 0.0
            continue

        # 特征值分解
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 从大到小排序

        # 计算曲率代理
        sum_eigenvalues = np.sum(eigenvalues)
        if sum_eigenvalues > 1e-6:
            curvatures[i] = eigenvalues[2] / sum_eigenvalues
        else:
            curvatures[i] = 0.0

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
    """CL-RA S-TRPD: 语义-目标保留删减"""
    if block.shape[0] == 0: return block, labels, ids

    cfg = config['semantic_targeted_point_drop']
    thing_class_ids = np.array(cfg['thing_class_ids'], dtype=labels.dtype)

    R = np.linalg.norm(block[:, :3], axis=1)
    is_thing = np.isin(labels, thing_class_ids)
    drop_probabilities = np.zeros(block.shape[0], dtype=np.float32)

    P_base_stuff = cfg['max_drop_prob_stuff'] * 0.1
    drop_probabilities[~is_thing] = np.clip(
        P_base_stuff + cfg['depth_decay_factor'] * R[~is_thing], a_min=0, a_max=cfg['max_drop_prob_stuff']
    )
    drop_probabilities[is_thing] = cfg['max_drop_prob_things']

    points_to_keep_mask = np.random.rand(block.shape[0]) > drop_probabilities

    return block[points_to_keep_mask], labels[points_to_keep_mask], ids[points_to_keep_mask]


def apply_controlled_structure_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                         config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CL-RA CSRJ/RD-HJ: 结构分解和异构抖动组合"""
    if block.shape[0] < 10: return block, labels, ids

    cfg = config['controlled_structure_jittering']
    curvatures = calculate_local_curvature(block, k_neighbors=cfg['k_neighbors'])

    N = block.shape[0]
    R = np.linalg.norm(block[:, :3], axis=1)

    # 1. 向量化计算 sigma_final
    sigma_r_base = cfg['sigma_base'] * (1 + cfg['r_gradient_factor'] * R ** 2)

    curvature_factor = np.ones_like(curvatures)
    high_curvature_mask = curvatures > cfg['curvature_threshold']

    # 仅对高曲率点应用加权因子
    curvature_factor[high_curvature_mask] = 1.0 + cfg['alpha_edge'] * (
            curvatures[high_curvature_mask] - cfg['curvature_threshold']
    )

    sigma_final = sigma_r_base * curvature_factor

    # 2. 向量化生成噪声 (利用 NumPy 的广播功能)
    sigma_xy = sigma_final
    sigma_z = sigma_final * cfg['z_sensitivity_factor']

    jitter_x = np.random.normal(loc=0., scale=sigma_xy).astype(np.float32)
    jitter_y = np.random.normal(loc=0., scale=sigma_xy).astype(np.float32)
    jitter_z = np.random.normal(loc=0., scale=sigma_z).astype(np.float32)

    jitter_xyz = np.stack((jitter_x, jitter_y, jitter_z), axis=1)

    block[:, :3] += jitter_xyz
    return block, labels, ids