import numpy as np
from typing import Tuple, Dict, Any


def apply_rotate_scale(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """通用旋转和尺度变换"""
    cfg = config['rotate_scale']
    theta = np.random.uniform(cfg['min_angle'], cfg['max_angle'])
    scale_factor = np.random.uniform(cfg['min_scale'], cfg['max_scale'])
    rot_mat = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    block[:, :3] = np.dot(block[:, :3], rot_mat) * scale_factor
    return block, labels, ids


def apply_flip_axis(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿 X/Y 轴翻转"""
    cfg = config['flip_axis']
    if np.random.rand() < cfg['prob_x']:
        block[:, 0] *= -1
    if np.random.rand() < cfg['prob_y']:
        block[:, 1] *= -1
    return block, labels, ids


def apply_random_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """通用随机抖动 """
    cfg = config['random_general_jittering']
    jittering = np.random.normal(loc=0., scale=cfg['scale'], size=(block.shape[0], 3)).astype(np.float32)
    jittering = np.clip(jittering, a_min=-cfg['max_clip'], a_max=cfg['max_clip'])
    block[:, :3] += jittering
    return block, labels, ids


def apply_random_drop_out(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机均匀删除"""
    cfg = config['random_drop_out']
    idxes = np.arange(block.shape[0])
    ratio = np.random.random() * (cfg['max_ratio'] - cfg['min_ratio']) + cfg['min_ratio']
    keep_count = int(ratio * block.shape[0])
    if keep_count == 0 and block.shape[0] > 0:
        keep_count = 1

    idxes = np.random.choice(idxes, keep_count, replace=False)
    return block[idxes], labels[idxes], ids[idxes]


def apply_add_noise_points(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """添加随机噪声点 """
    cfg = config['add_random_noise_points']
    ignore_label = config.get('ignore_label', 255)

    if block.shape[0] == 0:
        return block, labels, ids

    xmin, xmax = block[:, 0].min(), block[:, 0].max()
    ymin, ymax = block[:, 1].min(), block[:, 1].max()
    zmin, zmax = block[:, 2].min(), block[:, 2].max()
    imin, imax = block[:, 3].min(), block[:, 3].max()

    noise_num = int(np.random.random() * (cfg['max_num'] - cfg['min_num']) + cfg['min_num'])

    if noise_num == 0:
        return block, labels, ids

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


def apply_physical_attenuation_model(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PMA: 物理衰减模型，模拟强度衰减和概率丢点。
    """
    cfg = config['physical_attenuation_model']

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    coords = block[:, :3]
    distances = np.linalg.norm(coords, axis=1)

    alpha_min = cfg.get('alpha_min', 0.005)
    alpha_max = cfg.get('alpha_max', 0.05)

    # 随机采样消光系数 alpha
    alpha = np.random.uniform(alpha_min, alpha_max)

    # 1. 强度衰减: I' = I * exp(-2 * alpha * R)
    attenuation_factor = np.exp(-2 * alpha * distances)
    block[:, 3] *= attenuation_factor

    # 2. 概率丢点: P_drop = 1 - exp(-2 * alpha * R)
    # 计算保留概率 P_keep = exp(-2 * alpha * R)
    keep_prob = np.exp(-2 * alpha * distances)

    random_samples = np.random.rand(block.shape[0])
    keep_mask = random_samples <= keep_prob

    # 应用丢点
    block = block[keep_mask]
    labels = labels[keep_mask]
    ids = ids[keep_mask]

    return block, labels, ids


def apply_selective_range_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        RJ: 全局范围抖动 (Global Range Jittering)。
        对点云中所有点应用距离依赖的抖动，并进行裁剪。
        这个函数是从原来的 apply_selective_range_jittering 修改而来。
        """
    # 注意: 如果您在 YAML 中保留了 'selective_range_jittering' 的名字，这里仍然使用它
    cfg = config['selective_range_jittering']

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    coords = block[:, :3]
    distances = np.linalg.norm(coords, axis=1)

    # ----------------------------------------------------------------
    # --- 关键修改：移除 1. 区域选择 (Selection Mask) 逻辑 ---
    # 我们将 mask_angles 设为 True，使 Range Jittering 应用于所有点
    # ----------------------------------------------------------------

    # --- 2. 应用 Range Jittering (RJ) 到所有点 ---
    # 由于是全局应用，selected_coords 和 selected_distances 就是 coords 和 distances
    selected_coords = coords
    selected_distances = distances

    # 如果没有点，直接返回（虽然已经被 if block.shape[0] == 0 捕获，但保留严谨性）
    if selected_coords.shape[0] == 0:
        return block, labels, ids

    # 保留距离依赖的 std 计算 (这是您的特色)
    base_std = cfg.get('jitter_base_std', 0.01)
    dist_factor = cfg.get('jitter_dist_factor', 0.0005)

    # 计算距离依赖的抖动强度 (std)
    jitter_std = base_std + dist_factor * selected_distances

    # 生成噪声 (delta_R)
    delta_R = np.random.randn(selected_coords.shape[0]) * jitter_std

    # --- 3. 裁剪 (Clipping) ---
    clip_range = cfg.get('clip_range', [-0.05, 0.05])
    delta_R = np.clip(delta_R, clip_range[0], clip_range[1])

    # 4. 计算新的距离 R'
    new_R = selected_distances + delta_R

    # 避免 R < 0
    new_R = np.maximum(new_R, 0.01)

    # 5. 转换回 XYZ (保持角度不变)
    # 避免除以零
    selected_distances[selected_distances == 0] = 1e-6

    # 计算缩放因子，将所有点的距离从 R 变为 R'
    scaling_factor = new_R / selected_distances
    selected_coords *= scaling_factor[:, np.newaxis]

    block[:, :3] = selected_coords

    return block, labels, ids



def apply_intensity_channel_distortion(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                       config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ICD: 强度通道扰动，削弱模型对强度的依赖。
    """
    cfg = config['intensity_channel_distortion']

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    intensity = block[:, 3]

    gamma_prob = cfg.get('gamma_prob', 0.5)

    if np.random.rand() < gamma_prob:
        # 1. Gamma 校正
        gamma_min = cfg.get('gamma_min', 0.7)
        gamma_max = cfg.get('gamma_max', 1.3)
        gamma = np.random.uniform(gamma_min, gamma_max)

        intensity = np.power(intensity, gamma)
    else:
        # 2. 对比度/亮度扰动 (可选，这里简化为随机加性噪声)
        # 模拟强度信噪比下降，可以考虑对低强度点施加更高噪声
        noise_std = 0.05  # 经验值
        intensity += np.random.randn(intensity.shape[0]) * noise_std

    # 裁剪和更新
    # 假设强度范围是 [0, 1] 或类似的归一化范围
    intensity = np.clip(intensity, 0, intensity.max())
    block[:, 3] = intensity

    return block, labels, ids


things_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 13, 17, 18]

import numpy as np
from typing import Dict, Any, Tuple

things_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 13, 17, 18]

def _voxelize_coords(coords: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Returns integer voxel indices for coords (N,3).
    """
    #
    return np.floor(coords / voxel_size).astype(np.int32)


def _compute_voxel_occupancies(coords: np.ndarray, voxel_size: float) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Return a dict mapping voxel tuple -> indices list and per-point voxel-id index.
    """
    v = _voxelize_coords(coords, voxel_size)
    # create single integer key per voxel for dictionary hashing
    # Ensure keys are hashable strings
    keys = [f"{a}_{b}_{c}" for a, b, c in v]
    voxel2idx = {}
    point_voxel_key = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        point_voxel_key[i] = k
        if k not in voxel2idx:
            voxel2idx[k] = []
        voxel2idx[k].append(i)
    return voxel2idx, point_voxel_key


# ---------------------------------------------------------------------
# 1) Geometry-consistent Selective Jitter (GSJ) - Optimized
# ---------------------------------------------------------------------
def apply_geometry_selective_jitter(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Geometry-aware selective jitter (GSJ).
    Applies jitter along the local surface normal direction to preserve geometry.
    NOTE: The final radial-scale clamp is removed to retain the normal-guided effect.
    """
    cfg = config.get('geometry_selective_jitter', {})
    p_frame = cfg.get('prob', 0.5)  # frame-level apply prob
    jitter_prob = cfg.get('jitter_prob', 0.25)  # per-point candidate prob
    base_std = cfg.get('base_std', 0.008)  # meters
    dist_factor = cfg.get('dist_factor', 0.0006)
    cluster_voxel = cfg.get('cluster_voxel_size', 2.0)
    rng = np.random

    if block is None or block.shape[0] == 0:
        return block, labels, ids

    if rng.rand() > p_frame:
        return block, labels, ids

    coords = block[:, :3].copy().astype(np.float32)
    N = coords.shape[0]

    # --- 1) Build coarse voxels and compute per-voxel normals (PCA) ---
    voxel2idx, point_voxel_key = _compute_voxel_occupancies(coords, voxel_size=cluster_voxel)
    normals = np.zeros((N, 3), dtype=np.float32)

    for key, idx_list in voxel2idx.items():
        idxs = np.array(idx_list, dtype=np.int32)
        if idxs.size < 3:
            # Not enough points for PCA, assume flat surface (Z-axis is up)
            normals[idxs] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            continue

        pts = coords[idxs]
        centroid = pts.mean(axis=0)
        cov = (pts - centroid).T @ (pts - centroid)

        try:
            # SVD: vt[-1] is the eigenvector corresponding to the smallest eigenvalue (normal)
            _, s, vt = np.linalg.svd(cov)
            normal = vt[-1]
            # Normalize and assign
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            normals[idxs] = normal.astype(np.float32)
        except Exception as e:
            # Fallback for numerical instability
            # logging.warning(f"SVD failed: {e}. Defaulting to Z-normal.")
            normals[idxs] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # --- 2) Select candidate points to jitter ---
    select_mask = (rng.rand(N) < jitter_prob)
    if select_mask.sum() == 0:
        return block, labels, ids

    # --- 3) Correlated Jitter applied along Normals (The core GSJ step) ---
    sel_idxs = np.where(select_mask)[0]
    sel_voxel_keys = [point_voxel_key[i] for i in sel_idxs]
    vox2sel = {}

    for local_idx, vk in enumerate(sel_voxel_keys):
        if vk not in vox2sel:
            vox2sel[vk] = []
        vox2sel[vk].append(sel_idxs[local_idx])

    for vk, g_indices in vox2sel.items():
        g_indices = np.array(g_indices, dtype=np.int32)

        # Calculate distance-dependent sigma for the cluster
        mean_r = np.linalg.norm(coords[g_indices], axis=1).mean() if g_indices.size > 0 else 0.0
        sigma = base_std + dist_factor * mean_r

        # Sample scalar offset ONCE per cluster (correlated jitter)
        scalar = rng.randn() * sigma
        normals_cluster = normals[g_indices]

        # Apply offset along normal direction
        coords[g_indices] = coords[g_indices] + normals_cluster * scalar

    # --- 4) Radial-scale clamp (REMOVED) ---
    # The final block update uses the coords modified by GSJ directly.
    block_out = block.copy()
    block_out[:, :3] = coords.astype(np.float32)

    return block_out, labels, ids


def _compute_structure_strength(coords: np.ndarray, cluster_voxel: float = 2.0) -> np.ndarray:
    """
    Computes a 'structure strength' proxy (inverse of the smallest eigenvalue from PCA)
    for all points based on coarse voxel groups.
    A higher value means the point belongs to a strong planar or linear structure (high protection).
    """
    N = coords.shape[0]
    # Smallest eigenvalue (lambda_0) is proportional to curvature. Low lambda_0 means flat/linear structure.
    lambda0 = np.ones(N, dtype=np.float32) * 1e-4  # Initialize with small value

    voxel2idx, _ = _compute_voxel_occupancies(coords, voxel_size=cluster_voxel)

    for key, idx_list in voxel2idx.items():
        idxs = np.array(idx_list, dtype=np.int32)
        if idxs.size < 5:  # Need at least 5 points for a stable PCA estimate
            continue

        pts = coords[idxs]
        centroid = pts.mean(axis=0)
        cov = (pts - centroid).T @ (pts - centroid)

        try:
            # SVD: s contains the square roots of the eigenvalues (or eigenvalues if using np.linalg.eig)
            # np.linalg.svd returns singular values, which relate to eigenvalues (s^2 = lambda)
            u, s, vt = np.linalg.svd(cov)

            # Eigenvalues are related to singular values squared: lambda = s^2
            eigenvalues = np.sort(s ** 2)

            # Smallest eigenvalue (lambda_0) is the measure of flatness/linearity
            # We use it directly: Small lambda_0 means high structure
            lambda0[idxs] = eigenvalues[0].astype(np.float32)

        except Exception:
            # Fallback for numerical instability
            continue

    # 归一化 lambda0 (曲率)
    # 使用 max(lambda0) 而不是 max(lambda0[lambda0 > 0]) 来避免极值
    lambda0_norm = lambda0 / (np.max(lambda0) + 1e-8)

    # 结构强度 (Structure Strength): 1 - Curvature_norm
    # 结构强度越高 (接近1)，则曲率越低 (越平面/边缘)，需要保护
    structure_strength = 1.0 - lambda0_norm

    # 保证强度在 [0, 1] 范围内
    return np.clip(structure_strength, 0.0, 1.0)


# ---------------------------------------------------------------------
# 2) Semantic Aware Point Drop (SAPD) Function
# ---------------------------------------------------------------------

def apply_semantic_aware_point_drop(
        block: np.ndarray,
        labels: np.ndarray,
        ids: np.ndarray,
        config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    语义结构感知丢点 (SAPD)。
    结合了距离偏置丢点 (DBPD) 和结构支撑保护。
    """
    cfg = config.get('semantic_aware_point_drop', {})
    rng = np.random

    # 框架级参数
    p_frame = cfg.get('prob', 0.2)
    base_drop = cfg.get('base_drop', 0.04)
    max_drop = cfg.get('max_drop', 0.2)
    density_voxel_size = cfg.get('density_voxel_size', 2.0)
    gamma = cfg.get('gamma', 0.3)
    r_scale = cfg.get('r_scale', 60.0)

    # 保护参数
    protect_classes = cfg.get('protect_small_classes', [])
    protect_scale = cfg.get('protect_scale', 0.5)
    # 新增：几何保护因子 (Structure Protection Factor)
    geo_protect_factor = cfg.get('geo_protect_factor', 0.7)  # 结构越强，丢点概率降低的程度

    if rng.rand() > p_frame or block.shape[0] == 0:
        return block, labels, ids

    coords = block[:, :3].copy()
    N = coords.shape[0]

    # --- Step 1: 距离偏置丢点概率 (Distance-Biased Drop Prob) ---
    R = np.linalg.norm(coords, axis=1)
    # 基于距离的权重 (远距离点权重高)
    dist_weight = np.clip(R / r_scale, 0.0, 1.0)

    # 基础丢点概率: P_drop_base = base + (max - base) * W_dist
    drop_prob = base_drop + (max_drop - base_drop) * dist_weight

    # --- Step 2: 几何结构感知保护 (Geometry-Aware Protection) ---
    # 计算点的结构强度 (Structure Strength: 0=噪声/弱结构, 1=平面/边缘)
    structure_strength = _compute_structure_strength(coords, cluster_voxel=density_voxel_size)

    # 结构感知保护因子: 结构越强 (接近1)，保护因子越小 (丢点概率被更多地降低)
    # P_drop_geo = P_drop_base * [1 - geo_protect_factor * Structure_Strength]
    # 例: 结构强度=1, geo_protect_factor=0.7 -> P_drop_geo = P_drop_base * 0.3 (丢点概率降至30%)
    # 例: 结构强度=0, geo_protect_factor=0.7 -> P_drop_geo = P_drop_base * 1.0 (不保护)
    protection_mask = 1.0 - geo_protect_factor * structure_strength
    drop_prob *= protection_mask

    # --- Step 3: 小物体类别保护 (Small Class Protection) ---
    if labels is not None and protect_classes:
        small_mask = np.isin(labels, np.array(protect_classes, dtype=labels.dtype))
        drop_prob[small_mask] *= protect_scale

    # --- Step 4: 随机采样丢弃 ---
    keep_mask = rng.rand(N) > drop_prob

    # 保证至少保留一个点，防止空帧
    if keep_mask.sum() == 0:
        keep_mask[rng.randint(0, N)] = True

    # 应用 mask
    block_out = block[keep_mask]
    labels_out = labels[keep_mask]
    ids_out = ids[keep_mask]

    return block_out, labels_out, ids_out


def apply_distance_biased_point_drop(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Distance-biased point drop: far & sparse regions more likely to be dropped.
    Same signature as other apply_* functions.
    """
    cfg = config.get('distance_biased_point_drop', {})
    p_frame = cfg.get('prob', 0.5)
    base_drop = cfg.get('base_drop', 0.04)
    max_drop = cfg.get('max_drop', 0.5)
    r_scale = cfg.get('r_scale', 60.0)       # meters (controls distance effect)
    gamma = cfg.get('gamma', 0.3)           # weight for density effect
    density_voxel = cfg.get('density_voxel_size', 2.0)  # voxel size for density estimation (meters)
    protect_small_classes = cfg.get('protect_small_classes', None)  # list or None
    ignore_label = config.get('ignore_label', 255)
    rng = np.random

    if block is None or block.shape[0] == 0:
        return block, labels, ids

    if rng.rand() > p_frame:
        return block, labels, ids

    coords = block[:, :3].astype(np.float32)
    N = coords.shape[0]

    # distance-based weight (sigmoid-like mapping)
    dists = np.linalg.norm(coords, axis=1)
    dist_norm = dists / max(r_scale, 1e-6)
    # smooth mapping in [0,1]
    dist_weight = 1.0 / (1.0 + np.exp(- (dist_norm - 0.5) * 6.0))

    # density estimation via voxel occupancy (coarse)
    voxel2idx, _ = _compute_voxel_occupancies(coords, voxel_size=density_voxel)
    # per-point density estimate: number of points in that voxel
    densities = np.zeros(N, dtype=np.float32)
    for vk, idxs in voxel2idx.items():
        count = len(idxs)
        densities[idxs] = count
    # normalize density to [0,1] (higher means denser)
    if densities.max() > densities.min():
        densities = (densities - densities.min()) / (densities.max() - densities.min())
    else:
        densities = np.ones_like(densities) * 0.5

    # combine into drop probability
    drop_prob = base_drop + (max_drop - base_drop) * dist_weight
    drop_prob = np.clip(drop_prob + gamma * (1.0 - densities), 0.0, 1.0)

    # if labels given and protect_small_classes requested, reduce drop prob for those points
    if labels is not None and protect_small_classes:
        small_mask = np.isin(labels, np.array(protect_small_classes, dtype=labels.dtype))
        # reduce drop prob (protect) for small classes
        drop_prob[small_mask] *= cfg.get('protect_scale', 0.5)  # keep 50% of their original drop_prob

    keep_mask = rng.rand(N) > drop_prob
    if keep_mask.sum() == 0:
        # ensure at least one point kept
        keep_mask[rng.randint(0, N)] = True

    block_out = block[keep_mask]
    labels_out = labels[keep_mask] if labels is not None else labels
    ids_out = ids[keep_mask] if ids is not None else ids

    return block_out, labels_out, ids_out


def apply_intensity_jitter(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add random noise to intensity to simulate rain/snow reflection variation
    """
    cfg = config.get('intensity_jitter', {})
    p_frame = cfg.get('prob', 0.5)
    scale = cfg.get('scale', 0.1)
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > p_frame:
        return block, labels, ids

    block_out = block.copy()
    block_out[:, 3] += rng.uniform(-scale, scale, size=block_out.shape[0])
    block_out[:, 3] = np.clip(block_out[:, 3], 0.0, 1.0)
    return block_out, labels, ids


def apply_occlusion_patch(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        模拟局部遮挡（Occlusion） 的数据增强方法。它会在点云中随机选几个小区域，把这些区域里的点全部删除，从而模拟真实场景中可能出现的遮挡情况，比如行人被车挡住、物体部分被遮挡、或者 LiDAR 扫描不到某些区域。
    """
    cfg = config.get('occlusion_patch', {})
    p_frame = cfg.get('prob', 0.5)
    patch_size = cfg.get('patch_size', 2.0)
    num_patches = cfg.get('num_patches', 1)
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > p_frame:
        return block, labels, ids

    coords = block[:, :3]
    keep_mask = np.ones(coords.shape[0], dtype=bool)
    for _ in range(num_patches):
        center = coords[rng.randint(0, coords.shape[0])]
        # drop points inside box
        mask = np.all(np.abs(coords - center) < patch_size / 2, axis=1)
        keep_mask[mask] = False

    if keep_mask.sum() == 0:
        keep_mask[rng.randint(0, coords.shape[0])] = True

    block_out = block[keep_mask]
    labels_out = labels[keep_mask] if labels is not None else labels
    ids_out = ids[keep_mask] if ids is not None else ids
    return block_out, labels_out, ids_out

def angle_in_range(phi, low, high):
    """Checks if angle phi is in range [low, high], handling the +/- pi wrap-around."""
    diff = (phi - low) % (2 * np.pi)
    range_width = (high - low) % (2 * np.pi)
    return diff <= range_width

def apply_nonuniform_region_perturbation(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                         config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NRP: 非均匀区域扰动增强 (Non-uniform Region Perturbation)
    包含 DSP (深度选择性扰动) 和 SSP (扫描线选择性扰动).
    """
    cfg = config.get('nonuniform_region_perturbation', {})
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > cfg.get('prob', 1.0):
        return block, labels, ids

    coords = block[:, :3].copy()
    N = coords.shape[0]

    # --------------------------------------------------------
    # (1) 深度选择性扰动 (DSP)
    # --------------------------------------------------------
    dsp_cfg = cfg.get('depth_selective_perturbation', {})
    if rng.rand() < dsp_cfg.get('prob', 0.5):
        dists = np.linalg.norm(coords, axis=1)

        # 动态选择深度区间
        d_min_cfg, d_max_cfg = dsp_cfg.get('d_range', [20.0, 80.0])
        # 随机选择一个漂移区间 [d_l, d_h]
        d_l = rng.uniform(d_min_cfg, d_max_cfg - 10)
        d_h = rng.uniform(d_l + 5, d_max_cfg)

        # 选取在该深度区间内的点
        depth_mask = (dists >= d_l) & (dists <= d_h)
        sel_coords = coords[depth_mask]

        if sel_coords.shape[0] > 0:
            sel_dists = dists[depth_mask]  # 这是一个长度为 M 的向量
            base_std = dsp_cfg.get('base_std', 0.01)
            dist_factor = dsp_cfg.get('dist_factor', 0.0001)

            # 计算深度依赖的 sigma (远距离扰动更大)
            sigma_vector = base_std + dist_factor * sel_dists
            epsilon = rng.normal(loc=0., scale=sigma_vector[:, None], size=sel_coords.shape).astype(np.float32)

            coords[depth_mask] += epsilon

    # --------------------------------------------------------
    # (2) 扫描线选择性扰动 (SSP)
    # --------------------------------------------------------
    ssp_cfg = cfg.get('scanline_selective_perturbation', {})
    if rng.rand() < ssp_cfg.get('prob', 0.3):

        # 转换为极坐标 (计算角度)
        rho = np.linalg.norm(coords[:, :2], axis=1)
        phi = np.arctan2(coords[:, 1], coords[:, 0])  # 角度

        num_segments = ssp_cfg.get('num_segments', 3)
        angle_width = ssp_cfg.get('angle_width', np.pi / 36)  # 5度

        for _ in range(num_segments):
            # 随机选择一个中心角度
            theta_c = rng.uniform(-np.pi, np.pi)
            theta_l = theta_c - angle_width / 2
            theta_h = theta_c + angle_width / 2

            # 选取在该角度区间内的点 (环绕处理)
            angle_mask = angle_in_range(phi, theta_l, theta_h)

            sel_coords = coords[angle_mask]

            if sel_coords.shape[0] > 0:
                ssp_std = ssp_cfg.get('ssp_std', 0.02)
                # 施加高斯扰动
                epsilon = rng.normal(loc=0., scale=ssp_std, size=sel_coords.shape).astype(np.float32)
                coords[angle_mask] += epsilon

    block[:, :3] = coords
    return block, labels, ids


def apply_depth_adaptive_sparsity_augmentation(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                               config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DSA: 深度自适应点稀疏增强 (Depth-adaptive Sparsity Augmentation)
    包含 D-Drop (深度自适应点丢失) 和 Clustered-Drop (局部簇状点丢失).
    """
    cfg = config.get('depth_adaptive_sparsity_augmentation', {})
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > cfg.get('prob', 1.0):
        return block, labels, ids

    coords = block[:, :3].copy()
    current_labels = labels.copy()
    current_ids = ids.copy()
    N = coords.shape[0]

    # 初始化总的保留掩码 (True表示保留)
    keep_mask = np.ones(N, dtype=bool)

    # --------------------------------------------------------
    # (1) 深度自适应点丢失 (D-Drop)
    # --------------------------------------------------------
    ddrop_cfg = cfg.get('depth_aware_drop', {})
    if rng.rand() < ddrop_cfg.get('prob', 0.6):
        dists = np.linalg.norm(coords, axis=1)
        d_min = ddrop_cfg.get('d_min', 0.5)
        d_max = ddrop_cfg.get('d_max', 100.0)  # 假设最大感知距离
        alpha = ddrop_cfg.get('alpha', 0.4)  # 最大丢弃概率系数

        # 归一化距离 [0, 1]
        dist_norm = (dists - d_min) / (d_max - d_min + 1e-6)
        dist_norm = np.clip(dist_norm, 0.0, 1.0)

        # 深度自适应丢弃概率 p_drop(d)
        p_drop = alpha * dist_norm

        # 仅对 D-Drop 产生的点进行丢弃
        ddrop_mask = rng.rand(N) > p_drop
        keep_mask &= ddrop_mask  # 整合到总掩码

    # --------------------------------------------------------
    # (2) 局部簇状点丢失 (Clustered-Drop)
    # --------------------------------------------------------
    cdrop_cfg = cfg.get('clustered_drop', {})
    if rng.rand() < cdrop_cfg.get('prob', 0.4):
        num_clusters = cdrop_cfg.get('num_clusters', 2)
        radius = cdrop_cfg.get('radius', 1.5)

        for _ in range(num_clusters):
            # 随机选择一个中心点 p_c
            if keep_mask.sum() == 0:
                break

            active_coords = coords[keep_mask]
            center_idx_in_active = rng.randint(0, active_coords.shape[0])
            p_c = active_coords[center_idx_in_active]

            # 计算距离并创建新的丢弃掩码
            dists_to_center = np.linalg.norm(coords - p_c, axis=1)
            # cluster_drop_mask: True表示要丢弃 (即 keep_mask 设为 False)
            cluster_drop_mask = dists_to_center < radius

            # 将簇状丢弃应用到总保留掩码
            keep_mask[cluster_drop_mask] = False

    # --------------------------------------------------------
    # (3) 应用总保留掩码并返回
    # --------------------------------------------------------
    block_original = block.copy()
    labels_original = labels.copy()
    ids_original = ids.copy()

    # 确保至少保留一个点
    if keep_mask.sum() == 0:
        keep_mask[rng.randint(0, N)] = True
    N = block_original.shape[0]
    MIN_RATIO_THRESHOLD = 0.05
    N_kept = keep_mask.sum()
    if N_kept / N < MIN_RATIO_THRESHOLD:
        return block_original, labels_original, ids_original
    block_out = block[keep_mask]
    labels_out = current_labels[keep_mask]
    ids_out = current_ids[keep_mask]

    return block_out, labels_out, ids_out