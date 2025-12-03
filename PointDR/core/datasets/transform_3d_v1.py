import numpy as np
from typing import Tuple, Dict, Any

# 辅助函数，用于结构强度计算
def _voxelize_coords(coords: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Returns integer voxel indices for coords (N,3).
    """
    return np.floor(coords / voxel_size).astype(np.int32)


def _compute_voxel_occupancies(coords: np.ndarray, voxel_size: float) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Return a dict mapping voxel tuple -> indices list and per-point voxel-id index.
    """
    v = _voxelize_coords(coords, voxel_size)
    keys = [f"{a}_{b}_{c}" for a, b, c in v]
    voxel2idx = {}
    point_voxel_key = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        point_voxel_key[i] = k
        if k not in voxel2idx:
            voxel2idx[k] = []
        voxel2idx[k].append(i)
    return voxel2idx, point_voxel_key


def _compute_structure_strength(coords: np.ndarray, cluster_voxel: float = 2.0) -> np.ndarray:
    """
    Computes a 'structure strength' proxy (inverse of the smallest eigenvalue from PCA)
    for all points based on coarse voxel groups.
    A higher value means the point belongs to a strong planar or linear structure (high protection).
    """
    N = coords.shape[0]
    lambda0 = np.ones(N, dtype=np.float32) * 1e-4

    voxel2idx, _ = _compute_voxel_occupancies(coords, voxel_size=cluster_voxel)

    for key, idx_list in voxel2idx.items():
        idxs = np.array(idx_list, dtype=np.int32)
        if idxs.size < 5:
            continue

        pts = coords[idxs]
        centroid = pts.mean(axis=0)
        cov = (pts - centroid).T @ (pts - centroid)

        try:
            u, s, vt = np.linalg.svd(cov)
            eigenvalues = np.sort(s ** 2)
            lambda0[idxs] = eigenvalues[0].astype(np.float32)

        except Exception:
            continue

    lambda0_norm = lambda0 / (np.max(lambda0) + 1e-8)
    structure_strength = 1.0 - lambda0_norm

    return np.clip(structure_strength, 0.0, 1.0)


things_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 13, 17, 18]


def apply_rotate_scale(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """通用旋转和尺度变换，structure_strength 不变"""
    cfg = config['rotate_scale']
    theta = np.random.uniform(cfg['min_angle'], cfg['max_angle'])
    scale_factor = np.random.uniform(cfg['min_scale'], cfg['max_scale'])
    rot_mat = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    block[:, :3] = np.dot(block[:, :3], rot_mat) * scale_factor
    return block, labels, ids, strength


def apply_flip_axis(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """沿 X/Y 轴翻转，structure_strength 不变"""
    cfg = config['flip_axis']
    if np.random.rand() < cfg['prob_x']:
        block[:, 0] *= -1
    if np.random.rand() < cfg['prob_y']:
        block[:, 1] *= -1
    return block, labels, ids, strength


def apply_random_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """通用随机抖动，structure_strength 不变"""
    cfg = config['random_general_jittering']
    jittering = np.random.normal(loc=0., scale=cfg['scale'], size=(block.shape[0], 3)).astype(np.float32)
    jittering = np.clip(jittering, a_min=-cfg['max_clip'], a_max=cfg['max_clip'])
    block[:, :3] += jittering
    return block, labels, ids, strength


def apply_random_drop_out(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """随机均匀删除，structure_strength 同步删除"""
    cfg = config['random_drop_out']
    idxes = np.arange(block.shape[0])
    ratio = np.random.random() * (cfg['max_ratio'] - cfg['min_ratio']) + cfg['min_ratio']
    keep_count = int(ratio * block.shape[0])
    if keep_count == 0 and block.shape[0] > 0:
        keep_count = 1

    idxes = np.random.choice(idxes, keep_count, replace=False)
    return block[idxes], labels[idxes], ids[idxes], strength[idxes]


def apply_add_noise_points(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """添加随机噪声点，structure_strength 拼接新值"""
    cfg = config['add_random_noise_points']
    ignore_label = config.get('ignore_label', 255)

    if block.shape[0] == 0:
        return block, labels, ids, strength

    noise_num = int(np.random.random() * (cfg['max_num'] - cfg['min_num']) + cfg['min_num'])

    if noise_num == 0:
        return block, labels, ids, strength

    xmin, xmax = block[:, 0].min(), block[:, 0].max()
    ymin, ymax = block[:, 1].min(), block[:, 1].max()
    zmin, zmax = block[:, 2].min(), block[:, 2].max()
    imin, imax = block[:, 3].min(), block[:, 3].max()

    noise_x = np.random.uniform(xmin, xmax, noise_num).astype(np.float32)
    noise_y = np.random.uniform(ymin, ymax, noise_num).astype(np.float32)
    noise_z = np.random.uniform(zmin, zmax, noise_num).astype(np.float32)
    noise_i = np.random.normal(loc=(imin + imax) / 2, scale=cfg['intensity_scale'], size=noise_num).astype(np.float32)

    noise = np.stack((noise_x, noise_y, noise_z, noise_i), axis=1)

    noise_labels = np.ones(noise.shape[0], dtype=labels.dtype) * ignore_label
    noise_ids = np.ones(noise.shape[0], dtype=ids.dtype) * (-1)
    # 为噪声点设置低结构强度 (0.1)
    noise_strength = np.ones(noise.shape[0], dtype=strength.dtype) * 0.1

    block = np.concatenate((block, noise), axis=0)
    labels = np.concatenate((labels, noise_labels), axis=0)
    ids = np.concatenate((ids, noise_ids), axis=0)
    # 同步拼接 strength
    strength = np.concatenate((strength, noise_strength), axis=0)

    return block, labels, ids, strength


def apply_physical_attenuation_model(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray,
                                     config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PMA: 物理衰减模型，模拟强度衰减和概率丢点。
    已修改：同步接收并返回 strength 数组。
    """
    cfg = config['physical_attenuation_model']
    rng = np.random

    if block is None or block.shape[0] == 0:
        return block, labels, ids, strength

    # 提前退出条件 2: 随机数未达到触发概率
    if rng.rand() >= cfg.get('prob', 1.0):
        return block, labels, ids, strength

    block_copy = block.copy()
    labels_copy = labels.copy() if labels is not None else None
    ids_copy = ids.copy() if ids is not None else None
    strength_copy = strength.copy()  # 复制 strength 数组

    coords = block_copy[:, :3]
    distances = np.linalg.norm(coords, axis=1)

    alpha_min = cfg.get('alpha_min', 0.005)
    alpha_max = cfg.get('alpha_max', 0.05)

    # 随机采样消光系数 alpha
    alpha = rng.uniform(alpha_min, alpha_max)

    # 1. 强度衰减: I' = I * exp(-2 * alpha * R)
    attenuation_factor = np.exp(-2 * alpha * distances)
    block_copy[:, 3] *= attenuation_factor

    # 2. 概率丢点: P_drop = 1 - exp(-2 * alpha * R)
    # 计算保留概率 P_keep = exp(-2 * alpha * R)
    keep_prob = np.exp(-2 * alpha * distances)

    random_samples = rng.rand(block_copy.shape[0])
    keep_mask = random_samples <= keep_prob

    # 应用丢点
    block_out = block_copy[keep_mask]
    labels_out = labels_copy[keep_mask] if labels_copy is not None else labels
    ids_out = ids_copy[keep_mask] if ids_copy is not None else ids

    strength_out = strength_copy[keep_mask]

    return block_out, labels_out, ids_out, strength_out


def apply_selective_range_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        RJ: 全局范围抖动，structure_strength 不变。
    """
    cfg = config['selective_range_jittering']

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids, strength

    coords = block[:, :3]
    distances = np.linalg.norm(coords, axis=1)
    selected_distances = distances

    if selected_distances.shape[0] == 0:
        return block, labels, ids, strength

    base_std = cfg.get('jitter_base_std', 0.01)
    dist_factor = cfg.get('jitter_dist_factor', 0.0005)
    jitter_std = base_std + dist_factor * selected_distances

    delta_R = np.random.randn(selected_distances.shape[0]) * jitter_std

    clip_range = cfg.get('clip_range', [-0.05, 0.05])
    delta_R = np.clip(delta_R, clip_range[0], clip_range[1])

    new_R = selected_distances + delta_R
    new_R = np.maximum(new_R, 0.01)

    selected_distances[selected_distances == 0] = 1e-6

    scaling_factor = new_R / selected_distances
    coords *= scaling_factor[:, np.newaxis]

    block_out = block.copy()
    block_out[:, :3] = coords

    return block_out, labels, ids, strength


def apply_intensity_channel_distortion(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                       strength: np.ndarray, config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ICD: 强度通道扰动，structure_strength 不变。
    """
    cfg = config['intensity_channel_distortion']

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids, strength

    intensity = block[:, 3]
    gamma_prob = cfg.get('gamma_prob', 0.5)

    if np.random.rand() < gamma_prob:
        gamma_min = cfg.get('gamma_min', 0.7)
        gamma_max = cfg.get('gamma_max', 1.3)
        gamma = np.random.uniform(gamma_min, gamma_max)
        intensity = np.power(intensity, gamma)
    else:
        noise_std = 0.05
        intensity += np.random.randn(intensity.shape[0]) * noise_std

    intensity = np.clip(intensity, 0, intensity.max())
    block[:, 3] = intensity

    return block, labels, ids, strength


def apply_geometry_selective_jitter(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> \
Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    cfg = config.get('geometry_selective_jitter', {})
    p_frame = cfg.get('prob', 0.5)          # frame-level apply prob
    jitter_prob = cfg.get('jitter_prob', 0.25)  # per-point candidate prob
    base_std = cfg.get('base_std', 0.008)   # meters
    dist_factor = cfg.get('dist_factor', 0.0006)
    cluster_voxel = cfg.get('cluster_voxel_size', 1.0)
    rng = np.random

    if block is None or block.shape[0] == 0:
        return block, labels, ids, strength

    if rng.rand() > p_frame:
        return block, labels, ids, strength

    coords = block[:, :3].copy().astype(np.float32)
    N = coords.shape[0]

    # --- 1) Build coarse voxels and compute per-voxel normals (PCA) ---
    voxel2idx, point_voxel_key = _compute_voxel_occupancies(coords, voxel_size=cluster_voxel)
    normals = np.zeros((N, 3), dtype=np.float32)

    for key, idx_list in voxel2idx.items():
        idxs = np.array(idx_list, dtype=np.int32)
        # SVD/PCA 需要至少 3 个点
        if idxs.size < 3:
            # Not enough points for PCA, assume flat surface (Z-axis is up)
            normals[idxs] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            continue

        pts = coords[idxs]
        centroid = pts.mean(axis=0)
        # 中心化后的协方差矩阵：(pts - centroid).T @ (pts - centroid)
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
            normals[idxs] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # --- 2) Select candidate points to jitter ---
    select_mask = (rng.rand(N) < jitter_prob)
    if select_mask.sum() == 0:
        # 保持 strength 同步返回
        return block, labels, ids, strength

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

    # --- 4) Final Block Update ---
    block_out = block.copy()
    block_out[:, :3] = coords.astype(np.float32)

    # GSJ 旨在保持结构，故 strength 不变，直接返回
    return block_out, labels, ids, strength


def apply_semantic_aware_point_drop(
        block: np.ndarray,
        labels: np.ndarray,
        ids: np.ndarray,
        strength: np.ndarray, # 接收结构强度
        config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    语义结构感知丢点 (SAPD)，structure_strength 同步删除。
    使用传入的 strength 数组进行几何保护。
    """
    cfg = config.get('semantic_aware_point_drop', {})
    rng = np.random

    p_frame = cfg.get('prob', 0.2)
    base_drop = cfg.get('base_drop', 0.04)
    max_drop = cfg.get('max_drop', 0.2)
    r_scale = cfg.get('r_scale', 60.0)
    geo_protect_factor = cfg.get('geo_protect_factor', 0.7)
    protect_classes = cfg.get('protect_small_classes', [])
    protect_scale = cfg.get('protect_scale', 0.5)

    if rng.rand() > p_frame or block.shape[0] == 0:
        return block, labels, ids, strength

    coords = block[:, :3].copy()
    N = coords.shape[0]

    # --- Step 1: 距离偏置丢点概率 (Distance-Biased Drop Prob) ---
    R = np.linalg.norm(coords, axis=1)
    dist_weight = np.clip(R / r_scale, 0.0, 1.0)
    drop_prob = base_drop + (max_drop - base_drop) * dist_weight

    # --- Step 2: 几何结构感知保护 (使用传入的 strength) ---
    protection_mask = 1.0 - geo_protect_factor * strength
    drop_prob *= protection_mask

    # --- Step 3: 小物体类别保护 ---
    if labels is not None and protect_classes:
        small_mask = np.isin(labels, np.array(protect_classes, dtype=labels.dtype))
        drop_prob[small_mask] *= protect_scale

    # --- Step 4: 随机采样丢弃 ---
    keep_mask = rng.rand(N) > drop_prob

    if keep_mask.sum() == 0:
        keep_mask[rng.randint(0, N)] = True

    block_out = block[keep_mask]
    labels_out = labels[keep_mask]
    ids_out = ids[keep_mask]
    strength_out = strength[keep_mask] # 同步删除

    return block_out, labels_out, ids_out, strength_out


def apply_distance_biased_point_drop(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray,
                                     config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Distance-biased point drop: far & sparse regions more likely to be dropped.
    Modified to receive and return the strength array.
    """
    cfg = config.get('distance_biased_point_drop', {})
    p_frame = cfg.get('prob', 0.5)
    base_drop = cfg.get('base_drop', 0.04)
    max_drop = cfg.get('max_drop', 0.5)
    r_scale = cfg.get('r_scale', 60.0)  # meters (controls distance effect)
    gamma = cfg.get('gamma', 0.3)  # weight for density effect
    density_voxel = cfg.get('density_voxel_size', 1.0)  # voxel size for density estimation (meters)
    protect_small_classes = cfg.get('protect_small_classes', None)  # list or None
    ignore_label = config.get('ignore_label', 255)
    rng = np.random

    if block is None or block.shape[0] == 0:
        # 确保在空输入时，强度同步返回
        return block, labels, ids, strength

    if rng.rand() > p_frame:
        # 确保在不应用增强时，强度同步返回
        return block, labels, ids, strength

    coords = block[:, :3].astype(np.float32)
    N = coords.shape[0]

    # --- 1. Distance-based weight (sigmoid-like mapping) ---
    dists = np.linalg.norm(coords, axis=1)
    dist_norm = dists / max(r_scale, 1e-6)
    # smooth mapping in [0,1]
    dist_weight = 1.0 / (1.0 + np.exp(- (dist_norm - 0.5) * 6.0))

    # --- 2. Density estimation via voxel occupancy (coarse) ---
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

    # --- 3. Combine into final drop probability ---
    drop_prob = base_drop + (max_drop - base_drop) * dist_weight
    # Additive weight from low density: (1.0 - densities) is low density score
    drop_prob = np.clip(drop_prob + gamma * (1.0 - densities), 0.0, 1.0)

    # --- 4. Protection for small classes ---
    if labels is not None and protect_small_classes:
        small_mask = np.isin(labels, np.array(protect_small_classes, dtype=labels.dtype))
        # reduce drop prob (protect) for small classes
        drop_prob[small_mask] *= cfg.get('protect_scale', 0.5)

    # --- 5. Apply drop mask ---
    keep_mask = rng.rand(N) > drop_prob
    if keep_mask.sum() == 0:
        # ensure at least one point is kept to avoid empty array errors
        keep_mask[rng.randint(0, N)] = True

    # --- 6. Output the filtered arrays ---
    block_out = block[keep_mask]
    labels_out = labels[keep_mask] if labels is not None else labels
    ids_out = ids[keep_mask] if ids is not None else ids

    # *** 关键修改：同步过滤 strength 数组 ***
    strength_out = strength[keep_mask]

    return block_out, labels_out, ids_out, strength_out


def apply_intensity_jitter(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Add random noise to intensity, structure_strength 不变。
    """
    cfg = config.get('intensity_jitter', {})
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > cfg.get('prob', 0.5):
        return block, labels, ids, strength

    block_out = block.copy()
    block_out[:, 3] += rng.uniform(-cfg.get('scale', 0.1), cfg.get('scale', 0.1), size=block_out.shape[0])
    block_out[:, 3] = np.clip(block_out[:, 3], 0.0, 1.0)
    return block_out, labels, ids, strength


def apply_occlusion_patch(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟局部遮挡（Occlusion） 的数据增强方法，structure_strength 同步删除。
    """
    cfg = config.get('occlusion_patch', {})
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > cfg.get('prob', 0.5):
        return block, labels, ids, strength

    coords = block[:, :3]
    keep_mask = np.ones(coords.shape[0], dtype=bool)

    for _ in range(cfg.get('num_patches', 1)):
        center = coords[rng.randint(0, coords.shape[0])]
        mask = np.all(np.abs(coords - center) < cfg.get('patch_size', 2.0) / 2, axis=1)
        keep_mask[mask] = False

    if keep_mask.sum() == 0:
        keep_mask[rng.randint(0, coords.shape[0])] = True

    block_out = block[keep_mask]
    labels_out = labels[keep_mask] if labels is not None else labels
    ids_out = ids[keep_mask] if ids is not None else ids
    strength_out = strength[keep_mask] # 同步删除

    return block_out, labels_out, ids_out, strength_out



def angle_in_range(phi, low, high):
    """Checks if angle phi is in range [low, high], handling the +/- pi wrap-around."""
    diff = (phi - low) % (2 * np.pi)
    range_width = (high - low) % (2 * np.pi)
    return diff <= range_width

def apply_nonuniform_region_perturbation(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                         strength: np.ndarray, config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NRP: 非均匀区域扰动增强 (Non-uniform Region Perturbation)
    DSP + SSP
    支持 strength 同步更新
    """
    cfg = config.get('nonuniform_region_perturbation', {})
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > cfg.get('prob', 1.0):
        return block, labels, ids, strength

    coords = block[:, :3].copy()
    N = coords.shape[0]

    # ----------------- DSP -----------------
    dsp_cfg = cfg.get('depth_selective_perturbation', {})
    if rng.rand() < dsp_cfg.get('prob', 0.5):
        dists = np.linalg.norm(coords, axis=1)
        d_min_cfg, d_max_cfg = dsp_cfg.get('d_range', [20.0, 80.0])
        d_l = rng.uniform(d_min_cfg, d_max_cfg - 10)
        d_h = rng.uniform(d_l + 5, d_max_cfg)
        depth_mask = (dists >= d_l) & (dists <= d_h)
        sel_coords = coords[depth_mask]
        if sel_coords.shape[0] > 0:
            base_std = dsp_cfg.get('base_std', 0.01)
            dist_factor = dsp_cfg.get('dist_factor', 0.0001)
            sigma_vector = base_std + dist_factor * dists[depth_mask]
            epsilon = rng.normal(0., sigma_vector[:, None], sel_coords.shape).astype(np.float32)
            coords[depth_mask] += epsilon
            strength[depth_mask] *= 0.95  # 扰动点略微衰减

    # ----------------- SSP -----------------
    ssp_cfg = cfg.get('scanline_selective_perturbation', {})
    if rng.rand() < ssp_cfg.get('prob', 0.3):
        phi = np.arctan2(coords[:, 1], coords[:, 0])
        num_segments = ssp_cfg.get('num_segments', 3)
        angle_width = ssp_cfg.get('angle_width', np.pi / 36)
        for _ in range(num_segments):
            theta_c = rng.uniform(-np.pi, np.pi)
            theta_l, theta_h = theta_c - angle_width / 2, theta_c + angle_width / 2
            angle_mask = angle_in_range(phi, theta_l, theta_h)
            sel_coords = coords[angle_mask]
            if sel_coords.shape[0] > 0:
                ssp_std = ssp_cfg.get('ssp_std', 0.02)
                epsilon = rng.normal(0., ssp_std, sel_coords.shape).astype(np.float32)
                coords[angle_mask] += epsilon
                strength[angle_mask] *= 0.95  # 扰动点略微衰减

    block[:, :3] = coords
    return block, labels, ids, strength

def apply_depth_adaptive_sparsity_augmentation(block: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                                               strength: np.ndarray, config: Dict[str, Any]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    DSA: 深度自适应点稀疏增强
    D-Drop + Clustered-Drop
    支持 strength 同步更新
    """
    cfg = config.get('depth_adaptive_sparsity_augmentation', {})
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > cfg.get('prob', 1.0):
        return block, labels, ids, strength

    coords = block[:, :3].copy()
    N = coords.shape[0]
    keep_mask = np.ones(N, dtype=bool)

    # ----------------- D-Drop -----------------
    ddrop_cfg = cfg.get('depth_aware_drop', {})
    if rng.rand() < ddrop_cfg.get('prob', 0.6):
        dists = np.linalg.norm(coords, axis=1)
        d_min = ddrop_cfg.get('d_min', 0.5)
        d_max = ddrop_cfg.get('d_max', 100.0)
        alpha = ddrop_cfg.get('alpha', 0.4)
        dist_norm = np.clip((dists - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
        p_drop = alpha * dist_norm
        ddrop_mask = rng.rand(N) > p_drop
        keep_mask &= ddrop_mask
        strength[~ddrop_mask] = 0.0  # 删除点强度置0

    # ----------------- Clustered-Drop -----------------
    cdrop_cfg = cfg.get('clustered_drop', {})
    if rng.rand() < cdrop_cfg.get('prob', 0.4):
        num_clusters = cdrop_cfg.get('num_clusters', 2)
        radius = cdrop_cfg.get('radius', 1.5)
        for _ in range(num_clusters):
            if keep_mask.sum() == 0:
                break
            active_coords = coords[keep_mask]
            center_idx_in_active = rng.randint(0, active_coords.shape[0])
            p_c = active_coords[center_idx_in_active]
            cluster_drop_mask = np.linalg.norm(coords - p_c, axis=1) < radius
            keep_mask[cluster_drop_mask] = False
            strength[cluster_drop_mask] = 0.0  # 删除点强度置0

    # 确保至少保留一个点
    if keep_mask.sum() == 0:
        keep_mask[rng.randint(0, N)] = True

    block_out = block[keep_mask]
    labels_out = labels[keep_mask]
    ids_out = ids[keep_mask]
    strength_out = strength[keep_mask]

    return block_out, labels_out, ids_out, strength_out

def apply_occlusion_patch(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, strength: np.ndarray,
                          config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟局部遮挡（Occlusion） 的数据增强方法
    被遮挡点 strength 置 0
    """
    cfg = config.get('occlusion_patch', {})
    rng = np.random

    if block is None or block.shape[0] == 0 or rng.rand() > cfg.get('prob', 0.5):
        return block, labels, ids, strength

    coords = block[:, :3]
    keep_mask = np.ones(coords.shape[0], dtype=bool)

    for _ in range(cfg.get('num_patches', 1)):
        center = coords[rng.randint(0, coords.shape[0])]
        mask = np.all(np.abs(coords - center) < cfg.get('patch_size', 2.0) / 2, axis=1)
        keep_mask[mask] = False
        strength[mask] = 0.0  # 同步置0

    if keep_mask.sum() == 0:
        keep_mask[rng.randint(0, coords.shape[0])] = True

    block_out = block[keep_mask]
    labels_out = labels[keep_mask] if labels is not None else labels
    ids_out = ids[keep_mask] if ids is not None else ids
    strength_out = strength[keep_mask]

    return block_out, labels_out, ids_out, strength_out