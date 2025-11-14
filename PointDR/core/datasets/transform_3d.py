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
    """通用随机抖动 (旧 Aug6)"""
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

def apply_weather_layered_augmentation(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    天气分层增强：
    - 雪、雨、雾、薄雾
    - 每种天气按概率执行
    - 对小目标 (things_class_ids) 使用高概率扰动
    - 对大结构 (stuff) 使用低概率轻微扰动
    """
    cfg = config.get('weather_layered_augmentation', {})
    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    weather_cfg = cfg.get('weather_prob', {
        'snow': 0.3,
        'rain': 0.3,
        'fog': 0.2,
        'thin_fog': 0.2
    })

    coords = block[:, :3].copy()
    feats = block[:, 3:].copy() if block.shape[1] > 3 else None

    # ------------------------
    # 遍历天气类型
    # ------------------------
    for weather, p in weather_cfg.items():
        if np.random.rand() >= p:
            continue  # 不触发该天气增强

        # ------------------------
        # 1. 点丢失增强
        # ------------------------
        drop_prob_small = cfg.get('drop_prob_small', 0.3)
        drop_prob_large = cfg.get('drop_prob_large', 0.05)

        # 保证 mask 与当前 labels 对齐
        num_points = labels.shape[0]
        small_mask = np.isin(labels, things_class_ids)
        large_mask = ~small_mask
        keep_prob = np.ones(num_points)
        keep_prob[small_mask] *= 1 - drop_prob_small
        keep_prob[large_mask] *= 1 - drop_prob_large
        keep_mask = np.random.rand(num_points) < keep_prob

        coords = coords[keep_mask]
        labels = labels[keep_mask]
        ids = ids[keep_mask]
        if feats is not None:
            feats = feats[keep_mask]

        # ------------------------
        # 2. 局部簇扰动
        # ------------------------
        cluster_size = cfg.get('cluster_size', 10)
        sigma_local = cfg.get('sigma_local', 0.02)
        num_clusters = cfg.get('num_clusters', 5)

        num_points = labels.shape[0]
        for _ in range(num_clusters):
            if num_points == 0:
                break
            center_idx = np.random.randint(0, num_points)
            dists = np.linalg.norm(coords - coords[center_idx], axis=1)
            cluster_mask = np.argsort(dists)[:cluster_size]

            keep_mask_cluster = np.ones(num_points, dtype=bool)
            keep_mask_cluster[cluster_mask] = False

            coords = coords[keep_mask_cluster]
            labels = labels[keep_mask_cluster]
            ids = ids[keep_mask_cluster]
            if feats is not None:
                feats = feats[keep_mask_cluster]

            num_points = labels.shape[0]

        # 局部微扰
        if num_points > 0:
            coords += np.random.normal(0, sigma_local, coords.shape)

        # ------------------------
        # 3. 深度漂移（浓雾/薄雾）
        # ------------------------
        if weather in ['fog', 'thin_fog'] and num_points > 0:
            base_sigma = cfg.get('depth_sigma', 0.02)
            distances = np.linalg.norm(coords, axis=1)
            max_dist = distances.max() + 1e-6
            coords[:, 2] += np.random.normal(0, base_sigma * distances / max_dist, size=num_points)

        # ------------------------
        # 4. 随机孤立点增强（雪/雨）
        # ------------------------
        if weather in ['snow', 'rain'] and num_points > 0:
            noise_ratio = cfg.get('noise_ratio', 0.005)
            num_noise = int(coords.shape[0] * noise_ratio)
            if num_noise > 0:
                xyz_min = coords.min(0)
                xyz_max = coords.max(0)
                noise_xyz = np.random.uniform(xyz_min, xyz_max, size=(num_noise, 3))
                coords = np.concatenate([coords, noise_xyz], axis=0)
                labels = np.concatenate([labels, np.full(num_noise, 255)], axis=0)
                ids = np.concatenate([ids, np.arange(ids.max()+1, ids.max()+1+num_noise)], axis=0)
                if feats is not None:
                    feats = np.concatenate([feats, np.zeros((num_noise, feats.shape[1]), dtype=feats.dtype)], axis=0)

    # ------------------------
    # 5. 构造新的 block 返回
    # ------------------------
    if feats is not None:
        block_new = np.hstack([coords, feats])
    else:
        block_new = coords

    block_new = block_new.astype(np.float32)

    return block_new, labels, ids



# ---------------------------------------------------------------------
# Helper utilities (used by both augmentations)
# ---------------------------------------------------------------------
def _voxelize_coords(coords: np.ndarray, voxel_size: float):
    """
    Returns integer voxel indices for coords (N,3).
    """
    return np.floor(coords / voxel_size).astype(np.int32)

def _compute_voxel_occupancies(coords: np.ndarray, voxel_size: float):
    """
    Return a dict mapping voxel tuple -> indices list and per-point voxel-id index.
    """
    v = _voxelize_coords(coords, voxel_size)
    # create single integer key per voxel for dictionary hashing
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
# 1) Geometry-consistent Selective Jitter (GSJ)
# ---------------------------------------------------------------------
def apply_geometry_selective_jitter(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Geometry-aware selective jitter.
    block: (N, >=3) float32 with at least xyz in [:,0:3] and optionally intensity at [:,3].
    labels, ids: 1D arrays aligned with block rows.
    config: top-level config dict; this function expects config['geometry_selective_jitter'] to contain parameters.

    Returns modified (block, labels, ids).
    """
    cfg = config.get('geometry_selective_jitter', {})
    p_frame = cfg.get('prob', 0.5)                # frame-level apply prob
    jitter_prob = cfg.get('jitter_prob', 0.25)   # per-point candidate prob
    base_std = cfg.get('base_std', 0.008)        # meters
    dist_factor = cfg.get('dist_factor', 0.0006)
    cluster_voxel = cfg.get('cluster_voxel_size', 2.0)  # meters, coarse clustering resolution
    cluster_min_points = cfg.get('cluster_min_points', 20)
    min_sf = cfg.get('min_scaling', 0.95)
    max_sf = cfg.get('max_scaling', 1.05)
    normal_k = cfg.get('normal_k', 12)           # neigh count for local plane estimation (per-voxel)
    rng = np.random

    if block is None or block.shape[0] == 0:
        return block, labels, ids

    if rng.rand() > p_frame:
        return block, labels, ids

    coords = block[:, :3].astype(np.float32)
    N = coords.shape[0]

    # --- 1) build coarse voxels to do local PCA (approx normals) and clustering ---
    # Use coarse voxelization to group nearby points; this avoids O(N^2) kNN
    voxel2idx, point_voxel_key = _compute_voxel_occupancies(coords, voxel_size=cluster_voxel)

    # compute per-voxel normals by PCA of points in voxel
    normals = np.zeros((N, 3), dtype=np.float32)
    for key, idx_list in voxel2idx.items():
        idxs = np.array(idx_list, dtype=np.int32)
        if idxs.size < 3:
            normals[idxs] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            continue
        pts = coords[idxs]
        # PCA (covariance), smallest eigenvector is normal
        centroid = pts.mean(axis=0)
        cov = (pts - centroid).T @ (pts - centroid)
        try:
            _, s, vt = np.linalg.svd(cov)
            normal = vt[-1]
            # fix NaN or zero normal
            if not np.all(np.isfinite(normal)):
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            normals[idxs] = normal.astype(np.float32)
        except Exception:
            normals[idxs] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # --- 2) select candidate points to jitter ---
    select_mask = (rng.rand(N) < jitter_prob)
    if select_mask.sum() == 0:
        return block, labels, ids

    # --- 3) cluster selected points by coarse voxel again (reuse voxel keys) ---
    # Build mapping from voxel key to indices among selected points
    sel_idxs = np.where(select_mask)[0]
    sel_voxel_keys = [point_voxel_key[i] for i in sel_idxs]
    vox2sel = {}
    for local_idx, vk in enumerate(sel_voxel_keys):
        if vk not in vox2sel:
            vox2sel[vk] = []
        vox2sel[vk].append(sel_idxs[local_idx])

    # For each selected voxel group (acts as a cluster), sample a scalar offset and apply along normals
    for vk, g_indices in vox2sel.items():
        g_indices = np.array(g_indices, dtype=np.int32)
        # mean distance of this cluster
        mean_r = np.linalg.norm(coords[g_indices], axis=1).mean() if g_indices.size > 0 else 0.0
        sigma = base_std + dist_factor * mean_r
        # scalar offset sampled once per cluster (correlated jitter)
        scalar = rng.randn() * sigma
        normals_cluster = normals[g_indices]
        # apply along normal direction
        coords[g_indices] = coords[g_indices] + normals_cluster * scalar

    # --- 4) radial-scale clamp to avoid extreme deformation ---
    orig_r = np.linalg.norm(block[:, :3], axis=1) + 1e-8
    new_r = np.linalg.norm(coords, axis=1)
    scales = np.clip(new_r / orig_r, min_sf, max_sf)
    coords = block[:, :3] * scales[:, None]

    # write back coords into block
    block_out = block.copy()
    block_out[:, :3] = coords.astype(np.float32)

    return block_out, labels, ids


# ---------------------------------------------------------------------
# 2) Distance-biased Point Drop (DBPD)
# ---------------------------------------------------------------------
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

