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
