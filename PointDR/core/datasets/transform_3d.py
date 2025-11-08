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


def apply_beamwise_semantic_jitter(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Beamwise Semantic Jittering (BSJ) data augmentation.
    对随机选定的光束中的 'Things' 类施加更强的几何扰动。

    Args:
        block (np.ndarray): 点云数据，形状 (N, F)。前三列为 (X, Y, Z)。
        labels (np.ndarray): 语义标签，形状 (N, )。
        ids (np.ndarray): 实例ID，形状 (N, )。
        config (Dict[str, Any]): 包含增强参数的配置字典。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 扰动后的点云、标签和ID。
    """
    cfg = config['beamwise_semantic_jitter']

    if np.random.rand() > cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    # --- 1. 参数提取与准备 ---
    instance_classes = np.array(cfg['thing_class_ids'])

    # 将角度从度数转换为弧度
    angle_range = np.array(cfg['angle_range']) / 180.0 * np.pi
    sector_size = np.array(cfg['sector_size']) / 180.0 * np.pi

    stuff_std = np.array(cfg['stuff_jitter_std'], dtype=np.float32)
    things_std = np.array(cfg['things_jitter_std'], dtype=np.float32)

    X, Y, Z = block[:, 0], block[:, 1], block[:, 2]

    # --- 2. 光束选择 (Beam Selection) ---
    azimuth_span = np.random.uniform(sector_size[0], sector_size[1])
    start_angle = np.random.uniform(angle_range[0], angle_range[1] - azimuth_span)
    end_angle = start_angle + azimuth_span

    # 计算方位角 (Yaw)
    yaw = np.arctan2(Y, X)
    yaw[yaw < 0] += 2 * np.pi  # 归一化到 [0, 2pi]

    # 创建光束选择掩码 (Sector Mask)
    if end_angle > 2 * np.pi:
        end_angle -= 2 * np.pi
        sector_mask = (yaw >= start_angle) | (yaw < end_angle)
    else:
        sector_mask = (yaw >= start_angle) & (yaw < end_angle)

    # --- 3. 语义加权 Jitter 掩码 ---
    is_thing = np.isin(labels, instance_classes)

    thing_in_sector_mask = sector_mask & is_thing
    stuff_in_sector_mask = sector_mask & (~is_thing)

    jitter_xyz = np.zeros_like(block[:, :3], dtype=np.float32)

    # 4. 应用 Jitter (XYZ 方向)

    # Things (高强度)
    if thing_in_sector_mask.sum() > 0:
        thing_noise = np.random.normal(loc=0, scale=things_std, size=(thing_in_sector_mask.sum(), 3))
        jitter_xyz[thing_in_sector_mask] = thing_noise

    # Stuff (低强度)
    if stuff_in_sector_mask.sum() > 0:
        stuff_noise = np.random.normal(loc=0, scale=stuff_std, size=(stuff_in_sector_mask.sum(), 3))
        jitter_xyz[stuff_in_sector_mask] = stuff_noise

    # --- 5. 更新点云坐标 ---
    block[:, :3] += jitter_xyz

    return block, labels, ids


def apply_beamwise_semantic_drop(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Beamwise Semantic Drop (BSD) data augmentation.
    在随机选定的光束中，对 'Things' 类施加高丢弃率，对 'Stuff' 类施加低丢弃率。

    Args:
        block (np.ndarray): 点云数据，形状 (N, F)。
        labels (np.ndarray): 语义标签，形状 (N, )。
        ids (np.ndarray): 实例ID，形状 (N, )。
        config (Dict[str, Any]): 包含增强参数的配置字典。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 丢弃后的点云、标签和ID。
    """
    cfg = config['beamwise_semantic_drop']

    if np.random.rand() > cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    # --- 1. 参数提取与准备 ---
    instance_classes = np.array(cfg['thing_class_ids'])

    angle_range = np.array(cfg['angle_range']) / 180.0 * np.pi
    sector_size = np.array(cfg['sector_size']) / 180.0 * np.pi

    thing_drop_ratio = cfg['thing_drop_ratio']
    stuff_drop_ratio = cfg['stuff_drop_ratio']

    X, Y = block[:, 0], block[:, 1]

    # --- 2. 光束选择 (Beam Selection) ---
    azimuth_span = np.random.uniform(sector_size[0], sector_size[1])
    start_angle = np.random.uniform(angle_range[0], angle_range[1] - azimuth_span)
    end_angle = start_angle + azimuth_span

    # 计算方位角 (Yaw)
    yaw = np.arctan2(Y, X)
    yaw[yaw < 0] += 2 * np.pi

    # 创建光束选择掩码 (Sector Mask)
    if end_angle > 2 * np.pi:
        end_angle -= 2 * np.pi
        sector_mask = (yaw >= start_angle) | (yaw < end_angle)
    else:
        sector_mask = (yaw >= start_angle) & (yaw < end_angle)

    # --- 3. 语义加权丢弃概率 ---
    is_thing = np.isin(labels, instance_classes)

    P_drop = np.zeros_like(yaw, dtype=np.float32)

    # Things in sector (高丢弃率)
    P_drop[sector_mask & is_thing] = thing_drop_ratio

    # Stuff in sector (低丢弃率)
    P_drop[sector_mask & (~is_thing)] = stuff_drop_ratio

    # --- 4. 执行点丢弃 ---
    rand_vals = np.random.rand(block.shape[0])
    keep_mask = (rand_vals >= P_drop)

    # 更新数据
    block = block[keep_mask]
    labels = labels[keep_mask]
    ids = ids[keep_mask]

    return block, labels, ids


def apply_range_dependent_jittering(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Range-Dependent Jittering (RDJ): 几何扰动强度随距离增加。

    Args:
        block: 点云数据 (N, 4), [x, y, z, intensity]
        labels: 语义标签 (N,)
        ids: 实例 ID (N,)
        config: 包含 'range_dependent_jittering' 配置的字典

    Returns:
        tuple: 抖动后的点云、标签和实例 ID
    """
    cfg = config['range_dependent_jittering']

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    # 1. 计算距离 (Euclidean distance from origin)
    # 坐标 (x, y, z) 在前三列
    coords = block[:, :3]
    distances = np.linalg.norm(coords, axis=1)

    # 2. 计算每个点的抖动强度 (Standard Deviation)
    # 公式: sigma_i = base_std + dist_factor * d_i
    base_std = cfg.get('base_std', 0.01)
    dist_factor = cfg.get('dist_factor', 0.0005)

    jitter_std = base_std + dist_factor * distances

    # 3. 生成抖动噪声 (N x 3 矩阵)
    # 使用 np.newaxis 确保 jitter_std 广播到 N x 3
    noise = np.random.randn(*coords.shape) * jitter_std[:, np.newaxis]

    # 4. 应用抖动
    block[:, :3] = coords + noise

    return block, labels, ids


def apply_global_outlier_scaling(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global Outlier Scaling (GOS): 添加高比例全局噪声点。

    Args:
        block: 点云数据 (N, 4), [x, y, z, intensity]
        labels: 语义标签 (N,)
        ids: 实例 ID (N,)
        config: 包含 'global_outlier_scaling' 配置的字典

    Returns:
        tuple: 添加噪声后的点云、标签和实例 ID
    """
    cfg = config['global_outlier_scaling']
    ignore_label = config.get('ignore_label', 255)

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    current_num = block.shape[0]
    # 使用训练集的最大点数作为参考（如果 config 中有的话）
    max_total_num = config.get('num_points', current_num)

    # 1. 确定噪声点数量
    min_ratio = cfg.get('min_ratio', 0.05)
    max_ratio = cfg.get('max_ratio', 0.30)

    # 噪声点数量 N_g = Ratio * N_current
    # 这里我们使用当前点云数量和最大点数来确定噪声数量的上下限
    min_noise_num = int(current_num * min_ratio)
    max_noise_num = int(max_total_num * max_ratio)

    # 噪声点数量 N_g
    noise_num = int(np.random.uniform(min_noise_num, max_noise_num))

    if noise_num == 0:
        return block, labels, ids

    # 2. 确定噪声的边界和强度
    xmin, xmax = block[:, 0].min(), block[:, 0].max()
    ymin, ymax = block[:, 1].min(), block[:, 1].max()
    zmin, zmax = block[:, 2].min(), block[:, 2].max()
    # 假定强度在 [0, 1] 归一化或使用原点的强度范围
    # 为了简化，我们使用一个经验的强度范围或基于当前点云的统计
    imin, imax = block[:, 3].min(), block[:, 3].max()

    intensity_scale = cfg.get('intensity_scale', 1.0)

    # 3. 生成坐标噪声 (全局均匀分布)
    noise_x = np.random.uniform(xmin, xmax, noise_num).astype(np.float32)
    noise_y = np.random.uniform(ymin, ymax, noise_num).astype(np.float32)
    noise_z = np.random.uniform(zmin, zmax, noise_num).astype(np.float32)

    # 4. 生成强度噪声 (集中在平均值附近，模拟散射点)
    # 使用高斯分布，均值取当前强度的平均值，方差由 scale 控制
    avg_intensity = (imin + imax) / 2
    noise_i = np.random.normal(loc=avg_intensity, scale=intensity_scale, size=noise_num).astype(np.float32)
    # 限制强度在合理范围，例如 [0, 1] 或 [0, imax]
    noise_i = np.clip(noise_i, 0, imax)

    noise = np.stack((noise_x, noise_y, noise_z, noise_i), axis=1)

    # 5. 拼接
    noise_labels = np.ones(noise.shape[0], dtype=labels.dtype) * ignore_label
    noise_ids = np.ones(noise.shape[0], dtype=ids.dtype) * (-1)

    block = np.concatenate((block, noise), axis=0)
    labels = np.concatenate((labels, noise_labels), axis=0)
    ids = np.concatenate((ids, noise_ids), axis=0)

    return block, labels, ids


def apply_vulnerable_region_drop(block: np.ndarray, labels: np.ndarray, ids: np.ndarray, config: Dict[str, Any]) -> \
Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vulnerable Region Drop (VRD): 基于启发式规则 (远程/Things) 的加权丢点。

    Args:
        block: 点云数据 (N, 4), [x, y, z, intensity]
        labels: 语义标签 (N,)
        ids: 实例 ID (N,)
        config: 包含 'vulnerable_region_drop' 配置的字典

    Returns:
        tuple: 丢点后的点云、标签和实例 ID
    """
    cfg = config['vulnerable_region_drop']

    if np.random.rand() >= cfg.get('prob', 1.0) or block.shape[0] == 0:
        return block, labels, ids

    # 1. 获取配置参数
    base_drop_ratio = cfg.get('base_drop_ratio', 0.10)
    thing_boost_factor = cfg.get('thing_boost_factor', 2.0)
    remote_boost_factor = cfg.get('remote_boost_factor', 1.5)
    thing_class_ids = cfg.get('thing_class_ids', [])
    remote_distance_thresh = cfg.get('remote_distance_thresh', 40.0)  # 40米作为远程阈值

    current_num = block.shape[0]

    # 2. 计算距离和初始化权重
    coords = block[:, :3]
    distances = np.linalg.norm(coords, axis=1)

    # 丢点权重越高，该点被选中的概率越高
    drop_weights = np.ones(current_num, dtype=np.float32)

    # 3. 应用远程加权 (Distance Weighting)
    # 距离 > 阈值的点权重增加
    remote_mask = distances > remote_distance_thresh
    drop_weights[remote_mask] *= remote_boost_factor

    # 4. 应用 Things 加权 (Semantic Weighting)
    # 移动物体 (Things) 的标签权重增加
    thing_mask = np.isin(labels, thing_class_ids)
    drop_weights[thing_mask] *= thing_boost_factor

    # 5. 确定丢点数量
    # 丢点总数量 = N_total * base_drop_ratio
    num_to_drop = int(current_num * base_drop_ratio)
    if num_to_drop == 0:
        return block, labels, ids

    # 6. 归一化权重并采样要保留的点
    # 权重越高，被选中的概率越高，所以我们采样要 '丢弃' 的点
    # 为了避免浮点数问题，将权重归一化为概率分布
    drop_probs = drop_weights / np.sum(drop_weights)

    # 采样要丢弃的点的索引 (无放回采样)
    drop_indices = np.random.choice(current_num, size=num_to_drop, replace=False, p=drop_probs)

    # 7. 构造保留点的掩码
    keep_mask = np.ones(current_num, dtype=bool)
    keep_mask[drop_indices] = False

    # 8. 应用丢点
    block = block[keep_mask]
    labels = labels[keep_mask]
    ids = ids[keep_mask]

    return block, labels, ids