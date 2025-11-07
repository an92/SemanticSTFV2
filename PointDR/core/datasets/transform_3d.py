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