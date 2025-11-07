

import torch
import torch.nn.functional as F

try:
    # 导入 Open3D 的 PyTorch ML 运算模块
    from open3d.ml.torch.ops import knn_search
except ImportError:
    print("Warning: open3d.ml.torch.ops.knn_search not found. Ensure Open3D ML is correctly installed.")
    # 如果找不到，您可以切换到 PyTorch3D 或其他库


def calculate_density_weights(inputs, K: int, epsilon: float = 1e-6) -> torch.Tensor:
    """
    计算基于 Open3D KNN 的局部密度权重 S。
    S 定义为与稀疏度（d_K，到第 K 个邻居的距离）正相关的权重。
    """

    # ！！！ 1. 提取浮点坐标 (N, 3) ！！！
    # 假设 inputs 是 SparseTensor，您需要根据您的数据流找到原始的浮点坐标。
    # 这是一个占位符，您需要根据实际数据结构进行修正：
    # 例如：points_coords = inputs.F[:, :3]  # 如果特征 F 包含了坐标

    # 示例占位符 (假设 N=1000, 且在 CUDA 上)
    N_points = inputs.C.shape[0]  # 从 SparseTensor 获取点数
    points_coords = torch.rand(N_points, 3, device=inputs.device) * 50  # 替换为实际坐标

    # 2. KNN 搜索 (使用 Open3D ML Ops)
    # Open3D KNN 搜索需要输入：
    #   query_points (N, 3)
    #   support_points (N, 3) - 通常与 query_points 相同 (自搜索)
    #   K (int)
    # 它返回 distances (N, K) 和 indices (N, K)

    # 因为是自搜索，query 和 support 都是 points_coords
    # 这里的 batch_idx 假设为 0，因为 SparseTensor 已经在一个批次内

    # batch_splits (N_batches,) -> 这里我们假设 batch_size=1，所以 [N_points]
    batch_splits = torch.tensor([N_points], dtype=torch.int32, device=inputs.device)

    try:
        # 执行 KNN 搜索
        # Note: Open3D ML ops usually expects float32
        distances, _ = knn_search(
            points_coords.contiguous(),
            points_coords.contiguous(),
            batch_splits,
            batch_splits,
            K
        )
    except NameError:
        # 如果 knn_search 导入失败，这里需要报错或使用其他 CPU/GPU KNN 替代
        raise RuntimeError("Open3D ML ops knn_search is not available. Please check Open3D ML installation.")

    # 3. 提取到第 K 个邻居的距离 d_K (稀疏度度量)
    # distances 形状为 (N, K)。第 K 个邻居是索引 K-1
    # d_K 形状为 (N,)
    d_K = distances[:, K - 1]

    # 4. 转化为密度权重 S

    # 稀疏度 d_K 越大（距离越远），权重 S 应该越高。
    # 使用指数形式放大稀疏区域的权重。
    S = torch.exp(d_K)

    # 归一化 S (使平均权重接近 1.0)
    S = S / (S.mean() + epsilon)

    # 确保 S 的形状是 (N, 1)
    S = S.unsqueeze(1)

    return S

# --- [辅助函数：UWC 不确定性权重 W 计算] ---

def calculate_uncertainty_weights(outputs: torch.Tensor, alpha: float, epsilon: float = 1e-8) -> torch.Tensor:
    """
    [UWC] 根据预测熵计算不确定性权重 W。
    W = exp(-alpha * U)，其中 U = 预测熵 H(P)。

    Args:
        outputs (torch.Tensor): 模型的原始 logits (pred_2)，形状 (N, C)。
        alpha (float): 熵的缩放超参数。
        epsilon (float): 用于防止 log(0) 的极小值。

    Returns:
        torch.Tensor: 不确定性权重 W，形状为 (N)。
    """
    # 1. 计算 softmax 概率 P
    P = F.softmax(outputs, dim=1)

    # 2. 计算预测熵 U (Entropy Loss) H(P)
    # H(P) = - sum(P * log(P))
    # outputs 形状 (N, C)，P 形状 (N, C)，U 形状 (N, 1)
    U = -torch.sum(P * torch.log(P + epsilon), dim=1, keepdim=True)

    # 3. 计算 UWC 权重 W
    # W = exp(-alpha * U)
    # W 形状 (N, 1)
    W = torch.exp(-alpha * U)

    return W.squeeze(1)  # 返回 (N) 形状，与逐点损失匹配
