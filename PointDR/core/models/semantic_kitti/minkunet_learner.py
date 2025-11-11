from typing import Tuple

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch.nn.functional as F


__all__ = ['MinkUNet_Learner']

from torchsparse import SparseTensor


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class LearnableJitterModule(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, max_sigma: float = 0.05) -> None:
        super().__init__()
        self.max_sigma = max_sigma
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1), nn.Sigmoid()
        )
        nn.init.constant_(self.mlp[-2].bias, -2.0)

    def forward(self, points_F: torch.Tensor) -> torch.Tensor:
        coords = points_F[:, :3]
        sigma_factor = self.mlp(coords)
        sigma = sigma_factor * self.max_sigma
        noise = torch.randn_like(coords) * sigma
        jittered_F = points_F.clone()
        jittered_F[:, :3] = coords + noise
        return jittered_F


class AdversarialDropModule(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1), nn.Sigmoid()
        )

    def forward(
            self,
            lidar: 'SparseTensor',
            L_aug: torch.Tensor,
            H_aug: torch.Tensor,
            hard_threshold: float = 0.5,
            dual_forward: bool = True,
    ) -> Tuple['SparseTensor', torch.Tensor, torch.Tensor]:
        """
        Args:
            lidar: SparseTensor 输入点云
            L_aug, H_aug: 上游 supervision scalar
            hard_threshold: 控制 Gumbel-soft 硬化温度
            dual_forward: 是否启用双路径 (soft + hard) forward
        """
        points_F = lidar.F
        N_total = points_F.shape[0]
        device = points_F.device

        # --- 拼接输入特征 ---
        L_H_state = torch.cat([
            L_aug.unsqueeze(0).expand(N_total, 1),
            H_aug.unsqueeze(0).expand(N_total, 1)
        ], dim=1)
        policy_input = torch.cat([points_F, L_H_state], dim=1)

        p_drop = self.policy_net(policy_input)  # (N, 1) 丢弃概率

        # ============================================================
        # ✅ Gumbel-Soft + Straight-Through Estimator
        # ============================================================
        if dual_forward:
            # 软采样：可微的 Drop mask
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(p_drop) + 1e-9) + 1e-9)
            drop_mask_soft = torch.sigmoid((torch.log(p_drop + 1e-9) - gumbel_noise) / hard_threshold)
            # 硬采样：实际索引使用
            drop_mask_hard = (drop_mask_soft > 0.5).float()
            drop_mask = drop_mask_soft + (drop_mask_hard - drop_mask_soft).detach()
        else:
            # 普通 Bernoulli-ST
            drop_mask_hard = (torch.rand_like(p_drop) > p_drop).float()
            drop_mask = p_drop + (drop_mask_hard - p_drop).detach()

        # ============================================================
        # ✅ 硬索引：用于生成新的 SparseTensor
        # ============================================================
        keep_indices = torch.nonzero(drop_mask_hard.squeeze()).squeeze(1)
        if keep_indices.numel() == 0:
            dropped_lidar = SparseTensor(
                torch.empty(0, points_F.shape[1], device=device),
                torch.empty(0, lidar.C.shape[1], dtype=lidar.C.dtype, device=device)
            )
            actual_drop_ratio = torch.tensor(1.0, device=device)
        else:
            dropped_lidar = SparseTensor(
                points_F[keep_indices],
                lidar.C[keep_indices]
            )
            actual_drop_ratio = 1.0 - (keep_indices.numel() / N_total)

        # ============================================================
        # ✅ 可微 Drop Ratio (STE)
        # ============================================================
        drop_ratio_soft = p_drop.mean()
        drop_ratio_ste = drop_ratio_soft + (actual_drop_ratio - drop_ratio_soft).detach()

        # ============================================================
        # ✅ 双路径正则：soft mask 熵约束（增强梯度学习信号）
        # ============================================================
        if dual_forward:
            mask_entropy = - (drop_mask_soft * torch.log(drop_mask_soft + 1e-9)
                              + (1 - drop_mask_soft) * torch.log(1 - drop_mask_soft + 1e-9))
            mask_entropy = mask_entropy.mean()
            drop_ratio_ste = drop_ratio_ste + 0.05 * mask_entropy  # 调整 0.05 可平衡可微性与稳定性

        return dropped_lidar, drop_ratio_ste, keep_indices

# 辅助函数：计算熵 (同 LPD 论文)
def compute_entropy(logits):
    # logits shape: (N_points, Num_Classes)
    probs = torch.softmax(logits, dim=-1)
    # H = - (1/N) * sum(P * log(P))
    # 这里的平均是针对所有点
    entropy = - (probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
    return entropy  # Scalar


class MinkUNet_Learner(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

        # 确保配置存在且是字典
        ljm_config = kwargs['ljm_config']
        adm_config = kwargs['adm_config']

        ljm_in_channels = ljm_config.get('in_channels', 4)

        self.ljm = LearnableJitterModule(
            in_channels=ljm_in_channels,
            hidden_channels=ljm_config.get('hidden_channels', 32),  # 默认为 32
            max_sigma=ljm_config.get('max_sigma', 0.05)  # 默认为 0.05
        )

        adm_in_channels = adm_config.get('in_channels', 4)

        self.adm = AdversarialDropModule(
            in_channels=adm_in_channels,
            hidden_channels=adm_config.get('hidden_channels', 32)
        )

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        return out, None