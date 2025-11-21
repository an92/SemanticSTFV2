import torch
import torch.nn as nn
import torchsparse.nn as spnn
from torchsparse import SparseTensor
import torchsparse

__all__ = ['MinkUNet_PAMix']


class PhysicsAdapter(nn.Module):
    """
    将物理天气参数映射为特征混合系数 lambda
    """

    def __init__(self, num_features):
        super().__init__()
        # 输入维度 3: [Fog, Rain, Intensity]
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_features),  # 输出通道级的混合权重
            nn.Sigmoid()  # 归一化到 0~1 之间
        )

    def forward(self, B, device, dtype):
        # 1. 在 GPU 上模拟物理参数 (Simulate Physics Params)
        # fog: [0, 1], rain: [0, 1], attenuation: [0, 1]
        physics_params = torch.rand(B, 3, device=device, dtype=dtype)

        # 2. 映射到混合系数 lambda
        # 输出形状: (B, C) -> 实现了通道级的精细控制 (Channel-wise Control)
        lambdas = self.mlp(physics_params)
        return lambdas


class PAMix(nn.Module):
    def __init__(self, num_features, p=0.5, eps=1e-3):
        super().__init__()
        self.p = p
        self.eps = eps
        self.num_features = num_features

        self.physics_adapter = PhysicsAdapter(num_features)

    def forward(self, input_feat: SparseTensor) -> SparseTensor:
        # 训练阶段且概率满足时执行
        if not self.training or torch.rand(1, device=input_feat.F.device) >= self.p:
            return input_feat

        F = input_feat.F
        coords = input_feat.C
        device = F.device
        dtype = F.dtype

        batch_indices = coords[:, 0].long()
        if batch_indices.numel() == 0: return input_feat
        B = int(batch_indices.max().item()) + 1
        if B < 2: return input_feat

        # 计算 Batch-wise 统计量 (mu, sigma) - [Robust Implementation]
        F_sum = torch.zeros(B, F.shape[1], device=device, dtype=dtype).scatter_add_(
            0,
            batch_indices.unsqueeze(1).repeat(1, F.shape[1]),
            F
        )
        N_points_i = torch.bincount(batch_indices, minlength=B).float().to(dtype)
        mu_i = F_sum / N_points_i.unsqueeze(1).clamp(min=self.eps)

        F_centered = F - mu_i[batch_indices]

        F_sq_sum = torch.zeros(B, F.shape[1], device=device, dtype=dtype).scatter_add_(
            0,
            batch_indices.unsqueeze(1).repeat(1, F.shape[1]),
            F_centered.pow(2).to(dtype)  # Critical for AMP
        )
        sigma_i = torch.sqrt(F_sq_sum / N_points_i.unsqueeze(1).clamp(min=self.eps) + self.eps)

        # 物理感知混合
        rand_index = torch.randperm(B, device=device)

        mu_target = mu_i[rand_index]
        sigma_target = sigma_i[rand_index]

        # lambdas 形状: (B, C)
        lambdas = self.physics_adapter(B, device, dtype)

        # 混合公式: 物理天气参数决定了要从目标域迁移多少风格过来
        # mu_mix = lambda * mu + (1-lambda) * mu_target
        mu_mix = lambdas * mu_i + (1.0 - lambdas) * mu_target
        sigma_mix = lambdas * sigma_i + (1.0 - lambdas) * sigma_target

        # 风格迁移 (Style Transfer)
        sigma_mix_points = sigma_mix[batch_indices].clamp(min=self.eps)
        mu_mix_points = mu_mix[batch_indices]

        F_mix = F_centered / sigma_i[batch_indices] * sigma_mix_points + mu_mix_points

        F_mix = torch.clamp(F_mix, -10, 10)

        input_feat.F = F_mix
        return input_feat

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x): return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x): return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc),
        )
        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, stride=stride, dilation=1),
                spnn.BatchNorm(outc),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class MinkUNet_PAMix(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

        self.pamix2 = PAMix(num_features=cs[2], p=0.5)
        self.pamix3 = PAMix(num_features=cs[3], p=0.5)

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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: SparseTensor):
        x0 = self.stem(x)
        x1 = self.stage1(x0)

        x2 = self.stage2(x1)
        x2 = self.pamix2(x2)  # 插入 PAMix Stage 2

        x3 = self.stage3(x2)
        x3 = self.pamix3(x3)  # 插入 PAMix Stage 3

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