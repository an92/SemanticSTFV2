import torch
import torch.nn as nn
import torchsparse.nn as spnn
from torch.distributions.beta import Beta
from torchsparse import SparseTensor
import torchsparse.nn.functional as F_sparse
import torchsparse

__all__ = ['MinkUNet_Robust']


class SparseMixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=1.0, eps=1e-3):
        super().__init__()
        self.p = p
        self.eps = eps
        self.register_buffer('alpha_tensor', torch.tensor([alpha]))

    def forward(self, input_feat: SparseTensor) -> SparseTensor:
        if not self.training or torch.rand(1, device=input_feat.F.device) >= self.p:
            return input_feat

        F = input_feat.F
        coords = input_feat.C
        device = F.device

        batch_indices = coords[:, 0].long()
        if batch_indices.numel() == 0:
            return input_feat

        B = int(batch_indices.max().item()) + 1
        if B < 2:
            return input_feat

        alpha = self.alpha_tensor
        beta_dist = Beta(alpha, alpha)

        # batch-wise mean
        F_sum = torch.zeros(B, F.shape[1], device=device, dtype=F.dtype).scatter_add_(
            0,
            batch_indices.unsqueeze(1).repeat(1, F.shape[1]),
            F
        )
        N_points_i = torch.bincount(batch_indices, minlength=B).float().to(F.dtype)
        mu_i = F_sum / N_points_i.unsqueeze(1).clamp(min=self.eps)

        F_centered = F - mu_i[batch_indices]

        # batch-wise std
        F_sq_sum = torch.zeros(B, F.shape[1], device=device, dtype=F.dtype).scatter_add_(
            0,
            batch_indices.unsqueeze(1).repeat(1, F.shape[1]),
            F_centered.pow(2).to(F.dtype)
        )
        sigma_i = torch.sqrt(F_sq_sum / N_points_i.unsqueeze(1).clamp(min=self.eps) + self.eps)

        # Random mix
        rand_index = torch.randperm(B, device=device)
        lambda_val = beta_dist.sample().item()
        mu_mix = lambda_val * mu_i + (1.0 - lambda_val) * mu_i[rand_index]
        sigma_mix = lambda_val * sigma_i + (1.0 - lambda_val) * sigma_i[rand_index]

        # Apply mixed style
        sigma_mix_points = sigma_mix[batch_indices].clamp(min=self.eps)
        F_mix = F_centered / sigma_i[batch_indices] * sigma_mix_points + mu_mix[batch_indices]

        # Clamp to avoid extreme values
        F_mix = torch.clamp(F_mix, -10, 10)

        # 直接修改 F，保持原 SparseTensor 的 cmaps
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

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


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


class MinkUNet_Robust(nn.Module):
    def __init__(self, num_classes, cr=1.0, run_up=True):
        super().__init__()
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = run_up

        self.mixstyle2 = SparseMixStyle(p=0.5, alpha=0.2)
        self.mixstyle3 = SparseMixStyle(p=0.5, alpha=0.2)

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlock(cs[0], cs[1]),
            ResidualBlock(cs[1], cs[1])
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlock(cs[1], cs[2]),
            ResidualBlock(cs[2], cs[2])
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2),
            ResidualBlock(cs[2], cs[3]),
            ResidualBlock(cs[3], cs[3])
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2),
            ResidualBlock(cs[3], cs[4]),
            ResidualBlock(cs[4], cs[4])
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5]),
                ResidualBlock(cs[5], cs[5])
            )
        ])
        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6]),
                ResidualBlock(cs[6], cs[6])
            )
        ])
        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7]),
                ResidualBlock(cs[7], cs[7])
            )
        ])
        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8]),
                ResidualBlock(cs[8], cs[8])
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], num_classes))
        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True)
            )
        ])

        self.dropout = nn.Dropout(0.3, True)
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

        # SparseMixStyle
        x2 = self.mixstyle2(x2)
        x3 = self.stage3(x2)
        x3 = self.mixstyle3(x3)
        x4 = self.stage4(x3)

        # Upsample and skip connections
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

        feat = y4.F

        out = self.classifier(feat)
        
        return out, None
