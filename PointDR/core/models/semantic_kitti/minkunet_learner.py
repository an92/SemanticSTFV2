import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch.nn.functional as F

__all__ = ['MinkUNet_Learner']

# ---------------------------
# 基本卷积 / 残差 / 转置卷积
# ---------------------------
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
                spnn.Conv3d(inc, outc, kernel_size=1, stride=stride),
                spnn.BatchNorm(outc),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


# ---------------------------
# DBAG Generator
# ---------------------------
class DBAGGenerator(nn.Module):
    """
    输出连续扰动权重或 Gumbel-Softmax 离散掩码
    """
    def __init__(self, inc, outc=1, use_gumbel=False, tau=1.0):
        super().__init__()
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(inc, inc // 2),
            nn.BatchNorm1d(inc // 2),
            nn.ReLU(True),
            nn.Linear(inc // 2, outc)
        )

    def forward(self, x_f):
        logits = self.net(x_f)
        if self.use_gumbel and self.training:
            mask = F.gumbel_softmax(torch.cat([logits, -logits], dim=-1),
                                    tau=self.tau, hard=True)[..., 0:1]
        else:
            mask = torch.sigmoid(logits)
        return mask


# ---------------------------
# MultiScale BAWA
# ---------------------------
class MultiScaleBAWA(nn.Module):
    """
    多尺度卷积模拟小波特征
    """
    def __init__(self, inc):
        super().__init__()
        self.branch3 = BasicConvolutionBlock(inc, inc, ks=3)
        self.branch5 = BasicConvolutionBlock(inc, inc, ks=5)
        self.branch7 = BasicConvolutionBlock(inc, inc, ks=7)

    def forward(self, x):
        f3 = self.branch3(x)
        f5 = self.branch5(x)
        f7 = self.branch7(x)
        out = f3 + f5 + f7
        return out


# ---------------------------
# MinkUNet Learner
# ---------------------------
class MinkUNet_Learner(nn.Module):
    def __init__(self, num_classes, cr=1.0, run_up=True, use_gumbel=False):
        super().__init__()
        self.run_up = run_up
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        # Stem
        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True)
        )

        # Downsample stages
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

        # Upsample stages
        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5]+cs[3], cs[5]),
                ResidualBlock(cs[5], cs[5])
            )
        ])
        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6]+cs[2], cs[6]),
                ResidualBlock(cs[6], cs[6])
            )
        ])
        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7]+cs[1], cs[7]),
                ResidualBlock(cs[7], cs[7])
            )
        ])
        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8]+cs[0], cs[8]),
                ResidualBlock(cs[8], cs[8])
            )
        ])

        self.classifier = nn.Linear(cs[8], num_classes)

        # --- DBAG Generators ---
        self.generator_x0 = DBAGGenerator(inc=cs[0], use_gumbel=use_gumbel)
        self.generator_x1 = DBAGGenerator(inc=cs[1], use_gumbel=use_gumbel)

        # --- BAWA multi-scale features ---
        self.bawa_x0 = MultiScaleBAWA(cs[0])
        self.bawa_x1 = MultiScaleBAWA(cs[1])
        self.bawa_x2 = MultiScaleBAWA(cs[2])

        self.dropout = nn.Dropout(0.3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, perturb_mode='clean', mask_weights_in=None):
        """
        perturb_mode: 'clean', 'generate', 'apply'
        mask_weights_in: 外部提供 mask，用于 apply 模式
        返回：
            out: 分类输出 N x num_classes
            bawa_feats: [x0_bawa, x1_bawa, x2_bawa]
            mask_weights: [mask_x0, mask_x1]
        """
        # --- Stem ---
        x0 = self.stem(x)

        # --- Downsample ---
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # --- DBAG ---
        mask_weights = []
        if self.training and perturb_mode == 'generate':
            mask_x0 = self.generator_x0(x0.F)
            x0.F = x0.F * mask_x0.expand(-1, x0.F.shape[1])
            x0.F = self.dropout(x0.F)
            mask_x1 = self.generator_x1(x1.F)
            x1.F = x1.F * mask_x1.expand(-1, x1.F.shape[1])
            x1.F = self.dropout(x1.F)
            mask_weights = [mask_x0, mask_x1]

        elif perturb_mode == 'apply' and mask_weights_in is not None:
            mask_x0, mask_x1 = mask_weights_in
            x0.F = x0.F * mask_x0.expand(-1, x0.F.shape[1])
            x1.F = x1.F * mask_x1.expand(-1, x1.F.shape[1])
            mask_weights = [mask_x0, mask_x1]
        else:
            mask_weights = [torch.ones_like(x0.F[:, :1]), torch.ones_like(x1.F[:, :1])]

        # --- Upsample ---
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

        # --- BAWA multi-scale features ---
        x0_bawa = self.bawa_x0(x0)
        x1_bawa = self.bawa_x1(x1)
        x2_bawa = self.bawa_x2(x2)

        bawa_feats = [x0_bawa, x1_bawa, x2_bawa]

        return out, bawa_feats, mask_weights

    def get_bawa_features(self, x):
        """
        用于在 Trainer 中获取 Clean ST 的 BAWA 特征。
        """
        # --- Stem ---
        x0 = self.stem(x)
        # --- Downsample ---
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        # x3 = self.stage3(x2)
        # x4 = self.stage4(x3)

        # --- BAWA multi-scale features ---
        x0_bawa = self.bawa_x0(x0)
        x1_bawa = self.bawa_x1(x1)
        x2_bawa = self.bawa_x2(x2)

        return [x0_bawa, x1_bawa, x2_bawa]
