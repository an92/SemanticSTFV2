import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

__all__ = ['MinkUNet_Learner']

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

class MinkUNet_Learner(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
        )

        # Encoder
        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlock(cs[0], cs[1]),
            ResidualBlock(cs[1], cs[1]),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlock(cs[1], cs[2]),
            ResidualBlock(cs[2], cs[2]),
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2),
            ResidualBlock(cs[2], cs[3]),
            ResidualBlock(cs[3], cs[3]),
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2),
            ResidualBlock(cs[3], cs[4]),
            ResidualBlock(cs[4], cs[4]),
        )

        # Decoder
        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5]),
                ResidualBlock(cs[5], cs[5]),
            )
        ])
        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6]),
                ResidualBlock(cs[6], cs[6]),
            )
        ])
        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7]),
                ResidualBlock(cs[7], cs[7]),
            )
        ])
        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8]),
                ResidualBlock(cs[8], cs[8]),
            )
        ])

        self.classifier = nn.Sequential(
            nn.Linear(cs[8], kwargs['num_classes'])
        )

        # Pointwise transforms & projection head
        self.point_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(cs[0], cs[4]), nn.BatchNorm1d(cs[4]), nn.ReLU(True)),
            nn.Sequential(nn.Linear(cs[4], cs[6]), nn.BatchNorm1d(cs[6]), nn.ReLU(True)),
            nn.Sequential(nn.Linear(cs[6], cs[8]), nn.BatchNorm1d(cs[8]), nn.ReLU(True))
        ])
        self.proj = nn.Sequential(
            nn.Linear(cs[8], cs[8]),
            nn.ReLU(inplace=True),
            nn.Linear(cs[8], 128)
        )

        # Momentum memory banks
        num_classes = kwargs['num_classes']
        proj_dim = 128
        self.m = 0.99
        self.m_global = 0.999
        self.gamma_acp = kwargs['gamma_acp']

        self.register_buffer("memo_bank_B", torch.zeros(num_classes, proj_dim))
        self.register_buffer("memo_bank_G", torch.zeros(num_classes, proj_dim))
        self.register_buffer("class_counts", torch.zeros(num_classes))
        self.r_median =  kwargs['r_median']

        self.weight_initialization()

    @torch.no_grad()
    def momentum_update_B(self, feat_proto_B, init=False):
        if init:
            self.memo_bank_B = feat_proto_B
        else:
            self.memo_bank_B = self.memo_bank_B * self.m + feat_proto_B * (1. - self.m)

    @torch.no_grad()
    def momentum_update_G(self, feat_proto_G, init=False):
        if init:
            self.memo_bank_G = feat_proto_G
        else:
            self.memo_bank_G = self.memo_bank_G * self.m_global + feat_proto_G * (1. - self.m_global)

    @torch.no_grad()
    def get_adaptive_prototype(self, targets_1: torch.Tensor, current_batch_counts: torch.Tensor):
        self.class_counts += current_batch_counts.cpu().to(self.class_counts.device)
        non_zero_counts = self.class_counts[self.class_counts > 0]
        if non_zero_counts.numel() > 0:
            self.r_median = non_zero_counts.median()

        R_c = current_batch_counts.to(self.class_counts.device)
        diff = R_c - self.r_median
        active_classes = (R_c > 0)
        alpha_c = torch.ones_like(R_c) * 0.5
        if self.r_median > 0:
            alpha_c[active_classes] = torch.sigmoid(self.gamma_acp * (diff[active_classes] / self.r_median))
        alpha_c = alpha_c.view(1, -1).to(self.memo_bank_B.device)  # shape: 1 x C

        P_B = self.memo_bank_B.T.detach()  # D x C
        P_G = self.memo_bank_G.T.detach()  # D x C
        P_adaptive = P_B * alpha_c + P_G * (1.0 - alpha_c)  # broadcasting safe

        return P_adaptive, alpha_c.mean().item()

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
        feat = self.proj(y4.F)
        return out, feat
