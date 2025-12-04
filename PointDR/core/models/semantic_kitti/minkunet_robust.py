import torch
import torch.nn as nn
import torchsparse.nn as spnn
import torchsparse
from torch.autograd import Function


__all__ = ['MinkUNet_Robust']

from PointDR.core.models.utils import BasicConvolutionBlock, ResidualBlock, BasicDeconvolutionBlock

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha):
    return GradientReversalLayer.apply(x, alpha)

class WeatherEncoder(nn.Module):
    def __init__(self, inc, outc=128):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(inc, inc, ks=3, stride=1, dilation=1),
            spnn.Conv3d(inc, outc, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.net(x)

class MinkUNet_Robust(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

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

        self.proj = nn.Sequential(
            nn.Linear(cs[8], cs[8]),
            nn.ReLU(inplace=True),
            nn.Linear(cs[8], 128))

        self.E_W = WeatherEncoder(cs[3], 128)

        self.semantic_proj = nn.Sequential(
            nn.Linear(cs[8], cs[8]),
            nn.BatchNorm1d(cs[8]),
            nn.ReLU(True),
            nn.Linear(cs[8], 128)
        )
        self.W_decoder = nn.Sequential(
            BasicDeconvolutionBlock(128, 128, ks=2, stride=2),  # x3 -> x2 resolution
            BasicDeconvolutionBlock(128, 128, ks=2, stride=2),  # x2 -> x1 resolution
            BasicDeconvolutionBlock(128, 128, ks=2, stride=2),  # x1 -> x0/y4 resolution
        )

        self.m = 0.99  # momentum update rate
        self.register_buffer("memo_bank", torch.randn(kwargs['num_classes'], 128))
        self.memo_bank = self.memo_bank * 0.

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    @torch.no_grad()
    def momentum_update_key_encoder(self, feat, init=False):
        """
        Momentum update of the memo_bank
        """
        if init:
            self.memo_bank = feat
        else:
            self.memo_bank = self.memo_bank * self.m + feat * (1. - self.m)


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

        F_y4 = y4.F

        out = self.classifier(F_y4)

        feat_abstract = self.proj(F_y4)

        feat_semantic = self.semantic_proj(F_y4)  # (N, 128)
        feat_W_sparse = self.E_W(x3)

        feat_W_upsample = self.W_decoder(feat_W_sparse)

        feat_W = feat_W_upsample.F  # (N, 128)

        return out, feat_abstract, feat_W, feat_semantic

