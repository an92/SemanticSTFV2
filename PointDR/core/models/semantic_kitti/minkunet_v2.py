import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchsparse
import torchsparse.nn as spnn

__all__ = ['MinkUNetV2', 'MinkUNetWithPrototype']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
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
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkUNetV2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.backbone_output_dim = cs[8]  # 记录输出维度供 Wrapper 使用

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1)
        )

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
            nn.Sequential(nn.Linear(cs[0], cs[4]), nn.BatchNorm1d(cs[4]), nn.ReLU(True)),
            nn.Sequential(nn.Linear(cs[4], cs[6]), nn.BatchNorm1d(cs[6]), nn.ReLU(True)),
            nn.Sequential(nn.Linear(cs[6], cs[8]), nn.BatchNorm1d(cs[8]), nn.ReLU(True))
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

        return out, y4.F


class MinkUNetWithPrototype(nn.Module):
    """
    最终稳定版：包含 Loss/Update 顺序修复和冷启动保护。
    """

    def __init__(self, backbone, feature_dim=48, num_classes=19, momentum=0.99):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.momentum = momentum

        # 保持您的原始维度逻辑 (您已确认没问题)
        if hasattr(backbone, 'backbone_output_dim'):
            in_dim = backbone.backbone_output_dim
        else:
            in_dim = 48

            # Projection Head
        self.proj_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, feature_dim)
        )

        # Prototypes
        self.register_buffer("prototypes", torch.randn(num_classes, feature_dim))
        self.register_buffer("is_proto_init", torch.zeros(num_classes, dtype=torch.bool))

        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    def forward(self, x, targets=None):
        logits, raw_feats = self.backbone(x)

        if not self.training or targets is None:
            return logits, raw_feats, {}

        embed_feats = self.proj_head(raw_feats)
        embed_feats = F.normalize(embed_feats, p=2, dim=1)

        mask = targets != 255
        proto_loss = torch.tensor(0.0, device=logits.device)

        if mask.any():
            feats_valid = embed_feats[mask]
            targets_valid = targets[mask]

            # =======================================================
            # [关键修正] 冷启动保护：只对“已初始化”的类别算 Loss
            # =======================================================
            initialized_mask = self.is_proto_init[targets_valid]

            # 只有当存在已初始化的类别时，才计算 Loss
            if initialized_mask.any():
                feats_calc = feats_valid[initialized_mask]
                targets_calc = targets_valid[initialized_mask]

                # 取出旧原型 (Detached)
                proto_old = self.prototypes[targets_calc].detach()

                # 计算相似度与 Loss (Loss 在 Update 之前计算)
                sim = (feats_calc * proto_old).sum(dim=1)
                proto_loss = (1.0 - sim).mean()

            # =======================================================
            # 更新原型 (Update Prototypes)
            # =======================================================
            with torch.no_grad():
                unique_classes = targets_valid.unique()
                for cls in unique_classes:
                    cls_mask = targets_valid == cls
                    feat_mean = feats_valid[cls_mask].mean(dim=0)

                    if dist.is_initialized():
                        dist.all_reduce(feat_mean)
                        feat_mean /= dist.get_world_size()

                    feat_mean = F.normalize(feat_mean, p=2, dim=0)

                    if not self.is_proto_init[cls]:
                        self.prototypes[cls] = feat_mean
                        self.is_proto_init[cls] = True
                    else:
                        self.prototypes[cls] = self.momentum * self.prototypes[cls] + \
                                               (1 - self.momentum) * feat_mean
                        self.prototypes[cls] = F.normalize(self.prototypes[cls], p=2, dim=0)

        return logits, raw_feats, {'proto_loss': proto_loss}