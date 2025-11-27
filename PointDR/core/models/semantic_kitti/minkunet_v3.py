import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse
import torchsparse.nn as spnn

__all__ = ['MinkUNetV3']


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


def create_proj_head(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim * 2),
        nn.BatchNorm1d(output_dim * 2),
        nn.ReLU(True),
        nn.Linear(output_dim * 2, output_dim),
    )


def create_aug_classifier(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(True),
        nn.Linear(64, 2),    # 预测 0 (原始) 或 1 (增强)
    )


class MinkUNetV3(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

        self.stem = nn.Sequential(spnn.Conv3d(4, cs[0], kernel_size=3, stride=1), spnn.BatchNorm(cs[0]), spnn.ReLU(True), spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1), spnn.BatchNorm(cs[0]),
                                  spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1), ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
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

        self.num_classes = kwargs['num_classes']
        self.content_dim = kwargs.get('content_dim', 128)
        self.style_dim = kwargs.get('style_dim', 64)
        self.G_dim = 3    # 几何统计量维度 (线性度, 平面度, 散射度)
        self.decouple_layers = {
            'x3': cs[3],    # Stage 3 skip connection
            'x4': cs[4],    # Stage 4 output
            'y2': cs[6],    # Up2 output
            'y3': cs[7],    # Up3 output
            'y4': cs[8],    # Final output
        }

        self.content_dim = 128
        self.style_dim = 128
        self.num_classes = 19

        self.content_heads = nn.ModuleDict({name: create_proj_head(inc, self.content_dim) for name, inc in self.decouple_layers.items()},)
        self.style_heads = nn.ModuleDict({name: create_proj_head(inc + 3, self.style_dim) for name, inc in self.decouple_layers.items()},)
        self.aug_classifiers = nn.ModuleDict({name: create_aug_classifier(self.style_dim) for name in self.decouple_layers.keys()},)

        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.content_dim))
        self.register_buffer("is_proto_init", torch.zeros(self.num_classes, dtype=torch.bool))

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, spnn.BatchNorm)):
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

        # 分类头
        logits = self.classifier(y4.F)

        feature_map = {'x3': x3.F, 'x4': x4.F, 'y2': y2.F, 'y3': y3.F, 'y4': y4.F}
        coord_map = {'x3': x3.C, 'x4': x4.C, 'y2': y2.C, 'y3': y3.C, 'y4': y4.C}

        # compute content projections here
        f_content = {}
        for name, F_raw in feature_map.items():
            f_c = self.content_heads[name](F_raw)
            # normalize content at final layer y4 (used by proto loss / contrastive later)
            if name == 'y4':
                f_c = F.normalize(f_c, p=2, dim=1)
            f_content[name] = f_c

        # IMPORTANT: do NOT compute style projection here — Trainer will compute G and then call style_heads.
        # So we return f_style as the raw UNet features (F_raw) so trainer can concat G and call style_heads.
        f_style_raw = {name: F_raw for name, F_raw in feature_map.items()}

        decoupled_features = {
            'f_content': f_content,  # projected content features
            'f_style': f_style_raw,  # raw UNet features to be used with G in Trainer
            'coords': coord_map,  # coordinates for each layer (SparseTensor.C)
        }

        return {
            'logits': logits,
            'decoupled_features': decoupled_features,
            'prototypes': self.prototypes,
        }
