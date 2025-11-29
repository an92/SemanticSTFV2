import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from PointDR.core.models.utils import BasicConvolutionBlock, ResidualBlock, BasicDeconvolutionBlock

__all__ = ['MinkUNetV3']


class FeatureDisentangle(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.content_proj = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU(), nn.Linear(in_channels, in_channels))

        self.style_proj = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU(), nn.Linear(in_channels, in_channels))

        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        # modulation 参数 γ, β
        self.gamma_gen = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        self.beta_gen = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh()  # 约束 beta 在 [-1, 1] 之间
        )
    def forward(self, f):
        f_content = self.content_proj(f)
        f_style = self.style_proj(f)

        # modulation 参数
        gamma_raw = self.gamma_gen(f_style)
        beta = self.beta_gen(f_style)

        gamma = 1.0 + 0.5 * (gamma_raw - 0.5)  # 从 [0,1] 映射到 [0.5,1.5]

        # feature modulation
        f_content_norm = self.norm(f_content)
        f_modulated = gamma * f_content_norm + beta

        f_final = f + f_modulated

        return f_content, f_style, f_final


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
        self.decouple_layers = {
            'y3': cs[7],
            'y4': cs[8],
        }

        self.disentangle = nn.ModuleDict({
            'y3': FeatureDisentangle(cs[7]),
            'y4': FeatureDisentangle(cs[8]),
        })


        self.num_classes = kwargs['num_classes']

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

        feature_map = {'y3': y3.F, 'y4': y4.F}
        coord_map = {'y3': y3.C, 'y4': y4.C}

        f_content_dict = {}
        f_style_dict = {}
        f_final_dict = {}

        # Feature Disentangle + Residual Modulation
        for name, F_raw in feature_map.items():
            f_c, f_s, f_final = self.disentangle[name](F_raw)
            f_content_dict[name] = f_c
            f_style_dict[name] = f_s
            f_final_dict[name] = f_final

        logits = self.classifier(f_final_dict['y4'])

        decoupled_features = {
            'f_content': f_content_dict,
            'f_style': f_style_dict,
            'f_final': f_final_dict,
            'coords': coord_map
        }

        return {
            'logits': logits,
            'decoupled_features': decoupled_features
        }

