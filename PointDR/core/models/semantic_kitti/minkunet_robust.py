import torch
import torch.nn as nn
import torchsparse.nn as spnn
import torchsparse
from PointDR.core.models.utils import BasicConvolutionBlock, ResidualBlock, BasicDeconvolutionBlock
import torch.nn.functional as F

__all__ = ['MinkUNet_Robust']

class DeformableResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, dilation=1):
        super().__init__()
        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=stride, dilation=dilation)
        self.bn1 = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU(True)
        self.conv2 = spnn.Conv3d(out_channels, out_channels, kernel_size=ks, stride=1, dilation=dilation)
        self.bn2 = spnn.BatchNorm(out_channels)
        self.attn = nn.MultiheadAttention(out_channels, num_heads=4)  # 简单示例，实际可用可变形实现

        if in_channels != out_channels:
            self.downsample = spnn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None


def knn_bruteforce(query_coords, key_coords, K):
    """
    query_coords: (Nq, 3)
    key_coords: (Nk, 3)
    returns idx (Nq, K) of nearest neighbors in key_coords for each query
    NOTE: naive O(Nq * Nk) mem/computation - replace in production!
    """
    if key_coords.shape[0] == 0:
        return torch.zeros((query_coords.shape[0], 0), dtype=torch.long, device=query_coords.device)
    diff = query_coords.unsqueeze(1) - key_coords.unsqueeze(0)    # (Nq, Nk, 3)
    d2 = (diff**2).sum(-1)    # (Nq, Nk)
    k = min(K, key_coords.shape[0])
    idx = torch.topk(d2, k=k, largest=False).indices    # (Nq, k)
    return idx


class SparseLocalVoxelGeometryFeatureModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SparseLocalVoxelGeometryFeatureModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x is a torchsparse.SparseTensor with .F and .C
        feat = x.F    # (N_points, in_channels)
        newF = self.mlp(feat)    # (N_points, out_channels)
        out = torchsparse.SparseTensor(
            coords=x.C,
            feats=x.F.clone(),
            stride=x.s
        )
        out.F = newF
        return out


class SparseDeformableAttention(nn.Module):

    def __init__(self, in_channels, num_heads=4, K=8, n_levels=3):
        super().__init__()
        assert in_channels % num_heads == 0
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.K = K
        self.n_levels = n_levels
        self.head_dim = in_channels // num_heads

        # projections
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        # offset predictor: predict (num_heads * K * 3) per query (initially zeros)
        self.offset_fc = nn.Linear(in_channels, num_heads * K * 3)

        # per-level scalar weights
        self.level_weights = nn.Parameter(torch.ones(n_levels))

        self._reset_parameters()

    def _reset_parameters(self):
        # init offsets to 0 -> start as regular attention
        nn.init.constant_(self.offset_fc.weight, 0.0)
        nn.init.constant_(self.offset_fc.bias, 0.0)
        nn.init.constant_(self.level_weights, 1.0)

    def forward(self, query: 'SparseTensor', keys: list):
        """
        query: SparseTensor (Nq x C)
        keys: list of SparseTensors [level0, level1, ...], each Nk x C
        returns: SparseTensor with same coords as query (coords preserved)
        """
        q_feats = query.F    # (Nq, C)
        q_coords_all = query.C    # (Nq, 4)
        device = q_feats.device
        Nq = q_feats.shape[0]
        C = self.in_channels

        # linear projections
        Q = self.q_proj(q_feats).view(Nq, self.num_heads, self.head_dim)    # (Nq, H, Dh)
        offsets = self.offset_fc(q_feats).view(Nq, self.num_heads, self.K, 3)    # (Nq, H, K, 3)

        accum = q_feats.new_zeros((Nq, C))
        level_ws = F.softmax(self.level_weights, dim=0)    # (n_levels,)
        batch_idx_q = q_coords_all[:, 0].long()
        q_xyz = q_coords_all[:, 1:].float()

        # iterate levels
        for lvl, key in enumerate(keys):
            key_coords = key.C    # (Nk,4)
            key_feats = key.F    # (Nk, C)
            batch_idx_k = key_coords[:, 0].long()
            k_xyz = key_coords[:, 1:].float()

            per_level_out = q_feats.new_zeros((Nq, C))    # accumulate level outputs in global index space

            # process per batch separately
            unique_batches = torch.unique(batch_idx_q)
            for b in unique_batches:
                mask_q = (batch_idx_q == b)
                mask_k = (batch_idx_k == b)
                if mask_q.sum() == 0 or mask_k.sum() == 0:
                    continue

                q_idx_global = torch.nonzero(mask_q).squeeze(1)    # indices in global arrays
                q_xyz_b = q_xyz[mask_q]    # (nq_b, 3)
                k_xyz_b = k_xyz[mask_k]    # (nk_b, 3)
                k_feats_b = key_feats[mask_k]    # (nk_b, C)

                # naive knn: returns (nq_b, k) indices into k_xyz_b
                nn_idx = knn_bruteforce(q_xyz_b, k_xyz_b, self.K)    # (nq_b, K)

                # sample corresponding key features: (nq_b, K, C)
                k_feats_sampled = k_feats_b[nn_idx]    # fancy indexing

                # projections
                K_proj = self.k_proj(k_feats_sampled)    # (nq_b, K, C)
                V_proj = self.v_proj(k_feats_sampled)    # (nq_b, K, C)

                # reshape for heads
                n_q_b = K_proj.shape[0]
                Kh = K_proj.view(n_q_b, K_proj.shape[1], self.num_heads, self.head_dim)    # (nq_b,K,H,Dh)
                Vh = V_proj.view(n_q_b, K_proj.shape[1], self.num_heads, self.head_dim)    # same

                Qb = Q[q_idx_global]    # (nq_b, H, Dh)
                # attention logits: dot(Q, K) per head -> (nq_b, H, K)
                attn_logits = torch.einsum('nhd,nkhd->nhk', Qb, Kh) / (self.head_dim**0.5)

                # deformable bias: use offset norm as a simple regularizer (smaller offset preferred initially)
                offs_b = offsets[q_idx_global].norm(dim=-1)    # (nq_b, H, K)
                attn_logits = attn_logits - offs_b

                attn = F.softmax(attn_logits, dim=-1)    # (nq_b, H, K)

                # weighted sum over K and heads -> (nq_b, H, Dh)
                out_heads = torch.einsum('nhk,nkhd->nhd', attn, Vh)
                out_heads = out_heads.reshape(n_q_b, -1)    # (nq_b, C)

                # place results into per_level_out at global indices
                per_level_out[q_idx_global] = out_heads

            accum = accum + level_ws[lvl] * per_level_out

        out = self.out_proj(accum)    # (Nq, C)
        out = out + q_feats    # residual
        out_sparse = query.clone()
        out_sparse.F = out
        return out_sparse


class MinkUNet_Robust(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.num_classes = kwargs['num_classes']

        self.use_geometry = kwargs.get('use_geometry', True)
        self.use_deform_attn = kwargs.get('use_deform_attn', True)
        self.deform_K = kwargs.get('deform_K', 8)
        self.deform_heads = kwargs.get('deform_heads', 4)

        if self.use_geometry:
            self.geometry_feature_module = SparseLocalVoxelGeometryFeatureModule(in_channels=4, out_channels=cs[1])
            stem_in_channels = 4 + cs[1]
        else:
            self.geometry_feature_module = None
            stem_in_channels = 4

        self.stem = nn.Sequential(
            spnn.Conv3d(stem_in_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
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

        self.proj = nn.Sequential(
            nn.Linear(cs[8], cs[8]),
            nn.ReLU(inplace=True),
            nn.Linear(cs[8], 128),
        )

        if self.use_deform_attn:
            self.sparse_deform_attn = SparseDeformableAttention(in_channels=cs[4], num_heads=self.deform_heads, K=self.deform_K, n_levels=3)
        else:
            self.sparse_deform_attn = None

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_geometry and self.geometry_feature_module is not None:
            geom = self.geometry_feature_module(x)
            x = torchsparse.cat([x, geom])


        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        if self.sparse_deform_attn is not None:
            keys = [x2, x3, x4]
            x4 = self.sparse_deform_attn(x4, keys)


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

        F_y4 = y4.F    # (N, C_out)

        out = self.classifier(F_y4)

        # feat = self.proj(F_y4)

        return out, None
