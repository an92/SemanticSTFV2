import numpy as np
import torch
import torch_cluster
from torch import nn
from torch.cuda import amp
from torch_cluster import knn
from torch_scatter import scatter_mean
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
import torch.nn.functional as F

import time
from typing import Any, Dict, List, Optional, Callable
from torch.utils.data import DataLoader

from torchpack.callbacks import (Callback, Callbacks)
from torchpack.train.exception import StopTraining
from torchpack.train.summary import Summary
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from core.callbacks import MeanIoU
import tqdm

__all__ = ['MinkUnetV3Trainer']


def uncertainty_weight(logits, dim=1, temp=1.0):
    probs = torch.softmax(logits / temp, dim=dim)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)
    C = logits.size(dim)
    # 避免 log(0)
    max_entropy = torch.log(torch.tensor(C, dtype=logits.dtype, device=logits.device) + 1e-8)
    weight = 1.0 - (entropy / max_entropy)
    return weight.detach()


class LocalGeometryCalculator:
    def __init__(self, k_neighbors: int = 16, device='cuda'):
        self.k = k_neighbors
        self.G_dim = 3
        self.device = device

    @torch.no_grad()
    def calculate(self, coords: torch.Tensor) -> torch.Tensor:
        N = coords.size(0)
        device = coords.device

        if N == 0:
            return torch.empty((0, 3), device=device, dtype=torch.float32)

        # 1. 强制坐标为 float32 保证精度和 KNN 兼容性
        P = coords[:, :3].float().contiguous()
        B = coords[:, 3].long()

        # 2. KNN 搜索
        actual_k = self.k+1
        row, col = knn(P, P, k=actual_k, batch_x=B, batch_y=B)

        mask_self = row != col
        row, col = row[mask_self], col[mask_self]

        # 构建邻居索引
        unique_row, counts = torch.unique(row, return_counts=True)
        max_count = counts.max().item()

        neighbors = torch.zeros((N, max_count, 3), device=device, dtype=P.dtype)
        for i, r in enumerate(unique_row):
            idxs = col[row == r]
            neighbors[r, :idxs.numel()] = P[idxs]

        k_used = neighbors.shape[1]
        if k_used < 3:
            # 点太少，返回默认几何特征
            return torch.tensor([0., 0., 1.], device=device).repeat(N, 1)

        # --- 计算协方差矩阵 ---
        centroid = neighbors.mean(dim=1, keepdim=True)
        diffs = neighbors - centroid  # (N, k_used, 3)
        C = torch.einsum('nki,nkj->nij', diffs, diffs) / k_used

        # 对称化，数值稳定
        eps = 1e-6
        C = (C + C.transpose(1, 2)) * 0.5 + torch.eye(3, device=device).unsqueeze(0) * eps
        C = C.to(torch.float32)

        # --- 特征值分解 ---
        L, _ = torch.linalg.eigh(C)  # ascending
        L = L.flip(dims=(-1,))  # descending

        lam1 = L[:, 0].clamp_min(1e-6)
        lam2 = L[:, 1].clamp_min(1e-10)
        lam3 = L[:, 2].clamp_min(1e-10)

        Linearity = (lam1 - lam2) / lam1
        Planarity = (lam2 - lam3) / lam1
        Scattering = lam3 / lam1

        G_all = torch.stack([Linearity, Planarity, Scattering], dim=1)
        G_all = torch.nan_to_num(G_all, nan=0.0, posinf=1.0, neginf=0.0)

        return G_all  # (N,3)


class MinkUnetV3Trainer(Trainer):

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        scheduler: Scheduler,
        num_workers: int,
        seed: int,
        amp_enabled: bool = False,
        lambda_proto: float = 0.0,
        lambda_orth: float = 0.1,
        lambda_style: float = 0.01,
        temp_uncertainty: float = 0.5,
        disentangle_start_epoch: int = 3,
        lambda_aug: float = 1.0,
        decouple_layers: List[str] = None,    # 默认由你传入，None -> ['y3','y4']
    ) -> None:
        # --- keep basic fields ---
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

        # --- hyperparams ---
        self.lambda_proto = lambda_proto
        self.lambda_orth = lambda_orth
        self.lambda_style = lambda_style
        self.temp_uncertainty = temp_uncertainty
        self.ignore_label = 255
        self.disentangle_start_epoch = disentangle_start_epoch
        self.lambda_aug = lambda_aug
        self.num_sample_aux = 4096

        self.decouple_layers = decouple_layers if decouple_layers is not None else ['y3', 'y4']

        self.local_geometry_calculator = LocalGeometryCalculator(k_neighbors=16)

        self.criterion_reduction_none = nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction='none')

        self.summary = None
        self.callbacks = None

    @torch.no_grad()
    def _update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor, momentum: float = 0.99):
        return

    def _calculate_hybrid_proto_loss(self, feats, targets, model):
        return torch.tensor(0., device=feats.device)

    def _before_epoch(self) -> None:
        self.model.train()
        try:
            self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        except Exception:
            pass
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()
        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        batch_size = int(inputs.C[:, -1].max().item() + 1)

        if 'is_augmented' in feed_dict:
            is_augmented = feed_dict['is_augmented'].long().cuda(non_blocking=True)
        else:
            is_augmented = torch.zeros((batch_size,), dtype=torch.long).cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            model_output = self.model(inputs)
            outputs = model_output['logits']
            decoupled_output = model_output['decoupled_features']

            if outputs.requires_grad:
                L_CE_W = self.criterion(outputs, targets)
                # valid_mask = targets != self.ignore_label
                # if not valid_mask.any():
                #     return {'outputs': outputs, 'targets': targets}
                #
                # # --- 1. CE loss with uncertainty weight (Main Loss) ---
                # logits_v = outputs[valid_mask]
                # targets_v = targets[valid_mask]
                # loss_ce_per_point = self.criterion_reduction_none(logits_v, targets_v)
                #
                # weight_v = uncertainty_weight(logits_v, dim=1, temp=self.temp_uncertainty)
                # # 钳制分母，防止除零
                # den = weight_v.sum().clamp_min(1.0)
                # L_CE_W = (loss_ce_per_point * weight_v).sum() / den

                # --- 2. Initialize disentangle losses ---
                L_Orth, L_Style, L_Aug = torch.tensor(0., device=outputs.device), torch.tensor(0.,
                                                                                               device=outputs.device), torch.tensor(
                    0., device=outputs.device)

                if self.epoch_num >= self.disentangle_start_epoch:

                    NUM_SAMPLE = self.num_sample_aux

                    for name in self.decouple_layers:
                        f_content_l = decoupled_output['f_content'][name]  # (N, C)
                        F_raw_l = decoupled_output['f_style'][name]  # (N, C)
                        coords_l = decoupled_output['coords'][name]  # (N, 4)

                        N_layer = f_content_l.shape[0]

                        if N_layer > NUM_SAMPLE:
                            perm = torch.randperm(N_layer, device=outputs.device)[:NUM_SAMPLE]
                            f_content_sub = f_content_l[perm]
                            F_raw_sub = F_raw_l[perm]
                            coords_sub = coords_l[perm]
                        else:
                            f_content_sub = f_content_l
                            F_raw_sub = F_raw_l
                            coords_sub = coords_l

                        # 1. 计算几何特征 (只对采样的点算)
                        G_sub = self.local_geometry_calculator.calculate(coords_sub)  # (M, 3)
                        if G_sub.dtype != F_raw_sub.dtype:
                            G_sub = G_sub.to(F_raw_sub.dtype)

                        # 2. 通过 Style Head
                        F_raw_sub_detached = F_raw_sub.detach()
                        f_style_sub_detached = self.model.style_heads[name](
                            torch.cat([F_raw_sub_detached, G_sub], dim=1))
                        f_style_sub_for_orth = self.model.style_heads[name](torch.cat([F_raw_sub, G_sub], dim=1))
                        f_c_norm = F.normalize(f_content_sub, p=2, dim=1)
                        f_s_norm_orth = F.normalize(f_style_sub_for_orth, p=2, dim=1)  # 使用连接梯度的风格特征

                        dot = torch.sum(f_c_norm * f_s_norm_orth, dim=1)
                        L_Orth += torch.mean(dot ** 2)

                        L_Style += torch.mean(f_style_sub_detached.pow(2).clamp(max=100.0))

                        aug_logits = self.model.aug_classifiers[name](f_style_sub_detached)


                        scene_ids = coords_sub[:, 3].long()

                        # 使用 scatter_mean 聚合 (dim_size=batch_size 保证输出维度正确)
                        mean_logits = scatter_mean(aug_logits, scene_ids, dim=0, dim_size=batch_size)

                        L_Aug += F.cross_entropy(mean_logits, is_augmented)

                    num_layers = len(self.decouple_layers)
                    L_Orth = L_Orth / num_layers
                    L_Style = L_Style / num_layers
                    L_Aug = L_Aug / num_layers

                # --- total loss ---
                loss = L_CE_W + self.lambda_orth * L_Orth + self.lambda_style * L_Style + self.lambda_aug * L_Aug

                if torch.isnan(loss):
                    logger.warning("Loss is NaN (likely from auxiliary terms). Skipping step.")
                    return {'outputs': outputs, 'targets': targets}

                # --- backward ---
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()

                # --- summary logging ---
                self.summary.add_scalar('L_CE_W', float(L_CE_W.item()))
                self.summary.add_scalar('L_Orth', float(L_Orth.item()))
                self.summary.add_scalar('L_StyleReg', float(L_Style.item()))
                self.summary.add_scalar('L_Aug', float(L_Aug.item()))
                self.summary.add_scalar('loss', float(loss.item()))

                return {'outputs': outputs, 'targets': targets}
            else:
                invs = feed_dict['inverse_map']
                all_labels = feed_dict['targets_mapped']
                _outputs = []
                _targets = []
                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)
                outputs = torch.cat(_outputs, 0)
                targets = torch.cat(_targets, 0)
                return {'outputs': outputs, 'targets': targets}

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

    def train(
        self,
        dataflow: DataLoader,
        *,
        num_epochs: int = 9999999,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.dataflow = dataflow
        self.steps_per_epoch = len(self.dataflow)
        self.num_epochs = num_epochs

        if callbacks is None:
            callbacks = []
        self.callbacks = Callbacks(callbacks)
        self.summary = Summary()

        try:
            self.callbacks.set_trainer(self)
            self.summary.set_trainer(self)

            self.epoch_num = 0
            self.global_step = 0

            train_time = time.perf_counter()
            self.before_train()

            while self.epoch_num < self.num_epochs:
                self.epoch_num += 1
                self.local_step = 0

                logger.info('Epoch {}/{} started.'.format(self.epoch_num, self.num_epochs))
                epoch_time = time.perf_counter()
                self.before_epoch()

                for feed_dict in self.dataflow:
                    self.local_step += 1
                    self.global_step += 1

                    self.before_step(feed_dict)
                    output_dict = self.run_step(feed_dict)
                    self.after_step(output_dict)

                    self.trigger_step()

                self.after_epoch()
                logger.info('Training finished in {}.'.format(humanize.naturaldelta(time.perf_counter() - epoch_time)))

                self.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(humanize.naturaldelta(time.perf_counter() - epoch_time)))

            logger.success('{} epochs of training finished in {}.'.format(self.num_epochs, humanize.naturaldelta(time.perf_counter() - train_time)))
        except StopTraining as e:
            logger.info('Training was stopped by {}.'.format(str(e)))
        finally:
            self.after_train()

    def save_model(self, path):
        assert '.pt' in path, "Checkpoint save path is wrong"
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, path)


def evaluate(val_loader, model):
    mIoU = MeanIoU(name=f'iou/test_', num_classes=19, ignore_label=255)
    mIoU.before_epoch()

    with torch.no_grad():
        for feed_dict in tqdm.tqdm(val_loader, ncols=0):
            _inputs = dict()
            for key, value in feed_dict.items():
                if not 'name' in key:
                    _inputs[key] = value.cuda()
            inputs = _inputs['lidar']
            model_output = model(inputs)
            outputs = model_output['logits']

            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs = []
            _targets = []
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
            outputs = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)
            assert not outputs.requires_grad, "produced grad, wrong"
            output_dict = {'outputs': outputs, 'targets': targets}
            mIoU.after_step(output_dict)
    mIoU.after_epoch()
