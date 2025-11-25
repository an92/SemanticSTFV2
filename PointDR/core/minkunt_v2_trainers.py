import numpy as np
import torch
from torch import nn
from torch.cuda import amp
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

__all__ = ['MinkUnetV2Trainer']


def uncertainty_weight(logits, dim=1, temp=1.0):
    probs = torch.softmax(logits / temp, dim=dim)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)
    C = logits.size(dim)
    # 避免 log(0)
    max_entropy = torch.log(torch.tensor(C, dtype=logits.dtype, device=logits.device) + 1e-8)
    weight = 1.0 - (entropy / max_entropy)
    return weight.detach()


class MinkUnetV2Trainer(Trainer):

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        scheduler: Scheduler,
        num_workers: int,
        seed: int,
        amp_enabled: bool = False,
        lambda_proto: float = 100.0,
        lambda_orth: float = 0.1,
        lambda_style: float = 0.01,
        temp_uncertainty: float = 0.5,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

        self.eval_interval = 500

        self.lambda_proto = lambda_proto
        self.lambda_orth = lambda_orth
        self.lambda_style = lambda_style
        self.temp_uncertainty = temp_uncertainty
        self.ignore_label = 255

        self.criterion_reduction_none = nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction='none')

    @torch.no_grad()
    def _update_prototypes(self, feats, targets, model, momentum=0.99):

        device = feats.device
        num_classes, feat_dim = model.prototypes.shape
        momentum = getattr(model, "proto_momentum", 0.99)

        proto_sum = torch.zeros_like(model.prototypes, device=device)
        proto_count = torch.zeros(num_classes, device=device)

        targets_long = targets.long()
        proto_sum.index_add_(0, targets_long, feats)
        proto_count.index_add_(0, targets_long, torch.ones_like(targets_long, dtype=feats.dtype))

        valid_mask = proto_count > 0
        if not valid_mask.any():
            return

        batch_protos = torch.zeros_like(model.prototypes, device=device)
        batch_protos[valid_mask] = proto_sum[valid_mask] / proto_count[valid_mask].unsqueeze(1)

        new_init_mask = valid_mask & (~model.is_proto_init)
        if new_init_mask.any():
            batch_to_init = batch_protos[new_init_mask]
            norm = batch_to_init.norm(dim=1, keepdim=True).clamp_min(1e-6)
            model.prototypes[new_init_mask] = batch_to_init / norm
            model.is_proto_init[new_init_mask] = True

        ema_mask = valid_mask & model.is_proto_init
        if ema_mask.any():
            model.prototypes[ema_mask] = F.normalize(
                momentum * model.prototypes[ema_mask] + (1 - momentum) * batch_protos[ema_mask],
                dim=1
            )


    def _calculate_hybrid_proto_loss(self, feats, targets, model):
        """
        计算修复后的混合原型损失 (InfoNCE L_attraction + ReLU L_repulsion)。
        """

        self._update_prototypes(feats, targets, model)

        # 2. 取已初始化的原型
        proto_mask = model.is_proto_init
        if not proto_mask.any():
            return torch.tensor(0., device=feats.device)

        prototypes = model.prototypes[proto_mask]  # [K, C]
        # 只选择当前 batch 中存在的类别
        batch_classes = torch.unique(targets)
        valid_classes_mask = proto_mask.clone()
        valid_classes_mask[
            ~torch.isin(torch.arange(model.prototypes.shape[0], device=feats.device), batch_classes)] = False
        if valid_classes_mask.sum() == 0:
            return torch.tensor(0., device=feats.device)

        prototypes = model.prototypes[valid_classes_mask]

        # 3. 相似度 / tau
        tau = getattr(model, "tau", 0.1)
        sim_matrix = torch.matmul(feats, prototypes.t()) / tau

        # --- L_attraction (InfoNCE) ---
        # targets 映射到 prototypes 的索引
        class_map = torch.zeros(model.prototypes.shape[0], dtype=torch.long, device=feats.device) - 1
        class_map[valid_classes_mask] = torch.arange(valid_classes_mask.sum(), device=feats.device)
        valid_targets = class_map[targets.long()]
        # 避免-1 index
        mask = valid_targets >= 0
        if mask.sum() == 0:
            return torch.tensor(0., device=feats.device)

        L_attraction = F.nll_loss(F.log_softmax(sim_matrix[mask], dim=1), valid_targets[mask])

        # --- L_repulsion (Hard Negative ReLU)
        sim_matrix_hard = sim_matrix.clone()
        sim_matrix_hard[torch.arange(sim_matrix_hard.size(0), device=feats.device), valid_targets.clamp_min(0)] = -1e4
        neg_sim, _ = sim_matrix_hard.max(dim=1)
        L_repulsion = torch.relu(neg_sim).mean()

        return L_attraction + getattr(model, "lambda_repulsion", 0.5) * L_repulsion

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()

        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            outputs, f_content, f_style = self.model(inputs)

            if outputs.requires_grad:

                valid_mask = targets != self.ignore_label

                if valid_mask.any():
                    logits_v = outputs[valid_mask]
                    targets_v = targets[valid_mask]
                    f_content_v = f_content[valid_mask]
                    f_style_v = f_style[valid_mask]

                    # 1. L_CE (不确定性加权)
                    dynamic_weights = uncertainty_weight(logits_v, dim=1, temp=self.temp_uncertainty)
                    loss_ce_per_point = self.criterion_reduction_none(logits_v, targets_v)
                    den = dynamic_weights.sum().clamp_min(1e-6)
                    L_CE_W = (loss_ce_per_point * dynamic_weights).sum() / den

                    # 2. L_Proto (混合对比式)
                    L_Proto_Hybrid = self._calculate_hybrid_proto_loss(f_content_v, targets_v, self.model)

                    # 3. L_Orth (正交约束)
                    dot_product = torch.sum(f_content_v * f_style_v, dim=1)
                    L_Orth = torch.mean(dot_product**2)

                    # 4. L_StyleReg (风格抑制)
                    L_StyleReg = torch.mean(f_style_v**2)

                    loss = L_CE_W + self.lambda_proto * L_Proto_Hybrid + \
                           self.lambda_orth * L_Orth + self.lambda_style * L_StyleReg

                    self.summary.add_scalar('L_CE_W', L_CE_W.item())
                    self.summary.add_scalar('L_Proto_Hybrid', L_Proto_Hybrid.item())
                    self.summary.add_scalar('L_Orth', L_Orth.item())
                    self.summary.add_scalar('L_StyleReg', L_StyleReg.item())
                    self.summary.add_scalar('loss', loss.item())

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                else:
                    self.summary.add_scalar('skipped_steps', 1)

                return {
                    'outputs': outputs,
                    'targets': targets,
                }
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

                return {
                    'outputs': outputs,
                    'targets': targets,
                }

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
            outputs, _, _ = model(inputs)

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
