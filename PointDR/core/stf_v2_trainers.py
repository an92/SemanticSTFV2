import os
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

import time
from typing import Any, Dict, List, Optional, Callable
from torch.utils.data import DataLoader

from torchpack.callbacks import (Callback, Callbacks)
from torchpack.train.exception import StopTraining
from torchpack.train.summary import Summary
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.config import configs
from core.callbacks import MeanIoU
import pdb
import tqdm

__all__ = ['STFV2Trainer']


class STFV2Trainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False) -> None:
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

        # DUACL & CSCG 超参数
        self.lamda_uacl = 0.1  # DUACL 损失权重 (原 lamda)
        self.lamda_struc = 0.1  # CSCG 结构损失权重
        self.T = 0.07  # InfoNCE 温度
        self.uacl_weight_scale = 1.0  # DUACL 不确定性权重缩放因子 (用于L2距离)

        # *** 改进点：动态 alpha 调整的起始和结束值 ***
        self.uacl_alpha_dist_start = 0.3  # L2 距离权重起始比例
        self.uacl_alpha_dist_end = 0.7  # L2 距离权重结束比例

        self.IGNORE_LABEL = 255  # 假设的忽略标签
        self.NUM_CLASSES = configs.data.num_classes

        # 假设总步数来自 config，如果没有，需要手动设置一个较大的值
        self.MAX_STEPS = getattr(configs.train, 'max_steps', 1000000)

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()

        # 弱视图 (X^w)
        inputs_1 = _inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            outputs_1, feat_1 = self.model(inputs_1)
            if outputs_1.requires_grad:
                loss_ce = self.criterion(outputs_1, targets_1)  # L_ce

        if outputs_1.requires_grad:
            # 强视图 (X^s)
            inputs_2 = _inputs['lidar_2']
            targets_2 = feed_dict['targets_2'].F.long().cuda(non_blocking=True)
            pred_2, feat_2 = self.model(inputs_2)

            # ---- 1. DUACL 动态校准损失 (loss_uacl) ----
            feat_1 = nn.functional.normalize(feat_1, dim=1)
            feat_2 = nn.functional.normalize(feat_2, dim=1)

            # 弱视图特征聚合 (Prototypes from X^w) - 用于 Memory Bank 更新和 U_i 估计目标
            feat1_proto = torch.zeros((self.NUM_CLASSES, feat_1.shape[1])).cuda()
            for ii in range(self.NUM_CLASSES):
                mask = (targets_1 == ii)
                if mask.sum():
                    feat1_proto[ii] = feat_1[mask].mean(dim=0)
            feat1_proto = (feat1_proto + 1e-8).cuda()

            # 不确定性 U_i 估计和动态权重 W_i 计算
            valid_mask_2 = (targets_2 != self.IGNORE_LABEL)
            if valid_mask_2.sum() > 0:

                # --- A. L2 距离不确定性 (W_dist) ---
                feat2_proto_map = torch.zeros_like(feat_2)

                # *** 核心修改一：使用 Memory Bank 的稳定原型 M_proto ***
                M_proto = self.model.memo_bank.T.detach()

                for ii in range(self.NUM_CLASSES):
                    mask = (targets_2 == ii)
                    if mask.sum():
                        # 使用 Memory Bank 中的长期原型作为目标
                        feat2_proto_map[mask] = M_proto[ii].unsqueeze(0)

                        # U_i: L2 距离作为不确定性代理
                U_i = torch.linalg.norm(feat_2[valid_mask_2] - feat2_proto_map[valid_mask_2], dim=1)

                # 动态权重 W_dist: 1 + scale * U_i_normalized
                U_i_normalized = U_i / (U_i.mean() + 1e-8)
                W_dist = 1.0 + self.uacl_weight_scale * U_i_normalized

                # --- B. 预测熵不确定性 (W_entropy) ---
                probs_2 = nn.functional.softmax(pred_2, dim=1)[valid_mask_2]
                log_probs_2 = torch.log(probs_2 + 1e-8)  # 避免log(0)
                H_i = -torch.sum(probs_2 * log_probs_2, dim=1)  # 熵
                H_i_normalized = H_i / (H_i.mean() + 1e-8)
                W_entropy = 1.0 + self.uacl_weight_scale * H_i_normalized

                # --- C. 融合动态权重 W_i *** 核心修改二：动态 alpha 调整 ***

                # 计算当前训练进度比例 (0.0 到 1.0)
                progress = (self.global_step / self.MAX_STEPS)

                # 动态调整 alpha: 从 start (倾向Entropy) 线性过渡到 end (倾向L2 Distance)
                current_alpha_dist = self.uacl_alpha_dist_start + \
                                     (self.uacl_alpha_dist_end - self.uacl_alpha_dist_start) * progress
                current_alpha_dist = min(current_alpha_dist, self.uacl_alpha_dist_end)

                W_i = current_alpha_dist * W_dist + (1.0 - current_alpha_dist) * W_entropy
            else:
                W_i = torch.tensor([]).cuda()

            # InfoNCE Loss 计算
            logits = torch.mm(feat_2, self.model.memo_bank.T.detach())
            logits /= self.T

            targets_valid = targets_2[valid_mask_2]
            logits_valid = logits[valid_mask_2]

            # 应用融合后的动态权重 W_i 到 InfoNCE Loss
            if W_i.numel() > 0:
                log_prob = nn.functional.log_softmax(logits_valid, dim=1)
                positive_log_prob = log_prob[torch.arange(log_prob.size(0)), targets_valid]
                loss_uacl = (- W_i * positive_log_prob).mean()
            else:
                loss_uacl = torch.tensor(0.0, device=feat_1.device)

            # ---- 2. CSCG 结构一致性引导损失 (loss_struc) ----
            P_struct = torch.mm(feat1_proto.detach(), feat1_proto.T.detach())
            P_struct = nn.functional.normalize(P_struct, dim=1)

            feat2_proto_batch = torch.zeros((self.NUM_CLASSES, feat_2.shape[1])).cuda()
            for ii in range(self.NUM_CLASSES):
                mask = (targets_2 == ii)
                if mask.sum():
                    feat2_proto_batch[ii] = feat_2[mask].mean(dim=0)
            feat2_proto_batch = nn.functional.normalize(feat2_proto_batch, dim=1)
            P_current = torch.mm(feat2_proto_batch, feat2_proto_batch.T)

            loss_struc = nn.functional.mse_loss(P_current, P_struct)

            # momentum update memory bank
            self.model.momentum_update_key_encoder(feat1_proto, init=(self.global_step == 1))

            loss = loss_ce + self.lamda_uacl * loss_uacl + self.lamda_struc * loss_struc

            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_ce', loss_ce.item())
            self.summary.add_scalar('loss_uacl', loss_uacl.item())
            self.summary.add_scalar('loss_struc', loss_struc.item())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return {'outputs': outputs_1, 'targets': targets_1}
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs = []
            _targets = []
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (inputs_1.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = outputs_1[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
            outputs = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)

            return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()

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

    def train(self,
              dataflow: DataLoader,
              *,
              num_epochs: int = 9999999,
              callbacks: Optional[List[Callback]] = None) -> None:
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

                logger.info('Epoch {}/{} started.'.format(
                    self.epoch_num, self.num_epochs))
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

            logger.success('{} epochs of training finished in {}.'.format(self.num_epochs, humanize.naturaldelta(
                time.perf_counter() - train_time)))
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
            # targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
            outputs = model(inputs)

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