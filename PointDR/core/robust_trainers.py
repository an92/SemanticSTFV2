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
import tqdm

__all__ = ['RobustTrainer']


class RobustTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False,
                 things_class_ids: list = [],
                 num_classes: int = 19,
                 lamda_CL: float = 0.1,
                 lamda_SCRL: float = 0.05,
                 lamda_FeatConsist: float = 0.08,  # 新增 FeatConsist 权重
                 repulsion_margin: float = 1.0) -> None:
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

        self.lamda_CL = lamda_CL
        self.lamda_SCRL = lamda_SCRL
        self.lamda_FeatConsist = lamda_FeatConsist
        self.T = 0.07
        self.repulsion_margin = repulsion_margin

        thing_class_ids_list = things_class_ids
        self.thing_class_ids = torch.tensor(thing_class_ids_list, device='cuda')
        all_classes = set(range(num_classes))
        self.stuff_class_ids = torch.tensor(list(all_classes - set(thing_class_ids_list)), device='cuda')

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

        inputs_1 = _inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)

        # --- 1. Weak View: Supervised Loss (L_sup) ---
        with amp.autocast(enabled=self.amp_enabled):
            outputs_1, feat_1 = self.model(inputs_1)
            if outputs_1.requires_grad:
                loss_1 = self.criterion(outputs_1, targets_1)

        if outputs_1.requires_grad:
            inputs_2 = _inputs['lidar_2']
            targets_2 = feed_dict['targets_2'].F.long().cuda(non_blocking=True)

            # --- 2. Strong View: Self-Supervised Losses (L_CL, L_SCRL, L_FeatConsist) ---
            with amp.autocast(enabled=self.amp_enabled):

                pred_2, feat_2 = self.model(inputs_2)

                # 归一化特征 (用于 CL 和 SCRL)
                feat_1_norm = nn.functional.normalize(feat_1, dim=1)
                feat_2_norm = nn.functional.normalize(feat_2, dim=1)

                # ----------------------------------------------------
                # 3. PointDR 对比损失 (L_CL) 和 Target 更新
                # ----------------------------------------------------
                # Step 3.1: 弱视图原型 P_W_norm (用于更新 Memo Bank)
                feat1_proto_W_norm = torch.zeros((configs.data.num_classes, feat_1_norm.shape[1]), device='cuda')
                for ii in range(configs.data.num_classes):
                    mask = (targets_1 == ii)
                    if mask.sum():
                        feat1_proto_W_norm[ii] = feat_1_norm[mask].mean(dim=0)
                feat1_proto_W_norm = nn.functional.normalize(feat1_proto_W_norm, dim=1)

                # Step 3.2: InfoNCE (L_CL)
                logits_CL = torch.mm(feat_2_norm, self.model.memo_bank.T.detach())
                logits_CL /= self.T
                loss_CL = self.criterion(logits_CL, targets_2)

                # Step 3.3: Momentum Update Memo Bank (使用归一化原型)
                self.model.momentum_update_key_encoder(feat1_proto_W_norm.detach(), init=(self.global_step == 1))

                # ----------------------------------------------------
                # 4. 结构化对比关系学习 (SCRL) 损失
                # ----------------------------------------------------
                # SCRL 使用归一化特征的原型 feat2_proto_S 和 Memo Bank (feat1_proto_W_stable)

                # Step 4.1: 强增强归一化原型 P_S_norm (Online)
                feat2_proto_S_norm = torch.zeros((configs.data.num_classes, feat_2_norm.shape[1]), device='cuda')
                for ii in range(configs.data.num_classes):
                    mask = (targets_2 == ii)
                    if mask.sum():
                        feat2_proto_S_norm[ii] = feat_2_norm[mask].mean(dim=0)
                feat2_proto_S_norm = nn.functional.normalize(feat2_proto_S_norm, dim=1)

                # Step 4.2: Target 稳定原型 P_W_stable_norm (Memo Bank)
                feat1_proto_W_stable_norm = self.model.memo_bank.detach()

                # Step 4.3: 计算关系矩阵 M_S 和 M_W (相似度矩阵 - 归一化特征)
                M_S = torch.mm(feat2_proto_S_norm, feat2_proto_S_norm.T)
                M_W = torch.mm(feat1_proto_W_stable_norm, feat1_proto_W_stable_norm.T)

                # Step 4.4 & 4.5: SCRL 损失 (不变)
                loss_consistency = torch.mean((M_S - M_W) ** 2)
                loss_repulsion = torch.zeros(1, device='cuda', dtype=feat_2.dtype)
                for i in self.thing_class_ids:
                    for j in self.stuff_class_ids:
                        D_ij = torch.clamp(M_S[i, j] - M_W[i, j], min=0)
                        distance = torch.sqrt(2.0 - 2.0 * torch.clamp(M_S[i, j], min=-1, max=1))
                        repulsion_term = D_ij * torch.clamp(self.repulsion_margin - distance, min=0)
                        loss_repulsion += repulsion_term
                loss_SCRL = self.lamda_SCRL * (loss_consistency + loss_repulsion)

                # ----------------------------------------------------
                # 5. 特征空间一致性损失 (L_FeatConsist)
                # ----------------------------------------------------

                # Step 5.1: 强视图非归一化原型 P_S_raw (Online)
                feat2_proto_S_raw = torch.zeros((configs.data.num_classes, feat_2.shape[1]), device='cuda')
                for ii in range(configs.data.num_classes):
                    mask = (targets_2 == ii)
                    if mask.sum():
                        feat2_proto_S_raw[ii] = feat_2[mask].mean(dim=0)

                # Step 5.2: 弱视图非归一化原型 P_W_raw (Target - 稳定特征)
                feat1_stable = feat_1.detach()  # 使用弱视图的稳定特征作为目标
                feat1_proto_W_raw = torch.zeros((configs.data.num_classes, feat1_stable.shape[1]), device='cuda')
                for ii in range(configs.data.num_classes):
                    mask = (targets_1 == ii)
                    if mask.sum():
                        feat1_proto_W_raw[ii] = feat1_stable[mask].mean(dim=0)

                # Step 5.3: 类别原型之间的 L2 欧氏距离损失
                # 仅对至少在一个视图中出现的有效类别计算损失 (避免全零向量的 L2 损失)
                valid_classes_mask = (feat2_proto_S_raw.abs().sum(dim=1) > 0) | (feat1_proto_W_raw.abs().sum(dim=1) > 0)

                if valid_classes_mask.sum() > 0:
                    loss_FeatConsist = torch.mean(
                        (feat2_proto_S_raw[valid_classes_mask] - feat1_proto_W_raw[valid_classes_mask]) ** 2
                    )
                else:
                    loss_FeatConsist = torch.zeros(1, device='cuda', dtype=feat_2.dtype)

                loss = loss_1 + self.lamda_CL * loss_CL + loss_SCRL + self.lamda_FeatConsist * loss_FeatConsist

                self.summary.add_scalar('loss', loss.item())
                self.summary.add_scalar('loss_1_sup', loss_1.item())
                self.summary.add_scalar('loss_CL', loss_CL.item())
                self.summary.add_scalar('loss_SCRL_consist', loss_consistency.item())
                self.summary.add_scalar('loss_SCRL_rep', loss_repulsion.item())
                self.summary.add_scalar('loss_FeatConsist', loss_FeatConsist.item())  # 记录新损失

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
            outputs, _ = model(inputs)

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