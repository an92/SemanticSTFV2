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
# 假设 configs 模块已正确导入
from torchpack.utils.config import configs
# 假设 core.callbacks.MeanIoU 已在项目中定义
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
                 things_weights: float = 5.0,  # L_Focus 权重 (RCDW)
                 stuff_weights: float = 1.0,  # L_Focus 权重
                 lambda_cl: float = 0.1,  # L_CL 权重 (原 self.lamda)
                 lambda_gsp: float = 0.05  # L_GSP 权重
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

        # GAFA-Lite 核心权重
        self.lamda = lambda_cl  # L_CL 权重
        self.T = 0.07  # 对比学习温度

        self.things_class_ids = things_class_ids
        self.thing_adv_weight = things_weights
        self.stuff_adv_weight = stuff_weights
        self.lambda_gsp = lambda_gsp

    def _before_epoch(self) -> None:
        self.model.train()
        # 确保 dataflow 和 dataflow.sampler 存在
        if hasattr(self, 'dataflow') and hasattr(self.dataflow, 'sampler'):
            self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()

        inputs_1 = _inputs['lidar']  # 弱视图 x^W
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)

        # 强视图输入和标签
        inputs_2 = _inputs['lidar_2']
        targets_2 = feed_dict['targets_2'].F.long().cuda(non_blocking=True)

        targets_cuda = targets_1  # 主要使用 targets_1 作为监督和对比目标

        with amp.autocast(enabled=self.amp_enabled):
            # 1. 前向传播：弱视图 x^W (L_Sup, Memory Bank Update)
            outputs_1, feat_1, _ = self.model(inputs_1)

            if outputs_1.requires_grad:

                # --- [GAFA: L_Focus] 类别聚焦损失 ---

                # [修复 1/2: 解决 TypeError - 临时创建 reduction='none' 损失函数]
                # 1. 安全获取 ignore_index (来自原始 criterion)
                try:
                    ignore_index = self.criterion.ignore_index
                except AttributeError:
                    # 如果没有该属性，则使用配置中的默认值
                    ignore_index = configs.data.ignore_label

                    # 2. 临时创建一个 reduction='none' 的实例，用于逐点损失计算
                # 必须将其放在 CUDA 上
                criterion_none = nn.CrossEntropyLoss(
                    ignore_index=ignore_index,
                    reduction='none'
                ).cuda()

                # 1.1 计算逐点分割损失 (用于加权)
                loss_1_seg_pointwise = criterion_none(outputs_1, targets_1)

                # 1.2 RCDW (类别聚焦加权)
                W_Focus = torch.ones_like(targets_1, dtype=torch.float32) * self.stuff_adv_weight
                for thing_id in self.things_class_ids:
                    W_Focus[targets_1 == thing_id] = self.thing_adv_weight

                # 屏蔽掉 ignore_label
                ignore_mask = (targets_1 != ignore_index).float()
                W_Focus = W_Focus * ignore_mask

                # 计算加权损失（只对非忽略标签的像素进行平均）
                # loss_1_seg_pointwise 是一个稀疏张量，这里使用 sum() / sum(mask) 实现 mean()
                loss_Focus = (loss_1_seg_pointwise * W_Focus).sum() / ignore_mask.sum().clamp(min=1)

                loss_Sup = loss_Focus  # L_Sup 被 L_Focus 取代

                # 2. 前向传播：强视图 x^S (L_CL Query, L_GSP Query)
                pred_2, feat_2, _ = self.model(inputs_2)

                # 3. [GAFA: L_CL] 特征一致性损失 (MoCo/PointDR 风格)
                feat_1_norm = nn.functional.normalize(feat_1.detach(), dim=1)  # Key Feature (Detached)
                feat_2_norm = nn.functional.normalize(feat_2, dim=1)  # Query Feature

                # Memory Bank 原型更新
                feat1_proto = torch.zeros((configs.data.num_classes, feat_1_norm.shape[1]), device='cuda')
                for ii in range(configs.data.num_classes):
                    mask = (targets_cuda == ii)
                    if mask.sum():
                        feat1_proto[ii] = feat_1_norm[mask].mean(dim=0)
                feat1_proto = (feat1_proto + 1e-8).cuda()
                self.model.momentum_update_key_encoder(feat1_proto, init=(self.global_step == 1))

                # L_CL 计算: 使用原 self.criterion (默认 reduction='mean')
                logits = torch.mm(feat_2_norm, self.model.memo_bank.T.detach())
                logits /= self.T
                loss_cl = self.criterion(logits, targets_2)

                # --- 4. [GAFA: L_GSP] 几何结构保持损失 ---
                # [修复 2/2: 解决 RuntimeError - 使用 Batch-Level 对齐]

                # 计算批次平均特征（仅对非零特征）
                feat_1_avg = feat_1.mean(dim=0)
                feat_2_avg = feat_2.mean(dim=0)

                # L_GSP (使用批次特征平均 L2 损失，避免尺寸不匹配)
                loss_gsp = torch.mean(torch.pow(feat_1_avg - feat_2_avg, 2))

                # 5. 最终损失
                loss = loss_Sup + self.lamda * loss_cl + self.lambda_gsp * loss_gsp

                self.summary.add_scalar('loss', loss.item())
                self.summary.add_scalar('loss_Sup', loss_Sup.item())
                self.summary.add_scalar('loss_CL', loss_cl.item())
                self.summary.add_scalar('loss_GSP', loss_gsp.item())

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                return {'outputs': outputs_1, 'targets': targets_1}

            else:
                # 评估模式
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
    # 假设 configs.data.num_classes 和 configs.data.ignore_label 存在
    mIoU = MeanIoU(name=f'iou/test_', num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label)
    mIoU.before_epoch()

    with torch.no_grad():
        for feed_dict in tqdm.tqdm(val_loader, ncols=0):
            _inputs = dict()
            for key, value in feed_dict.items():
                if not 'name' in key:
                    _inputs[key] = value.cuda()
            inputs = _inputs['lidar']
            # 评估时，只返回 outputs 和 feature
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