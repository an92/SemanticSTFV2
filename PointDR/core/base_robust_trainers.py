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
import tqdm

__all__ = ['BaseRobustTrainer']

# ARSISA 框架的类别感知阈值参数 (已优化为基于语义)
# ** Stuff 类别（背景/常见）使用高阈值，确保伪标签质量 **
CAT_STUFF_THRESHOLD = 0.90
# ** Things 类别（稀有/重要）使用低阈值，确保召回率 **
CAT_THINGS_THRESHOLD = 0.80


class BaseRobustTrainer(Trainer):

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
                 lambda_cl: float = 0.1,  # L_Cons^CAT 权重
                 lambda_gsp: float = 0.05  # L_PSA 权重
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
        self.num_classes = configs.data.num_classes
        self.ignore_index = configs.data.ignore_label

        self.eval_interval = 500

        # ARSISA 核心权重
        self.lamda_cons = lambda_cl
        self.lambda_psa = lambda_gsp

        self.things_class_ids = things_class_ids
        self.thing_adv_weight = things_weights
        self.stuff_adv_weight = stuff_weights

        # L2 损失，用于 L_PSA
        self.criterion_l2 = nn.MSELoss(reduction='none').cuda()

        # 在初始化时，预计算类别阈值张量 (C1: 类别感知阈值)
        self.cat_thresholds = torch.ones(self.num_classes, device='cuda') * CAT_STUFF_THRESHOLD
        for thing_id in self.things_class_ids:
            if 0 <= thing_id < self.num_classes:
                self.cat_thresholds[thing_id] = CAT_THINGS_THRESHOLD

    def get_prototypes(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        [C2] 计算批次内的类别原型 (语义中心)
        (代码保持不变)
        """
        feat_dim = features.shape[1]
        prototypes = torch.zeros((self.num_classes, feat_dim), device='cuda')

        for class_id in range(self.num_classes):
            mask = (targets == class_id)
            if mask.sum() > 0:
                prototypes[class_id] = features[mask].mean(dim=0)

        return prototypes

    def get_cat_mask(self, prob_w: torch.Tensor) -> torch.Tensor:
        """
        [C1] 类别感知自适应阈值 (CAT) 掩码计算
        - 根据 self.things_class_ids 来查找对应阈值。
        """
        # 1. 获取 Max Confidence 和伪标签
        max_confidence, pseudo_label = torch.max(prob_w, dim=1)

        # 2. 查找每个点的伪标签对应的阈值 T_c
        # 使用伪标签 pseudo_label (形状: [N]) 索引预计算的 self.cat_thresholds (形状: [C])
        point_thresholds = self.cat_thresholds[pseudo_label]

        # 3. 生成 CAT 掩码
        # 只有当置信度 > 对应类别的阈值时，才通过
        cat_mask = (max_confidence > point_thresholds).float()

        return cat_mask, pseudo_label

    def _before_epoch(self) -> None:
        self.model.train()
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

        inputs_2 = _inputs['lidar_2']

        with amp.autocast(enabled=self.amp_enabled):
            # 1. 前向传播：弱视图 x^W (用于生成伪标签和原型 W)
            outputs_1, feat_1, _ = self.model(inputs_1)

            if outputs_1.requires_grad:

                # --- 1. L_Focus (C3) ---
                criterion_none = nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index,
                    reduction='none'
                ).cuda()
                loss_1_seg_pointwise = criterion_none(outputs_1, targets_1)

                W_Focus = torch.ones_like(targets_1, dtype=torch.float32) * self.stuff_adv_weight
                for thing_id in self.things_class_ids:
                    W_Focus[targets_1 == thing_id] = self.thing_adv_weight

                ignore_mask = (targets_1 != self.ignore_index).float()
                W_Focus = W_Focus * ignore_mask

                loss_Focus = (loss_1_seg_pointwise * W_Focus).sum() / ignore_mask.sum().clamp(min=1)
                loss_Sup = loss_Focus

                # 2. 前向传播：强视图 x^S (用于 L_Cons^CAT 和原型 S)
                pred_2, feat_2, _ = self.model(inputs_2)

                # --- 3. L_Cons^CAT (C1) ---

                # 3.1 从弱视图 logits (outputs_1) 计算置信度
                prob_1 = nn.functional.softmax(outputs_1.detach(), dim=1)

                # [核心 C1] 获取类别感知掩码和伪标签
                # CAT mask 根据伪标签的类别来动态调整阈值
                cat_mask, pseudo_label = self.get_cat_mask(prob_1)

                # 3.2 计算逐点一致性损失 (L_CE(P^S, y_hat^W))
                loss_cons_pointwise = criterion_none(pred_2, pseudo_label)

                # 3.3 计算加权的 L_Cons (仅高置信度且非忽略标签的点参与)
                total_mask_cons = cat_mask * ignore_mask
                num_valid_cons = total_mask_cons.sum().clamp(min=1)
                loss_cons = (loss_cons_pointwise * total_mask_cons).sum() / num_valid_cons

                # --- 4. L_PSA (C2) ---

                # [核心 C2] 原型级结构对齐 (PSA)

                # 4.1 归一化特征
                feat_1_norm = nn.functional.normalize(feat_1, dim=1)
                feat_2_norm = nn.functional.normalize(feat_2, dim=1)

                # 4.2 计算原型 (使用 targets_1 真实标签来定位特征)
                P_W = self.get_prototypes(feat_1_norm, targets_1)
                P_S = self.get_prototypes(feat_2_norm, targets_1)

                # 4.3 L_PSA 计算 (L2 损失)
                valid_prototype_mask = (P_W.sum(dim=1).abs() > 1e-6).float()  # 确保有特征参与计算

                loss_psa_pointwise = self.criterion_l2(P_W, P_S).mean(dim=1)

                # 加权 PSA 损失
                loss_psa = (loss_psa_pointwise * valid_prototype_mask).sum() / valid_prototype_mask.sum().clamp(min=1)

                # --- 5. 最终损失 ---
                loss = loss_Sup + self.lamda_cons * loss_cons + self.lambda_psa * loss_psa

                self.summary.add_scalar('loss', loss.item())
                self.summary.add_scalar('loss_Sup', loss_Sup.item())
                self.summary.add_scalar('loss_Cons', loss_cons.item())
                self.summary.add_scalar('loss_PSA', loss_psa.item())

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                return {'outputs': outputs_1, 'targets': targets_1}

            else:
                # 评估模式 (代码保持不变)
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
        if 'scaler' in state_dict:
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
    mIoU = MeanIoU(name=f'iou/test_', num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label)
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