import numpy as np
import torch
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
import torch.nn as nn

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


class CategoryChannelSensitivity:
    """
    类别-通道敏感矩阵 (CCSM)
    用于抑制对 source 域特定的 channel，从而提升 DG 泛化能力
    """

    def __init__(self, num_classes, feature_dim, reduction=0.8, device='cuda'):
        """
        Args:
            num_classes: 类别数
            feature_dim: 特征维度（channel数）
            reduction: 高敏感通道抑制比例
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.reduction = reduction
        self.device = device
        # 保存每个类别每个通道的重要性
        self.registered_importance = torch.zeros(num_classes, feature_dim, device=device)

    @torch.no_grad()
    def update_importance(self, features, targets):
        """
        统计当前 batch 每个类别每个通道的平均激活
        Args:
            features: [N, C] 特征
            targets: [N] 类别标签
        """
        for c in range(self.num_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                self.registered_importance[c] = features[mask].abs().mean(0)

    def compute_weights(self, targets, percentile=80):
        """
        根据统计的 importance 生成通道权重
        对敏感通道做轻度衰减
        Args:
            targets: [N] batch 中每个点的类别
            percentile: top k% 高敏感通道被抑制
        Returns:
            weights: [N, C] 点特征通道权重
        """
        N, C = targets.shape[0], self.feature_dim
        weights = torch.ones((N, C), device=self.device)

        for c in range(self.num_classes):
            mask = (targets == c)  # [N]
            if mask.sum() == 0:
                continue
            th = torch.quantile(self.registered_importance[c], percentile / 100.0)
            ch_mask = self.registered_importance[c] >= th  # [C]

            # 扩展 mask 用于广播
            mask_expand = mask.unsqueeze(1)  # [N, 1]
            ch_mask_expand = ch_mask.unsqueeze(0)  # [1, C]
            combined_mask = mask_expand & ch_mask_expand  # [N, C]

            weights[combined_mask] *= self.reduction

        return weights

    def apply_weights(self, features, targets, percentile=80):
        """
        应用通道权重到特征
        """
        weights = self.compute_weights(targets, percentile)
        return features * weights


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
        reduction: float = 0.8,
        percentile: int = 80,
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
        self.ignore_label = 255
        self.num_classes = 19
        self.feature_dim = 48
        self.reduction = reduction
        self.percentile = percentile

        self.ccsm = CategoryChannelSensitivity(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            reduction=self.reduction,
        )

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

            outputs, feat = self.model(inputs)

            if outputs.requires_grad:
                with torch.no_grad():
                    self.ccsm.update_importance(feat.detach(), targets)

                feat_weighted = self.ccsm.apply_weights(feat, targets,percentile=self.percentile)
                outputs_weighted = self.model.classifier(feat_weighted)
                loss = self.criterion(outputs_weighted, targets)

        if outputs.requires_grad:
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return {
                'outputs': outputs_weighted,
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
