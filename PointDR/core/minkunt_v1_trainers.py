import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
from core.callbacks import MeanIoU
import tqdm

__all__ = ['MinkUnetV1Trainer']


class MinkUnetV1Trainer(Trainer):

    def __init__(
            self,
            model: nn.Module,
            criterion: Callable,
            optimizer: Optimizer,
            scheduler: Scheduler,
            num_workers: int,
            seed: int,
            amp_enabled: bool = False,
            lambda_ortho: float=0.1,
            lambda_weather: float=0.5,
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

        self.lambda_ortho = lambda_ortho # 正交损失权重
        self.lambda_weather = lambda_weather  # 天气判别损失权重

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {k: v.cuda() for k, v in feed_dict.items() if 'name' not in k and 'ids' not in k}
        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        if self.model.training:
            inputs_strong = _inputs['lidar_2']
            targets_strong = feed_dict['targets_2'].F.long().cuda(non_blocking=True)

            current_epoch = self.epoch_num
            target_lambda_ortho = self.lambda_ortho

            if current_epoch <= 3:
                # 前 3 个 epoch，关闭正交约束
                actual_lambda_ortho = 0.0
            elif current_epoch <= 7:
                # 4 个 epoch 线性过渡到目标值 (4, 5, 6, 7)
                ratio = (current_epoch - 3) / 4.0  # 比例从 0.0 到 1.0
                actual_lambda_ortho = target_lambda_ortho * ratio
            else:
                # 稳定阶段，使用配置中的最大值 0.1
                actual_lambda_ortho = target_lambda_ortho

            # --------------------- ------
            # Step 1: Strong view forward (为了获取梯度)
            # ---------------------------
            with amp.autocast(enabled=self.amp_enabled):
                # 获取 Strong View 的特征和 Domain Pred
                _, pred_d_strong, feat_strong_F = self.model(inputs_strong, return_feat=True)

                label_strong = torch.ones(pred_d_strong.shape[0], dtype=torch.long, device=pred_d_strong.device)
                loss_d_strong = F.cross_entropy(pred_d_strong, label_strong)

            # ---------------------------
            # Step 2: Gradient Attribution (生成掩码)
            # ---------------------------
            grads = torch.autograd.grad(loss_d_strong, feat_strong_F, retain_graph=True)[0]
            sensitivity = torch.mean(torch.abs(grads), dim=0)
            channel_mask = (1.0 - sensitivity).detach()

            # ---------------------------
            # Step 3: Weak View Forward
            # ---------------------------
            with amp.autocast(enabled=self.amp_enabled):
                # out_weak 使用了掩码特征; pred_d_weak 使用了原始特征 (在模型内部处理)
                out_weak, pred_d_weak = self.model(inputs, force_mask=channel_mask)

                # 1. Segmentation Loss
                if hasattr(inputs, 'inverse_map'):
                    invs = inputs.inverse_map
                    out_weak_full = torch.zeros_like(targets, dtype=out_weak.dtype, device=out_weak.device)
                    out_weak_full[invs.F[:, 0]] = out_weak
                    loss_seg_weak = self.criterion(out_weak_full, targets)
                else:
                    assert out_weak.shape[0] == targets.shape[0]
                    loss_seg_weak = self.criterion(out_weak, targets)

                # 2. Domain Loss (Weak)
                label_weak = torch.zeros(pred_d_weak.shape[0], dtype=torch.long, device=pred_d_weak.device)
                loss_d_weak = F.cross_entropy(pred_d_weak, label_weak)

                loss_domain = 0.5 * (loss_d_strong + loss_d_weak)

            # ---------------------------
            # Step 4: Strong View Consistency (Seg)
            # ---------------------------
            with amp.autocast(enabled=self.amp_enabled):
                # Strong View Seg Output
                out_strong, _ = self.model(inputs_strong, force_mask=channel_mask)

                loss_seg_strong = self.criterion(out_strong, targets_strong)
                loss_seg = loss_seg_weak + loss_seg_strong

                # 正交约束
                feat_strong_refined = feat_strong_F * channel_mask
                ortho_loss = torch.abs(F.cosine_similarity(feat_strong_refined, grads.detach(), dim=1)).mean()

                # 总 Loss
                total_loss = loss_seg + self.lambda_weather * loss_domain + actual_lambda_ortho * ortho_loss

            self.summary.add_scalar('loss/total', total_loss.item())
            self.summary.add_scalar('loss/seg', loss_seg.item())
            self.summary.add_scalar('loss/domain', loss_domain.item())
            self.summary.add_scalar('loss/ortho', ortho_loss.item())
            self.summary.add_scalar('stats/mask_mean', channel_mask.mean().item())

            self.summary.add_scalar('stats/actual_lambda_ortho', actual_lambda_ortho)

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            return {
                'outputs': out_weak,
                'targets': targets,
            }

        else:
            with amp.autocast(enabled=self.amp_enabled):
                outputs, _ = self.model(inputs)

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
            # [修正] Eval 时也需要保持输出格式一致，用 _ 接收多余的返回值
            outputs, _, _ = model(inputs, return_feat=True)

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