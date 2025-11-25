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
        lambda_proto: float = 0.0,
        lambda_orth: float = 0.1,
        lambda_style: float = 0.01,
        temp_uncertainty: float = 0.5,
        disentangle_start_epoch: int=5,
        lambda_aug: float=1.0,
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
        self.disentangle_start_epoch =disentangle_start_epoch
        self.lambda_aug = lambda_aug

        self.criterion_reduction_none = nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction='none')

    @torch.no_grad()
    def _update_prototypes(self, feats, targets, model, momentum=0.99):
        pass


    def _calculate_hybrid_proto_loss(self, feats, targets, model):
        return torch.tensor(0., device=feats.device)

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

        if 'is_augmented' in feed_dict:
            is_augmented = feed_dict['is_augmented'].long().cuda(non_blocking=True)
        else:
            batch_size = inputs.C[:, -1].max() + 1
            is_augmented = torch.zeros((batch_size,), dtype=torch.long).cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            outputs, f_content, f_style, aug_logits = self.model(inputs)

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

                    # 2. 初始化辅助损失
                    L_Proto_Hybrid = torch.tensor(0., device=targets.device)
                    L_Orth = torch.tensor(0., device=targets.device)
                    L_StyleReg = torch.tensor(0., device=targets.device)
                    L_Aug = torch.tensor(0., device=targets.device)

                    if self.epoch_num >= self.disentangle_start_epoch:

                        # L_Proto (现在为 0)
                        L_Proto_Hybrid = self._calculate_hybrid_proto_loss(f_content_v, targets_v, self.model)

                        # L_Orth (正交约束)
                        dot_product = torch.sum(f_content_v * f_style_v, dim=1)
                        L_Orth = torch.mean(dot_product ** 2)

                        # L_StyleReg (风格抑制)
                        L_StyleReg = torch.mean(f_style_v ** 2)

                        # L_Aug (Augmentation 分类损失)
                        batch_size = is_augmented.size(0)
                        L_Aug_list = []
                        for i in range(batch_size):
                            scene_mask = (inputs.C[:, -1] == i).cuda()

                            # 对该场景的 Aug Logits 进行平均池化
                            if scene_mask.any():
                                # aug_logits 是点云级别的，取场景内的均值作为场景特征
                                mean_aug_logits = aug_logits[scene_mask].mean(dim=0).unsqueeze(0)
                                # 监督该场景的 Aug Flag (is_augmented[i] 是该场景的标签)
                                L_Aug_list.append(F.cross_entropy(mean_aug_logits, is_augmented[i].unsqueeze(0)))

                        if L_Aug_list:
                            L_Aug = torch.stack(L_Aug_list).mean()
                        else:
                            L_Aug = torch.tensor(0., device=targets.device)

                    loss = L_CE_W + self.lambda_proto * L_Proto_Hybrid + \
                           self.lambda_orth * L_Orth + self.lambda_style * L_StyleReg + \
                           self.lambda_aug * L_Aug

                    self.summary.add_scalar('L_CE_W', L_CE_W.item())
                    self.summary.add_scalar('L_Proto_Hybrid', L_Proto_Hybrid.item())
                    self.summary.add_scalar('L_Orth', L_Orth.item())
                    self.summary.add_scalar('L_StyleReg', L_StyleReg.item())
                    self.summary.add_scalar('L_Aug', L_Aug.item())  # *** 记录 L_Aug ***
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
            outputs, _, _ , _= model(inputs)

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
