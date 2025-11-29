import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch_scatter import scatter_mean
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
from torch.autograd import Function

__all__ = ['MinkUnetV3Trainer']


class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def uncertainty_weight(logits, dim=1, temp=1.0):
    probs = torch.softmax(logits / temp, dim=dim)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=dim)
    C = logits.size(dim)
    max_entropy = torch.log(torch.tensor(C, dtype=logits.dtype, device=logits.device) + 1e-8)
    weight = 1.0 - (entropy / max_entropy)
    return weight.detach()


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
            lambda_aug: float = 0.1,
            decouple_layers: List[str] = None,
            disentangle_start_epoch: int=5,
            temp_uncertainty: float=1.0
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

        self.ignore_label = 255
        self.lambda_aug = lambda_aug
        self.num_sample_aux = 4096
        self.temp_uncertainty = temp_uncertainty
        self.disentangle_start_epoch = disentangle_start_epoch

        self.decouple_layers = decouple_layers if decouple_layers is not None else ['y3', 'y4']

        self.criterion_reduction_none = nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction='none')
        self.criterion_ce_no_ig = nn.CrossEntropyLoss()  # 用于 L_Aug

        self.summary = None
        self.callbacks = None

    def _before_epoch(self) -> None:
        self.model.train()
        try:
            self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        except Exception:
            pass
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()
        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        batch_size = int(inputs.C[:, -1].max().item() + 1)

        if 'is_augmented' in feed_dict:
            # is_augmented: (Batch_Size) 0代表原始域，1代表增强域
            is_augmented = feed_dict['is_augmented'].long().cuda(non_blocking=True)
        else:
            is_augmented = torch.zeros((batch_size,), dtype=torch.long).cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            model_output = self.model(inputs)
            outputs = model_output['logits']
            decoupled_output = model_output['decoupled_features']

            if outputs.requires_grad:
                valid_mask = targets != self.ignore_label
                # --- 1. CE loss with uncertainty weight (Main Loss) ---
                logits_v = outputs[valid_mask]
                targets_v = targets[valid_mask]

                loss_ce_per_point = self.criterion_reduction_none(logits_v, targets_v)

                weight_v = uncertainty_weight(logits_v, dim=1, temp=self.temp_uncertainty)
                den = weight_v.sum().clamp_min(1.0)
                L_CE_W = (loss_ce_per_point * weight_v).sum() / den

                # L_CE_W = self.criterion(outputs, targets)

                # --- 2. Initialize disentangle losses ---
                L_Aug = torch.tensor(0., device=outputs.device)

                if self.epoch_num >= self.disentangle_start_epoch and self.lambda_aug > 0:

                    NUM_SAMPLE = self.num_sample_aux

                    name = 'y4'
                    f_style_l = decoupled_output['f_style'][name]  # (N, C)

                    N_layer = f_style_l.shape[0]

                    if N_layer > NUM_SAMPLE:
                        perm = torch.randperm(N_layer, device=outputs.device)[:NUM_SAMPLE]
                        f_style_sub = f_style_l[perm]
                        coords_sub = decoupled_output['coords'][name][perm]
                    else:
                        f_style_sub = f_style_l
                        coords_sub = decoupled_output['coords'][name]


                    f_style_rev = GRL.apply(f_style_sub, torch.tensor(1.0, device=f_style_sub.device))

                    aug_logits = self.model.aug_classifiers[name](f_style_rev)

                    scene_ids = coords_sub[:, 3].long()
                    mean_logits = scatter_mean(aug_logits, scene_ids, dim=0, dim_size=batch_size)

                    L_Aug = self.criterion_ce_no_ig(mean_logits, is_augmented)

                loss = L_CE_W + self.lambda_aug * L_Aug

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()

                # --- summary logging ---
                self.summary.add_scalar('L_CE_W', float(L_CE_W.item()))
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