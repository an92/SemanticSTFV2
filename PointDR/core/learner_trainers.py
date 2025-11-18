import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

import time
from typing import Any, Dict, List, Optional, Callable
from torch.utils.data import DataLoader

from torchpack.callbacks import Callback, Callbacks
from torchpack.train.exception import StopTraining
from torchpack.train.summary import Summary
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.config import configs
from core.callbacks import MeanIoU
import tqdm

__all__ = ['MinkUnetLearnerTrainer']

class MinkUnetLearnerTrainer(Trainer):
    """
    DG-oriented Trainer for STF adverse point cloud datasets.
    Combines:
        - CE Loss
        - Adaptive Class-aware Prototype (AC Prototype)
        - Feature Consistency (L2)
        - Point-wise infoNCE (memory bank / momentum)
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False,
                 lamda_proto: float = 0.1,
                 lamda_cons: float = 0.05,
                 lamda_info: float = 0.05,
                ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

        # Loss weights
        self.lamda_proto = lamda_proto
        self.lamda_cons = lamda_cons
        self.lamda_info = lamda_info
        self.T = 0.07

    # ---------------- Helper ----------------
    def compute_prototype(self, feat: torch.Tensor, targets: torch.Tensor):
        """Compute normalized class-wise prototypes."""
        feat = nn.functional.normalize(feat, dim=1)
        num_classes = configs.data.num_classes
        feat_proto = torch.zeros((num_classes, feat.shape[1]), device=feat.device)
        class_counts = torch.zeros(num_classes, device=feat.device)

        for ii in range(num_classes):
            mask = (targets == ii)
            if mask.sum():
                feat_proto[ii] = feat[mask].mean(dim=0)
                class_counts[ii] = mask.sum()

        feat_proto[class_counts > 0] = nn.functional.normalize(feat_proto[class_counts > 0], dim=1)
        return (feat_proto + 1e-8), class_counts

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id
        )

    def _after_epoch(self) -> None:
        self.model.eval()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        # ---------------- Prepare Inputs ----------------
        _inputs = {k: v.cuda() for k, v in feed_dict.items() if 'name' not in k and 'ids' not in k}
        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            # ---------------- Forward pass ----------------
            outputs, feat = self.model(inputs)
            loss_ce = self.criterion(outputs, targets) if outputs.requires_grad else None

        if outputs.requires_grad:
            # ---------------- Data Augmentation / Feature Consistency ----------------
            inputs_aug = _inputs.get('lidar_2', None)
            if inputs_aug is not None:
                outputs_aug, feat_aug = self.model(inputs_aug)

                # KDTree correspondence
                coords_1 = inputs.C[:, :3].float().cpu().numpy()
                coords_2 = inputs_aug.C[:, :3].float().cpu().numpy()
                from scipy.spatial import cKDTree
                tree = cKDTree(coords_1)
                _, correspond_idx = tree.query(coords_2, k=1)
                correspond_idx = torch.from_numpy(correspond_idx).long().cuda()

                # Mask: only valid labels
                targets_aug = feed_dict['targets_2'].F.long().cuda(non_blocking=True)
                valid_mask = (targets[correspond_idx] != 255) & (targets_aug != 255)

                # FP16 / dtype safe
                loss_consistency = nn.functional.mse_loss(
                    feat_aug[valid_mask].to(outputs.dtype),
                    feat.detach()[correspond_idx[valid_mask]].to(outputs.dtype)
                )
            else:
                loss_consistency = torch.zeros(1, device=feat.device, dtype=outputs.dtype)

            # ---------------- Adaptive Class-aware Prototype ----------------
            feat_norm = nn.functional.normalize(feat, dim=1)
            feat_proto, class_counts = self.compute_prototype(feat_norm, targets)
            self.model.momentum_update_B(feat_proto, init=(self.global_step == 1))
            self.model.momentum_update_G(feat_proto, init=(self.global_step == 1))
            P_adaptive_T, avg_alpha = self.model.get_adaptive_prototype(targets, class_counts)

            logits_proto = torch.mm(feat_norm.to(P_adaptive_T.dtype), P_adaptive_T) / self.T
            mask_proto = (targets != 255)
            loss_proto = self.criterion(logits_proto[mask_proto], targets[mask_proto])

            # ---------------- Point-wise infoNCE / memory bank ----------------
            if hasattr(self.model, 'memo_bank'):
                feat_norm = nn.functional.normalize(feat, dim=1)
                memo_bank_T_matched = self.model.memo_bank.T.detach().to(feat_norm.dtype)
                logits_info = torch.mm(feat_norm, memo_bank_T_matched) / self.T
                mask_info = (targets != 255)
                loss_info = self.criterion(logits_info[mask_info], targets[mask_info])
                # Momentum update memory bank
                self.model.momentum_update_key_encoder(feat_proto, init=(self.global_step == 1))
            else:
                loss_info = torch.zeros(1, device=feat.device, dtype=outputs.dtype)

            # ---------------- Total Loss ----------------
            loss = loss_ce + self.lamda_proto * loss_proto + self.lamda_cons * loss_consistency + self.lamda_info * loss_info

            # Summary
            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_ce', loss_ce.item())
            self.summary.add_scalar('loss_proto', loss_proto.item())
            self.summary.add_scalar('loss_consistency', loss_consistency.item())
            self.summary.add_scalar('loss_info', loss_info.item())
            self.summary.add_scalar('AC_Prototype/avg_alpha', avg_alpha)

            # ---------------- Backprop ----------------
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            return {'outputs': outputs, 'targets': targets}

        # ---------------- Inference / Mapping ----------------
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs, _targets = [], []
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

    # ---------------- State Dict ----------------
    def _state_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

    # ---------------- Train Loop ----------------
    def train(self, dataflow: DataLoader, *, num_epochs: int = 9999999, callbacks: Optional[List[Callback]] = None) -> None:
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
                logger.info(f'Epoch {self.epoch_num}/{self.num_epochs} started.')
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
                logger.info(f'Training finished in {humanize.naturaldelta(time.perf_counter() - epoch_time)}')
                self.trigger_epoch()
                logger.info(f'Epoch finished in {humanize.naturaldelta(time.perf_counter() - epoch_time)}')

            logger.success(f'{self.num_epochs} epochs of training finished in {humanize.naturaldelta(time.perf_counter() - train_time)}')
        except StopTraining as e:
            logger.info(f'Training was stopped by {str(e)}.')
        finally:
            self.after_train()

    # ---------------- Save ----------------
    def save_model(self, path):
        assert '.pt' in path, "Checkpoint save path is wrong"
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)


# ---------------- Evaluate ----------------
def evaluate(val_loader, model):
    mIoU = MeanIoU(name='iou/test_', num_classes=19, ignore_label=255)
    mIoU.before_epoch()

    with torch.no_grad():
        for feed_dict in tqdm.tqdm(val_loader, ncols=0):
            _inputs = {k: v.cuda() for k, v in feed_dict.items() if 'name' not in k}
            inputs = _inputs['lidar']
            outputs = model(inputs)

            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs, _targets = [], []
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
            mIoU.after_step({'outputs': outputs, 'targets': targets})

    mIoU.after_epoch()
