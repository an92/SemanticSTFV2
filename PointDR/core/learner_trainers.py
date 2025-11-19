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
                 thing_weight: float = 2.0,
                 mu: float = 0.001,
                 lam: float =0.1,
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


        self.T = 0.07

        self.thing_weight = thing_weight  # NTN: Weight multiplier for safety-critical (things) classes
        self.mu = mu
        self.lamda = lam


    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id
        )

    def _after_epoch(self) -> None:
        self.model.eval()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()

        inputs_1 = _inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)
        with amp.autocast(enabled=self.amp_enabled):
            outputs_1, feat_1 = self.model(inputs_1)
            if outputs_1.requires_grad:
                loss_1 = self.criterion(outputs_1, targets_1)

        if outputs_1.requires_grad:
            # consistency loss
            inputs_2 = _inputs['lidar_2']
            targets_2 = feed_dict['targets_2'].F.long().cuda(non_blocking=True)
            pred_2, feat_2 = self.model(inputs_2)
            # ---- point-wise infoNCE loss -----
            # step 1: get mean feature (prototypes) for each class in weak view
            feat_1 = nn.functional.normalize(feat_1, dim=1)
            feat_2 = nn.functional.normalize(feat_2, dim=1)
            feat1_proto = torch.zeros((configs.data.num_classes, feat_1.shape[1]))
            for ii in range(configs.data.num_classes):
                mask = (targets_1 == ii)
                if mask.sum():
                    feat1_proto[ii] = feat_1[mask].mean(dim=0)
            feat1_proto = (feat1_proto + 1e-8).cuda()
            # step 2: get similarity
            logits = torch.mm(feat_2, self.model.memo_bank.T.detach())
            logits /= self.T  # apply temperature

            # --- MODIFICATION 1: NTN-inspired Class Weighting for Contrastive Loss (loss_2) ---
            # Safety-critical classes (SemanticKITTI things classes: car, person, cyclist, etc.)
            # IDs: [0-7, 13, 17, 18]
            things_class_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 13, 17, 18], device=targets_2.device)
            class_weights_tensor = torch.ones(configs.data.num_classes, device=targets_2.device)
            # Apply higher weight to 'things' classes
            for cls_id in things_class_ids:
                if cls_id < configs.data.num_classes:
                    class_weights_tensor[cls_id] = self.thing_weight

                    # Create a weighted CrossEntropyLoss for the contrastive head (loss_2)
            # NOTE: Assumes self.criterion is not an instance of nn.CrossEntropyLoss or similar
            criterion_weighted = nn.CrossEntropyLoss(
                weight=class_weights_tensor,
                ignore_index=255,
                reduction='mean'
            ).cuda()

            loss_2 = criterion_weighted(logits, targets_2)


            # momentum update memory bank
            self.model.momentum_update_key_encoder(feat1_proto, init=(self.global_step==1))

            # --- MODIFICATION 2: DGUIL-inspired Entropy Regularization (loss_3) ---
            # Minimize the entropy of the perturbed view's prediction (pred_2)
            softmax_pred_2 = torch.softmax(pred_2, dim=1)
            # Entropy = - sum(p * log(p))
            # Use a small constant (1e-6) for numerical stability
            entropy = (softmax_pred_2 * torch.log(softmax_pred_2 + 1e-6)).sum(dim=1)
            # Mask out ignore label (255) points
            valid_mask = (targets_2 != 255)
            # Only consider the uncertainty of valid points
            loss_3 = -entropy[valid_mask].mean()

            loss = loss_1 + self.lamda * loss_2 + self.mu * loss_3
            # ----------------------------------------------


            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_1', loss_1.item())
            self.summary.add_scalar('loss_2', loss_2.item())
            self.summary.add_scalar('loss_3', loss_3.item())

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
