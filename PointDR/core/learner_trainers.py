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
from torch_cluster import knn_graph
from torch_scatter import scatter_mean

from core.callbacks import MeanIoU
import tqdm

__all__ = ['MinkUnetLearnerTrainer']


class MinkUnetLearnerTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False,
                 lamda_ct: float=0.1,
                 lamda_sc: float=0.05,
                 lamda_snc: float=0.05,
                 k_snc: int=16) -> None:
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
        self.T = 0.07

        # Loss weights
        self.lamda_ct = lamda_ct
        self.lamda_sc = lamda_sc
        self.lamda_snc = lamda_snc

        # SNC params
        self.k_snc = k_snc

    # ---------------- Helper ----------------
    def compute_prototype(self, feat: torch.Tensor, targets: torch.Tensor):
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
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()

        inputs_1 = _inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            outputs_1, feat_1 = self.model(inputs_1)
            loss_1 = self.criterion(outputs_1, targets_1) if outputs_1.requires_grad else None

        if outputs_1.requires_grad:
            ids_1 = feed_dict['ids_1'].F.long().cuda(non_blocking=True)
            # ---------------- Consistency Loss ----------------
            inputs_2 = _inputs['lidar_2']
            targets_2 = feed_dict['targets_2'].F.long().cuda(non_blocking=True)
            ids_2 = feed_dict['ids_2'].F.long().cuda(non_blocking=True)
            pred_2, feat_2 = self.model(inputs_2)

            feat_1_norm = nn.functional.normalize(feat_1, dim=1)
            feat_2_norm = nn.functional.normalize(feat_2, dim=1)

            # ---------- Sparse Structural Consistency Loss (L_sc) ----------
            ids_1_pos_mask = ids_1 >= 0
            ids_2_pos_mask = ids_2 >= 0
            ids_1_pos = ids_1[ids_1_pos_mask]
            ids_2_pos = ids_2[ids_2_pos_mask]

            feat_1_pos = feat_1_norm[ids_1_pos_mask]
            feat_2_pos = feat_2_norm[ids_2_pos_mask]
            targets_1_pos = targets_1[ids_1_pos_mask]

            id_to_idx_1 = -torch.ones(ids_1_pos.max() + 1, dtype=torch.long, device=ids_1.device)
            id_to_idx_1[ids_1_pos] = torch.arange(ids_1_pos.size(0), device=ids_1.device)

            id_to_idx_2 = -torch.ones(ids_2_pos.max() + 1, dtype=torch.long, device=ids_2.device)
            id_to_idx_2[ids_2_pos] = torch.arange(ids_2_pos.size(0), device=ids_2.device)

            # 找到两视图中都存在的 id
            common_ids = torch.tensor(list(set(ids_1_pos.tolist()) & set(ids_2_pos.tolist())),
                                      device=ids_1.device, dtype=torch.long)

            idx_1_matched = id_to_idx_1[common_ids]
            idx_2_matched = id_to_idx_2[common_ids]

            valid_mask = (idx_1_matched >= 0) & (idx_2_matched >= 0)
            idx_1_matched = idx_1_matched[valid_mask]
            idx_2_matched = idx_2_matched[valid_mask]

            feat_1_matched = feat_1_pos[idx_1_matched]
            feat_2_matched = feat_2_pos[idx_2_matched]
            targets_1_matched = targets_1_pos[idx_1_matched]

            valid_label_mask = (targets_1_matched != 255)
            feat_1_valid = feat_1_matched[valid_label_mask]
            feat_2_valid = feat_2_matched[valid_label_mask]

            loss_sc = nn.functional.smooth_l1_loss(
                feat_2_valid, feat_1_valid.detach(), reduction='mean'
            ) if feat_1_valid.size(0) > 0 else torch.zeros(1, device=feat_1.device)


            # ---------- Sparse Neighborhood Consistency Loss (L_snc) ----------
            loss_snc = torch.zeros(1, device=feat_1.device)
            if feat_1_valid.size(0) > 0:
                coords_1 = inputs_1.C[:, :3].float()
                coords_1_valid = coords_1[idx_1_matched[valid_label_mask]]
                edge_index = knn_graph(coords_1_valid, k=self.k_snc, batch=None, loop=True)
                # 聚合邻居
                neighbor_feats = scatter_mean(feat_1_valid[edge_index[1]], edge_index[0], dim=0)
                loss_snc = nn.functional.smooth_l1_loss(feat_2_valid, neighbor_feats, reduction='mean')

            # ---------- Adaptive Class-aware Prototype Loss (L_ct) ----------
            feat1_proto, class_counts_1 = self.compute_prototype(feat_1_norm, targets_1)
            feat2_proto, _ = self.compute_prototype(feat_2_norm, targets_2)

            self.model.momentum_update_B(feat1_proto, init=(self.global_step == 1))
            self.model.momentum_update_G(feat2_proto, init=(self.global_step == 1))

            P_adaptive_T, avg_alpha = self.model.get_adaptive_prototype(targets_1, class_counts_1)
            logits = torch.mm(feat_2_norm, P_adaptive_T) / self.T
            mask = (targets_2 != 255)
            loss_2 = self.criterion(logits[mask], targets_2[mask])

            # ---------- Final Loss ----------
            loss = loss_1 + self.lamda_ct * loss_2 + self.lamda_sc * loss_sc + self.lamda_snc * loss_snc

            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_1_ce', loss_1.item())
            self.summary.add_scalar('loss_2_ct', loss_2.item())
            self.summary.add_scalar('loss_3_sc', loss_sc.item())
            self.summary.add_scalar('loss_4_snc', loss_snc.item())
            self.summary.add_scalar('AC_Prototype/avg_alpha', avg_alpha)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return {'outputs': outputs_1, 'targets': targets_1}

        # ---------------- Inference / Mapping ----------------
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs, _targets = [], []
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

    # ---------------- State ----------------
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

    # ---------------- Train ----------------
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
