import numpy as np
import torch
import torch.nn.functional as F
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
from core.callbacks import MeanIoU
import tqdm

__all__ = ['MinkUnetLearnerTrainer']

from torchsparse import SparseTensor


class MinkUnetLearnerTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False,
                 lambda_mask: float = 0.1,
                 lambda_bawa: float = 1.0):
        """
        LearnerTrainer for MinkUNet + DBAG + BAWA.
        """
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
        self.lambda_mask = lambda_mask
        self.lambda_bawa = lambda_bawa

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id
        )

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' in key or 'ids' in key:
                continue
            if isinstance(value, torch.Tensor):
                _inputs[key] = value.cuda(non_blocking=True)
            elif isinstance(value, SparseTensor):
                _inputs[key] = value.cuda()
            elif isinstance(value, list):
                new_list = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        new_list.append(v.cuda(non_blocking=True))
                    elif isinstance(v, SparseTensor):
                        new_list.append(v.cuda())
                    else:
                        new_list.append(v)
                _inputs[key] = new_list
            else:
                _inputs[key] = value

        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        if self.model.training:
            perturb_mode = 'generate'
            with amp.autocast(enabled=self.amp_enabled):
                outputs, bawa_features, mask_weights = self.model(inputs, perturb_mode=perturb_mode)

                loss_seg = self.criterion(outputs, targets)
                loss_mask = 0.0
                if mask_weights is not None:
                    loss_mask = sum(((m - 0.5) ** 2).mean() for m in mask_weights)

                loss_bawa = torch.zeros(1, device=outputs.device)
                bawa_clean_st = _inputs['bawa_clean']  # Batch-Wide ST_clean

                with torch.no_grad():
                    bawa_features_clean = self.model.get_bawa_features(bawa_clean_st)

                if 'bawa_clean' in feed_dict:
                    aug_inv_st = _inputs['inverse_map']
                    clean_inv_st = _inputs['inverse_map_clean']

                    aug_orig_ids_full = aug_inv_st.F.squeeze().long()
                    clean_orig_ids = clean_inv_st.F.squeeze().long()

                    valid_clean_mask = clean_orig_ids >= 0
                    valid_clean_orig_ids = clean_orig_ids[valid_clean_mask]

                    clean_feature_indices = torch.arange(len(clean_orig_ids), device=outputs.device)[valid_clean_mask]

                    if len(valid_clean_orig_ids) > 0:
                        # 保护：负 ID 临时设为 0
                        aug_orig_ids_safe = aug_orig_ids_full.clone()
                        aug_orig_ids_safe[aug_orig_ids_safe < 0] = 0

                        max_orig_id = max(aug_orig_ids_safe.max().item(), valid_clean_orig_ids.max().item())

                        orig2clean_map = torch.full((max_orig_id + 1,), -1, dtype=torch.long, device=outputs.device)
                        orig2clean_map[valid_clean_orig_ids] = clean_feature_indices

                        total_overlap_points = 0.0

                        for k, (f_aug_st, f_clean_st) in enumerate(zip(bawa_features, bawa_features_clean)):
                            N_current_aug = f_aug_st.F.shape[0]

                            # 取当前特征层对应的 ID
                            aug_orig_ids_current = aug_orig_ids_full[:N_current_aug].long()

                            # 过滤掉负值
                            valid_aug_mask_current = aug_orig_ids_current >= 0
                            aug_orig_ids_current = aug_orig_ids_current[valid_aug_mask_current]

                            # 过滤掉超过查找表长度的 ID
                            map_size = orig2clean_map.shape[0]
                            safe_mask = aug_orig_ids_current < map_size
                            aug_orig_ids_current = aug_orig_ids_current[safe_mask]
                            valid_aug_indices = valid_aug_mask_current.nonzero(as_tuple=True)[0][safe_mask]

                            if len(aug_orig_ids_current) == 0:
                                continue  # 没有有效点，跳过

                            # 查找对应 Clean Feature Index
                            mapped_clean_indices_valid = orig2clean_map[aug_orig_ids_current]

                            # 只保留有效索引
                            valid_mask = mapped_clean_indices_valid >= 0
                            final_aug_indices = valid_aug_indices[valid_mask]
                            final_clean_indices = mapped_clean_indices_valid[valid_mask]

                            if len(final_aug_indices) == 0:
                                continue

                            f_aug_valid = f_aug_st.F[final_aug_indices]
                            aligned_clean_F = f_clean_st.F[final_clean_indices]

                            diff = f_aug_valid - aligned_clean_F
                            loss_bawa += (diff ** 2).sum()
                            total_overlap_points += len(final_aug_indices)

                        # 平均化
                        if total_overlap_points > 0:
                            loss_bawa = (self.lambda_bawa * loss_bawa) / total_overlap_points
                        else:
                            loss_bawa = torch.zeros(1, device=outputs.device)
                    else:
                        loss_bawa = torch.zeros(1, device=outputs.device)

                loss_total = loss_seg + self.lambda_mask * loss_mask + loss_bawa

            self.summary.add_scalar('loss_total', loss_total.item())
            self.summary.add_scalar('loss_seg', loss_seg.item())
            self.summary.add_scalar('loss_mask', loss_mask if isinstance(loss_mask, float) else loss_mask.item())
            self.summary.add_scalar('loss_bawa', loss_bawa if isinstance(loss_bawa, float) else loss_bawa.item())

            self.optimizer.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            return {
                'outputs': outputs,
                'targets': targets,
                'mask_weights': mask_weights,
                'bawa_features': bawa_features,
            }

        else:
            with torch.no_grad():
                outputs, _, _ = self.model(inputs, perturb_mode='clean')

            # 映射到原点云
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

            outputs_cat = torch.cat(_outputs, 0)
            targets_cat = torch.cat(_targets, 0)

            return {
                'outputs': outputs_cat,
                'targets': targets_cat,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'model': self.model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path)
        self._load_state_dict(state_dict)

    def train(self,
              dataflow: DataLoader,
              *,
              num_epochs: int = 9999999,
              callbacks: Optional[List[Callback]] = None) -> None:
        """
        完整训练循环
        """
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
                    output_dict = self._run_step(feed_dict)
                    self.after_step(output_dict)

                    self.trigger_step()

                self._after_epoch()
                logger.info(f'Epoch finished in {humanize.naturaldelta(time.perf_counter() - epoch_time)}')

                self.trigger_epoch()

            logger.success(f'{self.num_epochs} epochs of training finished in {humanize.naturaldelta(time.perf_counter() - train_time)}')

        except StopTraining as e:
            logger.info(f'Training was stopped by {str(e)}.')
        finally:
            self.after_train()

    def save_model(self, path):
        assert path.endswith('.pt'), "Checkpoint save path must end with .pt"
        state_dict = self._state_dict()
        torch.save(state_dict, path)


def evaluate(val_loader: DataLoader, model: nn.Module, num_classes: int = 19):
    """
    点云 mIoU 评估函数
    """
    mIoU = MeanIoU(name='iou/test_', num_classes=num_classes, ignore_label=255)
    mIoU.before_epoch()

    with torch.no_grad():
        for feed_dict in tqdm.tqdm(val_loader, ncols=0):
            _inputs = {k: v.cuda(non_blocking=True) for k, v in feed_dict.items() if 'name' not in k}
            inputs = _inputs['lidar']

            outputs, _, _ = model(inputs, perturb_mode='clean')

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

            outputs_cat = torch.cat(_outputs, 0)
            targets_cat = torch.cat(_targets, 0)

            output_dict = {'outputs': outputs_cat, 'targets': targets_cat}
            mIoU.after_step(output_dict)

    mIoU.after_epoch()
    return mIoU
