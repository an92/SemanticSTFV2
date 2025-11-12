import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer

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

# kNN utility
from torch_geometric.nn import knn_graph  # 确保安装 torch_geometric

__all__ = ['MinkUnetLearnerTrainer']

THING_NAMES = set([
    'car', 'bicycle', 'motorcycle', 'truck', 'bus',
    'person', 'bicyclist', 'motorcyclist', 'fence',
    'pole', 'traffice-sign',
])


class MinkUnetLearnerTrainer(Trainer):

    def __init__(self, model: nn.Module, criterion: Callable, optimizer: torch.optim.Optimizer,
                 scheduler: Callable, num_workers: int, seed: int, amp_enabled: bool = False,
                 total_steps: int = 100000):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

        self.lamda_memory = 0.5
        self.lamda_fcr = 1.0
        self.lamda_cwcl = 0.5
        self.T = 0.07
        self.teacher_momentum = 0.99
        self.teacher_momentum_warmup = True
        self.total_steps = total_steps
        self.multi_proto = 2

        # 确保 teacher 初始化
        if hasattr(self.model, 'init_teacher'):
            self.model.init_teacher()

    def _before_epoch(self):
        self.model.train()
        if hasattr(self.dataflow.sampler, 'set_epoch'):
            self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda wid: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + wid
        )

    def _compute_memory_infoNCE(self, feat_weather, feat_proto, targets, sample_weights):
        """
        Memory-based InfoNCE (prototype memory)
        """
        logits_mem = torch.mm(feat_weather, feat_proto.T) / self.T
        mask_valid = (targets != 255)
        targets_clamped = targets.clone()
        targets_clamped[targets_clamped == 255] = 0
        log_probs = F.log_softmax(logits_mem, dim=1)
        nll = -log_probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)
        nll[~mask_valid] = 0.0
        loss = (nll * sample_weights).sum() / sample_weights[mask_valid].sum().clamp(min=1.0)
        return loss

    def _compute_CWCL_knn(self, feat_clean, feat_weather, targets, sample_weights, neg_sample_ratio=50):
        """
        高效 CWCL：
        - feat_clean: [N, C] 对齐点
        - feat_weather: [N, C] 对齐点
        - targets: [N]
        - sample_weights: [N]
        """
        N, C = feat_clean.shape

        # 1. 正样本相似度
        sim_pos = (feat_clean * feat_weather).sum(dim=1) / self.T  # [N]

        # 2. 随机负样本索引（每个点采样 neg_sample_ratio 个负样本）
        neg_idx = torch.randint(0, N, (N, neg_sample_ratio), device=feat_clean.device)
        feat_neg = feat_weather[neg_idx]  # [N, neg_sample_ratio, C]

        # 3. 正样本 + 负样本相似度
        feat_clean_exp = feat_clean.unsqueeze(1)  # [N,1,C]
        sim_all = torch.cat([sim_pos.unsqueeze(1), (feat_clean_exp * feat_neg).sum(dim=2) / self.T],
                            dim=1)  # [N, 1+neg_sample_ratio]

        # 4. log-softmax loss
        log_probs = F.log_softmax(sim_all, dim=1)
        loss_point = -log_probs[:, 0]  # 负 log likelihood 对正样本

        mask_valid = (targets != 255)
        loss = (loss_point * sample_weights).sum() / sample_weights[mask_valid].sum().clamp(min=1.0)
        return loss

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {k: v.cuda() for k, v in feed_dict.items() if 'name' not in k and 'ids' not in k}
        inputs_1 = _inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)
        inputs_2 = _inputs['lidar_2']
        targets_2 = feed_dict['targets_2'].F.long().cuda(non_blocking=True)
        inverse_map = feed_dict['inverse_map_dense'].F.long().cuda(non_blocking=True)
        ids_1 = feed_dict['ids_1'].F.long().cuda(non_blocking=True)
        ids_2 = feed_dict['ids_2'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):
            outputs_1, feat_clean_student = self.model(inputs_1)

            if outputs_1.requires_grad:
                # ------------------- 训练分支 -------------------
                loss_seg = self.criterion(outputs_1, targets_1) if outputs_1.requires_grad else torch.tensor(0.0,
                                                                                                             device=outputs_1.device,
                                                                                                             requires_grad=True)
                outputs_2, feat_weather_student = self.model(inputs_2)

                # teacher 特征
                with torch.no_grad():
                    feat_clean_teacher = self.model.teacher_backbone.get_features(inputs_1)
                    feat_clean_teacher_aligned = feat_clean_teacher[inverse_map]

                # 对齐两视图点
                ids_1_np = ids_1.cpu().numpy()
                ids_2_np = ids_2.cpu().numpy()
                common_ids, idx1, idx2 = np.intersect1d(ids_1_np, ids_2_np, return_indices=True)
                idx1 = torch.from_numpy(idx1).long().cuda()
                idx2 = torch.from_numpy(idx2).long().cuda()

                feat_weather_aligned = feat_weather_student[idx2]
                feat_teacher_aligned = feat_clean_teacher_aligned[idx2]
                targets_aligned = targets_2[idx2]
                valid_mask = (targets_aligned != 255)

                # FCR loss
                l_fcr = F.mse_loss(
                    feat_weather_aligned[valid_mask],
                    feat_teacher_aligned[valid_mask]
                ) if valid_mask.sum() > 0 else torch.tensor(0.0, device=feat_weather_student.device, requires_grad=True)

                # Memory prototype
                feat_clean_norm = F.normalize(feat_clean_student, dim=1)
                feat_weather_norm = F.normalize(feat_weather_student, dim=1)
                feat_proto = torch.zeros((self.model.num_classes * self.model.multi_proto, feat_clean_norm.shape[1]),
                                         device=feat_clean_norm.device)
                for ii in range(self.model.num_classes):
                    mask = (targets_1 == ii)
                    if mask.sum() == 0:
                        continue
                    pts = feat_clean_norm[mask]
                    if self.model.multi_proto == 1:
                        feat_proto[ii] = pts.mean(dim=0)
                    else:
                        pts_split = torch.chunk(pts, self.model.multi_proto, dim=0)
                        for j, sub in enumerate(pts_split):
                            idx = ii * self.model.multi_proto + j
                            feat_proto[idx] = sub.mean(dim=0)
                feat_proto = F.normalize(feat_proto + 1e-8, dim=1)

                # class-aware weights
                sample_weights = torch.ones_like(targets_aligned, dtype=torch.float32)
                try:
                    reverse_map = self.model.teacher_backbone.reverse_label_name_mapping
                    for name, idx in reverse_map.items():
                        sample_weights[targets_aligned == idx] = 1.0 if name in THING_NAMES else 0.2
                except Exception:
                    pass

                # memory loss
                loss_memory = self._compute_memory_infoNCE(
                    feat_weather_norm, feat_proto, targets_aligned, sample_weights
                )

                # CWCL loss
                feat_clean_aligned = feat_clean_norm[idx1]
                feat_weather_aligned_cw = feat_weather_norm[idx2]
                targets_aligned_final = targets_aligned
                sample_weights_aligned = sample_weights
                loss_cw = self._compute_CWCL_knn(
                    feat_clean_aligned,
                    feat_weather_aligned_cw,
                    targets_aligned_final,
                    sample_weights_aligned,
                    neg_sample_ratio=50
                )

                # 总 loss
                total_loss = loss_seg + self.lamda_memory * loss_memory + self.lamda_fcr * l_fcr + self.lamda_cwcl * loss_cw

                # 更新 memory bank
                self.model.momentum_update_key_encoder(feat_proto.detach(), momentum=0.9, init=(self.global_step == 1))

                # backward
                self.summary.add_scalar('loss', total_loss.item())
                self.summary.add_scalar('loss_seg', float(loss_seg))
                self.summary.add_scalar('loss_memory', float(loss_memory))
                self.summary.add_scalar('loss_fcr', float(l_fcr))
                self.summary.add_scalar('loss_cw', float(loss_cw))

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # teacher EMA
                if self.teacher_momentum_warmup:
                    m = 1.0 - (1.0 - self.teacher_momentum) * (
                                np.cos(np.pi * self.global_step / self.total_steps) + 1) / 2
                else:
                    m = self.teacher_momentum
                self.model.update_teacher(momentum=m)

                return {'outputs': outputs_1, 'targets': targets_1}

            else:
                # ------------------- 评估分支 -------------------
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
                outputs_cat = torch.cat(_outputs, 0)
                targets_cat = torch.cat(_targets, 0)

                return {'outputs': outputs_cat, 'targets': targets_cat}

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

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
                logger.info('Training finished in {}.'.format(humanize.naturaldelta(time.perf_counter() - epoch_time)))
                self.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(humanize.naturaldelta(time.perf_counter() - epoch_time)))

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
