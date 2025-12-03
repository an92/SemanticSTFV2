import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch_cluster import knn
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


    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False,
                 geo_weight: float = 0.01,  # L_geo 的权重 (Beta)
                 conf_threshold: float = 0.95,  # HCF 阈值
                 knn_k: int = 16,  # KNN 邻居数
                 consis_ratio: float = 0.8,  # KNN 连贯性比例
                 r_scale: float = 50.0  # 距离权重尺度
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

        self.geo_weight = geo_weight
        self.conf_threshold = conf_threshold
        self.knn_k = knn_k
        self.consis_ratio = consis_ratio
        self.r_scale = r_scale

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

        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

        with amp.autocast(enabled=self.amp_enabled):

            outputs, _ = self.model(inputs)

            if outputs.requires_grad:
                loss_ce = self.criterion(outputs, targets)
                loss = loss_ce

                if self.geo_weight > 0:
                    # 1. 获取几何信息和模型预测
                    coords = inputs.C[:, :3].float()  # (N, 3) 坐标
                    batch_index = inputs.C[:, -1].long()  # (N,) batch index
                    N = coords.shape[0]

                    prob = torch.softmax(outputs, dim=1)  # (N, C)
                    preds = outputs.argmax(dim=1)  # (N,)

                    # 2. 距离感知权重 (W_dist)
                    r = torch.linalg.norm(coords, dim=1)
                    weights_dist = torch.exp(r / self.r_scale).detach()
                    weights_dist = weights_dist / (weights_dist.mean() + 1e-6)  # 归一化

                    # 3. KNN 连贯性门控 (I_KNN)
                    row, col = knn(coords, coords, k=self.knn_k, batch_x=batch_index, batch_y=batch_index)

                    # 3.2. 检查局部一致性
                    current_preds = preds[row]
                    neighbor_preds = preds[col]
                    is_consistent = (current_preds == neighbor_preds).float()  # (E,)

                    # 3.3. 计算每个点的平均邻域一致性 (N,)
                    coherence_sum = torch.zeros(N, device=outputs.device)
                    coherence_count = torch.zeros(N, device=outputs.device)
                    coherence_sum.scatter_add_(0, row, is_consistent)
                    coherence_count.scatter_add_(0, row, torch.ones_like(is_consistent))
                    avg_coherence = coherence_sum / (coherence_count + 1e-6)  # (N,)

                    # 3.4. 连贯性门控: 只有高连贯性点的门控为 1
                    knn_gate = (avg_coherence >= self.consis_ratio).float()  # (N,)

                    # 4. 几何连贯性正则化损失
                    conf, _ = prob.max(dim=1)
                    valid_mask = (targets != 255)
                    hcf_mask = (conf > self.conf_threshold) & valid_mask

                    # 4.2. 联合掩码: (HCF) AND (KNN 连贯性)
                    final_mask = hcf_mask & (knn_gate.bool())


                    if final_mask.any():
                        entropy = -(prob * torch.log(prob + 1e-6)).sum(dim=1)  # (N,)

                        selected_entropy = entropy[final_mask]
                        selected_weights = weights_dist[final_mask]

                        loss_geo = (selected_entropy * selected_weights).sum() / (selected_weights.sum() + 1e-6)
                    else:
                        loss_geo = torch.zeros(1, device=outputs.device).squeeze()

                    loss = loss_ce + self.geo_weight * loss_geo

        if outputs.requires_grad:
            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_ce', loss_ce.item())
            self.summary.add_scalar('loss_geo', loss_geo.item())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
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
                logger.info(
                    'Training finished in {}.'.format(humanize.naturaldelta(time.perf_counter() - epoch_time)))

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

