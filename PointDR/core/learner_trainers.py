import time
from typing import Any, Dict, Callable, Optional, List
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from torchpack.utils.logging import logger
from torchpack.train.exception import StopTraining
from torchpack.train.summary import Summary

from torchsparse import SparseTensor


# ----------------------------
# 工具函数
# ----------------------------
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算每个点的熵，用于不确定性 H_aug/H_drop"""
    if logits is None or logits.numel() == 0:
        device = logits.device if logits is not None else 'cuda'
        return torch.tensor(0.0, device=device)
    probs = torch.softmax(logits, dim=-1)
    entropy = - (probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
    return entropy


# ----------------------------
# Trainer
# ----------------------------
class MinkUnetLearnerTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 optimizer_ljm: Optional[Optimizer] = None,
                 optimizer_adm: Optional[Optimizer] = None,
                 num_workers: int = 4,
                 seed: int = 1234,
                 amp_enabled: bool = False,
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
        self.global_step = 0

        self.ljm = getattr(model, 'ljm', None)
        self.adm = getattr(model, 'adm', None)

        self.optimizer_ljm = optimizer_ljm
        self.optimizer_adm = optimizer_adm

        # HRAD/LAA 超参数
        self.lambda_adv = 1.0
        self.lambda_drop_reg = 0.1
        self.target_drop_ratio = 0.1
        self.lambda_aux = 1.0
        self.soft_loss_weight = 0.1
        self.hard_threshold = 0.5
        self.adm_tau_start = 1.0
        self.adm_tau_end = 0.1
        self.adm_tau_anneal_steps = 20000

        self.eval_interval = 500

    def _before_epoch(self):
        self.model.train()
        if self.ljm: self.ljm.train()
        if self.adm: self.adm.train()

        # anneal tau
        if self.adm:
            gs = max(1, getattr(self, 'global_step', 1))
            ratio = min(1.0, gs / max(1, self.adm_tau_anneal_steps))
            tau = self.adm_tau_start * (1.0 - ratio) + self.adm_tau_end * ratio
            if hasattr(self.adm, 'set_tau'):
                self.adm.set_tau(tau)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {k: v.cuda() for k, v in feed_dict.items() if 'name' not in k and 'ids' not in k}
        inputs_orig: SparseTensor = _inputs['lidar']
        targets_F: torch.Tensor = feed_dict['targets'].F.long().cuda(non_blocking=True)

        is_laa_enabled = self.ljm and self.adm and self.optimizer_ljm and self.optimizer_adm

        if self.model.training and is_laa_enabled:
            # --- Stage 1: LJM 增强 ---
            points_F_aug = self.ljm(inputs_orig.F)
            inputs_aug = SparseTensor(points_F_aug, inputs_orig.C)

            # --- Stage 2: 上游 loss & entropy ---
            with amp.autocast(enabled=self.amp_enabled):
                logits_aug, _ = self.model(inputs_aug)
                L_aug = self.criterion(logits_aug, targets_F).mean()
                H_aug = compute_entropy(logits_aug)

            # --- Stage 3: ADM drop ---
            try:
                soft_lidar, drop_ratio_ste, keep_indices = self.adm(
                    inputs_aug, L_aug.detach(), H_aug.detach(),
                    hard_threshold=self.hard_threshold, dual_forward=True
                )
            except Exception as e:
                logger.warning(f"[ADM forward error] {e}")
                soft_lidar = inputs_aug
                drop_ratio_ste = torch.tensor(0.0, device=inputs_orig.F.device)
                keep_indices = torch.arange(inputs_aug.F.shape[0], device=inputs_orig.F.device)

            # --- Stage 4: soft forward ---
            try:
                # 这里 targets_F 也要对齐 drop mask
                targets_soft = targets_F[keep_indices]
                with amp.autocast(enabled=self.amp_enabled):
                    logits_soft, _ = self.model(soft_lidar)
                    loss_soft = self.criterion(logits_soft, targets_soft).mean()
            except Exception as e:
                logger.warning(f"[Soft forward error] {e}")
                logits_soft = None
                loss_soft = torch.tensor(0.0, device=targets_F.device)

            # --- Stage 5: hard forward (实际分割) ---
            try:
                hard_lidar = SparseTensor(inputs_orig.F[keep_indices], inputs_orig.C[keep_indices])
                with amp.autocast(enabled=self.amp_enabled):
                    logits_drop, _ = self.model(hard_lidar)
                    targets_F_drop = targets_F[keep_indices]
                    L_drop = self.criterion(logits_drop, targets_F_drop).mean()
                    H_drop = compute_entropy(logits_drop)
            except Exception as e:
                logger.warning(f"[Hard forward error] {e}")
                logits_drop = None
                L_drop = torch.tensor(0.0, device=targets_F.device)
                H_drop = torch.tensor(0.0, device=targets_F.device)
                targets_F_drop = targets_F

            # --- Stage 6: compose losses ---
            adv_loss = - (L_drop + H_drop)
            drop_reg_loss = self.lambda_drop_reg * (drop_ratio_ste - self.target_drop_ratio).clamp(min=0.0) ** 2
            aux_loss = torch.tensor(0.0, device=targets_F.device)
            adm_loss_total = self.lambda_adv * adv_loss + drop_reg_loss + self.lambda_aux * aux_loss + self.soft_loss_weight * loss_soft

            # L2 reg
            l2_reg = torch.tensor(0.0, device=targets_F.device)
            for group in self.optimizer_adm.param_groups + self.optimizer_ljm.param_groups:
                for p in group['params']:
                    if p.requires_grad: l2_reg += p.norm()
            adm_loss_total += 1e-6 * l2_reg

            seg_loss = L_drop

            # --- Stage 7: zero grad & backward ---
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_ljm.zero_grad(set_to_none=True)
            self.optimizer_adm.zero_grad(set_to_none=True)

            self.scaler.scale(adm_loss_total).backward(retain_graph=True)
            self.scaler.scale(seg_loss).backward()

            self.scaler.step(self.optimizer_adm)
            self.scaler.step(self.optimizer_ljm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            return {'outputs': logits_drop, 'targets': targets_F_drop}

        else:
            # Eval
            with amp.autocast(enabled=self.amp_enabled):
                outputs, _ = self.model(inputs_orig)
            return {'outputs': outputs, 'targets': targets_F}

    def _after_epoch(self):
        self.model.eval()
        if self.ljm: self.ljm.eval()
        if self.adm: self.adm.eval()

    def train(self, dataflow: DataLoader, *, num_epochs: int = 9999, callbacks: Optional[List] = None):
        self.dataflow = dataflow
        self.num_epochs = num_epochs
        self.callbacks = callbacks if callbacks else []

        self.summary = Summary()
        try:
            self.epoch_num = 0
            self.global_step = 0
            train_time = time.perf_counter()
            while self.epoch_num < self.num_epochs:
                self.epoch_num += 1
                self._before_epoch()
                data_iterator = tqdm.tqdm(self.dataflow, desc=f'Epoch {self.epoch_num}', ncols=80)
                for feed_dict in data_iterator:
                    self.global_step += 1
                    output_dict = self._run_step(feed_dict)
                self._after_epoch()
            logger.info(f"Training finished in {time.perf_counter() - train_time:.2f}s")
        except StopTraining as e:
            logger.info(f"Training stopped: {e}")
