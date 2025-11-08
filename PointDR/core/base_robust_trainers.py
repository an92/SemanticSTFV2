import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

import time
from typing import Any, Dict, List, Optional, Callable
from torch.utils.data import DataLoader
import torchsparse.nn as spnn
from torchpack.callbacks import (Callback, Callbacks)
from torchpack.train.exception import StopTraining
from torchpack.train.summary import Summary
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.config import configs
from core.callbacks import MeanIoU
from torchsparse import SparseTensor
import tqdm

__all__ = ['BaseRobustTrainer']



class SpatialAwareAdversarialGenerator(nn.Module):
    """
    空间感知型对抗性生成器 (G_Adv)。
    使用稀疏 3D 卷积在特征空间生成具有局部依赖性的扰动。
    """

    def __init__(self, input_dim: int, output_dim: int, ks: int = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            spnn.Conv3d(input_dim, input_dim * 2, kernel_size=ks, stride=1),
            spnn.BatchNorm(input_dim * 2),
            spnn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            spnn.Conv3d(input_dim * 2, output_dim, kernel_size=ks, stride=1),

        )
        # Tanh 激活函数 (用于约束扰动范围)
        self.tanh = nn.Tanh()

    def forward(self, x: SparseTensor) -> SparseTensor:
        """
        x: 稀疏张量 (SparseTensor)，包含特征 F_raw 和坐标 C。
        返回: 稀疏张量 (SparseTensor)，包含扰动 delta_F。
        """
        out = self.conv1(x)
        out = self.conv2(out)

        delta_F = self.tanh(out.F)

        return SparseTensor(delta_F, out.C, out.s)

class BaseRobustTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False,
                 adv_lambda: float = 0.05,  # 对抗损失 L_Adv_U 的权重
                 adv_epsilon: float = 0.1,
                 adv_lr: float = 1e-4) -> None:
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

        self.lamda = 0.1  # 原有的对比损失 L_con 权重
        self.T = 0.07

        self.adv_epsilon = adv_epsilon
        self.adv_lambda = adv_lambda

        self.adv_generator = SpatialAwareAdversarialGenerator(
            input_dim=48,
            output_dim=48
        ).cuda()

        # 实例化 G_Adv 优化器和 Scaler
        self.adv_generator_optimizer = torch.optim.Adam(
            self.adv_generator.parameters(),
            lr=adv_lr
        )
        self.adv_scaler = amp.GradScaler(enabled=self.amp_enabled)

    def _before_epoch(self) -> None:
        self.model.train()
        if self.adv_generator is not None:
            self.adv_generator.train()  # 确保 G_Adv 处于训练模式

        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()

        inputs_1 = _inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)

        loss_AdvU_generator = torch.zeros(1, device='cuda', dtype=torch.float32)

        with amp.autocast(enabled=self.amp_enabled):
            # outputs_1: logits, feat_1: proj(128维), feat_1_raw: y4.F (cs[8]维)
            outputs_1, feat_1, x_raw = self.model(inputs_1)

            if outputs_1.requires_grad:
                loss_1 = self.criterion(outputs_1, targets_1)

        if outputs_1.requires_grad:
            targets_cuda = targets_1  # 默认使用 clean view 标签作为对比学习目标

            x_raw_detached = SparseTensor(x_raw.F.detach(), x_raw.C, x_raw.s)
            x_raw_detached.F.requires_grad = True

            with amp.autocast(enabled=self.amp_enabled):
                # 1.1 优化 G_Adv (最大化不确定性 U)
                delta_x = self.adv_generator(x_raw_detached)

                # L_inf 约束和应用扰动 (在特征张量 F 上进行)
                delta_F_clamped = torch.clamp(delta_x.F, -self.adv_epsilon, self.adv_epsilon)
                feat_hard_raw_tensor = x_raw.F + delta_F_clamped

                # 计算不确定性 U (预测熵)
                outputs_hard_logits = self.model.predict_head(feat_hard_raw_tensor.float())
                outputs_hard_logsoftmax = nn.functional.log_softmax(outputs_hard_logits, dim=1)
                U = -torch.sum(torch.exp(outputs_hard_logsoftmax) * outputs_hard_logsoftmax, dim=1).mean()

                # L_Adv_U 损失：最大化 U -> 损失是 -U
                loss_AdvU_generator = -U

                # G_Adv 优化步骤
            self.adv_generator_optimizer.zero_grad()
            self.adv_scaler.scale(loss_AdvU_generator).backward(retain_graph=True)
            self.adv_scaler.step(self.adv_generator_optimizer)
            self.adv_scaler.update()

            # 2. 最终 Query 特征：使用硬样本的投影特征 F_hard
            feat_2_query = self.model.proj(feat_hard_raw_tensor.detach().float())

            # 3. 对比损失 (L_con) 计算 (与之前逻辑相同)
            feat_1_norm = nn.functional.normalize(feat_1.detach(), dim=1)
            feat_2_norm = nn.functional.normalize(feat_2_query, dim=1)

            feat1_proto = torch.zeros((configs.data.num_classes, feat_1_norm.shape[1]), device='cuda')
            for ii in range(configs.data.num_classes):
                mask = (targets_1 == ii)
                if mask.sum():
                    feat1_proto[ii] = feat_1_norm[mask].mean(dim=0)
            feat1_proto = (feat1_proto + 1e-8).cuda()

            logits = torch.mm(feat_2_norm, self.model.memo_bank.T.detach())
            logits /= self.T
            loss_2 = self.criterion(logits, targets_cuda)

            self.model.momentum_update_key_encoder(feat1_proto.detach(), init=(self.global_step == 1))

            # final loss
            loss = loss_1 + self.lamda * loss_2

            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('loss_1', loss_1.item())
            self.summary.add_scalar('loss_CL', loss_2.item())
            self.summary.add_scalar('loss_AdvU_G', loss_AdvU_generator.item())

            # 主模型优化步骤
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

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict['adv_generator'] = self.adv_generator.state_dict()
        state_dict['adv_optimizer'] = self.adv_generator_optimizer.state_dict()
        state_dict['adv_scaler'] = self.adv_scaler.state_dict()

        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.adv_generator.load_state_dict(state_dict['adv_generator'])
        self.adv_generator_optimizer.load_state_dict(state_dict['adv_optimizer'])
        self.adv_scaler.load_state_dict(state_dict['adv_scaler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass

    def train(self,
              dataflow: DataLoader,
              *,
              num_epochs: int = 9999999,
              callbacks: Optional[List[Callback]] = None) -> None:
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

                logger.info('Epoch {}/{} started.'.format(
                    self.epoch_num, self.num_epochs))
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
            outputs, _ , _= model(inputs)

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