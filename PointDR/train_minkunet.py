import argparse
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from PointDR.core.minkunt_trainers import MinkUnetTrainer
from PointDR.tools.util import auto_time_set_run_dir, BestEpochSaver, EpochSaver

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.callbacks import MeanIoU


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='/home/SemanticSTFV2/PointDR/configs/base_minkunet.yaml',
        help='config file',
    )
    parser.add_argument('--run-dir', default='minkunet', help='run directory')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.distributed:
        dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    args.run_dir = auto_time_set_run_dir(args.run_dir)

    configs.run_dir = args.run_dir
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2**32 - 1)

    seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = builder.make_dataset()
    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset[split], num_replicas=dist.size(), rank=dist.rank(), shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(dataset[split],
                                                      batch_size=configs.batch_size,
                                                      sampler=sampler,
                                                      num_workers=configs.workers_per_gpu,
                                                      pin_memory=True,
                                                      collate_fn=dataset[split].collate_fn)

    model = builder.make_model().cuda()
    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = MinkUnetTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_workers=configs.workers_per_gpu,
        seed=seed,
        amp_enabled=configs.amp_enabled,
    )
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[InferenceRunner(
            dataflow[split],
            callbacks=[
                MeanIoU(name=f'iou/{split}', num_classes=configs.data.num_classes, ignore_label=configs.data.ignore_label),
            ],
        ) for split in ['test']] + [
            BestEpochSaver('iou/test', filename='best_epoch'),
            EpochSaver(max_to_keep=None),
        ])


if __name__ == '__main__':
    main()
