import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from PointDR.tools.util import auto_time_set_run_dir, BestEpochSaver, EpochSaver
from PointDR.core.learner_trainers import MinkUnetLearnerTrainer
from core import builder
from core.callbacks import MeanIoU


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='/home/SemanticSTFV2/PointDR/configs/learner_minkunet_1.yaml',
        help='config file',
    )
    parser.add_argument('--run-dir', default='minkunet_learner_1', help='run directory')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    # ensure LOCAL_RANK env exists for compatibility
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # initialize distributed if requested
    if configs.distributed:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )

    # deterministic-ish run_dir across processes: use the same timestamp
    timestamp = int(time.time())
    args.run_dir = f"{args.run_dir}_{timestamp}"
    # Only rank0 will actually create directories / log experiment metadata
    if (not configs.distributed) or (dist.rank() == 0):
        args.run_dir = auto_time_set_run_dir(args.run_dir)
    # sync so every process has same value in configs.run_dir
    if configs.distributed:
        # broadcast run_dir string from rank0 to others using environment variable hack
        # (torchpack's dist doesn't expose a convenient broadcast-string helper here),
        # so we write to a temporary file in a shared filesystem if needed. Simpler approach:
        # store run_dir in configs (all processes constructed args.run_dir above with same timestamp
        # and auto_time_set_run_dir on rank0 only modifies slightly; for simplicity we use the
        # timestamped name for all processes and ensure rank0 created any needed folders below.)
        pass

    configs.run_dir = args.run_dir

    # only print full logs from rank0
    if (not configs.distributed) or (dist.rank() == 0):
        logger.info(' '.join([sys.executable] + sys.argv))
        logger.info(f'Experiment started: "{args.run_dir}".\n{configs}')

    # seeds (make reproducible across ranks)
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    # Compose per-process seed: include rank, workers_per_gpu and num_epochs for some variability
    rank = dist.rank() if configs.distributed else 0
    seed = int(configs.train.seed) + rank * int(configs.workers_per_gpu) * int(configs.num_epochs)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    # build dataset
    dataset = builder.make_dataset()
    dataflow = {}
    for split in dataset:
        if configs.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset[split],
                num_replicas=dist.size(),
                rank=dist.rank(),
                shuffle=(split == 'train')
            )
            shuffle = False
        else:
            sampler = None
            shuffle = (split == 'train')

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn
        )

    # build model, criterion, optimizer, scheduler
    model = builder.make_model().cuda()

    if configs.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank()],
            output_device=dist.local_rank(),
            find_unused_parameters=True
        )

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = MinkUnetLearnerTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_workers=configs.workers_per_gpu,
        seed=seed,
        amp_enabled=configs.amp_enabled,
        lamda_ct=configs.model.lamda_ct,
        lamda_sc=configs.model.lamda_sc,
        lamda_snc=configs.model.lamda_snc,
        k_snc=configs.model.k_snc,
    )

    # ensure rank0 creates run_dir if not created yet (safety)
    if configs.distributed and dist.rank() == 0:
        os.makedirs(configs.run_dir, exist_ok=True)
    if configs.distributed:
        dist.barrier()

    # Only rank0 will save logs / checkpoints in a human-friendly way handled by callbacks.
    # The trainer and callbacks are expected to obey configs.run_dir value.
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
            InferenceRunner(
                dataflow['test'],
                callbacks=[
                    MeanIoU(
                        name='iou/test',
                        num_classes=configs.data.num_classes,
                        ignore_label=configs.data.ignore_label,
                    ),
                ],
            ),
            BestEpochSaver('iou/test', filename='best_epoch'),
            EpochSaver(max_to_keep=None),
        ]
    )


if __name__ == '__main__':
    main()
