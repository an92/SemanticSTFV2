#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/SemanticSTFV2/"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


# 设置可见GPU
export CUDA_VISIBLE_DEVICES=3

#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#nohup torchrun --nproc_per_node=2 --master_port=29999 PointDR/train_aug.py > aug_pointdr.log 2>&1 &


nohup python PointDR/train_aug_minkunet.py > aug_minkunet.log 2>&1 &

