#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/SemanticSTFV2/"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

config_file='/home/SemanticSTFV2/PointDR/configs/aug_robust.yaml'
run_dir='aug_robust_weights'

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=4

#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#nohup torchrun --nproc_per_node=2 --master_port=29999 PointDR/train_aug.py > aug_pointdr.log 2>&1 &

nohup python PointDR/train_aug_robust.py \
      --config "$config_file" \
      --run-dir "$run_dir" > aug_robust_2.log 2>&1 &

