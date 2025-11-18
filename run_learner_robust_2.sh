#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/SemanticSTFV2/"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

config_file='/home/SemanticSTFV2/PointDR/configs/learner_minkunet_2.yaml'
run_dir='learner_robust_2'

# 设置可见GPU
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#
#nohup torchrun --nproc_per_node=4 --master_port=29999 PointDR/train_learner_minkunet.py > learner_.log 2>&1 &
#
##nohup python PointDR/train_learner_minkunet.py \
##      --config "$config_file" \
##      --run-dir "$run_dir" > learner_robust.log 2>&1 &
export CUDA_VISIBLE_DEVICES=4

nohup python PointDR/train_learner_minkunet.py \
      --config "$config_file" \
      --run-dir "$run_dir" > \
      learner_2.log 2>&1 &
