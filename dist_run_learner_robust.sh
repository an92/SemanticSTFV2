#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/SemanticSTFV2/"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

config_file='/home/SemanticSTFV2/PointDR/configs/learner_minkunet_1.yaml'
run_dir='learner_robust_dist'

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=2,3,4,5

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

nohup torchrun --nproc_per_node=4 \
      --nnodes=$NNODES \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$PORT \
      PointDR/dist_train_learner_minkunet.py \
      --config "$config_file" \
      --run-dir "$run_dir"  \
      > learner_dist.log 2>&1 &

#nohup python PointDR/train_learner_minkunet.py \
#      --config "$config_file" \
#      --run-dir "$run_dir" > learner_robust.log 2>&1 &
#export CUDA_VISIBLE_DEVICES=6
#
#nohup python PointDR/train_learner_minkunet.py > learner_.log 2>&1 &
