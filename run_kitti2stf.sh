#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/SemanticSTFV2/"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


# 设置可见GPU
export CUDA_VISIBLE_DEVICES=6

#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#nohup torchrun --nproc_per_node=8 --master_port=28888 train/train_mseg3d_deform_v10.py > mseg3d_deform_v10.log 2>&1 &

nohup python PointDR/train_kitti2stf.py > kitti2stf.log 2>&1 &
