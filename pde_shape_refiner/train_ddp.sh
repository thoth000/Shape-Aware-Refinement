#!/bin/bash

# OpenMPのスレッド数を1に設定して、負荷を削減する
export OMP_NUM_THREADS=1

# エラー回避
export NO_ALBUMENTATIONS_UPDATE=1

# マスターポートの設定
MASTER_PORT=$((50000 + RANDOM % 1000))  # 50000〜50999の範囲でランダムなポート番号

# 使用するGPUを指定する（例：4 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# トレーニング用引数の指定
MAX_EPOCH=40
BATCH_SIZE=4
RESOLUTION=592
LEARNING_RATE=1e-4
ETA_MIN=0
WEIGHT_DECAY=1e-5
CRITERION="BCE"

SCHEDULER="cosine_annealing"
EXP_DIR="exp"
EXP_NAME="exp_$(date +"%Y%m%d_%H%M%S")"  # exp_nameをタイムスタンプに基づいて設定
VAL_INTERVAL=1
THRESHOLD=0.5
NUM_WORKERS=4
DATASET="drive"
TRANSFORM="fr_unet"
DATASET_PATH="/home/sano/dataset/DRIVE" # should change to fit your environment
DATASET_OPT="512" # should change to fit your environment
PRETRAINED_PATH="/home/sano/shape-aware-refinement/pde_shape_refiner/models/improved_unet.pth" # should change to fit your environment

ALPHA=1.0 # loss ratio for BCE and clDice
EXP_NAME="alpha_1.0"

# PyTorch DDPでトレーニングを実行
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 --master_port=$MASTER_PORT train.py \
    --max_epoch $MAX_EPOCH \
    --batch $BATCH_SIZE \
    --resolution $RESOLUTION \
    --lr $LEARNING_RATE \
    --eta_min $ETA_MIN \
    --weight_decay $WEIGHT_DECAY \
    --criterion $CRITERION \
    --scheduler $SCHEDULER \
    --exp_dir $EXP_DIR \
    --exp_name $EXP_NAME \
    --val_interval $VAL_INTERVAL \
    --threshold $THRESHOLD \
    --num_workers $NUM_WORKERS \
    --dataset $DATASET \
    --transform $TRANSFORM \
    --dataset_path $DATASET_PATH \
    --dataset_opt $DATASET_OPT \
    --pretrained_path $PRETRAINED_PATH \
    --alpha $ALPHA