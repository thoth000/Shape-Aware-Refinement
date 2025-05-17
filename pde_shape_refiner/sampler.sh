#!/bin/bash

# エラー回避
export NO_ALBUMENTATIONS_UPDATE=1

# 使用するGPUを指定する（単一GPUで実行）
export CUDA_VISIBLE_DEVICES=0

# パラメータ設定
RESOLUTION=584
THRESHOLD=0.5
RESULT_DIR="result/sample"
DATASET_PATH="/home/sano/dataset/DRIVE"
DATASET_OPT="pad"
PRETRAINED_PATH="/home/sano/shape-aware-refinement/pde_shape_refiner/models/final_model.pth"
IMAGE_INDEX=0
NUM_ITERATIONS=100
GAMMA=0.1
SAVE_INTERVAL=10

# 実行時間のタイムスタンプを取得（結果ディレクトリを分けたい場合に使用）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# RESULT_DIR="${RESULT_DIR}_${TIMESTAMP}"

# 単一GPUでサンプラーを実行
python sampler.py \
    --resolution $RESOLUTION \
    --threshold $THRESHOLD \
    --result_dir $RESULT_DIR \
    --dataset_path $DATASET_PATH \
    --dataset_opt $DATASET_OPT \
    --pretrained_path $PRETRAINED_PATH \
    --image_index $IMAGE_INDEX \
    --num_iterations $NUM_ITERATIONS \
    --gamma $GAMMA \
    --save_interval $SAVE_INTERVAL
