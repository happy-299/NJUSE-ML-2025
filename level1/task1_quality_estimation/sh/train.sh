#!/bin/bash

# Level 1 Task 1 训练脚本

# 设置数据和输出目录
DATA_DIR="../../data/raw/Diff_Quality_Estimation"
OUTPUT_DIR="../../level1/checkpoints/task1"

# 模型参数
MODEL_NAME="microsoft/codereviewer"
MAX_SOURCE_LENGTH=512

# 训练参数
BATCH_SIZE=12
LEARNING_RATE=3e-4
NUM_EPOCHS=30
GRADIENT_ACCUMULATION_STEPS=3
WARMUP_STEPS=1000
SAVE_STEPS=3600
SEED=42

python ../train.py \
    --model_name_or_path ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --seed ${SEED}
