#!/bin/bash

# Level 1 Task 1 测试脚本

# 模型路径
MODEL_PATH="../../level1/checkpoints/task1/checkpoint-best"

# 测试参数
BATCH_SIZE=16
MAX_SOURCE_LENGTH=512
SEED=42

python ../test.py \
    --model_path ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --seed ${SEED}
