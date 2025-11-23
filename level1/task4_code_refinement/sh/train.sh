#!/bin/bash

# Level 1 Task 4: 代码修复生成 - 训练脚本

echo "Starting Task 4 Code Refinement Training..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 切换到项目根目录
cd "$(dirname "$0")/../../.."

# 训练参数
MODEL_NAME="microsoft/codereviewer"
OUTPUT_DIR="level1/checkpoints/task4"
BATCH_SIZE=8
LEARNING_RATE=3e-4
NUM_EPOCHS=10
GRADIENT_ACCUMULATION_STEPS=4
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=128
MAX_TRAIN_SAMPLES=50000
MAX_VALID_SAMPLES=5000

echo "Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Max Train Samples: $MAX_TRAIN_SAMPLES"
echo "  Max Valid Samples: $MAX_VALID_SAMPLES"
echo

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行训练
python level1/task4_code_refinement/train.py \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_source_length $MAX_SOURCE_LENGTH \
  --max_target_length $MAX_TARGET_LENGTH \
  --max_train_samples $MAX_TRAIN_SAMPLES \
  --max_valid_samples $MAX_VALID_SAMPLES \
  --warmup_steps 500 \
  --save_steps 1000 \
  --max_grad_norm 1.0 \
  --seed 42

echo "Training completed!"