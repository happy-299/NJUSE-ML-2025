#!/bin/bash

# Level 1 Task 4: 代码修复生成 - 快速测试脚本 (小样本)

echo "Starting Task 4 Quick Test (Small Sample)..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 切换到项目根目录
cd "$(dirname "$0")/../../.."

# 快速测试参数
MODEL_PATH="level1/checkpoints/task4/checkpoint-best"
BATCH_SIZE=4
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=128
MAX_TEST_SAMPLES=100  # 小样本快速测试
NUM_BEAMS=3

echo "Quick Testing Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Test Samples: $MAX_TEST_SAMPLES (SMALL SAMPLE for quick test)"
echo "  Num Beams: $NUM_BEAMS"
echo

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model not found at $MODEL_PATH"
  echo "Please train the model first using: bash level1/task4_code_refinement/sh/train.sh"
  exit 1
fi

echo "Running quick test with small sample for fast evaluation..."

# 执行快速测试
python level1/task4_code_refinement/test.py \
  --model_path "$MODEL_PATH" \
  --batch_size $BATCH_SIZE \
  --max_source_length $MAX_SOURCE_LENGTH \
  --max_target_length $MAX_TARGET_LENGTH \
  --max_test_samples $MAX_TEST_SAMPLES \
  --num_beams $NUM_BEAMS \
  --temperature 1.0 \
  --do_sample False

echo
echo "Quick testing completed!"
echo "This was a quick test with $MAX_TEST_SAMPLES samples."
echo "For full evaluation, use: bash level1/task4_code_refinement/sh/test.sh"
echo "Results saved in: outputs/level1/task4/"