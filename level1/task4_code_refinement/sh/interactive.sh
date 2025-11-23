#!/bin/bash

# Level 1 Task 4: 代码修复生成 - 交互推理脚本

echo "Starting Task 4 Code Refinement Interactive Mode..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 切换到项目根目录
cd "$(dirname "$0")/../../.."

# 推理参数
MODEL_PATH="level1/checkpoints/task4/checkpoint-best"
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=128
NUM_BEAMS=5

echo "Interactive Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Num Beams: $NUM_BEAMS"
echo

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model not found at $MODEL_PATH"
  echo "Please train the model first using: bash level1/task4_code_refinement/sh/train.sh"
  exit 1
fi

echo "Launching interactive mode..."
echo "You can input code and comments to get refined code suggestions."
echo

# 执行交互推理
python level1/task4_code_refinement/inference.py \
  --model_path "$MODEL_PATH" \
  --max_source_length $MAX_SOURCE_LENGTH \
  --max_target_length $MAX_TARGET_LENGTH \
  --num_beams $NUM_BEAMS \
  --temperature 1.0 \
  --interactive

echo "Interactive mode ended."