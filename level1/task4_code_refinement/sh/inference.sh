#!/bin/bash

# Level 1 Task 4: 代码修复生成 - 推理脚本

echo "Starting Task 4 Code Refinement Inference..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 切换到项目根目录
cd "$(dirname "$0")/../../.."

# 推理参数
MODEL_PATH="level1/checkpoints/task4/checkpoint-best"
INPUT_FILE="${1:-data/raw/ref-test.jsonl}"
OUTPUT_FILE="${2:-outputs/level1/task4/inference_results.jsonl}"
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=128
NUM_BEAMS=5

echo "Inference Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Input File: $INPUT_FILE"
echo "  Output File: $OUTPUT_FILE"
echo "  Num Beams: $NUM_BEAMS"
echo

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model not found at $MODEL_PATH"
  echo "Please train the model first using: bash level1/task4_code_refinement/sh/train.sh"
  exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file not found at $INPUT_FILE"
  echo "Usage: bash level1/task4_code_refinement/sh/inference.sh [input_file] [output_file]"
  exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_FILE")"

# 执行推理
python level1/task4_code_refinement/inference.py \
  --model_path "$MODEL_PATH" \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_source_length $MAX_SOURCE_LENGTH \
  --max_target_length $MAX_TARGET_LENGTH \
  --num_beams $NUM_BEAMS \
  --temperature 1.0

echo
echo "Inference completed!"
echo "Results saved to: $OUTPUT_FILE"