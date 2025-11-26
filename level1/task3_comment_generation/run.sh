#!/bin/bash

# 路径配置
DATA_DIR="../../data/raw/Comment_Generation"
PRETRAIN_MODEL_DIR="../checkpoints"  # 预训练模型路径 (pytorch_model.bin 等)
OUTPUT_DIR="../../outputs/level1/task3"

mkdir -p $OUTPUT_DIR

# 1. 训练
python train.py \
  --do_train \
  --model_type codet5 \
  --model_name_or_path $PRETRAIN_MODEL_DIR \
  --tokenizer_path $PRETRAIN_MODEL_DIR \
  --train_filename $DATA_DIR/train.jsonl \
  --dev_filename $DATA_DIR/valid.jsonl \
  --output_dir $OUTPUT_DIR \
  --max_source_length 300 \
  --max_target_length 128 \
  --train_batch_size 4 \
  --train_epochs 10 \
  --learning_rate 3e-4 \
  --beam_size 6 \
  --add_lang_ids \
  --raw_input \
  --save_steps 1000 \
  --log_steps 100

# 2. 推理
# 使用保存的最后一个 checkpoint
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint-last"

python inference.py \
  --do_test \
  --model_type codet5 \
  --model_name_or_path $CHECKPOINT_PATH \
  --tokenizer_path $CHECKPOINT_PATH \
  --test_filename $DATA_DIR/test.jsonl \
  --output_dir $OUTPUT_DIR \
  --max_source_length 300 \
  --max_target_length 128 \
  --eval_batch_size 16 \
  --beam_size 10 \
  --add_lang_ids \
  --raw_input

# 3. 计算指标
python calc_metrics.py \
  --preds_file $OUTPUT_DIR/predictions.txt \
  --golds_file $OUTPUT_DIR/references.txt