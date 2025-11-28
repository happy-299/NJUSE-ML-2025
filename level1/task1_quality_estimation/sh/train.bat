@echo off
REM Level 1 Task 1 训练脚本 (Windows)

REM 设置数据和输出目录
set DATA_DIR=..\..\data\raw\Diff_Quality_Estimation
set OUTPUT_DIR=..\..\outputs\level1\checkpoints\task1

REM 模型参数 (使用相对路径指向本地模型)
set MODEL_NAME=..\..\models\codereviewer
set MAX_SOURCE_LENGTH=256

REM 训练参数 (极简配置，约6小时完成)
set BATCH_SIZE=32
set LEARNING_RATE=5e-4
set NUM_EPOCHS=2
set GRADIENT_ACCUMULATION_STEPS=1
set WARMUP_STEPS=100
set SAVE_STEPS=200
set SEED=42

REM 采样数量
set SAMPLE_NUM=4000

REM 切换到上级目录运行 train.py
cd ..
python train.py ^
    --model_name_or_path %MODEL_NAME% ^
    --output_dir %OUTPUT_DIR% ^
    --batch_size %BATCH_SIZE% ^
    --learning_rate %LEARNING_RATE% ^
    --num_epochs %NUM_EPOCHS% ^
    --gradient_accumulation_steps %GRADIENT_ACCUMULATION_STEPS% ^
    --warmup_steps %WARMUP_STEPS% ^
    --save_steps %SAVE_STEPS% ^
    --max_source_length %MAX_SOURCE_LENGTH% ^
    --sample_num %SAMPLE_NUM% ^
    --seed %SEED%
