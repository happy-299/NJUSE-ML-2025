@echo off
REM Level 1 Task 1 测试脚本 (Windows)

REM 模型路径
set MODEL_PATH=..\..\outputs\level1\checkpoints\task1\checkpoint-best

REM 测试参数
set BATCH_SIZE=32
set MAX_SOURCE_LENGTH=256
set SEED=42
set SAMPLE_NUM=2000

REM 切换到上级目录运行 test.py
cd ..
python test.py ^
    --model_path %MODEL_PATH% ^
    --batch_size %BATCH_SIZE% ^
    --max_source_length %MAX_SOURCE_LENGTH% ^
    --sample_num %SAMPLE_NUM% ^
    --seed %SEED%
