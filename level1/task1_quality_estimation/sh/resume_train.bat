@echo off
REM Resume training from last checkpoint
REM Use this script to continue training after interruption

echo ============================================
echo Resume Training for Level 1 Task 1
echo Start time: %date% %time%
echo ============================================

set OUTPUT_DIR=..\..\outputs\level1\checkpoints\task1
set MODEL_NAME=..\..\models\codereviewer
set MAX_SOURCE_LENGTH=256

set TRAIN_BATCH_SIZE=32
set LEARNING_RATE=5e-4
set NUM_EPOCHS=2
set GRADIENT_ACCUMULATION_STEPS=1
set WARMUP_STEPS=100
set SAVE_STEPS=200
set SEED=42
set TRAIN_SAMPLE_NUM=4000

REM Check for latest checkpoint
if exist "%OUTPUT_DIR%\checkpoint-epoch-2" (
    echo Found epoch 2 checkpoint - Training already completed!
    echo You can run test.bat directly.
    pause
    exit /b 0
)

if exist "%OUTPUT_DIR%\checkpoint-epoch-1" (
    echo Resuming from epoch 1 checkpoint...
    set RESUME_FROM=%OUTPUT_DIR%\checkpoint-epoch-1
) else if exist "%OUTPUT_DIR%\checkpoint-best" (
    echo Resuming from best checkpoint...
    set RESUME_FROM=%OUTPUT_DIR%\checkpoint-best
) else (
    echo No checkpoint found. Starting fresh training...
    set RESUME_FROM=
)

cd ..

if defined RESUME_FROM (
    python train.py ^
        --model_name_or_path %RESUME_FROM% ^
        --output_dir %OUTPUT_DIR% ^
        --batch_size %TRAIN_BATCH_SIZE% ^
        --learning_rate %LEARNING_RATE% ^
        --num_epochs %NUM_EPOCHS% ^
        --gradient_accumulation_steps %GRADIENT_ACCUMULATION_STEPS% ^
        --warmup_steps %WARMUP_STEPS% ^
        --save_steps %SAVE_STEPS% ^
        --max_source_length %MAX_SOURCE_LENGTH% ^
        --sample_num %TRAIN_SAMPLE_NUM% ^
        --seed %SEED%
) else (
    python train.py ^
        --model_name_or_path %MODEL_NAME% ^
        --output_dir %OUTPUT_DIR% ^
        --batch_size %TRAIN_BATCH_SIZE% ^
        --learning_rate %LEARNING_RATE% ^
        --num_epochs %NUM_EPOCHS% ^
        --gradient_accumulation_steps %GRADIENT_ACCUMULATION_STEPS% ^
        --warmup_steps %WARMUP_STEPS% ^
        --save_steps %SAVE_STEPS% ^
        --max_source_length %MAX_SOURCE_LENGTH% ^
        --sample_num %TRAIN_SAMPLE_NUM% ^
        --seed %SEED%
)

if %errorlevel% neq 0 (
    echo Training failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Training completed!
echo End time: %date% %time%
echo ============================================
pause
