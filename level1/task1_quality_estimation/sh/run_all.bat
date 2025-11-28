@echo off
REM Level 1 Task 1 Automatic Script (Train + Test)
REM Estimated time: Training ~2 hours + Testing ~20 minutes

echo ============================================
echo Level 1 Task 1 - Code Quality Estimation
echo Start time: %date% %time%
echo ============================================

REM Set output directory
set OUTPUT_DIR=..\..\outputs\level1\checkpoints\task1

REM Model parameters
set MODEL_NAME=..\..\models\codereviewer
set MAX_SOURCE_LENGTH=256

REM ============ Training Phase ============
echo.
echo [1/2] Start training...
echo ============================================

REM Training config (~2 hours total)
REM - 3000 samples x 2 epochs = ~1.5h training
REM - Validation on 500 samples = ~30min total
set TRAIN_BATCH_SIZE=32
set LEARNING_RATE=5e-4
set NUM_EPOCHS=2
set GRADIENT_ACCUMULATION_STEPS=1
set WARMUP_STEPS=50
set SAVE_STEPS=500
set SEED=42
set TRAIN_SAMPLE_NUM=1000
set VALID_SAMPLE_NUM=300

cd ..
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
    --valid_sample_num %VALID_SAMPLE_NUM% ^
    --seed %SEED%

if %errorlevel% neq 0 (
    echo Train failed!
    pause
    exit /b 1
)

echo.
echo Train finished!
echo ============================================

REM ============ Testing Phase ============
echo.
echo [2/2] Start testing...
echo ============================================

REM Test config (~20 minutes)
set MODEL_PATH=%OUTPUT_DIR%\checkpoint-epoch-2
set TEST_BATCH_SIZE=32
set TEST_SAMPLE_NUM=400

python test.py ^
    --model_path %MODEL_PATH% ^
    --batch_size %TEST_BATCH_SIZE% ^
    --max_source_length %MAX_SOURCE_LENGTH% ^
    --sample_num %TEST_SAMPLE_NUM% ^
    --seed %SEED%

if %errorlevel% neq 0 (
    echo Test failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo All completed!
echo End time: %date% %time%
echo ============================================
echo.
echo Results saved to: outputs\level1\task1\test_results.json
echo.
pause
