@echo off
REM Level 2 Task 1 & Task 2 - Run with DeepSeek API
REM Code Quality Estimation and Code Localization

echo ============================================
echo Level 2: DeepSeek LLM Inference
echo Start time: %date% %time%
echo ============================================



REM ============ Task 1: Code Quality Estimation ============
echo.
echo [1/2] Task 1: Code Quality Estimation
echo ============================================

cd task1_quality_estimation
python inference.py ^
    --provider deepseek ^
    --model deepseek-chat ^
    --max_samples 20 ^
    --temperature 0.7 ^
    --output_file predictions.json

if %errorlevel% neq 0 (
    echo Task 1 inference failed!
    pause
    exit /b 1
)

echo.
echo Task 1 evaluation...
python evaluate.py --predictions_file predictions.json

echo.
echo Task 1 completed!
echo ============================================

REM ============ Task 2: Code Localization ============
echo.
echo [2/2] Task 2: Code Localization
echo ============================================

cd ..\task2_code_localization
python inference.py ^
    --provider deepseek ^
    --model deepseek-chat ^
    --max_samples 20 ^
    --temperature 0.7 ^
    --output_file predictions.json

if %errorlevel% neq 0 (
    echo Task 2 inference failed!
    pause
    exit /b 1
)

echo.
echo Task 2 evaluation...
python evaluate.py --predictions_file predictions.json

echo.
echo ============================================
echo All Level 2 tasks completed!
echo End time: %date% %time%
echo ============================================
echo.
echo Results saved to:
echo   - outputs/level2/task1/predictions.json
echo   - outputs/level2/task2/predictions.json
echo.
pause
