@echo off
REM =====================================================
REM  Role 2 (Part 2) — Full Pipeline Runner
REM  Double-click from anywhere — always runs from repo root
REM =====================================================

REM Anchor to the repo root regardless of where this bat is called from.
REM %~dp0 = directory of this bat file (part2\)
REM ..    = one level up = repo root
cd /d "%~dp0.."

echo.
echo =====================================================
echo  Student Placement Prediction — Part 2 Pipeline
echo  Working directory: %CD%
echo =====================================================
echo.

SET PYTHON=venv\Scripts\python.exe -X utf8

REM Step 1: Download dataset
echo [STEP 1/5] Downloading dataset ...
%PYTHON% download_dataset.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: download_dataset.py failed. & pause & exit /b 1)

REM Step 2: Preprocessing
echo.
echo [STEP 2/5] Running preprocessing ...
%PYTHON% preprocessing.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: preprocessing.py failed. & pause & exit /b 1)

REM Step 3: Logistic Regression
echo.
echo [STEP 3/5] Training Logistic Regression ...
%PYTHON% part2\logistic_regression_model.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: logistic_regression_model.py failed. & pause & exit /b 1)

REM Step 4: Random Forest
echo.
echo [STEP 4/5] Training Random Forest ...
%PYTHON% part2\random_forest_model.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: random_forest_model.py failed. & pause & exit /b 1)

REM Step 5a: Model Comparison
echo.
echo [STEP 5a/5] Running model comparison ...
%PYTHON% part2\model_comparison.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: model_comparison.py failed. & pause & exit /b 1)

REM Step 5b: Summary Report
echo.
echo [STEP 5b/5] Generating summary report ...
%PYTHON% part2\model_summary_report.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: model_summary_report.py failed. & pause & exit /b 1)

echo.
echo =====================================================
echo  PIPELINE COMPLETE — Check model_results\ folder
echo =====================================================
echo.
pause
