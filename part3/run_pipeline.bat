@echo off
REM =====================================================
REM  Role 3 (Part 3) — XGBoost Pipeline Runner
REM  Double-click from anywhere — always runs from repo root
REM =====================================================

REM Anchor to the repo root regardless of where this bat is called from.
REM %~dp0 = directory of this bat file (part3\)
REM ..    = one level up = repo root
cd /d "%~dp0.."

echo.
echo =====================================================
echo  Student Placement Prediction — Part 3 (XGBoost) Pipeline
echo  Working directory: %CD%
echo =====================================================
echo.

SET PYTHON=venv\Scripts\python.exe -X utf8

REM Step 1: Train XGBoost Model
echo.
echo [STEP 1/2] Training XGBoost Model (CUDA / CPU fallback) ...
%PYTHON% part3\xgboost_model.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: xgboost_model.py failed. & pause & exit /b 1)

REM Step 4: Smoke-test sample prediction
echo.
echo [STEP 4/4] Running smoke-test prediction ...
%PYTHON% part3\predict_sample.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: predict_sample.py failed. & pause & exit /b 1)

echo.
echo =====================================================
echo  PART 3 PIPELINE COMPLETE — Check part3\models\ and part3\model_results\
echo =====================================================
echo.
pause
