@echo off
REM =====================================================
REM  Role 4 (Part 4) — Explainability & Fairness Pipeline
REM  Double-click from anywhere — always runs from repo root
REM =====================================================

cd /d "%~dp0.."

echo.
echo =====================================================
echo  Student Placement Prediction — Part 4 (Explainability)
echo  Working directory: %CD%
echo =====================================================
echo.

SET PYTHON=venv\Scripts\python.exe -X utf8

REM Step 1: Ensure dependencies exist
echo [STEP 1/2] Checking dependencies ...
%PYTHON% -c "import shap" 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Installing shap ...
    %PYTHON% -m pip install shap>=0.44.0
)

REM Step 2: Run explainability pipeline
echo.
echo [STEP 2/2] Running explainability and fairness pipeline ...
%PYTHON% part4\explainability_fairness.py
IF %ERRORLEVEL% NEQ 0 (echo ERROR: explainability_fairness.py failed. & pause & exit /b 1)

echo.
echo =====================================================
echo  PART 4 PIPELINE COMPLETE — Check part4\explainability_results\
echo =====================================================
echo.
pause
