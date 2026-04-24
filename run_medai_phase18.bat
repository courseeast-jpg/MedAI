@echo off
setlocal

set STEP=python -m pytest tests
echo Running: %STEP%
python -m pytest tests
if errorlevel 1 goto :fail

set STEP=python scripts\run_phase11_integration_audit.py
echo Running: %STEP%
python scripts\run_phase11_integration_audit.py
if errorlevel 1 goto :fail

set STEP=python scripts\run_phase12_real_world_validation.py --dataset-dir test_data\final_batch_50 --quota-safe
echo Running: %STEP%
python scripts\run_phase12_real_world_validation.py --dataset-dir test_data\final_batch_50 --quota-safe
if errorlevel 1 goto :fail

set STEP=python scripts\run_phase17_dashboard.py --latest
echo Running: %STEP%
python scripts\run_phase17_dashboard.py --latest
if errorlevel 1 goto :fail

set STEP=python scripts\run_phase17_dashboard.py --export
echo Running: %STEP%
python scripts\run_phase17_dashboard.py --export
if errorlevel 1 goto :fail

echo Phase 18 one-command cycle completed successfully.
exit /b 0

:fail
echo Phase 18 one-command cycle failed at step: %STEP%
exit /b 1
