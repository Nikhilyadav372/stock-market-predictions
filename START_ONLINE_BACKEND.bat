@echo off
title AI Stock Forecasting Platform - Backend Tunnel
cd /d "%~dp0"
echo =========================================================
echo   Starting AI Stock Market Backend with Cloudflare Tunnel
echo =========================================================
echo.

if exist "backend\venv\Scripts\python.exe" (
    "backend\venv\Scripts\python.exe" run_online.py
) else (
    python run_online.py
)

pause
