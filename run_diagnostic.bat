@echo off
chcp 65001 > nul
cls
echo ================================================================================
echo   DIAGNOSTICO DEL PROYECTO TRADING IA
echo ================================================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python scripts\diagnostic_report.py
    pause
) else (
    echo ERROR: No se encontro el entorno virtual .venv
    echo Por favor ejecuta primero: python -m venv .venv
    pause
    exit /b 1
)
