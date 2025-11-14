@echo off
REM Trading IA - Ejecutar Plataforma Principal
REM ==========================================
REM Este script ejecuta la plataforma principal de Trading IA
REM con la interfaz gráfica completa.

echo.
echo ===============================================
echo     [START] TRADING IA - PLATAFORMA PRINCIPAL
echo ===============================================
echo.

REM Verificar si estamos en el directorio correcto
if not exist "src\main_platform.py" (
    echo ❌ Error: No se encuentra src\main_platform.py
    echo 💡 Asegúrate de ejecutar este script desde la raíz del proyecto TradingIA
    pause
    exit /b 1
)

REM Verificar si existe el entorno virtual
if exist ".venv\Scripts\activate.bat" (
    echo ✅ Entorno virtual encontrado
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️  No se encontró entorno virtual, usando Python del sistema
)

REM Verificar si las dependencias están instaladas
echo 📦 Verificando dependencias...
python -c "import PySide6, pandas, numpy" 2>nul
if errorlevel 1 (
    echo ❌ Error: Dependencias no instaladas
    echo 💡 Ejecuta: pip install -r requirements_platform.txt
    pause
    exit /b 1
)

echo ✅ Dependencias verificadas
echo.

REM Ejecutar la plataforma principal
echo [START] Iniciando Trading IA Platform...
echo.
python src\main_platform.py

REM Si hay error, mostrar mensaje
if errorlevel 1 (
    echo.
    echo ❌ Error al ejecutar la plataforma
    echo 💡 Verifica los logs en la carpeta logs/
)

echo.
pause