# Trading IA - Ejecutar Plataforma Principal
# ==========================================
# Script PowerShell para ejecutar la plataforma principal de Trading IA

param(
    [switch]$NoVirtualEnv,
    [switch]$Help
)

if ($Help) {
    Write-Host "Trading IA - Ejecutar Plataforma Principal" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host "Este script ejecuta la plataforma principal de Trading IA con interfaz grafica."
    Write-Host ""
    Write-Host "Uso:" -ForegroundColor Yellow
    Write-Host "  .\start_platform.ps1              # Ejecutar normalmente"
    Write-Host "  .\start_platform.ps1 -NoVirtualEnv # Ejecutar sin entorno virtual"
    Write-Host "  .\start_platform.ps1 -Help         # Mostrar esta ayuda"
    Write-Host ""
    Write-Host "Requisitos:" -ForegroundColor Yellow
    Write-Host "  - Python 3.8+ instalado"
    Write-Host "  - Dependencias instaladas (pip install -r requirements_platform.txt)"
    Write-Host "  - Archivo .env configurado con credenciales de Alpaca"
    exit 0
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Magenta
Write-Host "    START TRADING IA - PLATAFORMA PRINCIPAL" -ForegroundColor Magenta
Write-Host "===============================================" -ForegroundColor Magenta
Write-Host ""

# Verificar si estamos en el directorio correcto
if (!(Test-Path "src\main_platform.py")) {
    Write-Host "ERROR: No se encuentra src\main_platform.py" -ForegroundColor Red
    Write-Host "INFO: Asegurate de ejecutar este script desde la raiz del proyecto TradingIA" -ForegroundColor Yellow
    Read-Host "Presiona Enter para continuar"
    exit 1
}

# Verificar entorno virtual (a menos que se especifique -NoVirtualEnv)
if (!$NoVirtualEnv -and (Test-Path ".venv\Scripts\activate.ps1")) {
    Write-Host "OK: Activando entorno virtual..." -ForegroundColor Green
    & ".venv\Scripts\activate.ps1"
} elseif (!$NoVirtualEnv) {
    Write-Host "WARN: No se encontro entorno virtual, usando Python del sistema" -ForegroundColor Yellow
}

# Verificar dependencias criticas
Write-Host "Verificando dependencias..." -ForegroundColor Blue
try {
    python -c "import PySide6, pandas, numpy, sys; print('Python OK')" 2>$null
    Write-Host "OK: Dependencias verificadas" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Dependencias no instaladas o Python no encontrado" -ForegroundColor Red
    Write-Host "INFO: Ejecuta: pip install -r requirements_platform.txt" -ForegroundColor Yellow
    Read-Host "Presiona Enter para continuar"
    exit 1
}

# Verificar archivo .env
if (!(Test-Path ".env")) {
    Write-Host "WARN: No se encontro archivo .env" -ForegroundColor Yellow
    Write-Host "INFO: Crea un archivo .env con tus credenciales de Alpaca" -ForegroundColor Yellow
    Write-Host "   Copia .env.example y configura tus claves API" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "START: Iniciando Trading IA Platform..." -ForegroundColor Green
Write-Host "   (La ventana de la aplicacion se abrira en unos segundos)" -ForegroundColor Gray
Write-Host ""

try {
    # Ejecutar la plataforma principal
    python src\main_platform.py
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Error al ejecutar la plataforma (Codigo: $exitCode)" -ForegroundColor Red
        Write-Host "INFO: Verifica los logs en la carpeta logs/" -ForegroundColor Yellow
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: Error al ejecutar la plataforma: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host ""
    Read-Host "Press Enter to close"
}