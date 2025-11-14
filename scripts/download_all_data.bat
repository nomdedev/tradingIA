@echo off
REM Script batch para descargar todos los datos de BTC/USD necesarios
REM Uso: download_all_data.bat

echo === Descargando Datos BTC/USD para Trading IA ===
echo.

set START_DATE=2020-01-01
set END_DATE=2024-12-31

echo Descargando datos desde %START_DATE% hasta %END_DATE%
echo.

echo Descargando timeframe 5 minutos...
python scripts/download_btc_data.py --start-date %START_DATE% --end-date %END_DATE% --timeframe 5Min

echo.
echo Descargando timeframe 15 minutos...
python scripts/download_btc_data.py --start-date %START_DATE% --end-date %END_DATE% --timeframe 15Min

echo.
echo Descargando timeframe 1 hora...
python scripts/download_btc_data.py --start-date %START_DATE% --end-date %END_DATE% --timeframe 1Hour

echo.
echo Descargando timeframe 4 horas...
python scripts/download_btc_data.py --start-date %START_DATE% --end-date %END_DATE% --timeframe 4Hour

echo.
echo === Descarga Completada ===
echo Verificando archivos generados...
echo.

if exist "data\raw\btc_usd_5m.csv" (
    for %%A in ("data\raw\btc_usd_5m.csv") do echo ✅ btc_usd_5m.csv - %%~zA bytes
) else (
    echo ❌ btc_usd_5m.csv - No generado
)

if exist "data\raw\btc_usd_15m.csv" (
    for %%A in ("data\raw\btc_usd_15m.csv") do echo ✅ btc_usd_15m.csv - %%~zA bytes
) else (
    echo ❌ btc_usd_15m.csv - No generado
)

if exist "data\raw\btc_usd_1h.csv" (
    for %%A in ("data\raw\btc_usd_1h.csv") do echo ✅ btc_usd_1h.csv - %%~zA bytes
) else (
    echo ❌ btc_usd_1h.csv - No generado
)

if exist "data\raw\btc_usd_4h.csv" (
    for %%A in ("data\raw\btc_usd_4h.csv") do echo ✅ btc_usd_4h.csv - %%~zA bytes
) else (
    echo ❌ btc_usd_4h.csv - No generado
)

echo.
echo === Proceso Finalizado ===
echo Puedes ejecutar: python scripts/backtest_example.py
pause