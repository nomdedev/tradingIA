#!/usr/bin/env python3
"""
Script de verificación completa del proyecto - Checkpoint 1
Verifica que toda la infraestructura esté funcionando antes de implementar las IAs
"""

import os
import pandas as pd
import importlib

def print_separator(title=""):
    """Imprimir separador con título opcional."""
    if title:
        print(f"\n============================================================\n{title}\n============================================================")
    else:
        print("\n============================================================")

def check_directories():
    """Verificar estructura de directorios."""
    print_separator("VERIFICACIÓN DE DIRECTORIOS")

    required_dirs = [
        'data/raw', 'data/processed', 'agents', 'environments', 'strategies',
        'utils', 'models', 'models/checkpoints', 'results', 'results/logs',
        'results/figures', 'results/backtests', 'notebooks', 'tests'
    ]

    passed = 0
    total = len(required_dirs)

    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}")
            passed += 1
        else:
            print(f"✗ {dir_path}")

    return passed, total

def check_init_files():
    """Verificar archivos __init__.py."""
    print_separator("VERIFICACIÓN DE ARCHIVOS __init__.py")

    required_inits = [
        'agents/__init__.py',
        'environments/__init__.py',
        'strategies/__init__.py',
        'utils/__init__.py'
    ]

    passed = 0
    total = len(required_inits)

    for init_file in required_inits:
        if os.path.isfile(init_file):
            print(f"✓ __init__.py en {init_file.split('/')[0]}")
            passed += 1
        else:
            print(f"✗ Falta __init__.py en {init_file.split('/')[0]}")

    return passed, total

def check_raw_data():
    """Verificar datos raw."""
    print_separator("VERIFICACIÓN DE DATOS RAW")

    raw_dir = 'data/raw'
    if not os.path.isdir(raw_dir):
        print("✗ No existe directorio data/raw")
        return 0, 1

    # Buscar archivos que empiecen con SPY
    spy_files = [f for f in os.listdir(raw_dir) if f.startswith('SPY') and f.endswith('.csv')]

    if not spy_files:
        print("✗ No se encontraron archivos SPY en data/raw")
        return 0, 1

    try:
        # Verificar el primer archivo encontrado
        file_path = os.path.join(raw_dir, spy_files[0])
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        rows, cols = df.shape

        # Verificar requisitos mínimos (OHLCV + fecha como índice)
        if rows > 1000 and cols >= 5:  # OHLCV son 5 columnas
            print(f"✓ Datos raw: {len(spy_files)} archivo(s) encontrado(s), {rows} filas, {cols} columnas")
            return 1, 1
        else:
            print(f"✗ Datos raw insuficientes: {rows} filas, {cols} columnas (necesita >1000 filas, >=5 columnas OHLCV)")
            return 0, 1

    except Exception as e:
        print(f"✗ Error al cargar datos raw: {str(e)}")
        return 0, 1

def check_processed_data():
    """Verificar datos procesados."""
    print_separator("VERIFICACIÓN DE DATOS PROCESADOS")

    processed_file = 'data/processed/SPY_with_indicators.csv'

    if not os.path.isfile(processed_file):
        print("✗ No existe data/processed/SPY_with_indicators.csv")
        return 0, 1

    try:
        df = pd.read_csv(processed_file, index_col=0)

        rows, cols = df.shape
        critical_cols = ['Close', 'RSI_14', 'MACD_12_26_9', 'ATR_14', 'SMA_20', 'BBL_20_2.0']

        # Verificar filas y columnas
        checks_passed = 0
        total_checks = 4

        if rows > 1000:
            checks_passed += 1
        else:
            print(f"✗ Insuficientes filas: {rows} (necesita >1000)")

        if cols > 30:
            checks_passed += 1
        else:
            print(f"✗ Insuficientes columnas: {cols} (necesita >30)")

        # Verificar columnas críticas
        missing_cols = [col for col in critical_cols if col not in df.columns]
        if not missing_cols:
            checks_passed += 1
        else:
            print(f"✗ Faltan columnas críticas: {missing_cols}")

        # Verificar NaN en columnas críticas
        nan_percentage = df[critical_cols].isnull().mean().mean() * 100
        if nan_percentage < 5:
            checks_passed += 1
        else:
            print(f"✗ Demasiados NaN: {nan_percentage:.1f}% (máximo 5%)")

        if checks_passed == total_checks:
            print(f"✓ Datos procesados: {rows} filas, {cols} columnas, {nan_percentage:.1f}% NaN")
            return 1, 1
        else:
            print(f"✗ Problemas con datos procesados: {checks_passed}/{total_checks} checks pasaron")
            return 0, 1

    except Exception as e:
        print(f"✗ Error al cargar datos procesados: {str(e)}")
        return 0, 1

def check_libraries():
    """Verificar librerías críticas."""
    print_separator("VERIFICACIÓN DE LIBRERÍAS CRÍTICAS")

    critical_libs = [
        'pandas', 'numpy', 'stable_baselines3', 'deap', 'gym',
        'backtesting', 'yfinance', 'ta'
    ]

    passed = 0
    total = len(critical_libs)

    for lib in critical_libs:
        try:
            importlib.import_module(lib)
            print(f"✓ {lib}")
            passed += 1
        except ImportError:
            print(f"✗ {lib} no instalada")

    return passed, total

def main():
    """Función principal de verificación."""
    print_separator("VERIFICACIÓN CHECKPOINT 1")

    # Ejecutar todas las verificaciones
    results = []

    # Directorios
    dir_passed, dir_total = check_directories()
    results.append((dir_passed, dir_total))

    # __init__.py
    init_passed, init_total = check_init_files()
    results.append((init_passed, init_total))

    # Datos raw
    raw_passed, raw_total = check_raw_data()
    results.append((raw_passed, raw_total))

    # Datos procesados
    proc_passed, proc_total = check_processed_data()
    results.append((proc_passed, proc_total))

    # Librerías
    lib_passed, lib_total = check_libraries()
    results.append((lib_passed, lib_total))

    # Calcular totales
    total_passed = sum(p for p, t in results)
    total_checks = sum(t for p, t in results)

    # Resumen final
    print_separator(f"RESUMEN: {total_passed}/{total_checks} checks pasados")

    if total_passed == total_checks:
        print("🎉 ✅ CHECKPOINT 1 COMPLETADO - Todo listo para implementar las IAs")
    else:
        print("❌ CHECKPOINT 1 FALLIDO - Resolver problemas antes de continuar")
        print("\nProblemas encontrados:")
        print(f"- Directorios: {dir_passed}/{dir_total} pasaron")
        print(f"- __init__.py: {init_passed}/{init_total} pasaron")
        print(f"- Datos raw: {raw_passed}/{raw_total} pasaron")
        print(f"- Datos procesados: {proc_passed}/{proc_total} pasaron")
        print(f"- Librerías: {lib_passed}/{lib_total} pasaron")

if __name__ == "__main__":
    main()