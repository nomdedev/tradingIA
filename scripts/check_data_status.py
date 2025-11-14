#!/usr/bin/env python3
"""
Script de ejemplo para verificar el estado de los datos descargados
y mostrar información útil sobre los archivos disponibles.
"""

import os
from datetime import datetime

def check_data_status():
    """Verifica el estado de todos los archivos de datos BTC/USD"""
    timeframes = [
        ('5m', '5 minutos - High frequency scalping'),
        ('15m', '15 minutos - Intraday analysis'),
        ('1h', '1 hora - Swing trading'),
        ('4h', '4 horas - Position trading')
    ]

    print("=== Estado de Datos BTC/USD ===\n")

    total_files = 0
    total_size = 0

    for filename_tf, description in timeframes:
        filepath = f"data/raw/btc_usd_{filename_tf}.csv"

        if os.path.exists(filepath):
            # File exists - get stats
            file_size = os.path.getsize(filepath)
            total_size += file_size
            total_files += 1

            # Get record count
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    record_count = sum(1 for _ in f) - 1  # Subtract header
            except Exception:
                record_count = 0

            # Get last modified
            mod_time = os.path.getmtime(filepath)
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

            size_mb = file_size / (1024 * 1024)

            print(f"[OK] {filename_tf}: {description}")
            print(f"   [DATA] Ubicacion: {filepath}")
            print(f"   [RECORDS] Registros: {record_count:,}")
            print(f"   [SIZE] Tamano: {size_mb:.1f} MB")
            print(f"   [DATE] Modificado: {mod_date}")
            print()

        else:
            timeframe_arg = filename_tf.replace('m', 'Min').replace('h', 'Hour')
            print(f"[MISSING] {filename_tf}: {description}")
            print(f"   [PATH] Archivo faltante: {filepath}")
            print(f"   [RUN] Ejecuta: python scripts/download_btc_data.py --timeframe {timeframe_arg} --start-date 2020-01-01 --end-date 2024-01-01")
            print()

    # Summary
    print("=== Resumen ===")
    print(f"Archivos disponibles: {total_files}/4")
    total_size_mb = total_size / (1024 * 1024)
    print(f"Tamano total: {total_size_mb:.1f} MB")

    if total_files == 4:
        print("[SUCCESS] Todos los datos estan descargados!")
    else:
        missing = 4 - total_files
        print(f"[WARNING] Faltan {missing} archivo(s) por descargar")
        print("[INFO] Usa la pestana 'Data Download' en la plataforma o ejecuta:")
        print("   python scripts/download_btc_data.py --all-timeframes --start-date 2020-01-01 --end-date 2024-01-01")

if __name__ == "__main__":
    check_data_status()