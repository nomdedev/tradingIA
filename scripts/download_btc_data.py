#!/usr/bin/env python3
"""
Script para descargar datos históricos de BTC/USD desde Alpaca API
para backtesting de estrategias de trading.

Uso:
    python download_btc_data.py --start-date 2020-01-01 --end-date 2024-01-01 --timeframe 1Day

Argumentos:
    --start-date: Fecha de inicio (YYYY-MM-DD)
    --end-date: Fecha de fin (YYYY-MM-DD)
    --timeframe: Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
    --output: Archivo de salida (default: data/raw/btc_usd_{timeframe}.csv)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    print("Error: alpaca-trade-api no está instalado.")
    print("Instala con: pip install alpaca-trade-api")
    sys.exit(1)

def download_btc_data(start_date, end_date, timeframe='1Day', output_file=None):
    """
    Descarga datos históricos de BTC/USD desde Alpaca.

    Args:
        start_date (str): Fecha de inicio en formato YYYY-MM-DD
        end_date (str): Fecha de fin en formato YYYY-MM-DD
        timeframe (str): Timeframe para los datos
        output_file (str): Archivo de salida (opcional)
    """
    # Configurar API de Alpaca
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://api.alpaca.markets')

    if not api_key or not api_secret:
        print("Error: Variables de entorno ALPACA_API_KEY y ALPACA_API_SECRET no encontradas.")
        print("Asegúrate de tener un archivo .env con tus credenciales de Alpaca.")
        return None

    # Crear cliente de Alpaca
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    # Símbolo de BTC/USD en Alpaca
    symbol = 'BTC/USD'

    # Convertir fechas
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    print(f"Descargando datos de {symbol} desde {start_date} hasta {end_date}")
    print(f"Timeframe: {timeframe}")

    try:
        # Alpaca tiene límites de rate, así que descargamos en chunks si es necesario
        all_data = []
        current_start = start

        while current_start < end:
            chunk_end = min(current_start + timedelta(days=30), end)

            print(f"Descargando chunk: {current_start.date()} - {chunk_end.date()}")

            # Descargar datos
            bars = api.get_crypto_bars(
                symbol,
                timeframe,
                start=current_start.isoformat(),
                end=chunk_end.isoformat(),
                limit=10000  # Máximo por request
            ).df

            if not bars.empty:
                all_data.append(bars)
                print(f"  Obtenidos {len(bars)} registros")
            else:
                print("  No se encontraron datos para este período")

            current_start = chunk_end

            # Pequeña pausa para no exceder rate limits
            import time
            time.sleep(0.1)

        if not all_data:
            print("No se encontraron datos para el período especificado.")
            return None

        # Combinar todos los datos
        df = pd.concat(all_data)

        # Limpiar y formatear datos
        df = df.reset_index()
        df = df.rename(columns={
            'timestamp': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'trade_count': 'trade_count',
            'vwap': 'vwap'
        })

        # Convertir timezone si es necesario
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')

        # Ordenar por fecha
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'])

        # Determinar archivo de salida
        if output_file is None:
            timeframe_clean = timeframe.lower().replace('min', 'm').replace('hour', 'h').replace('day', 'd')
            output_file = f"data/raw/btc_usd_{timeframe_clean}.csv"

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Guardar datos
        df.to_csv(output_file, index=False)
        print(f"Datos guardados en: {output_file}")
        print(f"Total de registros: {len(df)}")
        print(f"Período: {df['datetime'].min()} - {df['datetime'].max()}")

        return df

    except Exception as e:
        print(f"Error descargando datos: {e}")
        return None

def download_all_timeframes(start_date, end_date):
    """
    Descarga datos de BTC/USD en todos los timeframes necesarios para la plataforma.

    Args:
        start_date (str): Fecha de inicio (YYYY-MM-DD)
        end_date (str): Fecha de fin (YYYY-MM-DD)
    """
    timeframes = [
        ('5Min', '5m'),
        ('15Min', '15m'),
        ('1Hour', '1h'),
        ('4Hour', '4h')
    ]

    print("=== Descargando Datos BTC/USD para Todos los Timeframes ===\n")

    for alpaca_tf, filename_tf in timeframes:
        print(f"Descargando timeframe: {alpaca_tf}")
        print("-" * 50)

        df = download_btc_data(start_date, end_date, alpaca_tf)

        if df is not None:
            print(f"✅ {alpaca_tf} completado - {len(df)} registros\n")
        else:
            print(f"❌ Error descargando {alpaca_tf}\n")

        # Pausa entre timeframes para evitar rate limits
        import time
        time.sleep(1)

    print("=== Descarga Completada ===")
    print("Archivos generados:")
    for _, filename_tf in timeframes:
        filepath = f"data/raw/btc_usd_{filename_tf}.csv"
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✅ {filepath} ({size:,} bytes)")
        else:
            print(f"  ❌ {filepath} (no generado)")

def main():
    parser = argparse.ArgumentParser(description='Descargar datos históricos de BTC/USD desde Alpaca')
    parser.add_argument('--start-date', required=True, help='Fecha de inicio (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='Fecha de fin (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1Day', choices=['1Min', '5Min', '15Min', '1Hour', '4Hour', '1Day'],
                       help='Timeframe para los datos')
    parser.add_argument('--output', help='Archivo de salida (opcional)')
    parser.add_argument('--all-timeframes', action='store_true',
                       help='Descargar todos los timeframes necesarios (5m, 15m, 1h, 4h)')

    args = parser.parse_args()

    # Validar fechas
    try:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start >= end:
            print("Error: La fecha de inicio debe ser anterior a la fecha de fin.")
            sys.exit(1)
    except ValueError:
        print("Error: Formato de fecha inválido. Use YYYY-MM-DD.")
        sys.exit(1)

    # Descargar datos
    if args.all_timeframes:
        download_all_timeframes(args.start_date, args.end_date)
    else:
        df = download_btc_data(args.start_date, args.end_date, args.timeframe, args.output)

        if df is not None:
            print("\nResumen de datos:")
            print(f"Precio de apertura promedio: ${df['open'].mean():.2f}")
            print(f"Precio máximo: ${df['high'].max():.2f}")
            print(f"Precio mínimo: ${df['low'].min():.2f}")
            print(f"Volumen total: {df['volume'].sum():,.0f}")
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()