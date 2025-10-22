#!/usr/bin/env python3
"""
Script para descargar datos de trading.
Descarga datos históricos de SPY y calcula indicadores técnicos.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import ta  # technical analysis library
from datetime import datetime, timedelta

def download_spy_data():
    """Descargar datos históricos de SPY"""
    print("Descargando datos de SPY...")

    # Descargar últimos 2 años de datos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 años

    spy = yf.Ticker("SPY")
    df = spy.history(start=start_date, end=end_date, interval="1d")

    print(f"Datos descargados: {len(df)} filas")
    return df

def calculate_indicators(df):
    """Calcular indicadores técnicos"""
    print("Calculando indicadores técnicos...")

    # RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD_12_26_9'] = macd.macd()
    df['MACDs_12_26_9'] = macd.macd_signal()
    df['MACDh_12_26_9'] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    df['BBB_20_2.0'] = bb.bollinger_pband()
    df['BBP_20_2.0'] = bb.bollinger_pband()

    # SMA
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()

    # EMA
    df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['STOCHk_14_3_3'] = stoch.stoch()
    df['STOCHd_14_3_3'] = stoch.stoch_signal()

    # Williams %R
    df['WILLR_14'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

    # Commodity Channel Index
    df['CCI_14'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()

    # Average True Range
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    # Volume indicators
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['AD'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()

    return df

def main():
    """Función principal"""
    print("Modulo de descarga de datos")
    print("Descargando datos de SPY con indicadores...")

    try:
        # Crear directorios
        os.makedirs("data/processed", exist_ok=True)

        # Descargar datos
        df = download_spy_data()

        # Calcular indicadores
        df = calculate_indicators(df)

        # Limpiar datos (eliminar filas con NaN)
        df_clean = df.dropna()

        # Guardar datos
        output_path = "data/processed/SPY_with_indicators.csv"
        df_clean.to_csv(output_path)

        print(f"Datos guardados en: {output_path}")
        print(f"Filas finales: {len(df_clean)}")
        print("Columnas:", list(df_clean.columns))

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()