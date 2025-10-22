#!/usr/bin/env python3
"""
Script para calcular indicadores técnicos usando pandas-ta.
Agrega múltiples indicadores técnicos a datos OHLCV.
"""

import pandas as pd
import ta
import numpy as np
import os
from pathlib import Path

def add_technical_indicators(df):
    """
    Agrega indicadores técnicos al DataFrame usando la librería 'ta'.

    Args:
        df (pd.DataFrame): DataFrame con columnas OHLCV

    Returns:
        pd.DataFrame: DataFrame con indicadores agregados
    """
    print("🔧 Calculando indicadores técnicos...")

    initial_rows = len(df)
    initial_cols = len(df.columns)

    # A) MOMENTUM INDICATORS
    print("  → Calculando RSI...")
    rsi_14 = ta.momentum.RSIIndicator(df['Close'], window=14)
    df['RSI_14'] = rsi_14.rsi()

    rsi_21 = ta.momentum.RSIIndicator(df['Close'], window=21)
    df['RSI_21'] = rsi_21.rsi()

    print("  → Calculando MACD...")
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df['MACDh_12_26_9'] = macd.macd_diff()
    df['MACDs_12_26_9'] = macd.macd_signal()

    print("  → Calculando Stochastic...")
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['STOCHk_14_3_3'] = stoch.stoch()
    df['STOCHd_14_3_3'] = stoch.stoch_signal()

    # B) VOLATILITY INDICATORS
    print("  → Calculando ATR...")
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_14'] = atr.average_true_range()

    print("  → Calculando Bollinger Bands...")
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    df['BBB_20_2.0'] = bb.bollinger_wband()
    df['BBP_20_2.0'] = bb.bollinger_pband()

    print("  → Calculando volatilidad 20d...")
    df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()

    # C) TREND INDICATORS
    print("  → Calculando SMAs...")
    sma_20 = ta.trend.SMAIndicator(df['Close'], window=20)
    df['SMA_20'] = sma_20.sma_indicator()

    sma_50 = ta.trend.SMAIndicator(df['Close'], window=50)
    df['SMA_50'] = sma_50.sma_indicator()

    sma_200 = ta.trend.SMAIndicator(df['Close'], window=200)
    df['SMA_200'] = sma_200.sma_indicator()

    print("  → Calculando EMA...")
    ema_12 = ta.trend.EMAIndicator(df['Close'], window=12)
    df['EMA_12'] = ema_12.ema_indicator()

    print("  → Calculando ADX...")
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX_14'] = adx.adx()
    df['DMP_14'] = adx.adx_pos()
    df['DMN_14'] = adx.adx_neg()

    # D) VOLUME INDICATORS
    print("  → Calculando OBV...")
    obv = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume'])
    df['OBV'] = obv.on_balance_volume()

    print("  → Calculando CMF...")
    cmf = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    df['CMF'] = cmf.chaikin_money_flow()

    # E) RETURNS
    print("  → Calculando returns...")
    df['returns_1d'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(5)
    df['returns_20d'] = df['Close'].pct_change(20)
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # F) DERIVED FEATURES
    print("  → Calculando features derivados...")
    df['SMA_20_50_ratio'] = df['SMA_20'] / df['SMA_50']
    df['price_to_sma20'] = df['Close'] / df['SMA_20']
    df['bb_position'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])

    # Limpiar NaN
    print("  → Eliminando filas con NaN...")
    df_clean = df.dropna()

    final_rows = len(df_clean)
    final_cols = len(df_clean.columns)

    print("\n📊 Resumen del procesamiento:")
    print(f"   Filas iniciales: {initial_rows}")
    print(f"   Filas con NaN eliminadas: {initial_rows - final_rows}")
    print(f"   Filas finales: {final_rows}")
    print(f"   Total columnas: {final_cols} (inicial: {initial_cols}, agregadas: {final_cols - initial_cols})")

    # Agregar indicadores avanzados
    print("\n🔬 Agregando indicadores avanzados...")
    from utils.advanced_indicators import add_all_advanced_indicators
    df_clean = add_all_advanced_indicators(df_clean)

    return df_clean

def find_latest_spy_file(raw_dir):
    """
    Encuentra el archivo SPY más reciente en el directorio raw.

    Args:
        raw_dir (str): Directorio donde buscar archivos

    Returns:
        str or None: Ruta del archivo más reciente, o None si no encuentra
    """
    raw_path = Path(raw_dir)
    spy_files = list(raw_path.glob("SPY*.csv"))

    if not spy_files:
        return None

    # Ordenar por fecha de modificación (más reciente primero)
    latest_file = max(spy_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def main():
    """
    Función principal: carga datos, calcula indicadores y guarda.
    """
    print("🚀 Iniciando cálculo de indicadores técnicos\n")

    # Directorios
    raw_dir = "data/raw"
    processed_dir = "data/processed"

    # Buscar archivo SPY más reciente
    spy_file = find_latest_spy_file(raw_dir)

    if spy_file is None:
        print(f"❌ No se encontraron archivos SPY en {raw_dir}")
        print("💡 Ejecuta primero download_data.py para descargar datos")
        return

    print(f"📂 Archivo encontrado: {spy_file}")

    try:
        # Cargar datos
        print("📥 Cargando datos...")
        df = pd.read_csv(spy_file, index_col=0, parse_dates=True)

        # Limpiar datos: remover filas con index no fecha
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna()

        # Manejar MultiIndex columns de yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Convertir a numérico y limpiar
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        print(f"   Rango temporal: {df.index.min()} a {df.index.max()}")

        # Calcular indicadores
        df_with_indicators = add_technical_indicators(df)

        # Crear directorio processed si no existe
        os.makedirs(processed_dir, exist_ok=True)

        # Guardar resultado
        output_file = os.path.join(processed_dir, "SPY_with_indicators.csv")
        df_with_indicators.to_csv(output_file)
        print(f"\n💾 Datos guardados en: {output_file}")

        # Mostrar muestra
        print("\n📋 Muestra de datos procesados (últimas 5 filas):")
        sample_cols = ['Close', 'RSI_14', 'MACD_12_26_9', 'ATR_14', 'SMA_20']
        available_cols = [col for col in sample_cols if col in df_with_indicators.columns]
        print(df_with_indicators[available_cols].tail())

        print("\n✅ Indicadores calculados y guardados exitosamente!")

    except Exception as e:
        print(f"❌ Error en el procesamiento: {e}")
        print("💡 Verifica que pandas-ta esté instalado correctamente")

if __name__ == "__main__":
    main()