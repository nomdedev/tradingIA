#!/usr/bin/env python3
"""
Indicadores técnicos avanzados para trading institucional.

Implementa Squeeze Momentum, IFVG (Implied Fair Value Gaps) y EMAs institucionales.
"""

import pandas as pd
import numpy as np
import ta


def calculate_squeeze_momentum(df):
    """
    Calcula Squeeze Momentum Indicator.

    Squeeze Momentum combina Bollinger Bands y Keltner Channels para detectar
    periodos de baja volatilidad (squeeze) y momentum posterior.

    Args:
        df: DataFrame con columnas OHLC

    Returns:
        DataFrame con columnas agregadas: squeeze_on, squeeze_momentum, squeeze_momentum_delta
    """
    df = df.copy()

    # Bollinger Bands (20, 2σ)
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()

    # Keltner Channels (20, 1.5×ATR)
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=20).average_true_range()
    ema_20 = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()

    df['KC_Upper'] = ema_20 + (atr * 1.5)
    df['KC_Lower'] = ema_20 - (atr * 1.5)
    df['KC_Middle'] = ema_20

    # Squeeze Detection: BB dentro de KC = squeeze ON
    df['squeeze_on'] = ((df['BB_Upper'] <= df['KC_Upper']) & (df['BB_Lower'] >= df['KC_Lower'])).astype(int)

    # Momentum Calculation
    # avg_of_highest_lowest = (highest_high + lowest_low) / 2 over lookback period
    highest_high = df['High'].rolling(window=20).max()
    lowest_low = df['Low'].rolling(window=20).min()
    avg_hl = (highest_high + lowest_low) / 2

    df['squeeze_momentum'] = ((df['Close'] - avg_hl) / df['Close']) * 100

    # Momentum Delta (cambio en momentum)
    df['squeeze_momentum_delta'] = df['squeeze_momentum'].diff()

    print(f"✓ Squeeze Momentum calculado: {df['squeeze_on'].sum()} periodos de squeeze detectados")

    return df


def detect_ifvg(df, lookback=50):
    """
    Detecta Implied Fair Value Gaps (IFVG).

    IFVG son gaps de valor justo implícito que ocurren cuando el precio
    no alcanza niveles previos, creando zonas de valor.

    Args:
        df: DataFrame con columnas OHLC
        lookback: Periodos hacia atrás para buscar IFVGs

    Returns:
        DataFrame con columnas agregadas: ifvg_bullish_nearby, ifvg_bearish_nearby
    """
    df = df.copy()

    # Inicializar columnas
    df['ifvg_bullish_nearby'] = 0
    df['ifvg_bearish_nearby'] = 0

    for i in range(len(df)):
        current_price = df.iloc[i]['Close']

        # Buscar IFVGs en ventana de lookback
        start_idx = max(0, i - lookback)
        window = df.iloc[start_idx:i+1]

        # Bullish IFVG: High actual < Low de vela +2
        # Bearish IFVG: Low actual > High de vela +2

        bullish_ifvgs = []
        bearish_ifvgs = []

        for j in range(len(window) - 2):
            high_current = window.iloc[j]['High']
            low_future = window.iloc[j+2]['Low']

            low_current = window.iloc[j]['Low']
            high_future = window.iloc[j+2]['High']

            # Bullish IFVG
            if high_current < low_future:
                ifvg_mid = (high_current + low_future) / 2
                bullish_ifvgs.append(ifvg_mid)

            # Bearish IFVG
            if low_current > high_future:
                ifvg_mid = (low_current + high_future) / 2
                bearish_ifvgs.append(ifvg_mid)

        # Verificar si precio actual está cerca de algún IFVG (dentro de 2%)
        for ifvg_mid in bullish_ifvgs:
            if abs(current_price - ifvg_mid) / current_price <= 0.02:
                df.iloc[i, df.columns.get_loc('ifvg_bullish_nearby')] = 1
                break

        for ifvg_mid in bearish_ifvgs:
            if abs(current_price - ifvg_mid) / current_price <= 0.02:
                df.iloc[i, df.columns.get_loc('ifvg_bearish_nearby')] = 1
                break

    bullish_count = df['ifvg_bullish_nearby'].sum()
    bearish_count = df['ifvg_bearish_nearby'].sum()

    print(f"✓ IFVG detectados: {bullish_count} bullish, {bearish_count} bearish")

    return df


def calculate_institutional_emas(df):
    """
    Calcula EMAs institucionales (22, 55, 200).

    Estas son las medias móviles utilizadas por instituciones financieras
    para determinar tendencias macro.

    Args:
        df: DataFrame con columna Close

    Returns:
        DataFrame con columnas agregadas: EMA_22, EMA_55, EMA_200,
                                         ema_alignment, price_above_ema200,
                                         price_above_ema55, price_above_ema22
    """
    df = df.copy()

    # Calcular EMAs
    df['EMA_22'] = ta.trend.EMAIndicator(df['Close'], window=22).ema_indicator()
    df['EMA_55'] = ta.trend.EMAIndicator(df['Close'], window=55).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()

    # EMA Alignment
    # 1 = bullish (EMA22 > EMA55 > EMA200)
    # -1 = bearish (EMA22 < EMA55 < EMA200)
    # 0 = neutral
    df['ema_alignment'] = 0
    bullish_mask = (df['EMA_22'] > df['EMA_55']) & (df['EMA_55'] > df['EMA_200'])
    bearish_mask = (df['EMA_22'] < df['EMA_55']) & (df['EMA_55'] < df['EMA_200'])

    df.loc[bullish_mask, 'ema_alignment'] = 1
    df.loc[bearish_mask, 'ema_alignment'] = -1

    # Price above EMAs
    df['price_above_ema200'] = (df['Close'] > df['EMA_200']).astype(int)
    df['price_above_ema55'] = (df['Close'] > df['EMA_55']).astype(int)
    df['price_above_ema22'] = (df['Close'] > df['EMA_22']).astype(int)

    bullish_periods = (df['ema_alignment'] == 1).sum()
    bearish_periods = (df['ema_alignment'] == -1).sum()
    neutral_periods = (df['ema_alignment'] == 0).sum()

    print(f"✓ EMAs institucionales calculadas: {bullish_periods} bullish, {bearish_periods} bearish, {neutral_periods} neutral")

    return df


def add_all_advanced_indicators(df):
    """
    Agrega todos los indicadores avanzados al DataFrame.

    Args:
        df: DataFrame con columnas OHLC

    Returns:
        DataFrame con todos los indicadores avanzados agregados
    """
    print("\n🧮 Calculando indicadores avanzados...")

    # Squeeze Momentum
    df = calculate_squeeze_momentum(df)

    # IFVG Detection
    df = detect_ifvg(df)

    # Institutional EMAs
    df = calculate_institutional_emas(df)

    # Limpiar NaN generados por indicadores
    df = df.dropna()

    print(f"✓ Todos los indicadores avanzados agregados. Columnas finales: {len(df.columns)}")
    print("   Nuevas columnas: squeeze_on, squeeze_momentum, squeeze_momentum_delta,")
    print("   ifvg_bullish_nearby, ifvg_bearish_nearby, EMA_22, EMA_55, EMA_200,")
    print("   ema_alignment, price_above_ema200, price_above_ema55, price_above_ema22")

    return df