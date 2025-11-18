#!/usr/bin/env python3
"""
Diagnóstico de Señales: Squeeze Momentum + ADX
Verificar por qué no se generan trades
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import talib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy


def diagnose_signals():
    """Diagnostic function to check signal generation"""

    print("🔍 DIAGNÓSTICO DE SEÑALES")
    print("=" * 80)

    # Load data
    df_5m = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df_5m.columns = df_5m.columns.str.lower()
    df_5m = df_5m.tail(1000)  # Smaller sample for diagnosis

    print(f"📊 Datos cargados: {len(df_5m)} barras")
    print()

    # Create strategy
    strategy = SqueezeMomentumADXTTMStrategy()
    strategy.min_adx_entry = 20
    strategy.max_adx_entry = 50
    strategy.min_squeeze_momentum = 0.3
    strategy.use_volume_filter = True
    strategy.volume_threshold = 1.2

    # Generate signals
    print("🔄 Generando señales...")
    signals = strategy.generate_signals({'5Min': df_5m})

    print(f"📊 Señales generadas:")
    print(f"   - Total entries: {signals['entries'].sum()}")
    print(f"   - Total exits: {signals['exits'].sum()}")
    print(f"   - Total signals: {len(signals['signals'][signals['signals'] != 0])}")
    print()

    # Check individual conditions
    df = df_5m.copy()
    df = strategy._calculate_squeeze_momentum(df)
    df = strategy._calculate_adx(df)
    df = strategy._calculate_ttm_waves(df)

    # Check each condition
    print("🔍 ANÁLISIS DE CONDICIONES:")
    print(f"   - Barras con ADX > 20: {len(df[df['adx'] > 20])}")
    print(f"   - Barras con ADX entre 20-50: {len(df[(df['adx'] >= 20) & (df['adx'] <= 50)])}")
    print(f"   - Barras con squeeze momentum > 0.3: {len(df[abs(df['linreg_value']) > 0.3])}")
    print(f"   - Barras con volume ratio > 1.2: {len(df[df['volume'] / df['volume'].rolling(20).mean() > 1.2])}")
    print()

    # Check signal distribution
    signal_counts = signals['signals'].value_counts()
    print("📊 DISTRIBUCIÓN DE SEÑALES:")
    for value, count in signal_counts.items():
        if value == 1:
            print(f"   - Señales LONG: {count}")
        elif value == -1:
            print(f"   - Señales SHORT: {count}")
        else:
            print(f"   - Sin señal: {count}")
    print()

    # Check recent signals
    recent_signals = signals['signals'].tail(50)
    signal_indices = recent_signals[recent_signals != 0].index
    if len(signal_indices) > 0:
        print("📊 ÚLTIMAS SEÑALES GENERADAS:")
        for idx in signal_indices[-5:]:  # Last 5 signals
            bar = df.loc[idx]
            signal = signals['signals'].loc[idx]
            direction = "LONG" if signal == 1 else "SHORT"
            print(f"   - {idx}: {direction} | ADX: {bar.get('adx', 0):.1f} | Squeeze: {bar.get('linreg_value', 0):.3f}")
    else:
        print("❌ No se encontraron señales recientes")
    print()

    # Check if squeeze conditions are met
    squeeze_off = df['sqz_off']
    momentum_positive = df['linreg_value'] > 0.1
    momentum_negative = df['linreg_value'] < -0.1
    adx_strong = df['adx'] > 20

    potential_longs = squeeze_off & momentum_positive & adx_strong
    potential_shorts = squeeze_off & momentum_negative & adx_strong

    print("🔍 ANÁLISIS DE CONDICIONES BASE:")
    print(f"   - Barras con squeeze OFF: {squeeze_off.sum()}")
    print(f"   - Barras con momentum positivo: {momentum_positive.sum()}")
    print(f"   - Barras con momentum negativo: {momentum_negative.sum()}")
    print(f"   - Barras con ADX fuerte: {adx_strong.sum()}")
    print(f"   - Potenciales LONG: {potential_longs.sum()}")
    print(f"   - Potenciales SHORT: {potential_shorts.sum()}")
    print()

    # Check TTM waves
    wave_a_bullish = df['hist_a'] > 0
    wave_b_bullish = df['hist_b'] > 0
    wave_a_bearish = df['hist_a'] < 0
    wave_b_bearish = df['hist_b'] < 0

    print("🔍 ANÁLISIS DE TTM WAVES:")
    print(f"   - Wave A bullish: {wave_a_bullish.sum()}")
    print(f"   - Wave B bullish: {wave_b_bullish.sum()}")
    print(f"   - Wave A bearish: {wave_a_bearish.sum()}")
    print(f"   - Wave B bearish: {wave_b_bearish.sum()}")
    print()

    # Check ADX trend
    adx_trend_up = (df['plus_di'] > df['minus_di']) & adx_strong
    adx_trend_down = (df['minus_di'] > df['plus_di']) & adx_strong

    print("🔍 ANÁLISIS DE ADX TREND:")
    print(f"   - ADX trend UP: {adx_trend_up.sum()}")
    print(f"   - ADX trend DOWN: {adx_trend_down.sum()}")
    print()

    # Final combined conditions
    buy_conditions = potential_longs & (adx_trend_up | (df['adx'] < strategy.key_level)) & (wave_a_bullish | wave_b_bullish)
    sell_conditions = potential_shorts & (adx_trend_down | (df['adx'] < strategy.key_level)) & (wave_a_bearish | wave_b_bearish)

    print("🔍 CONDICIONES FINALES COMBINADAS:")
    print(f"   - BUY conditions met: {buy_conditions.sum()}")
    print(f"   - SELL conditions met: {sell_conditions.sum()}")
    print()

    if buy_conditions.sum() == 0 and sell_conditions.sum() == 0:
        print("❌ PROBLEMA IDENTIFICADO:")
        print("   Las condiciones de entrada son demasiado restrictivas")
        print("   Recomendaciones:")
        print("   - Reducir umbral de momentum (actual: 0.1)")
        print("   - Reducir umbral de ADX (actual: 20)")
        print("   - Verificar cálculo de squeeze momentum")
        print("   - Revisar lógica de TTM waves")
    else:
        print("✅ Las condiciones base funcionan")
        print("   El problema puede estar en los filtros adicionales del backtest")


if __name__ == "__main__":
    diagnose_signals()