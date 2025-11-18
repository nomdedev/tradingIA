#!/usr/bin/env python3
"""
Test rápido para verificar generación de señales
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy

def test_signals():
    """Test signal generation"""
    print("🧪 TEST DE GENERACIÓN DE SEÑALES")
    print("=" * 50)

    # Load sample data
    df = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    df = df.tail(1000)  # Small sample for testing

    print(f"📊 Datos cargados: {len(df)} barras")

    # Test basic strategy
    strategy = SqueezeMomentumADXTTMStrategy()
    strategy.use_multitimeframe = False
    strategy.use_poc_filter = False
    strategy.use_adx_slope = False
    strategy.use_adx_divergence = False

    df_multi = {'5Min': df}

    print("🔄 Generando señales...")
    signals = strategy.generate_signals(df_multi)

    print(f"📈 Señales generadas:")
    print(f"   - Entries: {signals['entries'].sum()}")
    print(f"   - Exits: {signals['exits'].sum()}")
    print(f"   - Signals: {signals['signals'].sum()}")
    print(f"   - Trade scores: {signals['trade_scores'].sum()}")

    # Check if signals exist
    if signals['entries'].sum() > 0:
        print("✅ Señales encontradas - estrategia funcionando")
        # Show first few signals
        signal_indices = signals['entries'][signals['entries'] == 1].index[:5]
        print(f"📍 Primeras 5 señales en índices: {list(signal_indices)}")
    else:
        print("❌ No se generaron señales")
        # Debug: check strategy components
        print("🔍 Depurando componentes...")

        # Calculate indicators manually
        df_test = df.copy()
        df_test = strategy._calculate_squeeze_momentum(df_test)
        df_test = strategy._calculate_adx(df_test)
        df_test = strategy._calculate_ttm_waves(df_test)

        print(f"   - Squeeze momentum calculado: {df_test['linreg_value'].notna().sum()} valores")
        print(f"   - ADX calculado: {df_test['adx'].notna().sum()} valores")
        print(f"   - TTM waves calculado: {df_test['hist_a'].notna().sum()} valores")

        # Generate base signals
        base_signals = strategy._generate_base_signals(df_test)
        print(f"   - Señales base generadas: {base_signals.sum()}")

if __name__ == "__main__":
    test_signals()