#!/usr/bin/env python3
"""
Backtest ultra-simple para verificar funcionamiento básico
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy

def simple_backtest():
    """Simple backtest without advanced filters"""
    print("🧪 BACKTEST ULTRA-SIMPLE")
    print("=" * 50)

    # Load data
    df = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    df = df.tail(1000)  # Small sample

    print(f"📊 Datos: {len(df)} barras")

    # Basic strategy
    strategy = SqueezeMomentumADXTTMStrategy()
    strategy.use_multitimeframe = False
    strategy.use_poc_filter = False
    strategy.use_adx_slope = False
    strategy.use_adx_divergence = False

    df_multi = {'5Min': df}

    # Generate signals
    signals = strategy.generate_signals(df_multi)

    print(f"📈 Señales encontradas:")
    print(f"   - Total entries: {signals['entries'].sum()}")
    print(f"   - Total exits: {signals['exits'].sum()}")

    # Simple backtest logic
    capital = 10000
    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = 0

    for idx in range(len(df)):
        if not in_trade and signals['entries'].iloc[idx] == 1:
            # Enter trade
            entry_price = df.iloc[idx]['close']
            entry_idx = idx
            in_trade = True
            direction = signals['signals'].iloc[idx]
            print(f"📈 ENTRADA en {df.index[idx]}: precio={entry_price:.2f}, dirección={direction}")

        elif in_trade and signals['exits'].iloc[idx] == 1:
            # Exit trade
            exit_price = df.iloc[idx]['close']
            pnl = (exit_price - entry_price) * direction * (capital * 0.02 / entry_price)  # 2% risk
            pnl -= abs(pnl) * 0.001  # Commission
            capital += pnl

            trade = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'direction': direction,
                'duration': idx - entry_idx
            }
            trades.append(trade)

            print(f"📉 SALIDA en {df.index[idx]}: precio={exit_price:.2f}, pnl={pnl:.2f}, capital={capital:.2f}")
            in_trade = False

    # Results
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100

        print("\n📊 RESULTADOS:")
        print(f"   - Trades: {len(trades)}")
        print(f"   - Win Rate: {win_rate:.1f}%")
        print(f"   - P&L Total: ${total_pnl:.2f}")
        print(f"   - Capital Final: ${capital:.2f}")
        print(f"   - Retorno: {((capital - 10000) / 10000 * 100):.2f}%")
    else:
        print("❌ No se ejecutaron trades")

if __name__ == "__main__":
    simple_backtest()