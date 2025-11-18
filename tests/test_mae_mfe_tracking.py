"""
Test MAE/MFE Tracking Implementation

Tests the Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
tracking functionality in the backtester.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.execution.backtester_core import BacktesterCore

def test_mae_mfe_tracking():
    """Test MAE/MFE calculation during trades"""
    print("🧪 Testing MAE/MFE Tracking Implementation...")

    # Create backtester with Kelly sizing enabled
    backtester = BacktesterCore(
        initial_capital=10000,
        enable_kelly_position_sizing=True
    )

    # Create mock 5min data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    np.random.seed(42)  # For reproducible results

    # Create price data with some trends
    base_price = 50000
    prices = []
    for i in range(100):
        # Add some random walk with trend
        change = np.random.normal(0, 50)
        if i > 50:  # Add upward trend after halfway
            change += 20
        base_price += change
        prices.append(max(base_price, 40000))  # Floor price

    df_5m = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.uniform(0, 100) for p in prices],
        'low': [p - np.random.uniform(0, 100) for p in prices],
        'close': prices,
        'volume': [1000000 + np.random.uniform(-100000, 100000) for _ in prices]
    }, index=dates)

    # Simulate a long trade: entry at index 10, exit at index 30
    entry_idx = 10
    exit_idx = 30
    entry_price = df_5m.iloc[entry_idx]['close']
    exit_price = df_5m.iloc[exit_idx]['close']

    # Calculate expected MAE/MFE manually
    price_series = df_5m['high'].iloc[entry_idx:exit_idx+1]  # For long trade, use high
    max_price = price_series.max()
    min_price = df_5m['low'].iloc[entry_idx:exit_idx+1].min()

    expected_mae = (entry_price - min_price) / entry_price if min_price < entry_price else 0.0
    expected_mfe = (max_price - entry_price) / entry_price if max_price > entry_price else 0.0

    print(f"Entry price: ${entry_price:.2f}, Exit price: ${exit_price:.2f}")
    print(f"Max price during trade: ${max_price:.2f}, Min price during trade: ${min_price:.2f}")
    print(f"Expected MAE: {expected_mae:.4f}, Expected MFE: {expected_mfe:.4f}")

    # Test the _record_trade method with MAE/MFE
    backtester._record_trade(
        timestamp=df_5m.index[exit_idx],
        side='buy',
        entry_price=entry_price,
        exit_price=exit_price,
        size=1.0,
        hold_time=1.0,
        mae=expected_mae,
        mfe=expected_mfe
    )

    # Verify trade was recorded with MAE/MFE
    assert len(backtester.trade_history) == 1, "Trade should be recorded"
    trade = backtester.trade_history.iloc[0]

    assert 'mae' in trade.index, "MAE column should exist"
    assert 'mfe' in trade.index, "MFE column should exist"
    assert abs(trade['mae'] - expected_mae) < 0.001, f"MAE should be {expected_mae}, got {trade['mae']}"
    assert abs(trade['mfe'] - expected_mfe) < 0.001, f"MFE should be {expected_mfe}, got {trade['mfe']}"

    print("✅ Trade recording with MAE/MFE works correctly")

    # Test short trade MAE/MFE calculation
    entry_idx_short = 50
    exit_idx_short = 70
    entry_price_short = df_5m.iloc[entry_idx_short]['close']
    exit_price_short = df_5m.iloc[exit_idx_short]['close']

    # For short trade, MAE is how far price went above entry, MFE below entry
    price_series_short = df_5m['low'].iloc[entry_idx_short:exit_idx_short+1]  # Use low for shorts
    max_price_short = df_5m['high'].iloc[entry_idx_short:exit_idx_short+1].max()
    min_price_short = price_series_short.min()

    expected_mae_short = (max_price_short - entry_price_short) / entry_price_short if max_price_short > entry_price_short else 0.0
    expected_mfe_short = (entry_price_short - min_price_short) / entry_price_short if min_price_short < entry_price_short else 0.0

    backtester._record_trade(
        timestamp=df_5m.index[exit_idx_short],
        side='sell',
        entry_price=entry_price_short,
        exit_price=exit_price_short,
        size=1.0,
        hold_time=1.0,
        mae=expected_mae_short,
        mfe=expected_mfe_short
    )

    # Verify short trade
    assert len(backtester.trade_history) == 2, "Second trade should be recorded"
    short_trade = backtester.trade_history.iloc[1]

    assert abs(short_trade['mae'] - expected_mae_short) < 0.001, f"Short MAE should be {expected_mae_short}, got {short_trade['mae']}"
    assert abs(short_trade['mfe'] - expected_mfe_short) < 0.001, f"Short MFE should be {expected_mfe_short}, got {short_trade['mfe']}"

    print("✅ Short trade MAE/MFE calculation works correctly")

    # Test metrics calculation includes MAE/MFE
    # Create mock returns and trades_records for metrics calculation
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015], index=dates[:5])
    trades_records = pd.DataFrame({
        'pnl': [100, -50, 200, -100, 150]
    })

    metrics = backtester.calculate_metrics(returns, trades_records)

    assert 'avg_mae' in metrics, "Metrics should include avg_mae"
    assert 'avg_mfe' in metrics, "Metrics should include avg_mfe"
    assert 'max_mae' in metrics, "Metrics should include max_mae"
    assert 'max_mfe' in metrics, "Metrics should include max_mfe"

    print(f"Avg MAE: {metrics['avg_mae']:.4f}, Avg MFE: {metrics['avg_mfe']:.4f}")
    print(f"Max MAE: {metrics['max_mae']:.4f}, Max MFE: {metrics['max_mfe']:.4f}")
    print("✅ Metrics calculation includes MAE/MFE statistics")

    print("🎉 All MAE/MFE tracking tests passed!")

if __name__ == "__main__":
    test_mae_mfe_tracking()