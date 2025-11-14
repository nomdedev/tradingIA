#!/usr/bin/env python3
"""
Test script for Advanced Backtester

Tests the complete MTF BTC IFVG strategy backtesting system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtester import AdvancedBacktester

def create_sample_data(n_bars=1000):
    """Create sample BTC data for testing"""
    np.random.seed(42)

    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=5*i) for i in range(n_bars)]

    # Generate realistic BTC price data
    base_price = 45000
    returns = np.random.normal(0.0001, 0.003, n_bars)
    prices = base_price * (1 + returns).cumprod()

    # Create OHLCV data
    high_mult = 1 + np.abs(np.random.normal(0, 0.001, n_bars))
    low_mult = 1 - np.abs(np.random.normal(0, 0.001, n_bars))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })

    df.set_index('timestamp', inplace=True)

    # Add ATR for stop losses
    df['ATR'] = df['close'] * 0.02  # 2% ATR approximation

    # Add cross-TF filters (simplified)
    df['uptrend_1h'] = (df['close'] > df['close'].rolling(12).mean()).astype(bool)
    df['momentum_15m'] = (df['close'] > df['open']).astype(bool)
    df['vol_cross'] = (df['volume'] > df['volume'].rolling(20).mean()).astype(bool)

    return df

def test_basic_backtest():
    """Test basic backtest functionality"""
    print("🧪 Testing basic backtest...")

    # Create sample data
    df_5m = create_sample_data(500)

    # Create multi-TF data (simplified)
    df_15m = df_5m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    df_1h = df_5m.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    dfs = {'5m': df_5m, '15m': df_15m, '1h': df_1h}

    # Test parameters
    params = {
        'atr_multi': 0.2,
        'va_percent': 0.7,
        'vp_rows': 120,
        'vol_thresh': 1.2,
        'tp_rr': 2.2,
        'min_confidence': 0.6,
        'ema_fast_5m': 9,
        'ema_slow_5m': 21,
        'ema_fast_15m': 13,
        'ema_slow_15m': 34,
        'ema_fast_1h': 21,
        'ema_slow_1h': 55
    }

    # Initialize backtester
    backtester = AdvancedBacktester(capital=10000)

    try:
        # Run backtest
        result = backtester.run_optimized_backtest(dfs, params)

        print("✅ Basic backtest successful!")
        print(f"   Trades: {result['metrics']['total_trades']}")
        print(f"   Final Return: {result['metrics']['total_return']:.3f}")
        print(f"   Win Rate: {result['metrics']['win_rate']:.1%}")

        return True

    except Exception as e:
        print(f"❌ Basic backtest failed: {e}")
        return False

def test_monte_carlo():
    """Test Monte Carlo simulation"""
    print("\n🧪 Testing Monte Carlo simulation...")

    # Create sample data
    df_5m = create_sample_data(200)
    df_15m = df_5m.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_1h = df_5m.resample('1H').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    dfs = {'5m': df_5m, '15m': df_15m, '1h': df_1h}

    params = {
        'atr_multi': 0.2, 'va_percent': 0.7, 'vp_rows': 120,
        'vol_thresh': 1.2, 'tp_rr': 2.2, 'min_confidence': 0.6,
        'ema_fast_5m': 9, 'ema_slow_5m': 21, 'ema_fast_15m': 13,
        'ema_slow_15m': 34, 'ema_fast_1h': 21, 'ema_slow_1h': 55
    }

    backtester = AdvancedBacktester(capital=10000)

    try:
        mc_results = backtester.monte_carlo_simulation(dfs, params, n_runs=50, noise_pct=0.05)

        print("✅ Monte Carlo simulation successful!")
        print(f"   Mean Return: {mc_results['statistics']['total_return_mean']:.3f}")
        print(f"   Robustness: {mc_results['statistics']['robustness_score']:.3f}")
        return True

    except Exception as e:
        print(f"❌ Monte Carlo test failed: {e}")
        return False

def test_stress_scenarios():
    """Test stress scenario testing"""
    print("\n🧪 Testing stress scenarios...")

    # Create sample data
    df_5m = create_sample_data(200)
    df_15m = df_5m.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_1h = df_5m.resample('1H').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    dfs = {'5m': df_5m, '15m': df_15m, '1h': df_1h}

    params = {
        'atr_multi': 0.2, 'va_percent': 0.7, 'vp_rows': 120,
        'vol_thresh': 1.2, 'tp_rr': 2.2, 'min_confidence': 0.6,
        'ema_fast_5m': 9, 'ema_slow_5m': 21, 'ema_fast_15m': 13,
        'ema_slow_15m': 34, 'ema_fast_1h': 21, 'ema_slow_1h': 55
    }

    backtester = AdvancedBacktester(capital=10000)

    try:
        stress_results = backtester.stress_test_scenarios(dfs, params)

        print("✅ Stress testing successful!")
        print(f"   Survival Rate: {stress_results['summary']['survival_rate']:.1%}")
        print(f"   Stress Score: {stress_results['summary']['stress_score']:.3f}")
        return True

    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Advanced Backtester System")
    print("=" * 50)

    tests = [
        test_basic_backtest,
        test_monte_carlo,
        test_stress_scenarios
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())