#!/usr/bin/env python3
"""
Quick Test Script for New Trading Strategies
Tests LSTM and HFT strategies with sample data.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.lstm_strategy import LSTMStrategy
from strategies.hft_strategy import HFTStrategy


def create_sample_data(n_bars: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results

    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_bars)]

    # Generate realistic price data with trend and volatility
    base_price = 50000
    prices = [base_price]

    for i in range(1, n_bars):
        # Add trend component
        trend = 0.0001 * np.sin(i / 100)  # Slow trend
        # Add random walk
        change = np.random.normal(0, 0.01)  # 1% volatility
        # Add momentum
        momentum = 0.1 * (prices[-1] - prices[-2]) if len(prices) > 1 else 0

        new_price = prices[-1] * (1 + trend + change + momentum)
        prices.append(max(new_price, 1000))  # Floor price

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.randint(100, 10000)

        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def test_lstm_strategy():
    """Test LSTM strategy"""
    print("=" * 50)
    print("TESTING LSTM STRATEGY")
    print("=" * 50)

    # Create sample data
    df = create_sample_data(500)  # Smaller dataset for quick test
    print(f"Created sample data: {len(df)} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Initialize strategy
    strategy = LSTMStrategy(name="Test_LSTM")
    print(f"Strategy initialized: {strategy.name}")

    # Test signal generation (without training)
    print("\nTesting signal generation...")
    signal = strategy.predict_signal(df)
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.3f}")
    print(f"Reason: {signal['reason']}")

    # Test parameter access
    params = strategy.get_parameters()
    print(f"\nParameters: {len(params)} parameters configured")

    print("LSTM strategy test completed ✓")


def test_hft_strategy():
    """Test HFT strategy"""
    print("\n" + "=" * 50)
    print("TESTING HFT STRATEGY")
    print("=" * 50)

    # Create sample data
    df = create_sample_data(200)  # Smaller dataset for quick test
    print(f"Created sample data: {len(df)} bars")

    # Initialize strategy
    strategy = HFTStrategy(name="Test_HFT")
    print(f"Strategy initialized: {strategy.name}")

    # Test signal generation
    print("\nTesting signal generation...")
    signal = strategy.predict_signal(df)
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.3f}")
    print(f"Position Size: {signal.get('position_size', 'N/A')}")
    print(f"Reasons: {signal.get('reasons', [])}")

    # Test parameter access
    params = strategy.get_parameters()
    print(f"\nParameters: {len(params)} parameters configured")

    # Test signal generation for backtesting
    print("\nTesting backtest signal generation...")
    signals = strategy.generate_signals(df)
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']

    print(f"Total signals generated: {len(signals)}")
    print(f"Buy signals: {len(buy_signals)}")
    print(f"Sell signals: {len(sell_signals)}")

    print("HFT strategy test completed ✓")


def test_strategy_loading():
    """Test that strategies can be loaded from registry"""
    print("\n" + "=" * 50)
    print("TESTING STRATEGY LOADING")
    print("=" * 50)

    try:
        from config.strategies_registry import load_strategy_class

        # Test loading LSTM
        print("Loading LSTM strategy from registry...")
        lstm_class = load_strategy_class("LSTM")
        if lstm_class:
            print("LSTM strategy loaded successfully ✓")
        else:
            print("LSTM strategy not found in registry ✗")

        # Test loading HFT
        print("Loading HFT strategy from registry...")
        hft_class = load_strategy_class("HFT")
        if hft_class:
            print("HFT strategy loaded successfully ✓")
        else:
            print("HFT strategy not found in registry ✗")

    except ImportError as e:
        print(f"Strategy registry import failed: {e}")
        print("Testing direct imports instead...")

        # Test direct imports
        try:
            from strategies.lstm_strategy import LSTMStrategy
            print("LSTM strategy import successful ✓")
        except ImportError:
            print("LSTM strategy import failed ✗")

        try:
            from strategies.hft_strategy import HFTStrategy
            print("HFT strategy import successful ✓")
        except ImportError:
            print("HFT strategy import failed ✗")


def main():
    """Run all tests"""
    print("TRADING STRATEGIES TEST SUITE")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    print()

    try:
        # Test individual strategies
        test_lstm_strategy()
        test_hft_strategy()

        # Test loading
        test_strategy_loading()

        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 50)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()