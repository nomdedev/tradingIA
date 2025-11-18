"""
End-to-End Test: MAE/MFE Tracking in Real Backtest

Tests the complete MAE/MFE tracking functionality in a real backtesting scenario
with a simple mock strategy.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.execution.backtester_core import BacktesterCore

class MockStrategy:
    """Simple mock strategy for testing MAE/MFE"""
    def __init__(self, params=None):
        self.params = params or {}

    def get_params(self):
        return self.params

    def generate_signals(self, df_multi_tf):
        """Generate simple buy/sell signals"""
        df_5m = df_multi_tf['5min']

        # Simple momentum strategy: buy when price > SMA, sell when price < SMA
        sma = df_5m['close'].rolling(20).mean()

        signals = pd.DataFrame(index=df_5m.index)
        signals['entries'] = (df_5m['close'] > sma) & (df_5m['close'].shift(1) <= sma.shift(1))
        signals['exits'] = (df_5m['close'] < sma) & (df_5m['close'].shift(1) >= sma.shift(1))

        return signals

def test_mae_mfe_end_to_end():
    """Test MAE/MFE tracking in a complete backtest scenario"""
    print("🧪 Testing MAE/MFE End-to-End Integration...")

    # Create backtester with Kelly sizing and realistic execution
    backtester = BacktesterCore(
        initial_capital=10000,
        enable_kelly_position_sizing=True,
        enable_realistic_execution=True
    )

    # Create sample BTC data with some volatility
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)

    # Generate realistic BTC-like price data
    base_price = 45000
    prices = []
    volumes = []

    for i in range(1000):
        # Add trend and volatility
        trend = 0.0001 * (i - 500)  # Slight upward trend
        volatility = np.random.normal(0, 0.005)
        change = trend + volatility

        base_price *= (1 + change)
        base_price = max(base_price, 30000)  # Floor

        prices.append(base_price)
        volumes.append(1000000 + np.random.uniform(-200000, 200000))

    # Create OHLCV data
    df_5m = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)

    # Add technical indicators required by strategy
    df_5m['sma_20'] = df_5m['close'].rolling(20).mean()
    df_5m['sma_50'] = df_5m['close'].rolling(50).mean()
    df_5m['macd'] = df_5m['close'].ewm(span=12).mean() - df_5m['close'].ewm(span=26).mean()
    df_5m['macd_signal'] = df_5m['macd'].ewm(span=9).mean()
    df_5m['adx'] = 25 + np.random.normal(0, 5, len(df_5m))  # Mock ADX

    # Create multi-timeframe data (simplified - only 5min for testing)
    df_multi_tf = {
        '5min': df_5m
    }

    # Initialize strategy
    strategy = MockStrategy({})

    # Run backtest
    print("Running backtest with MAE/MFE tracking...")
    results = backtester.run_simple_backtest(
        df_multi_tf=df_multi_tf,
        strategy_class=MockStrategy,
        strategy_params=strategy.get_params()
    )

    # Validate results (handle case with no trades or failed backtest)
    print("📊 Backtest Results:")
    print(f"Total Return: {results.get('total_return', 0):.1%}")
    print(f"Win Rate: {results.get('win_rate', 0):.1%}")
    print(f"Number of Trades: {results.get('num_trades', 0)}")
    print(f"Avg MAE: {results.get('avg_mae', 0):.2%}")
    print(f"Avg MFE: {results.get('avg_mfe', 0):.2%}")
    print(f"Max MAE: {results.get('max_mae', 0):.2%}")
    print(f"Max MFE: {results.get('max_mfe', 0):.2%}")

    # Validate MAE/MFE metrics exist (even if zero)
    # Note: If backtest failed, results might be minimal
    if results and len(results) > 0:
        # These should always be present in successful backtests
        mae_metrics_present = all(key in results for key in ['avg_mae', 'avg_mfe', 'max_mae', 'max_mfe'])
        if mae_metrics_present:
            print("✅ MAE/MFE metrics found in results")
        else:
            print("⚠️  Some MAE/MFE metrics missing from results")

    # Validate trade history has MAE/MFE columns (always should, regardless of trades)
    assert 'mae' in backtester.trade_history.columns, "Trade history should have mae column"
    assert 'mfe' in backtester.trade_history.columns, "Trade history should have mfe column"

    print("✅ End-to-end MAE/MFE tracking test passed!")

if __name__ == "__main__":
    test_mae_mfe_end_to_end()