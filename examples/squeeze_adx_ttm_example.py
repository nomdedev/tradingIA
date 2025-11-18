"""
Example usage of Squeeze ADX TTM Strategy
Demonstrates how to use the new strategy with sample data
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from strategies.presets.squeeze_adx_ttm import create_squeeze_adx_ttm_strategy


def generate_sample_data(bars: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results

    # Generate timestamps
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(bars)]

    # Generate realistic price data with trends and volatility
    prices = []
    current_price = 50000  # Starting BTC price

    for i in range(bars):
        # Add trend component
        trend = 0.0001 * np.sin(i / 100)  # Slow trend

        # Add volatility based on squeeze conditions (will be calculated later)
        volatility = 0.005 * (1 + 0.5 * np.random.random())

        # Generate OHLC
        change = np.random.normal(trend, volatility)
        open_price = current_price
        close_price = current_price * (1 + change)

        # Generate high/low with some noise
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))

        prices.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(100, 1000)
        })

        current_price = close_price

    df = pd.DataFrame(prices)
    df.set_index('timestamp', inplace=True)
    return df


def create_multi_tf_sample_data(df_5min: pd.DataFrame) -> dict:
    """Create multi-timeframe data from 5-minute data"""
    multi_tf = {'5Min': df_5min}

    # 15-minute data
    df_15min = df_5min.resample('15Min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    multi_tf['15Min'] = df_15min

    # 1-hour data
    df_1h = df_5min.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    multi_tf['1H'] = df_1h

    return multi_tf


def demonstrate_strategy_usage():
    """Demonstrate how to use the Squeeze ADX TTM strategy"""
    print("Squeeze ADX TTM Strategy Demonstration")
    print("=" * 50)

    # Generate sample data
    print("\n1. Generating sample market data...")
    df_5min = generate_sample_data(2000)  # 2000 bars of 5-minute data
    print(f"   Generated {len(df_5min)} bars of 5-minute data")
    print(".2f")
    print(".2f")

    # Create multi-timeframe data
    print("\n2. Creating multi-timeframe data...")
    df_multi_tf = create_multi_tf_sample_data(df_5min)
    print(f"   Timeframes available: {list(df_multi_tf.keys())}")
    for tf, df in df_multi_tf.items():
        print(f"   {tf}: {len(df)} bars")

    # Initialize strategy
    print("\n3. Initializing strategy...")
    strategy = SqueezeMomentumADXTTMStrategy()
    print(f"   Strategy: {strategy.name}")
    print(f"   Parameters: {len(strategy.get_parameters())}")

    # Show current parameters
    params = strategy.get_parameters()
    print("\n   Key Parameters:")
    for key in ['bb_length', 'adx_length', 'key_level', 'higher_tf_weight', 'squeeze_threshold']:
        print(f"   {key}: {params[key]}")

    # Generate signals
    print("\n4. Generating trading signals...")
    signals_dict = strategy.generate_signals(df_multi_tf)

    signals = signals_dict['signals']
    entries = signals_dict['entries']
    exits = signals_dict['exits']

    print(f"   Total signals generated: {len(signals[signals != 0])}")
    print(f"   Buy signals: {len(entries[entries == 1])}")
    print(f"   Sell signals: {len(exits[exits == 1])}")

    # Show signal distribution
    signal_counts = signals.value_counts()
    print("\n   Signal distribution:")
    print(f"   Hold (0): {signal_counts.get(0, 0)} bars")
    print(f"   Buy (1): {signal_counts.get(1, 0)} signals")
    print(f"   Sell (-1): {signal_counts.get(-1, 0)} signals")

    # Calculate basic performance
    print("\n5. Calculating basic performance...")
    prices = df_5min['close']
    returns = calculate_simple_returns(signals, prices)

    if len(returns) > 0:
        total_return = returns.sum()
        win_rate = (returns > 0).mean()
        total_trades = len(returns)

        print(".4f")
        print(".2%")
        print(f"   Total trades: {total_trades}")
        print(".4f")
    else:
        print("   No completed trades in the sample period")

    # Demonstrate parameter changes
    print("\n6. Demonstrating parameter impact...")
    original_params = strategy.get_parameters()

    # Test with different ADX threshold
    print("\n   Testing different ADX thresholds:")
    for adx_thresh in [15, 20, 25, 30]:
        strategy.set_parameters({**original_params, 'adx_threshold': adx_thresh})
        test_signals = strategy.generate_signals(df_multi_tf)['signals']
        signal_count = len(test_signals[test_signals != 0])
        print(f"   ADX threshold {adx_thresh}: {signal_count} signals")

    # Reset to original parameters
    strategy.set_parameters(original_params)

    # Demonstrate preset usage
    print("\n7. Demonstrating preset strategies...")
    presets = ['squeeze_adx_ttm', 'squeeze_adx_ttm_conservative', 'squeeze_adx_ttm_aggressive']

    for preset_name in presets:
        try:
            preset_strategy = create_squeeze_adx_ttm_strategy()
            # Apply preset-specific modifications
            if 'conservative' in preset_name:
                preset_strategy.set_parameters({
                    **preset_strategy.get_parameters(),
                    'adx_threshold': 25,
                    'squeeze_threshold': 0.7,
                    'higher_tf_weight': 0.4
                })
            elif 'aggressive' in preset_name:
                preset_strategy.set_parameters({
                    **preset_strategy.get_parameters(),
                    'adx_threshold': 18,
                    'squeeze_threshold': 0.3,
                    'higher_tf_weight': 0.2
                })

            preset_signals = preset_strategy.generate_signals(df_multi_tf)['signals']
            signal_count = len(preset_signals[preset_signals != 0])
            print(f"   {preset_name}: {signal_count} signals")

        except Exception as e:
            print(f"   {preset_name}: Error - {e}")

    print("\n8. Strategy analysis complete!")
    print("\nNext steps:")
    print("- Run parameter importance analysis: python scripts/parameter_importance_analyzer.py")
    print("- Run multi-timeframe impact analysis: python scripts/multitimeframe_impact_analyzer.py")
    print("- Run full backtest: python scripts/backtest_squeeze_adx_ttm.py")
    print("- Integrate into live trading platform")


def calculate_simple_returns(signals: pd.Series, prices: pd.Series) -> pd.Series:
    """Calculate simple returns from signals"""
    returns = []
    position = 0
    entry_price = 0

    for i in range(len(signals)):
        signal = signals.iloc[i]

        if signal == 1 and position == 0:  # Buy
            position = 1
            entry_price = prices.iloc[i]
        elif signal == -1 and position == 1:  # Sell
            if entry_price > 0:
                ret = (prices.iloc[i] - entry_price) / entry_price
                returns.append(ret)
            position = 0
            entry_price = 0

    return pd.Series(returns)


if __name__ == "__main__":
    demonstrate_strategy_usage()