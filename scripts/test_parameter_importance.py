"""
Test Parameter Importance Analysis
Runs the parameter importance analyzer with sample data
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from scripts.parameter_importance_analyzer import run_parameter_importance_analysis


def generate_sample_data(bars: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results

    # Generate datetime index
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(minutes=5 * i) for i in range(bars)]

    # Generate realistic price data with trend and volatility
    base_price = 50000
    prices = []
    current_price = base_price

    for i in range(bars):
        # Add some trend
        trend = 0.0001 * np.sin(i / 50)  # Slow trend
        # Add volatility
        change = np.random.normal(0, 0.005) + trend
        current_price *= (1 + change)
        prices.append(current_price)

    prices = np.array(prices)

    # Generate OHLCV from prices
    high_mult = 1 + np.random.uniform(0, 0.01, bars)
    low_mult = 1 - np.random.uniform(0, 0.01, bars)
    volume_base = 1000000

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.002, bars)),
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': volume_base * (1 + np.random.uniform(0, 1, bars))
    })

    df.set_index('timestamp', inplace=True)
    return df


def create_multi_tf_data(df_5min: pd.DataFrame) -> dict:
    """Create multi-timeframe data dictionary"""
    # Resample to 15Min and 1H
    df_15min = df_5min.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    df_1h = df_5min.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return {
        '5Min': df_5min,
        '15Min': df_15min,
        '1H': df_1h
    }


def main():
    print("Parameter Importance Analysis Test")
    print("=" * 50)

    # Generate sample data
    print("1. Generating sample market data...")
    df_5min = generate_sample_data(2000)
    print(f"   Generated {len(df_5min)} bars of 5-minute data")
    print(".2f")
    print(".2f")

    # Create multi-timeframe data
    print("2. Creating multi-timeframe data...")
    df_multi_tf = create_multi_tf_data(df_5min)
    print(f"   Timeframes available: {list(df_multi_tf.keys())}")
    for tf, df in df_multi_tf.items():
        print(f"   {tf}: {len(df)} bars")

    # Initialize strategy
    print("3. Initializing strategy...")
    strategy = SqueezeMomentumADXTTMStrategy()
    print(f"   Strategy: {strategy.name}")

    # Run parameter importance analysis
    print("4. Running parameter importance analysis...")
    print("   This may take a few minutes...")

    try:
        run_parameter_importance_analysis(strategy, df_multi_tf)
        print("   Analysis completed successfully!")
    except Exception as e:
        print(f"   Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()