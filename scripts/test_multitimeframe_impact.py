"""
Test Multi-Timeframe Impact Analysis
Runs the multi-timeframe impact analyzer with sample data
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from scripts.multitimeframe_impact_analyzer import run_multitimeframe_analysis


def generate_sample_data(bars: int = 2000) -> pd.DataFrame:
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


def main():
    print("Multi-Timeframe Impact Analysis Test")
    print("=" * 50)

    # Generate sample data
    print("1. Generating sample market data...")
    df_5min = generate_sample_data(2000)
    print(f"   Generated {len(df_5min)} bars of 5-minute data")
    print(".2f")
    print(".2f")

    # Initialize strategy
    print("2. Initializing strategy...")
    strategy = SqueezeMomentumADXTTMStrategy()
    print(f"   Strategy: {strategy.name}")

    # Run multi-timeframe impact analysis
    print("3. Running multi-timeframe impact analysis...")
    print("   This may take a few minutes...")

    try:
        run_multitimeframe_analysis(strategy, df_5min)
        print("   Analysis completed successfully!")
    except Exception as e:
        print(f"   Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()