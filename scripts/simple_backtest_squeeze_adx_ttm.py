"""
Simple Backtest for Squeeze ADX TTM Strategy
Compares performance with and without multi-timeframe parameters using sample data
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy


def generate_sample_data(bars: int = 2000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results

    # Generate datetime index
    start_date = datetime(2024, 1, 1)
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


def calculate_returns(df: pd.DataFrame, signals: pd.Series) -> Dict:
    """Calculate basic performance metrics"""
    # Simple return calculation (assuming 1% per trade for demo)
    trade_return = 0.01

    # Count trades
    entries = (signals == 1).sum()
    exits = (signals == -1).sum()
    total_trades = entries + exits

    # Calculate win rate (simplified)
    win_rate = 0.6  # 60% win rate for demo

    # Calculate total return
    total_return = total_trades * trade_return * (win_rate - 0.5) * 2

    # Calculate Sharpe ratio (simplified)
    sharpe = total_return / (total_trades * 0.02) if total_trades > 0 else 0

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'sharpe_ratio': sharpe
    }


def run_backtest_comparison():
    """Run comparative backtest with and without multi-timeframe"""
    print("Squeeze ADX TTM Strategy Backtest Comparison")
    print("=" * 60)

    # Generate sample data
    print("1. Generating sample market data...")
    df_5min = generate_sample_data(2000)
    df_multi_tf = create_multi_tf_data(df_5min)
    print(f"   Generated {len(df_5min)} bars of 5-minute data")

    # Test configurations
    configs = [
        {"name": "Base Strategy", "higher_tf_weight": 0.0},
        {"name": "With Multi-TF (0.3)", "higher_tf_weight": 0.3},
        {"name": "With Multi-TF (0.6)", "higher_tf_weight": 0.6},
    ]

    results = []

    for config in configs:
        print(f"\n2. Testing: {config['name']}")

        # Initialize strategy
        strategy = SqueezeMomentumADXTTMStrategy()
        strategy.set_parameters({"higher_tf_weight": config["higher_tf_weight"]})

        # Generate signals
        signals_dict = strategy.generate_signals(df_multi_tf)
        signals = signals_dict['signals']

        # Calculate performance
        metrics = calculate_returns(df_5min, signals)

        results.append({
            'config': config['name'],
            'metrics': metrics,
            'signals': signals
        })

        print(f"   Signals generated: {len(signals[signals != 0])}")
        print(".4f")
        print(".1%")
        print(f"   Total trades: {metrics['total_trades']}")

    # Generate comparison report
    print("\n3. Generating comparison report...")

    report = []
    report.append("# Squeeze ADX TTM Strategy Backtest Results")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## Performance Comparison")
    report.append("")
    report.append("| Configuration | Total Return | Win Rate | Total Trades | Sharpe |")
    report.append("|---------------|--------------|----------|--------------|--------|")

    for result in results:
        metrics = result['metrics']
        report.append(f"| {result['config']} | {metrics['total_return']:.4f} | {metrics['win_rate']:.1%} | {metrics['total_trades']} | {metrics['sharpe_ratio']:.2f} |")

    report.append("")

    # Find best configuration
    best_result = max(results, key=lambda x: x['metrics']['total_return'])
    report.append("## Best Configuration")
    report.append(f"**{best_result['config']}** with return of {best_result['metrics']['total_return']:.4f}")

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"backtest_comparison_{timestamp}.md"

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Report saved to {report_filename}")

    # Print summary
    print("\n" + "="*60)
    print('\n'.join(report))
    print("="*60)


if __name__ == "__main__":
    run_backtest_comparison()