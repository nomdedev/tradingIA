"""
Example usage of the Optimized Backtester with advanced metrics.
Demonstrates parallel backtesting with comprehensive performance analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from core.execution.optimized_backtester import OptimizedBacktester

def momentum_strategy(data, fast_period=10, slow_period=20, threshold=0.02):
    """Momentum strategy with configurable parameters"""
    signals = pd.Series(0, index=data.index)

    # Calculate moving averages
    fast_ma = data['close'].rolling(fast_period).mean()
    slow_ma = data['close'].rolling(slow_period).mean()

    # Generate signals
    crossover_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    crossover_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    signals[crossover_up] = 1
    signals[crossover_down] = -1

    return signals

def main():
    """Demonstrate optimized backtester usage"""
    print("🚀 Optimized Backtester Demo")
    print("=" * 50)

    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    # Generate realistic price data with trend and volatility
    trend = np.linspace(0, 50, 500)
    noise = np.random.normal(0, 2, 500)
    prices = 100 + trend + noise.cumsum() * 0.1
    data = pd.DataFrame({'close': prices}, index=dates)

    print(f"📊 Generated {len(data)} days of sample data")
    print(".2f")
    print()

    # Initialize backtester
    backtester = OptimizedBacktester(initial_capital=10000, max_workers=4)

    # Define parameter sets for optimization
    parameter_sets = [
        {'fast_period': 5, 'slow_period': 15, 'threshold': 0.01},
        {'fast_period': 10, 'slow_period': 20, 'threshold': 0.02},
        {'fast_period': 15, 'slow_period': 30, 'threshold': 0.03},
        {'fast_period': 20, 'slow_period': 40, 'threshold': 0.01},
        {'fast_period': 8, 'slow_period': 25, 'threshold': 0.025},
    ]

    print(f"🔬 Running {len(parameter_sets)} parallel backtests...")
    print()

    # Run parallel backtests
    results = backtester.run_parallel_backtests(data, momentum_strategy, parameter_sets)

    # Analyze results
    print("📈 Backtest Results Summary")
    print("-" * 80)
    print("<12")
    print("-" * 80)

    best_result = None
    best_sharpe = -float('inf')

    for i, result in enumerate(results):
        params = result.parameter_set
        metrics = result.metrics

        # Find best Sharpe ratio
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_result = result

        print("<2")

    print()
    print("🏆 Best Strategy Parameters:")
    print(f"   Fast Period: {best_result.parameter_set['fast_period']}")
    print(f"   Slow Period: {best_result.parameter_set['slow_period']}")
    print(f"   Threshold: {best_result.parameter_set['threshold']}")
    print()

    print("📊 Best Strategy Metrics:")
    print("-" * 40)
    for key, value in best_result.metrics.items():
        if isinstance(value, float):
            if 'ratio' in key or 'factor' in key or 'rate' in key:
                print("<20")
            elif 'return' in key or 'drawdown' in key:
                print("<20")
            else:
                print("<20")
        else:
            print("<20")
    print()

    print("✅ Demo completed successfully!")
    print(f"   Total execution time: {sum(r.execution_time for r in results):.2f}s")
    print(f"   Average time per backtest: {np.mean([r.execution_time for r in results]):.3f}s")

if __name__ == "__main__":
    main()