"""
Strategy Framework Demo - Extensible Trading Strategy Framework
Demonstrates how to use the new strategy framework with registry and multiple strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from core.strategies import (
    StrategyRegistry, StrategyConfig,
    MomentumStrategy, MeanReversionStrategy, BreakoutStrategy
)
from core.execution.optimized_backtester import OptimizedBacktester

def create_sample_data(periods=500):
    """Create realistic sample market data"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')

    # Generate trending price data with volatility
    trend = np.linspace(0, 100, periods)
    noise = np.random.normal(0, 3, periods)
    # Add some cyclical behavior
    cycle = 10 * np.sin(np.linspace(0, 4*np.pi, periods))
    prices = 100 + trend + cycle + noise.cumsum() * 0.1

    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 1, periods),
        'high': prices + abs(np.random.normal(0, 2, periods)),
        'low': prices - abs(np.random.normal(0, 2, periods)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    }, index=dates)

    # Ensure OHLC relationships are correct
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data

def demo_strategy_registry():
    """Demonstrate strategy registry functionality"""
    print("🚀 Strategy Framework Demo")
    print("=" * 60)

    # Initialize registry
    registry = StrategyRegistry()

    # Register strategies
    registry.register_strategy(MomentumStrategy)
    registry.register_strategy(MeanReversionStrategy)
    registry.register_strategy(BreakoutStrategy)

    print("📋 Registered Strategies:")
    strategies = registry.list_strategies()
    for strategy_name in strategies:
        info = registry.get_strategy_info(strategy_name)
        print(f"   • {strategy_name}")
        print(f"     Required params: {info['required_parameters']}")
        print(f"     Description: {info['docstring'][:50]}..." if info['docstring'] else "")
    print()

def demo_strategy_creation():
    """Demonstrate strategy creation and configuration"""
    print("🔧 Strategy Creation & Configuration")
    print("-" * 40)

    registry = StrategyRegistry()
    registry.register_strategy(MomentumStrategy)
    registry.register_strategy(MeanReversionStrategy)

    # Create momentum strategy
    momentum_config = StrategyConfig(
        name="MomentumStrategy",
        description="Aggressive momentum strategy for trending markets",
        parameters={
            'fast_period': 8,
            'slow_period': 21,
            'momentum_threshold': 0.025
        },
        risk_management={
            'max_drawdown_threshold': 0.15,
            'volatility_threshold': 0.03
        },
        filters=['max_drawdown_filter', 'trend_filter']
    )

    momentum_strategy = registry.create_strategy(momentum_config)
    print(f"✅ Created: {momentum_strategy}")
    print(f"   Parameters: {momentum_strategy.config.parameters}")
    print(f"   Risk filters: {momentum_strategy.config.filters}")
    print()

    # Create mean reversion strategy
    mr_config = StrategyConfig(
        name="MeanReversionStrategy",
        description="Conservative mean reversion for range-bound markets",
        parameters={
            'lookback_period': 25,
            'entry_threshold': 1.8,
            'exit_threshold': 0.3
        },
        risk_management={
            'max_drawdown_threshold': 0.10
        },
        filters=['volatility_filter']
    )

    mr_strategy = registry.create_strategy(mr_config)
    print(f"✅ Created: {mr_strategy}")
    print(f"   Parameters: {mr_strategy.config.parameters}")
    print()

def demo_strategy_execution():
    """Demonstrate strategy execution with backtesting"""
    print("📊 Strategy Execution & Backtesting")
    print("-" * 40)

    # Create sample data
    data = create_sample_data(300)
    print(f"📈 Generated {len(data)} days of sample data")
    print(".2f")
    print()

    # Setup registry and strategies
    registry = StrategyRegistry()
    registry.register_strategy(MomentumStrategy)
    registry.register_strategy(MeanReversionStrategy)

    # Strategy configurations for optimization
    strategy_configs = [
        {
            'name': 'MomentumStrategy',
            'description': 'Conservative momentum',
            'parameters': {'fast_period': 10, 'slow_period': 20, 'momentum_threshold': 0.02},
            'risk_management': {'max_drawdown_threshold': 0.12},
            'filters': ['trend_filter']
        },
        {
            'name': 'MomentumStrategy',
            'description': 'Aggressive momentum',
            'parameters': {'fast_period': 5, 'slow_period': 15, 'momentum_threshold': 0.035},
            'risk_management': {'max_drawdown_threshold': 0.18},
            'filters': ['max_drawdown_filter']
        },
        {
            'name': 'MeanReversionStrategy',
            'description': 'Conservative mean reversion',
            'parameters': {'lookback_period': 20, 'entry_threshold': 2.0, 'exit_threshold': 0.5},
            'risk_management': {'max_drawdown_threshold': 0.08},
            'filters': ['volatility_filter']
        }
    ]

    # Create backtester
    backtester = OptimizedBacktester(initial_capital=10000, max_workers=3)

    print(f"🏃 Running {len(strategy_configs)} strategy backtests in parallel...")

    # Convert configs to StrategyConfig objects and create strategies
    strategies = []
    for config_dict in strategy_configs:
        config = StrategyConfig(**config_dict)
        strategy = registry.create_strategy(config)
        strategies.append(strategy)

    # Create parameter sets for backtesting (each strategy as a separate run)
    parameter_sets = []
    for i, strategy in enumerate(strategies):
        # For demo, we'll just run each strategy once with its configured parameters
        # In practice, you'd optimize parameters here
        param_set = {f'config_{i}': True}  # Dummy parameter to satisfy the interface
        parameter_sets.append(param_set)

    # Custom backtest function that uses our strategies
    def run_strategy_backtest(params):
        strategy_idx = int(list(params.keys())[0].split('_')[1])
        strategy = strategies[strategy_idx]

        # Calculate indicators and generate signals
        data_with_indicators = strategy.calculate_indicators(data)
        raw_signals = strategy.generate_signals(data_with_indicators)
        signals = strategy.apply_filters(data_with_indicators, raw_signals)

        # Run backtest
        result = backtester._run_backtest(data, signals)
        result.parameter_set = params
        return result

    # Run parallel backtests
    results = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_strategy_backtest, params) for params in parameter_sets]
        for future in futures:
            results.append(future.result())

    # Analyze results
    print("\n📊 Backtest Results Summary")
    print("-" * 70)
    print("<25")
    print("-" * 70)

    best_result = None
    best_sharpe = -float('inf')

    for i, result in enumerate(results):
        config = strategy_configs[i]
        metrics = result.metrics

        strategy_name = config['name']
        description = config['description']

        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_result = (i, result)

        print("<25")

    print()
    print("🏆 Best Performing Strategy:")
    best_idx, best_result = best_result
    best_config = strategy_configs[best_idx]
    print(f"   Strategy: {best_config['name']} ({best_config['description']})")
    print(f"   Parameters: {best_config['parameters']}")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"   Win Rate: {best_result.metrics['win_rate']:.1%}")
    print()

    print("✅ Strategy Framework Demo Completed!")
    print(f"   Strategies tested: {len(strategies)}")
    print(f"   Total backtest time: {sum(r.execution_time for r in results):.2f}s")
    print(f"   Average time per strategy: {np.mean([r.execution_time for r in results]):.3f}s")

def main():
    """Run complete strategy framework demonstration"""
    try:
        demo_strategy_registry()
        demo_strategy_creation()
        demo_strategy_execution()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()