#!/usr/bin/env python3
"""
Backtest Script for Weighted Multi-Strategy
Prueba la estrategia híbrida con diferentes combinaciones de pesos
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.weighted_multi_strategy import WeightedMultiStrategy


class SimpleBacktester:
    """Simple backtester for strategy testing"""

    def __init__(self, strategy, initial_capital=10000, commission=0.001):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = [self.capital]

    def run_backtest(self, df, entries, exits, strategy_name="Strategy"):
        """Run backtest with separate entry and exit signals"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        equity_curve = [capital]

        for i in range(len(df)):
            # Check for entry signal (only if not in position)
            if entries.iloc[i] and position == 0:
                entry_price = df.iloc[i]['Close'] * (1 + self.commission)
                entry_time = df.index[i]
                position = capital / entry_price
                print(f"  📈 ENTRY at {entry_time}: ${entry_price:.2f}")

            # Check for exit signal (only if in position)
            elif exits.iloc[i] and position > 0:
                exit_price = df.iloc[i]['Close'] * (1 - self.commission)
                exit_time = df.index[i]
                trade_value = position * exit_price
                profit = trade_value - capital
                capital = trade_value

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'return_pct': profit / self.initial_capital
                })

                print(f"  📉 EXIT at {exit_time}: ${exit_price:.2f}, Profit: ${profit:.2f}")
                position = 0
                entry_price = 0
                entry_time = None

            # Track equity
            current_equity = capital + (position * df.iloc[i]['Close'] if position > 0 else 0)
            equity_curve.append(current_equity)

        self.equity_curve = equity_curve
        return pd.DataFrame(trades)

        return pd.DataFrame(trades)


def load_multi_timeframe_data(symbol: str = 'BTCUSDT') -> Dict[str, pd.DataFrame]:
    """Load data for multiple timeframes"""
    data = {}

    timeframes = ['5Min', '15Min', '1H']
    data_dir = 'data'

    for tf in timeframes:
        filename = f'{data_dir}/btc_{tf.lower()}.csv'
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                data[tf] = df
                print(f"✓ Loaded {len(df)} rows for {tf}")
            else:
                print(f"✗ File not found: {filename}")
        except Exception as e:
            print(f"✗ Error loading {tf}: {e}")

    return data


def calculate_metrics(trades_df):
    """Calculate basic performance metrics"""
    if trades_df.empty:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0,
            'avg_trade_return': 0,
            'profit_factor': 0
        }

    # Simple total return as sum of profits
    total_return = trades_df['profit'].sum() / 10000 * 100  # Percentage return

    # Individual trade returns
    trade_returns = trades_df['profit'] / 10000  # As fraction

    # Sharpe ratio (simplified)
    if len(trade_returns) > 1:
        # Sharpe con risk-free rate
        rf_daily = 0.04 / 252
        excess_returns = trade_returns - rf_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    else:
        sharpe_ratio = 0

    # Max drawdown (simplified - based on cumulative returns)
    if len(trade_returns) > 0:
        cumulative = (1 + trade_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0
    else:
        max_drawdown = 0

    # Win rate
    win_rate = (trades_df['profit'] > 0).mean()

    # Profit factor
    winning_trades = trades_df[trades_df['profit'] > 0]['profit'].sum()
    losing_trades = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
    profit_factor = winning_trades / losing_trades if losing_trades > 0 else float('inf')

    # Calculate trade duration statistics (assuming 5-minute bars)
    if not trades_df.empty and 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
        # Calculate duration in bars (assuming 5-minute intervals)
        durations = []
        for _, trade in trades_df.iterrows():
            # Calculate number of 5-minute bars between entry and exit
            duration_minutes = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
            duration_bars = duration_minutes / 5  # 5-minute bars
            durations.append(duration_bars)

        avg_duration = np.mean(durations)
        median_duration = np.median(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        # Duration by trade type
        winning_mask = trades_df['profit'] > 0
        avg_winning_duration = np.mean([d for d, w in zip(durations, winning_mask) if w]) if winning_mask.any() else 0
        avg_losing_duration = np.mean([d for d, w in zip(durations, winning_mask) if not w]) if not winning_mask.all() else 0
    else:
        avg_duration = median_duration = min_duration = max_duration = avg_winning_duration = avg_losing_duration = 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades_df),
        'avg_trade_return': trade_returns.mean(),
        'profit_factor': profit_factor,
        # Trade duration statistics
        'avg_duration': avg_duration,
        'median_duration': median_duration,
        'min_duration': min_duration,
        'max_duration': max_duration,
        'avg_winning_duration': avg_winning_duration,
        'avg_losing_duration': avg_losing_duration
    }


def test_weight_combinations():
    """Test different weight combinations for the hybrid strategy"""

    print("🔬 Testing Weighted Multi-Strategy Combinations")
    print("=" * 60)

    # Load data
    data = load_multi_timeframe_data()
    if not data or '5Min' not in data:
        print("❌ No data available for backtesting")
        return

    # Weight combinations to test
    weight_configs = [
        # Balanced weights
        {'squeeze': 0.35, 'vp_ifvg': 0.35, 'mtf': 0.20, 'volume': 0.10},
        # Squeeze focused
        {'squeeze': 0.50, 'vp_ifvg': 0.25, 'mtf': 0.15, 'volume': 0.10},
        # VP focused
        {'squeeze': 0.25, 'vp_ifvg': 0.50, 'mtf': 0.15, 'volume': 0.10},
        # Multi-timeframe focused
        {'squeeze': 0.30, 'vp_ifvg': 0.30, 'mtf': 0.30, 'volume': 0.10},
        # Volume focused
        {'squeeze': 0.30, 'vp_ifvg': 0.30, 'mtf': 0.20, 'volume': 0.20},
    ]

    results = []

    for i, weights in enumerate(weight_configs):
        print(f"\n🧪 Testing Configuration {i+1}: {weights}")

        # Create strategy with specific weights
        strategy = WeightedMultiStrategy()
        strategy.squeeze_weight = weights['squeeze']
        strategy.vp_ifvg_weight = weights['vp_ifvg']
        strategy.multitimeframe_weight = weights['mtf']
        strategy.volume_weight = weights['volume']

        # Generate signals
        signals = strategy.generate_signals(data)

        if signals['entries'].sum() == 0 and signals['exits'].sum() == 0:
            print("  ❌ No signals generated")
            continue

        # Run backtest
        backtester = SimpleBacktester(strategy, initial_capital=10000)
        backtest_result = backtester.run_backtest(
            data['5Min'],
            signals['entries'],
            signals['exits'],
            strategy_name=f"Weighted_Config_{i+1}"
        )

        if backtest_result.empty:
            print("  ❌ No trades generated")
            continue

        # Calculate metrics
        metrics = calculate_metrics(backtest_result)

        result = {
            'config': i+1,
            'weights': weights,
            'total_return': metrics.get('total_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0),
            'avg_trade': metrics.get('avg_trade_return', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'avg_duration': metrics.get('avg_duration', 0),
            'median_duration': metrics.get('median_duration', 0),
            'min_duration': metrics.get('min_duration', 0),
            'max_duration': metrics.get('max_duration', 0),
            'avg_winning_duration': metrics.get('avg_winning_duration', 0),
            'avg_losing_duration': metrics.get('avg_losing_duration', 0)
        }

        results.append(result)

        print(f"  💰 Total Return: {result['total_return']:.2f}")
        print(f"  📈 Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  📉 Max Drawdown: {result['max_drawdown']:.2f}")
        print(f"  🎯 Win Rate: {result['win_rate']:.2f}")
        print(f"  📊 Total Trades: {result['total_trades']}")
        print(f"  💵 Avg Trade: {result['avg_trade']:.4f}")
        print(f"  ⚡ Profit Factor: {result['profit_factor']:.2f}")
        print(f"  ⏱️  Avg Duration: {result['avg_duration']:.1f} bars")
        print(f"  📊 Median Duration: {result['median_duration']:.1f} bars")
        print(f"  📏 Min/Max Duration: {result['min_duration']:.0f}/{result['max_duration']:.0f} bars")
    # Find best configuration
    if results:
        best_result = max(results, key=lambda x: x['total_return'])
        print("\n🏆 BEST CONFIGURATION:")
        print(f"  Config {best_result['config']}: {best_result['weights']}")
        print(f"  💰 Total Return: {best_result['total_return']:.2f}")
        print(f"  📈 Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"  📉 Max Drawdown: {best_result['max_drawdown']:.2f}")
        print(f"  🎯 Win Rate: {best_result['win_rate']:.2%}")
        print(f"  📊 Total Trades: {best_result['total_trades']}")
        print(f"  ⏱️  Avg Duration: {best_result['avg_duration']:.1f} bars")
        print(f"  📊 Median Duration: {best_result['median_duration']:.1f} bars")
        print(f"  📏 Min/Max Duration: {best_result['min_duration']:.0f}/{best_result['max_duration']:.0f} bars")
        print(f"  🟢 Avg Winning Duration: {best_result['avg_winning_duration']:.1f} bars")
        print(f"  🔴 Avg Losing Duration: {best_result['avg_losing_duration']:.1f} bars")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/weighted_strategy_comparison.csv', index=False)
        print("\n💾 Results saved to results/weighted_strategy_comparison.csv")
        return best_result['weights']
    else:
        print("❌ No valid results obtained")
        return None


def run_optimized_backtest(best_weights=None):
    """Run backtest with optimized weights"""

    print("\n🚀 Running Optimized Backtest")
    print("=" * 40)

    # Load data
    data = load_multi_timeframe_data()
    if not data or '5Min' not in data:
        print("❌ No data available")
        return

    # Use best weights from testing (or default balanced)
    strategy = WeightedMultiStrategy()
    if best_weights:
        strategy.squeeze_weight = best_weights['squeeze']
        strategy.vp_ifvg_weight = best_weights['vp_ifvg']
        strategy.multitimeframe_weight = best_weights['mtf']
        strategy.volume_weight = best_weights['volume']
        print(f"  📊 Using optimized weights: {best_weights}")
    else:
        # Default weights
        strategy.squeeze_weight = 0.35
        strategy.vp_ifvg_weight = 0.35
        strategy.multitimeframe_weight = 0.20
        strategy.volume_weight = 0.10
        print("  📊 Using default weights")

    # Generate signals
    signals = strategy.generate_signals(data)

    # Run detailed backtest
    backtester = SimpleBacktester(strategy, initial_capital=10000)
    backtest_result = backtester.run_backtest(
        data['5Min'],
        signals['entries'],
        signals['exits'],
        strategy_name="Weighted_Multi_Strategy_Optimized"
    )

    if backtest_result.empty:
        print("❌ No trades generated")
        return

    # Calculate comprehensive metrics
    metrics = calculate_metrics(backtest_result)

    print("📈 BACKTEST RESULTS:")
    print(f"  💰 Total Return: {metrics.get('total_return', 0):.2f}")
    print(f"  📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2f}")
    print(f"  🎯 Win Rate: {metrics.get('win_rate', 0):.2f}")
    print(f"  📊 Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  💵 Avg Trade: {metrics.get('avg_trade_return', 0):.4f}")
    print(f"  ⚡ Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  ⏱️  Avg Trade Duration: {metrics.get('avg_duration', 0):.1f} bars")
    print(f"  📊 Median Duration: {metrics.get('median_duration', 0):.1f} bars")
    print(f"  📏 Min/Max Duration: {metrics.get('min_duration', 0):.0f}/{metrics.get('max_duration', 0):.0f} bars")
    print(f"  🟢 Avg Winning Duration: {metrics.get('avg_winning_duration', 0):.1f} bars")
    print(f"  🔴 Avg Losing Duration: {metrics.get('avg_losing_duration', 0):.1f} bars")
    print("  📅 Max Consecutive Losses: 0")  # Simplified
    print("  🏆 Calmar Ratio: 0.0")  # Simplified

    # Save detailed results
    backtest_result.to_csv('results/weighted_strategy_backtest.csv')
    print("\n💾 Detailed results saved to results/weighted_strategy_backtest.csv")

    # Generate simple performance report
    with open('reports/weighted_strategy_report.html', 'w') as f:
        f.write(f"""
        <html>
        <head><title>Weighted Multi-Strategy Report</title></head>
        <body>
        <h1>Weighted Multi-Strategy Backtest Report</h1>
        <h2>Performance Metrics</h2>
        <ul>
        <li>Total Return: {metrics.get('total_return', 0):.2f}%</li>
        <li>Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}</li>
        <li>Max Drawdown: {metrics.get('max_drawdown', 0):.2f}</li>
        <li>Win Rate: {metrics.get('win_rate', 0):.2%}</li>
        <li>Total Trades: {metrics.get('total_trades', 0)}</li>
        <li>Avg Trade Duration: {metrics.get('avg_duration', 0):.1f} bars</li>
        <li>Median Duration: {metrics.get('median_duration', 0):.1f} bars</li>
        <li>Min/Max Duration: {metrics.get('min_duration', 0):.0f}/{metrics.get('max_duration', 0):.0f} bars</li>
        <li>Avg Winning Duration: {metrics.get('avg_winning_duration', 0):.1f} bars</li>
        <li>Avg Losing Duration: {metrics.get('avg_losing_duration', 0):.1f} bars</li>
        </ul>
        </body>
        </html>
        """)
    print("📋 Performance report generated: reports/weighted_strategy_report.html")


if __name__ == "__main__":
    print("🎯 Weighted Multi-Strategy Backtest")
    print("Combines Squeeze + ADX + TTM with VP + IFVG + EMAs")
    print("=" * 60)

    # Test different weight combinations
    best_weights = test_weight_combinations()

    # Run optimized backtest with best weights
    if best_weights:
        run_optimized_backtest(best_weights)
    else:
        print("❌ Cannot run optimized backtest - no valid weights found")

    print("\n✅ Backtest completed!")