#!/usr/bin/env python3
"""
Configurable Backtesting Script
Allows users to modify strategy parameters and run backtests with results and charts
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Try to import platform components
try:
    from core.backend_core import DataManager, StrategyEngine
    from core.execution.backtester_core import BacktesterCore
    from strategies import load_strategy
    PLATFORM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Platform components not available: {e}")
    PLATFORM_AVAILABLE = False

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class ConfigurableBacktester:
    """Configurable backtesting system with parameter modification and visualization"""

    def __init__(self):
        self.backtester = BacktesterCore()
        self.data_manager = DataManager()
        self.results_dir = project_root / 'results' / 'configurable_backtests'
        self.results_dir.mkdir(exist_ok=True)

    def load_data(self, symbol='BTCUSD', timeframe='5Min', start_date='2023-01-01', end_date='2024-12-31'):
        """Load data for backtesting"""
        print(f"📥 Loading {symbol} {timeframe} data from {start_date} to {end_date}...")

        # Load data using data manager
        data = self.data_manager.load_data(symbol, timeframe, start_date, end_date)

        if data is None or data.empty:
            print("❌ No data loaded. Please check data availability.")
            return None

        print(f"✅ Loaded {len(data)} bars")
        return data

    def run_backtest(self, strategy_name, strategy_params, data, output_name=None):
        """Run backtest with given parameters"""
        print(f"🚀 Running backtest for {strategy_name}...")

        # Load strategy
        strategy_class = load_strategy(strategy_name)
        if not strategy_class:
            print(f"❌ Strategy {strategy_name} not found")
            return None

        # Create strategy instance and set parameters
        strategy = strategy_class()
        if strategy_params:
            strategy.set_parameters(**strategy_params)
            print(f"✅ Applied parameters: {strategy_params}")

        # Run backtest
        result = self.backtester.run_simple_backtest(data, strategy_class, strategy_params)

        if not result:
            print("❌ Backtest failed")
            return None

        print("✅ Backtest completed")

        # Generate timestamp for output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_name:
            output_name = f"{strategy_name}_{timestamp}"

        # Save results
        self.save_results(result, output_name, strategy_name, strategy_params)

        return result

    def save_results(self, result, output_name, strategy_name, strategy_params):
        """Save backtest results with charts"""
        output_dir = self.results_dir / output_name
        output_dir.mkdir(exist_ok=True)

        # Save metrics as JSON
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(result.get('metrics', {}), f, indent=2, default=str)

        # Save parameters
        params_file = output_dir / 'parameters.json'
        with open(params_file, 'w') as f:
            json.dump({
                'strategy': strategy_name,
                'parameters': strategy_params,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # Generate and save charts
        self.generate_charts(result, output_dir)

        # Generate summary report
        self.generate_report(result, output_dir, strategy_name, strategy_params)

        print(f"💾 Results saved to: {output_dir}")

    def generate_charts(self, result, output_dir):
        """Generate performance charts"""
        if 'trades' not in result:
            return

        trades_df = pd.DataFrame(result['trades'])

        if trades_df.empty:
            return

        # 1. Equity curve
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results Analysis', fontsize=16)

        # Equity curve
        if 'cumulative_return' in trades_df.columns:
            axes[0,0].plot(trades_df['exit_time'], trades_df['cumulative_return'] * 100)
            axes[0,0].set_title('Equity Curve (%)')
            axes[0,0].set_ylabel('Return (%)')
            axes[0,0].grid(True, alpha=0.3)

        # 2. Trade P&L distribution
        if 'pnl' in trades_df.columns:
            axes[0,1].hist(trades_df['pnl'], bins=50, alpha=0.7, edgecolor='black')
            axes[0,1].set_title('Trade P&L Distribution')
            axes[0,1].set_xlabel('P&L ($)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)

        # 3. Win/Loss ratio over time
        if 'pnl' in trades_df.columns:
            trades_df['is_win'] = trades_df['pnl'] > 0
            rolling_win_rate = trades_df['is_win'].rolling(window=20, min_periods=1).mean()
            axes[1,0].plot(trades_df['exit_time'], rolling_win_rate * 100)
            axes[1,0].set_title('Rolling Win Rate (20 trades)')
            axes[1,0].set_ylabel('Win Rate (%)')
            axes[1,0].set_ylim(0, 100)
            axes[1,0].grid(True, alpha=0.3)

        # 4. Drawdown
        if 'drawdown' in trades_df.columns:
            axes[1,1].fill_between(trades_df['exit_time'], trades_df['drawdown'] * 100, 0, alpha=0.7, color='red')
            axes[1,1].set_title('Drawdown (%)')
            axes[1,1].set_ylabel('Drawdown (%)')
            axes[1,1].set_xlabel('Time')
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        chart_file = output_dir / 'performance_charts.png'
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Charts saved: {chart_file}")

    def generate_report(self, result, output_dir, strategy_name, strategy_params):
        """Generate summary report"""
        metrics = result.get('metrics', {})

        report = f"""
# Backtest Report: {strategy_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Strategy Parameters
```json
{json.dumps(strategy_params, indent=2)}
```

## Performance Metrics
- **Total Return:** {metrics.get('total_return', 0):.2%}
- **Win Rate:** {metrics.get('win_rate', 0):.1%}
- **Profit Factor:** {metrics.get('profit_factor', 0):.3f}
- **Expectancy:** ${metrics.get('expectancy', 0):.2f}
- **Total Trades:** {metrics.get('total_trades', 0)}
- **Max Drawdown:** {metrics.get('max_drawdown', 0):.2%}
- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.3f}

## Trade Statistics
- **Average Win:** ${metrics.get('avg_win', 0):.2f}
- **Average Loss:** ${metrics.get('avg_loss', 0):.2f}
- **Largest Win:** ${metrics.get('max_win', 0):.2f}
- **Largest Loss:** ${metrics.get('max_loss', 0):.2f}
- **Max Consecutive Wins:** {metrics.get('max_consecutive_wins', 0)}
- **Max Consecutive Losses:** {metrics.get('max_consecutive_losses', 0)}

## Analysis
{self.analyze_results(metrics)}
"""

        report_file = output_dir / 'report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Report saved: {report_file}")

    def analyze_results(self, metrics):
        """Provide analysis of results"""
        analysis = []

        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        expectancy = metrics.get('expectancy', 0)
        total_trades = metrics.get('total_trades', 0)

        if total_trades < 30:
            analysis.append("⚠️ **Warning:** Low number of trades. Results may not be statistically significant.")

        if win_rate > 0.55:
            analysis.append("✅ **Good Win Rate:** Above 55% indicates strong edge.")
        elif win_rate < 0.45:
            analysis.append("❌ **Poor Win Rate:** Below 45% suggests strategy needs improvement.")

        if profit_factor > 1.2:
            analysis.append("✅ **Excellent Profit Factor:** Strong risk-reward profile.")
        elif profit_factor < 1.0:
            analysis.append("❌ **Poor Profit Factor:** Strategy is losing money overall.")

        if expectancy > 0:
            analysis.append("✅ **Positive Expectancy:** Strategy is profitable per trade.")
        else:
            analysis.append("❌ **Negative Expectancy:** Strategy loses money per trade.")

        return "\n".join(analysis) if analysis else "No significant analysis available."


def main():
    parser = argparse.ArgumentParser(description='Configurable Backtesting System')
    parser.add_argument('--strategy', help='Strategy name (e.g., vp_ifvg_ema_ema15m50)')
    parser.add_argument('--config', help='Configuration name from backtest_configs.json')
    parser.add_argument('--params', help='JSON string of strategy parameters')
    parser.add_argument('--symbol', help='Trading symbol')
    parser.add_argument('--timeframe', help='Timeframe')
    parser.add_argument('--start-date', help='Start date')
    parser.add_argument('--end-date', help='End date')
    parser.add_argument('--output', help='Output directory name')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations')

    args = parser.parse_args()

    # Load configurations
    config_file = project_root / 'config' / 'backtest_configs.json'
    configs = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            configs = json.load(f)

    if args.list_configs:
        print("📋 Available Configurations:")
        for name, config in configs.get('backtest_configs', {}).items():
            print(f"  - {name}: {config.get('description', 'No description')}")
        return

    # Determine configuration
    if args.config and args.config in configs.get('backtest_configs', {}):
        config_data = configs['backtest_configs'][args.config]
        strategy_name = config_data['strategy']
        strategy_params = config_data['parameters']
        symbol = config_data['data']['symbol']
        timeframe = config_data['data']['timeframe']
        start_date = config_data['data']['start_date']
        end_date = config_data['data']['end_date']
        print(f"✅ Loaded config: {args.config}")
    else:
        if not args.strategy:
            print("❌ Must specify --strategy or --config")
            return
        strategy_name = args.strategy
        strategy_params = {}
        symbol = args.symbol or 'BTCUSD'
        timeframe = args.timeframe or '5Min'
        start_date = args.start_date or '2023-01-01'
        end_date = args.end_date or '2024-12-31'

    # Override with command line params if provided
    if args.params:
        try:
            strategy_params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing parameters: {e}")
            return

    # Override data settings if provided
    if args.symbol:
        symbol = args.symbol
    if args.timeframe:
        timeframe = args.timeframe
    if args.start_date:
        start_date = args.start_date
    if args.end_date:
        end_date = args.end_date

    # Initialize backtester
    backtester = ConfigurableBacktester()

    # Load data
    data = backtester.load_data(symbol, timeframe, start_date, end_date)
    if data is None:
        return

    # Run backtest
    result = backtester.run_backtest(strategy_name, strategy_params, data, args.output)

    if result:
        print("🎉 Backtest completed successfully!")
        output_dir = args.output or f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"📊 Results saved in: {backtester.results_dir / output_dir}")
    else:
        print("❌ Backtest failed")


if __name__ == '__main__':
    main()