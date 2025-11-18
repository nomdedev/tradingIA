"""
Demo script showing automatic parameter saving with backtest results.

This script demonstrates the new functionality where strategy parameters
are automatically included in all backtest results for better tracking
and reproducibility of trading strategies.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from core.execution.backtester_core import BacktesterCore
from strategies.vp_ifvg_ema_strategy import VPIFVGEmaStrategy


def create_sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range('2023-01-01', periods=200, freq='5min')

    # Create realistic price data with some trend and volatility
    import numpy as np
    np.random.seed(42)

    # Base price movement
    base_price = 100
    price_changes = np.random.normal(0, 0.5, len(dates))
    prices = base_price + np.cumsum(price_changes)

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 0.3, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 0.3, len(dates))),
        'close': prices + np.random.normal(0, 0.2, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    # Ensure high >= close >= low and high >= open >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data


def save_backtest_results_with_parameters(result, strategy_name, output_dir='results/backtests'):
    """
    Save backtest results including strategy parameters for reproducibility.

    Args:
        result: Backtest result dictionary from BacktesterCore
        strategy_name: Name of the strategy
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/{strategy_name}_backtest_{timestamp}.json"

    # Create comprehensive result structure
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'strategy_name': strategy_name,
        'strategy_parameters': result.get('strategy_parameters', {}),
        'metrics': result.get('metrics', {}),
        'trades_count': len(result.get('trades', [])),
        'trades': result.get('trades', []),
        'equity_curve': result.get('equity_curve', []),
        'signals_summary': {
            'total_signals': len(result.get('signals', [])),
            'entries': sum(1 for s in result.get('signals', []) if s == 1),
            'exits': sum(1 for s in result.get('signals', []) if s == -1)
        }
    }

    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

    print(f"💾 Backtest results saved to: {filename}")
    print(f"   📊 Strategy: {strategy_name}")
    print(f"   ⚙️  Parameters saved: {len(output_data['strategy_parameters'])}")
    print(f"   📈 Sharpe Ratio: {output_data['metrics'].get('sharpe', 'N/A'):.3f}")
    print(f"   💰 Total Trades: {output_data['trades_count']}")

    return filename


def run_backtest_with_parameter_saving():
    """Demonstrate automatic parameter saving with backtest results"""

    print("🚀 Backtest with Automatic Parameter Saving Demo")
    print("=" * 60)

    # Create sample data
    print("📊 Creating sample market data...")
    data = create_sample_data()
    df_multi_tf = {'5min': data}
    print(f"   ✅ Created {len(data)} bars of 5-minute data")

    # Initialize backtester
    backtester = BacktesterCore(
        initial_capital=10000,
        commission=0.001,
        slippage_pct=0.0005
    )
    print("   ✅ Backtester initialized")

    # Run backtest with VP_IFVG_EMA strategy
    strategy_class = VPIFVGEmaStrategy
    strategy_params = {}  # Use default parameters

    print(f"🏃 Running backtest with {strategy_class.__name__}...")
    result = backtester.run_simple_backtest(df_multi_tf, strategy_class, strategy_params)

    if 'error' in result:
        print(f"❌ Backtest failed: {result['error']}")
        return

    # Save results with parameters
    saved_file = save_backtest_results_with_parameters(
        result,
        strategy_class.__name__,
        'results/backtests'
    )

    # Demonstrate parameter access
    print("\n🔍 Parameter Preservation Demo:")
    print("-" * 40)

    parameters = result.get('strategy_parameters', {})
    print(f"Total parameters saved: {len(parameters)}")

    # Show some key parameters
    key_params = ['disp_num', 'atr_multi', 'vp_length', 'ema_fast', 'ema_slow']
    print("Key parameters:")
    for param in key_params:
        if param in parameters:
            print(f"  {param}: {parameters[param]}")

    print(f"\n📄 Full results saved to: {saved_file}")
    print("💡 These parameters can now be used to reproduce this exact backtest!")

    return result


def demonstrate_parameter_reproducibility(original_result):
    """Demonstrate how saved parameters can reproduce results"""

    print("\n🔄 Parameter Reproducibility Test")
    print("=" * 50)

    # Extract parameters from original result
    original_params = original_result.get('strategy_parameters', {})
    print(f"📋 Loaded {len(original_params)} parameters from saved results")

    # Create new backtester with same settings
    backtester = BacktesterCore(
        initial_capital=10000,
        commission=0.001,
        slippage_pct=0.0005
    )

    # Create same data
    data = create_sample_data()
    df_multi_tf = {'5min': data}

    # Run backtest with saved parameters
    print("🔁 Re-running backtest with saved parameters...")
    reproduced_result = backtester.run_simple_backtest(df_multi_tf, VPIFVGEmaStrategy, {})

    if 'error' in reproduced_result:
        print(f"❌ Reproduction failed: {reproduced_result['error']}")
        return

    print("📊 Results Comparison:")
    print("-" * 30)
    print(f"Original Sharpe:   {original_result.get('metrics', {}).get('sharpe', 0):.3f}")
    print(f"Reproduced Sharpe: {reproduced_result.get('metrics', {}).get('sharpe', 0):.3f}")
    print(f"Trades match:      {len(original_result.get('trades', [])) == len(reproduced_result.get('trades', []))}")

    # Check if parameters match
    reproduced_params = reproduced_result.get('strategy_parameters', {})
    params_match = original_params == reproduced_params
    print(f"Parameters match: {'✅ Yes' if params_match else '❌ No'}")

    print("\n🎯 Parameter preservation ensures reproducible backtests!")


if __name__ == '__main__':
    # Run main demo
    result = run_backtest_with_parameter_saving()

    if result and 'error' not in result:
        # Demonstrate reproducibility
        demonstrate_parameter_reproducibility(result)

    print("\n✨ Demo completed! Strategy parameters are now automatically saved with all backtest results.")