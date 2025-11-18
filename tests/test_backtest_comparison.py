"""
Comparison Test: Backtest Results with vs without Realistic Execution

Compares the same strategy with:
1. Simple execution (legacy)
2. Realistic execution (FASE 1: market impact + latency)

Shows the impact of realistic execution modeling on backtest metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import backtester
from core.execution.backtester_core import BacktesterCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_bars=1000):
    """Create sample BTC data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    
    # Simulate price movement
    np.random.seed(42)
    price = 50000
    prices = [price]
    
    for _ in range(n_bars - 1):
        change = np.random.normal(0, 0.002)  # 0.2% std
        price = price * (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': np.random.uniform(5, 15, n_bars)  # BTC volume
    })
    
    df.set_index('timestamp', inplace=True)
    
    # Add ATR
    df['atr'] = df['high'] - df['low']
    df['atr'] = df['atr'].rolling(14).mean()
    
    return df


class SimpleStrategy:
    """Simple moving average crossover strategy for testing"""
    
    def __init__(self, fast_period=20, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df_multi_tf):
        """Generate buy/sell signals"""
        df = df_multi_tf['5min'].copy()
        
        # Calculate moving averages
        df['ma_fast'] = df['close'].rolling(self.fast_period).mean()
        df['ma_slow'] = df['close'].rolling(self.slow_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
        df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1
        
        # Entry/exit signals
        df['entries'] = (df['signal'] == 1) & (df['signal'].shift(1) != 1)
        df['exits'] = (df['signal'] == -1) & (df['signal'].shift(1) != -1)
        
        return {
            'entries': df['entries'],
            'exits': df['exits'],
            'signals': df[['signal', 'entries', 'exits']]
        }
    
    def get_parameters(self):
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }


def run_comparison():
    """Run backtest comparison"""
    print("=" * 80)
    print("BACKTEST COMPARISON: Simple vs Realistic Execution")
    print("=" * 80)
    
    # Create sample data
    print("\n1. Creating sample data...")
    df = create_sample_data(n_bars=1000)
    df_multi_tf = {'5min': df}
    print(f"   ✓ Created {len(df)} bars of sample data")
    print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    # Initialize strategy
    print("\n2. Initializing strategy...")
    strategy = SimpleStrategy(fast_period=20, slow_period=50)
    strategy_params = strategy.get_parameters()
    print(f"   ✓ Strategy: MA Crossover ({strategy_params['fast_period']}/{strategy_params['slow_period']})")
    
    # Run with SIMPLE execution
    print("\n3. Running backtest with SIMPLE execution (legacy)...")
    backtester_simple = BacktesterCore(
        initial_capital=10000,
        commission=0.001,
        slippage_pct=0.001,
        enable_realistic_execution=False
    )
    
    results_simple = backtester_simple.run_simple_backtest(
        df_multi_tf=df_multi_tf,
        strategy_class=SimpleStrategy,
        strategy_params=strategy_params
    )
    
    if 'error' in results_simple:
        print(f"   ✗ Error: {results_simple['error']}")
        return
    
    print("   ✓ Simple backtest complete")
    
    # Run with REALISTIC execution
    print("\n4. Running backtest with REALISTIC execution (FASE 1)...")
    backtester_realistic = BacktesterCore(
        initial_capital=10000,
        commission=0.001,
        slippage_pct=0.001,
        enable_realistic_execution=True,
        latency_profile='retail_average'
    )
    
    results_realistic = backtester_realistic.run_simple_backtest(
        df_multi_tf=df_multi_tf,
        strategy_class=SimpleStrategy,
        strategy_params=strategy_params
    )
    
    if 'error' in results_realistic:
        print(f"   ✗ Error: {results_realistic['error']}")
        return
    
    print("   ✓ Realistic backtest complete")
    
    # Compare results
    print("\n5. Comparison Results")
    print("=" * 80)
    
    metrics_simple = results_simple['metrics']
    metrics_realistic = results_realistic['metrics']
    
    print(f"\n{'Metric':<20} {'Simple':<15} {'Realistic':<15} {'Change':<15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('sharpe', 'Sharpe Ratio', '.3f'),
        ('total_return', 'Total Return', '.2%'),
        ('win_rate', 'Win Rate', '.2%'),
        ('max_dd', 'Max Drawdown', '.2%'),
        ('num_trades', 'Total Trades', '.0f'),
        ('profit_factor', 'Profit Factor', '.2f'),
        ('sortino', 'Sortino Ratio', '.3f'),
        ('calmar', 'Calmar Ratio', '.3f')
    ]
    
    for metric_key, metric_name, fmt in metrics_to_compare:
        if metric_key in metrics_simple and metric_key in metrics_realistic:
            val_simple = metrics_simple[metric_key]
            val_realistic = metrics_realistic[metric_key]
            
            if val_simple != 0 and not np.isnan(val_simple) and not np.isinf(val_simple):
                change_pct = ((val_realistic - val_simple) / abs(val_simple)) * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
            
            if 'f' in fmt:
                val_simple_str = f"{val_simple:{fmt}}"
                val_realistic_str = f"{val_realistic:{fmt}}"
            elif '%' in fmt:
                val_simple_str = f"{val_simple*100:{fmt.replace('%', 'f')}}"
                val_realistic_str = f"{val_realistic*100:{fmt.replace('%', 'f')}}"
            else:
                val_simple_str = str(val_simple)
                val_realistic_str = str(val_realistic)
            
            print(f"{metric_name:<20} {val_simple_str:<15} {val_realistic_str:<15} {change_str:<15}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    sharpe_change = ((metrics_realistic['sharpe'] - metrics_simple['sharpe']) / 
                     abs(metrics_simple['sharpe'])) * 100 if metrics_simple['sharpe'] != 0 else 0
    return_change = ((metrics_realistic['total_return'] - metrics_simple['total_return']) / 
                     abs(metrics_simple['total_return'])) * 100 if metrics_simple['total_return'] != 0 else 0
    
    print(f"\nRealistic execution impact:")
    print(f"  Sharpe Ratio: {sharpe_change:+.1f}% ({metrics_simple['sharpe']:.3f} → {metrics_realistic['sharpe']:.3f})")
    print(f"  Total Return: {return_change:+.1f}% ({metrics_simple['total_return']:.2%} → {metrics_realistic['total_return']:.2%})")
    print(f"  Trades: {metrics_simple['num_trades']:.0f} → {metrics_realistic['num_trades']:.0f}")
    
    if sharpe_change < -10:
        print("\n⚠️  Significant degradation detected (>10% drop in Sharpe)")
        print("   This is EXPECTED and shows realistic execution costs are substantial.")
        print("   Without FASE 1, you would have overestimated strategy performance.")
    elif abs(sharpe_change) < 5:
        print("\n✓ Minimal impact (<5% change in Sharpe)")
        print("  Strategy may be trading infrequently or with small positions.")
    else:
        print(f"\n→ Moderate impact ({abs(sharpe_change):.1f}% change)")
        print("  Realistic costs are measurable but manageable.")
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. Realistic execution typically reduces Sharpe by 15-30%
2. Returns decrease due to market impact and latency costs
3. These costs are REAL and will occur in live trading
4. Better to discover this in backtest than live!

Next steps:
- Review trades with high impact costs
- Consider position sizing adjustments
- Test with different latency profiles
- Optimize entry/exit timing to minimize impact
""")
    
    print("=" * 80)
    
    return results_simple, results_realistic


if __name__ == "__main__":
    try:
        results_simple, results_realistic = run_comparison()
        print("\n✅ Comparison complete! Check results above.")
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
