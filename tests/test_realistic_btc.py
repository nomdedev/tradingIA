"""
Test realistic execution with actual BTC data

Uses real BTC-USD data to validate realistic execution modeling
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from core.execution.backtester_core import BacktesterCore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMAStrategy:
    """Moving average crossover strategy"""
    
    def __init__(self, fast_period=20, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df_multi_tf):
        df = df_multi_tf['5min'].copy()
        
        df['ma_fast'] = df['close'].rolling(self.fast_period).mean()
        df['ma_slow'] = df['close'].rolling(self.slow_period).mean()
        
        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
        df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1
        
        df['entries'] = (df['signal'] == 1) & (df['signal'].shift(1) != 1)
        df['exits'] = (df['signal'] == -1) & (df['signal'].shift(1) != -1)
        
        return {
            'entries': df['entries'],
            'exits': df['exits'],
            'signals': df[['signal', 'entries', 'exits']]
        }
    
    def get_parameters(self):
        return {'fast_period': self.fast_period, 'slow_period': self.slow_period}


def load_btc_data():
    """Load actual BTC data"""
    data_path = 'd:/martin/Proyectos/tradingIA/data/btc_5Min.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    # Rename columns to lowercase
    df.columns = [col.lower() if col != 'Unnamed: 0' else 'timestamp' for col in df.columns]
    
    # Set index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Calculate ATR if not present
    if 'atr' not in df.columns:
        df['atr'] = df['high'] - df['low']
        df['atr'] = df['atr'].rolling(14).mean()
    
    # Use last 2000 bars for reasonable test duration
    df = df.tail(2000)
    
    logger.info(f"Loaded {len(df)} bars of BTC data")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    return df


def run_test():
    print("=" * 80)
    print("REALISTIC EXECUTION TEST - BTC-USD")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading BTC data...")
    df = load_btc_data()
    if df is None:
        print("   ✗ Failed to load data")
        return
    print("   ✓ Data loaded successfully")
    
    df_multi_tf = {'5min': df}
    
    # Test with different latency profiles
    profiles = [
        ('co-located', 'Co-located (HFT)'),
        ('institutional', 'Institutional'),
        ('retail_average', 'Retail Average'),
        ('retail_slow', 'Retail Slow')
    ]
    
    print("\n2. Running backtests with different latency profiles...")
    print("-" * 80)
    
    results_all = {}
    
    for profile_key, profile_name in profiles:
        print(f"\n   Testing: {profile_name}")
        
        backtester = BacktesterCore(
            initial_capital=10000,
            commission=0.001,
            slippage_pct=0.001,
            enable_realistic_execution=True,
            latency_profile=profile_key
        )
        
        results = backtester.run_simple_backtest(
            df_multi_tf=df_multi_tf,
            strategy_class=SimpleMAStrategy,
            strategy_params={'fast_period': 20, 'slow_period': 50}
        )
        
        if 'error' not in results:
            results_all[profile_key] = results
            metrics = results['metrics']
            print(f"      Sharpe: {metrics['sharpe']:.3f}")
            print(f"      Return: {metrics['total_return']:.2%}")
            print(f"      Trades: {metrics['num_trades']:.0f}")
        else:
            print(f"      ✗ Error: {results['error']}")
    
    # Compare profiles
    if len(results_all) > 0:
        print("\n3. Latency Profile Comparison")
        print("=" * 80)
        print(f"\n{'Profile':<20} {'Sharpe':<12} {'Return':<12} {'Trades':<10}")
        print("-" * 80)
        
        for profile_key, profile_name in profiles:
            if profile_key in results_all:
                m = results_all[profile_key]['metrics']
                print(f"{profile_name:<20} {m['sharpe']:<12.3f} {m['total_return']:<12.2%} {m['num_trades']:<10.0f}")
        
        print("\n" + "=" * 80)
        print("INSIGHTS")
        print("=" * 80)
        print("""
Latency impact analysis:
- Co-located: Fastest execution (~3ms), minimal slippage
- Institutional: Fast execution (~20ms), low slippage
- Retail Average: Typical retail latency (~80ms), moderate slippage
- Retail Slow: Poor connection (~120ms), higher slippage

Key observations:
1. Higher latency = worse execution prices = lower returns
2. HFT strategies REQUIRE co-located/institutional latency
3. Swing trading less sensitive to latency
4. Retail traders should account for ~80-120ms delays

Recommendation:
Use 'retail_average' as baseline for realistic retail trading simulation.
""")
    
    print("=" * 80)
    print("\n✅ Test complete!")


if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
