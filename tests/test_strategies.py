"""
Test Script - Quick Strategy Validation
Run this to verify all strategies load correctly
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from strategies import load_strategy, list_available_strategies, STRATEGY_CATALOG
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(periods=200):
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
    
    # Generate realistic price movement
    close_prices = 40000 + np.cumsum(np.random.randn(periods) * 100)
    
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(periods) * 50,
        'high': close_prices + abs(np.random.randn(periods) * 100),
        'low': close_prices - abs(np.random.randn(periods) * 100),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, periods)
    }, index=dates)
    
    # Ensure high >= close >= low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def test_strategy(strategy_name, preset=None):
    """Test a single strategy"""
    print(f"\n{'='*60}")
    print(f"Testing: {strategy_name}" + (f" (preset: {preset})" if preset else ""))
    print(f"{'='*60}")
    
    try:
        # Load strategy
        strategy = load_strategy(strategy_name, preset)
        print(f"✅ Strategy loaded: {strategy.name}")
        
        # Show parameters
        params = strategy.get_parameters()
        print(f"📊 Parameters ({len(params)}):")
        for key, value in params.items():
            print(f"   - {key}: {value}")
        
        # Generate test data
        df = generate_sample_data()
        print(f"\n📈 Generated {len(df)} bars of test data")
        
        # Generate signals
        result = strategy.generate_signals(df)
        
        # Count signals
        buy_signals = len(result[result['signal'] == 1])
        sell_signals = len(result[result['signal'] == -1])
        total_signals = buy_signals + sell_signals
        
        print(f"\n🎯 Signal Generation:")
        print(f"   - BUY signals: {buy_signals}")
        print(f"   - SELL signals: {sell_signals}")
        print(f"   - Total: {total_signals}")
        
        if total_signals > 0:
            # Show signal strength stats
            signal_mask = result['signal'] != 0
            avg_strength = result.loc[signal_mask, 'signal_strength'].mean()
            print(f"   - Avg strength: {avg_strength:.2f}/5.0")
        
        print(f"\n✅ Strategy test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Strategy test FAILED")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner"""
    print("\n" + "="*70)
    print(" 🧪 STRATEGY VALIDATION TEST SUITE")
    print("="*70)
    
    # List all available strategies
    strategies = list_available_strategies()
    print(f"\n📚 Found {len(strategies)} strategies")
    
    # Show catalog info
    print("\n📖 Strategy Catalog:")
    for name in strategies:
        if name in STRATEGY_CATALOG:
            info = STRATEGY_CATALOG[name]
            print(f"\n   {info['name']}")
            print(f"   └─ {info['description']}")
            print(f"      Best for: {info['best_for']}")
            print(f"      Timeframes: {', '.join(info['timeframes'])}")
    
    # Test each strategy
    print("\n" + "="*70)
    print(" RUNNING STRATEGY TESTS")
    print("="*70)
    
    results = {}
    
    for strategy_name in strategies:
        # Test default configuration
        passed = test_strategy(strategy_name)
        results[strategy_name] = passed
        
        # Small delay for readability
        import time
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for r in results.values() if r)
    total_count = len(results)
    
    print(f"\n✅ PASSED: {passed_count}/{total_count}")
    print(f"❌ FAILED: {total_count - passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 ALL STRATEGIES VALIDATED SUCCESSFULLY!")
    else:
        print("\n⚠️  Some strategies failed validation")
        print("\nFailed strategies:")
        for name, passed in results.items():
            if not passed:
                print(f"   - {name}")
    
    print("\n" + "="*70)
    
    # Show usage example
    print("\n💡 USAGE EXAMPLE:")
    print("="*70)
    print("""
# In your code:
from strategies import load_strategy

# Load any strategy
strategy = load_strategy('rsi_mean_reversion')

# Or with preset
strategy = load_strategy('macd_momentum', preset='aggressive')

# Generate signals on your data
signals = strategy.generate_signals(your_dataframe)

# Use in backtesting
from backtesting import BacktestEngine
engine = BacktestEngine(strategy)
results = engine.run(your_dataframe, initial_capital=10000)
""")
    
    print("\n" + "="*70)
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
