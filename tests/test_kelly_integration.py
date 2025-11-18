#!/usr/bin/env python3
"""
Test Kelly Position Sizing Integration with BacktesterCore

Tests the integration of KellyPositionSizer with BacktesterCore.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import BacktesterCore
try:
    from core.execution.backtester_core import BacktesterCore
    BACKTESTER_AVAILABLE = True
except ImportError as e:
    print(f"❌ BacktesterCore not available: {e}")
    BACKTESTER_AVAILABLE = False

# Mock strategy class for testing
class MockStrategy:
    def __init__(self, params):
        self.params = params

    def generate_signals(self, df_multi_tf):
        # Generate simple mock signals
        df_5m = df_multi_tf['5min']
        signals = pd.DataFrame(index=df_5m.index)
        signals['entries'] = False
        signals['exits'] = False

        # Simple entry signal: price above SMA
        sma = df_5m['close'].rolling(20).mean()
        signals.loc[df_5m['close'] > sma, 'entries'] = True

        # Simple exit signal: price below SMA
        signals.loc[df_5m['close'] < sma, 'exits'] = True

        return signals


def create_mock_data():
    """Create mock market data for testing"""
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')

    # Generate realistic price data with trend and noise
    np.random.seed(42)
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    prices = trend + noise

    df_5m = pd.DataFrame({
        'open': prices,
        'high': prices + abs(np.random.normal(0, 1, len(dates))),
        'low': prices - abs(np.random.normal(0, 1, len(dates))),
        'close': prices + np.random.normal(0, 0.5, len(dates)),
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)

    # Ensure high >= close >= low >= 0
    df_5m['high'] = np.maximum(df_5m[['high', 'close']].max(axis=1), df_5m['open'])
    df_5m['low'] = np.minimum(df_5m[['low', 'close']].min(axis=1), df_5m['open'])
    df_5m['low'] = np.maximum(df_5m['low'], 0)

    return {'5min': df_5m}


def test_kelly_integration():
    """Test Kelly position sizing integration"""
    if not BACKTESTER_AVAILABLE:
        print("❌ Skipping test: BacktesterCore not available")
        return False

    print("🧪 Testing Kelly Position Sizing Integration...")

    try:
        # Test 1: Initialize with Kelly sizing enabled
        print("   Testing initialization...")
        backtester = BacktesterCore(
            initial_capital=10000,
            enable_kelly_position_sizing=True,
            kelly_fraction=0.5,
            max_position_pct=0.10
        )

        assert hasattr(backtester, 'kelly_sizer'), "Kelly sizer not initialized"
        assert backtester.enable_kelly_position_sizing, "Kelly sizing not enabled"
        print("✅ Kelly sizer initialization test passed")

        # Test 2: Test position size calculation
        print("   Testing position size calculation...")
        position_size = backtester._calculate_position_size(
            capital=10000,
            win_rate=0.6,
            win_loss_ratio=2.0
        )

        print(f"   Position size: ${position_size:.2f}")
        assert position_size > 0, "Position size should be positive"
        assert position_size <= 1000, "Position size should be <= $1000 (10% of capital)"
        print("✅ Position size calculation test passed")

        print("🎉 Basic Kelly integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_kelly_integration()
    if success:
        print("\n✅ Kelly Position Sizing integration is working correctly!")
    else:
        print("\n❌ Kelly Position Sizing integration has issues.")
        sys.exit(1)