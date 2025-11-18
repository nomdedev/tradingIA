#!/usr/bin/env python3
"""
Test Kelly Position Sizer

Tests for the Kelly Criterion implementation in position sizing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
import pandas as pd
from src.risk.kelly_sizer import KellyPositionSizer


class TestKellyPositionSizer:
    """Test cases for Kelly Position Sizer"""

    def test_kelly_basic_calculation(self):
        """Test basic Kelly calculation"""
        sizer = KellyPositionSizer(kelly_fraction=1.0)  # Full Kelly

        # Coin flip example: 50% win rate, 1:1 payoff
        result = sizer.calculate_kelly_fraction(win_rate=0.5, win_loss_ratio=1.0)

        # For fair coin flip, Kelly should be 0
        assert result.kelly_full == 0.0
        assert result.kelly_fraction == 0.0

    def test_kelly_positive_edge(self):
        """Test Kelly with positive edge"""
        sizer = KellyPositionSizer(kelly_fraction=1.0)

        # 60% win rate, 2:1 payoff (good edge)
        result = sizer.calculate_kelly_fraction(win_rate=0.6, win_loss_ratio=2.0)

        # Kelly formula: f = (bp - q) / b
        # b = 2, p = 0.6, q = 0.4
        # f = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4
        expected_kelly = 0.4

        assert abs(result.kelly_full - expected_kelly) < 0.001

    def test_kelly_conservative_fraction(self):
        """Test conservative Kelly fraction"""
        sizer = KellyPositionSizer(kelly_fraction=0.5)  # Half Kelly

        result = sizer.calculate_kelly_fraction(win_rate=0.6, win_loss_ratio=2.0)

        # Should be half of full Kelly
        assert abs(result.kelly_fraction - 0.2) < 0.001  # 0.4 * 0.5 = 0.2

    def test_kelly_negative_edge(self):
        """Test Kelly with negative edge"""
        sizer = KellyPositionSizer()

        # Bad strategy: 40% win rate, 1:2 loss ratio
        result = sizer.calculate_kelly_fraction(win_rate=0.4, win_loss_ratio=0.5)

        # Should return 0 (no position)
        assert result.kelly_fraction == 0.0

    def test_position_size_calculation(self):
        """Test position size calculation with capital"""
        sizer = KellyPositionSizer(kelly_fraction=0.5, max_position_pct=0.10)

        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.6,
            win_loss_ratio=2.0
        )

        # Kelly fraction should be 0.1 (half of 0.2)
        # Position should be 0.1 * 10000 = 1000
        assert abs(result['position_size'] - 1000.0) < 0.001
        assert abs(result['position_pct'] - 0.10) < 0.001

    def test_volatility_adjustment(self):
        """Test volatility adjustment"""
        sizer = KellyPositionSizer(volatility_adjustment=True)

        # High volatility should reduce position
        result_high_vol = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.6,
            win_loss_ratio=2.0,
            current_volatility=0.5  # High volatility
        )

        result_low_vol = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.6,
            win_loss_ratio=2.0,
            current_volatility=0.0  # Low volatility
        )

        # High volatility should result in smaller position
        assert result_high_vol['position_size'] < result_low_vol['position_size']

    def test_market_impact_adjustment(self):
        """Test market impact adjustment"""
        sizer = KellyPositionSizer()

        # Without market impact
        result_no_impact = sizer.calculate_kelly_fraction(
            win_rate=0.6,
            win_loss_ratio=2.0,
            market_impact_pct=0.0
        )

        # With market impact
        result_with_impact = sizer.calculate_kelly_fraction(
            win_rate=0.6,
            win_loss_ratio=2.0,
            market_impact_pct=0.10  # 10% market impact
        )

        # Market impact should reduce Kelly fraction
        assert result_with_impact.kelly_fraction < result_no_impact.kelly_fraction

    def test_bounds_enforcement(self):
        """Test position size bounds"""
        sizer = KellyPositionSizer(
            kelly_fraction=1.0,
            max_position_pct=0.05,  # Max 5%
            min_position_pct=0.01   # Min 1%
        )

        # Very aggressive Kelly (would be > 5%)
        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.8,      # High win rate
            win_loss_ratio=3.0  # High reward
        )

        # Should be capped at 5%
        assert result['position_pct'] <= 0.05

    def test_risk_warnings(self):
        """Test risk warnings"""
        sizer = KellyPositionSizer()

        # High risk scenario
        result = sizer.calculate_kelly_fraction(
            win_rate=0.7,
            win_loss_ratio=4.0  # Very high reward
        )

        warnings = sizer.get_risk_warnings(result)

        # Should have warnings for high Kelly
        assert len(warnings) > 0

    def test_edge_cases(self):
        """Test edge cases"""
        sizer = KellyPositionSizer()

        # Win rate = 0
        result = sizer.calculate_kelly_fraction(win_rate=0.0, win_loss_ratio=2.0)
        assert result.kelly_fraction == 0.0

        # Win rate = 1.0
        result = sizer.calculate_kelly_fraction(win_rate=1.0, win_loss_ratio=2.0)
        assert result.kelly_fraction > 0

        # Win/loss ratio = 0 (guaranteed loss)
        result = sizer.calculate_kelly_fraction(win_rate=0.6, win_loss_ratio=0.0)
        assert result.kelly_fraction == 0.0


if __name__ == "__main__":
    # Run basic tests
    test = TestKellyPositionSizer()

    print("🧪 Testing Kelly Position Sizer...")

    try:
        test.test_kelly_basic_calculation()
        print("✅ Basic calculation test passed")

        test.test_kelly_positive_edge()
        print("✅ Positive edge test passed")

        test.test_kelly_conservative_fraction()
        print("✅ Conservative fraction test passed")

        test.test_position_size_calculation()
        print("✅ Position size test passed")

        test.test_volatility_adjustment()
        print("✅ Volatility adjustment test passed")

        test.test_market_impact_adjustment()
        print("✅ Market impact test passed")

        print("\n🎉 All Kelly Position Sizer tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()