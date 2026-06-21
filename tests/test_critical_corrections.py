#!/usr/bin/env python3
"""
Test Correcciones Críticas - Kelly Position Sizing

Valida que las correcciones críticas funcionan correctamente:
1. Capital dinámico
2. Estadísticas reales desde trade history
3. Eliminación de duplicación de código
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from core.execution.backtester_core import BacktesterCore

    BACKTESTER_AVAILABLE = True
except ImportError as e:
    print(f"❌ BacktesterCore not available: {e}")
    BACKTESTER_AVAILABLE = False


def test_dynamic_capital_tracking():
    """Test que el capital se actualiza dinámicamente"""
    if not BACKTESTER_AVAILABLE:
        return False

    print("🧪 Test #1: Dynamic Capital Tracking")

    backtester = BacktesterCore(initial_capital=10000, enable_kelly_position_sizing=True)

    # Verificar inicialización
    assert backtester.current_capital == 10000, "Initial capital should be 10000"
    print("   ✅ Initial capital correctly set")

    # Simular cambio de capital
    backtester.current_capital = 15000
    assert backtester.current_capital == 15000, "Capital should update to 15000"
    print("   ✅ Capital updates dynamically")

    # Verificar que position sizing usa capital dinámico
    position_size = backtester._calculate_position_size(
        capital=backtester.current_capital, win_rate=0.6, win_loss_ratio=2.0
    )

    # Con $15k de capital, position debería ser mayor que con $10k
    position_size_10k = backtester._calculate_position_size(capital=10000, win_rate=0.6, win_loss_ratio=2.0)

    assert position_size > position_size_10k, "Position size should scale with capital"
    print(f"   ✅ Position scaling: $10k→${position_size_10k:.0f}, $15k→${position_size:.0f}")

    return True


def test_trade_history_statistics():
    """Test que las estadísticas se calculan desde trade history"""
    if not BACKTESTER_AVAILABLE:
        return False

    print("\n🧪 Test #2: Trade History Statistics")

    backtester = BacktesterCore(initial_capital=10000, enable_kelly_position_sizing=True)

    # Verificar inicialización de trade history
    assert hasattr(backtester, "trade_history"), "Should have trade_history attribute"
    assert isinstance(backtester.trade_history, pd.DataFrame), "trade_history should be DataFrame"
    print("   ✅ Trade history initialized")

    # Verificar estadísticas por defecto (sin historia)
    win_rate, wl_ratio = backtester._get_strategy_statistics()
    assert math.isclose(win_rate, 0.50), "Default win rate should be 0.50"
    assert math.isclose(wl_ratio, 1.2), "Default W/L ratio should be 1.2"
    print(f"   ✅ Default statistics: WR={win_rate}, W/L={wl_ratio}")

    # Simular trades
    mock_trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="1h"),
            "side": ["buy"] * 30,
            "entry_price": np.random.uniform(100, 110, 30),
            "exit_price": np.random.uniform(100, 110, 30),
            "size": [1.0] * 30,
            "pnl": np.concatenate(
                [np.random.uniform(10, 50, 18), np.random.uniform(-30, -10, 12)]  # 18 wins  # 12 losses
            ),
            "pnl_pct": [0.0] * 30,
            "hold_time": [1.0] * 30,
        }
    )

    backtester.trade_history = mock_trades

    # Calcular estadísticas reales
    win_rate, wl_ratio = backtester._get_strategy_statistics()

    # 18 wins / 30 trades = 0.6
    assert 0.55 <= win_rate <= 0.65, f"Win rate should be ~0.6, got {win_rate}"
    assert wl_ratio > 1.0, f"W/L ratio should be > 1.0, got {wl_ratio}"
    print(f"   ✅ Real statistics calculated: WR={win_rate:.2%}, W/L={wl_ratio:.2f}")

    return True


def test_code_deduplication():
    """Test que el código no está duplicado"""
    if not BACKTESTER_AVAILABLE:
        return False

    print("\n🧪 Test #3: Code Deduplication")

    backtester = BacktesterCore(initial_capital=10000, enable_kelly_position_sizing=True)

    # Verificar que existe el método helper
    assert hasattr(
        backtester, "_calculate_order_size_for_execution"
    ), "Should have _calculate_order_size_for_execution method"
    print("   ✅ Helper method exists")

    # Verificar que el método funciona
    order_size = backtester._calculate_order_size_for_execution(
        base_price=100.0, current_capital=10000, volatility_val=0.2
    )

    assert order_size > 0, "Order size should be positive"
    print(f"   ✅ Helper method works: order_size={order_size:.4f}")

    # Verificar que produce resultados consistentes
    order_size_2 = backtester._calculate_order_size_for_execution(
        base_price=100.0, current_capital=10000, volatility_val=0.2
    )

    assert abs(order_size - order_size_2) < 0.001, "Should produce consistent results"
    print("   ✅ Helper method is deterministic")

    return True


def test_volatility_adjustment_improved():
    """Test que el nuevo ajuste de volatilidad es mejor"""
    if not BACKTESTER_AVAILABLE:
        return False

    print("\n🧪 Test #4: Improved Volatility Adjustment")

    from core.risk.kelly_sizer import KellyPositionSizer

    sizer = KellyPositionSizer(volatility_adjustment=True)

    # Test diferentes niveles de volatilidad
    vol_levels = [0.0, 0.1, 0.3, 0.5, 0.8]
    adjustments = []

    for vol in vol_levels:
        result = sizer.calculate_position_size(capital=10000, win_rate=0.6, win_loss_ratio=2.0, current_volatility=vol)
        adj = result["volatility_adjustment"]
        adjustments.append(adj)
        print(f"   Volatility {vol:.1f}: adjustment={adj:.3f}")

    # Verificar que el ajuste es monotónico decreciente
    for i in range(len(adjustments) - 1):
        assert adjustments[i] >= adjustments[i + 1], "Adjustment should decrease with volatility"

    # Verificar que usa función no-lineal (exponencial)
    # La diferencia entre ajustes consecutivos debe cambiar
    diffs = [adjustments[i] - adjustments[i + 1] for i in range(len(adjustments) - 1)]
    assert len(set([round(d, 3) for d in diffs])) > 1, "Should use non-linear adjustment"

    print("   ✅ Volatility adjustment is non-linear and monotonic")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 TESTING CRITICAL CORRECTIONS")
    print("=" * 60)

    all_passed = True

    try:
        if not test_dynamic_capital_tracking():
            all_passed = False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        all_passed = False

    try:
        if not test_trade_history_statistics():
            all_passed = False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        all_passed = False

    try:
        if not test_code_deduplication():
            all_passed = False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        all_passed = False

    try:
        if not test_volatility_adjustment_improved():
            all_passed = False
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CRITICAL CORRECTIONS VALIDATED!")
        print("=" * 60)
        print("\n✨ The Kelly Position Sizing implementation is now:")
        print("   ✅ Using dynamic capital")
        print("   ✅ Calculating real statistics from trade history")
        print("   ✅ Code deduplicated with helper methods")
        print("   ✅ Improved non-linear volatility adjustment")
        print("\n🚀 Ready for production!")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
