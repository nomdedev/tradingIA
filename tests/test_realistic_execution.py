"""
Test Suite for Realistic Execution Models (Fase 1)

Tests:
1. Market Impact Model
2. Order Manager (various order types)
3. Latency Model

Validates that Fase 1 components work correctly before integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.execution.market_impact import MarketImpactModel, VolumeProfileAnalyzer
from src.execution.order_manager import OrderManager, OrderType, OrderSide, OrderStatus
from src.execution.latency_model import LatencyModel, LatencyProfile

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def test_market_impact():
    """Test Market Impact Model"""
    print("\n" + "=" * 70)
    print("TEST 1: Market Impact Model")
    print("=" * 70)

    model = MarketImpactModel()

    # Test case 1: Small order
    print("\n📊 Test 1a: Small order (1% of volume)")
    impact = model.calculate_impact(
        order_size=0.1,  # 0.1 BTC
        price=50000,
        avg_volume=10,  # 10 BTC volume
        volatility=0.02,  # 2% volatility
        bid_ask_spread=50,
        time_of_day=14
    )

    print(f"  Volume Ratio: {impact['volume_ratio']:.2%}")
    print(f"  Total Impact: {impact['total_impact_pct']:.4%}")
    print(f"  Impact Cost: ${impact['total_impact_dollars']:.2f}")
    
    assert impact['volume_ratio'] == 0.01, "Volume ratio calculation error"
    assert impact['total_impact_pct'] < 0.01, "Small order should have < 1% impact"
    print("  ✅ PASS: Small order impact is reasonable")

    # Test case 2: Large order
    print("\n📊 Test 1b: Large order (50% of volume)")
    impact_large = model.calculate_impact(
        order_size=5.0,  # 5 BTC
        price=50000,
        avg_volume=10,
        volatility=0.02,
        bid_ask_spread=50,
        time_of_day=14
    )

    print(f"  Volume Ratio: {impact_large['volume_ratio']:.2%}")
    print(f"  Total Impact: {impact_large['total_impact_pct']:.4%}")
    print(f"  Impact Cost: ${impact_large['total_impact_dollars']:.2f}")
    
    assert impact_large['volume_ratio'] == 0.5, "Volume ratio calculation error"
    assert impact_large['total_impact_pct'] > impact['total_impact_pct'], "Large order should have higher impact"
    print("  ✅ PASS: Large order has proportionally higher impact")

    # Test case 3: Optimal sizing
    print("\n📊 Test 1c: Optimal order sizing")
    optimal = model.estimate_optimal_order_size(
        available_capital=100000,
        price=50000,
        avg_volume=10,
        volatility=0.02,
        max_impact_pct=0.005
    )

    print(f"  Available Capital: $100,000")
    print(f"  Optimal Size: {optimal['optimal_size']:.4f} BTC")
    print(f"  Optimal Value: ${optimal['optimal_value']:,.2f}")
    print(f"  Expected Impact: {optimal['expected_impact_pct']:.4%}")
    print(f"  Capital Utilization: {optimal['capital_utilization']:.2%}")
    
    assert optimal['expected_impact_pct'] <= 0.005, "Optimal sizing should stay within max impact"
    assert optimal['optimal_size'] > 0, "Should recommend some position size"
    print("  ✅ PASS: Optimal sizing respects impact constraints")

    print("\n✅ Market Impact Model: ALL TESTS PASSED")


def test_order_manager():
    """Test Order Manager"""
    print("\n" + "=" * 70)
    print("TEST 2: Order Manager")
    print("=" * 70)

    manager = OrderManager(account_balance=10000, enable_partial_fills=True)

    # Test case 1: Market order execution
    print("\n📊 Test 2a: Market Order")
    market_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )

    assert market_order.status == OrderStatus.PENDING, "New order should be PENDING"
    print(f"  Created: {market_order.order_id}")

    # Process with market data
    filled = manager.process_orders(
        current_price=50000,
        current_timestamp=datetime.now(),
        high=50100,
        low=49900,
        volume=100,
        avg_volume=100
    )

    assert len(filled) > 0, "Market order should fill immediately"
    assert market_order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL], "Market order should be filled"
    print(f"  Status: {market_order.status.value}")
    print(f"  Filled: {market_order.filled_quantity:.4f} / {market_order.quantity}")
    print("  ✅ PASS: Market order executed")

    # Test case 2: Limit order execution
    print("\n📊 Test 2b: Limit Order")
    limit_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.2,
        price=49000
    )

    # Process with price above limit (shouldn't fill)
    manager.process_orders(
        current_price=50000,
        current_timestamp=datetime.now(),
        high=50100,
        low=49500,
        volume=100,
        avg_volume=100
    )
    assert limit_order.status == OrderStatus.PENDING, "Limit order shouldn't fill above limit price"
    print("  ✓ Correctly didn't fill above limit price")

    # Process with price at/below limit (should fill)
    manager.process_orders(
        current_price=48900,
        current_timestamp=datetime.now(),
        high=49100,
        low=48800,
        volume=100,
        avg_volume=100
    )
    assert limit_order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL], "Limit order should fill at/below limit"
    print(f"  ✓ Filled at limit: {limit_order.avg_fill_price:.2f}")
    print("  ✅ PASS: Limit order executed correctly")

    # Test case 3: Stop order
    print("\n📊 Test 2c: Stop Market Order")
    stop_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.1,
        stop_price=48000
    )

    # Process with price above stop (shouldn't trigger)
    manager.process_orders(
        current_price=49000,
        current_timestamp=datetime.now(),
        high=49500,
        low=48500,
        volume=100,
        avg_volume=100
    )
    assert stop_order.status == OrderStatus.PENDING, "Stop shouldn't trigger above stop price"
    print("  ✓ Correctly didn't trigger above stop")

    # Process with price at/below stop (should trigger)
    manager.process_orders(
        current_price=47500,
        current_timestamp=datetime.now(),
        high=48100,
        low=47000,
        volume=100,
        avg_volume=100
    )
    assert stop_order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL], "Stop should trigger below stop price"
    print(f"  ✓ Triggered and filled: {stop_order.avg_fill_price:.2f}")
    print("  ✅ PASS: Stop order executed correctly")

    # Test case 4: Trailing stop
    print("\n📊 Test 2d: Trailing Stop Order")
    trailing_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.TRAILING_STOP,
        quantity=0.1,
        trailing_offset=0.02  # 2% trail
    )

    # Price rises (trail should follow)
    for price in [50000, 51000, 52000]:
        manager.process_orders(
            current_price=price,
            current_timestamp=datetime.now(),
            high=price + 100,
            low=price - 100,
            volume=100,
            avg_volume=100
        )
        if trailing_order.highest_price:
            print(f"  Price: ${price:,} | Trail High: ${trailing_order.highest_price:,.0f}")

    assert trailing_order.highest_price == 52100, "Should track highest price"
    assert trailing_order.status == OrderStatus.PENDING, "Shouldn't trigger while rising"

    # Price drops below trail (should trigger)
    drop_price = trailing_order.highest_price * (1 - 0.03)  # Drop 3% (below 2% trail)
    manager.process_orders(
        current_price=drop_price,
        current_timestamp=datetime.now(),
        high=drop_price + 100,
        low=drop_price - 100,
        volume=100,
        avg_volume=100
    )

    assert trailing_order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL], "Should trigger on trail breach"
    print(f"  ✓ Triggered at: ${trailing_order.avg_fill_price:.2f}")
    print("  ✅ PASS: Trailing stop executed correctly")

    print("\n✅ Order Manager: ALL TESTS PASSED")


def test_latency_model():
    """Test Latency Model"""
    print("\n" + "=" * 70)
    print("TEST 3: Latency Model")
    print("=" * 70)

    # Test different profiles
    profiles_to_test = ['co-located', 'retail_average', 'mobile']

    print("\n📊 Test 3a: Latency Profiles")
    for profile_name in profiles_to_test:
        model = LatencyProfile.get_profile(profile_name)
        stats = model.get_latency_statistics(n_samples=100)

        print(f"\n  {profile_name}:")
        print(f"    Mean: {stats['mean']:.1f}ms")
        print(f"    P95: {stats['p95']:.1f}ms")

        # Validate expectations
        if profile_name == 'co-located':
            assert stats['mean'] < 10, "Co-located should have < 10ms latency"
        elif profile_name == 'retail_average':
            assert 40 < stats['mean'] < 150, "Retail average should be 40-150ms"
        elif profile_name == 'mobile':
            assert stats['mean'] > 100, "Mobile should have > 100ms latency"

    print("\n  ✅ PASS: All profiles have expected latency ranges")

    # Test scenario variations
    print("\n📊 Test 3b: Scenario Variations")
    model = LatencyProfile.get_profile('retail_average')

    # Use averages to account for randomness
    n_samples = 30
    
    normal_latencies = [model.calculate_total_latency('market', 1.0, 14) for _ in range(n_samples)]
    high_vol_latencies = [model.calculate_total_latency('market', 2.5, 14) for _ in range(n_samples)]
    open_latencies = [model.calculate_total_latency('market', 1.0, 9) for _ in range(n_samples)]
    limit_latencies = [model.calculate_total_latency('limit', 1.0, 14) for _ in range(n_samples)]
    
    avg_normal = np.mean(normal_latencies)
    avg_high_vol = np.mean(high_vol_latencies)
    avg_open = np.mean(open_latencies)
    avg_limit = np.mean(limit_latencies)
    
    print(f"  Normal market: {avg_normal:.1f}ms")
    print(f"  High volatility: {avg_high_vol:.1f}ms")
    print(f"  Market open: {avg_open:.1f}ms")
    print(f"  Limit order: {avg_limit:.1f}ms")

    # High vol should have higher latency (on average)
    assert avg_high_vol > avg_normal * 1.2, "High volatility should increase latency by ~20%+"
    print("\n  ✅ PASS: Volatility correctly affects latency")

    # Market hours should have higher latency (test with averages to account for randomness)
    midday_latencies = [model.calculate_total_latency('market', 1.0, 12) for _ in range(n_samples)]
    avg_midday = np.mean(midday_latencies)
    # Market open (hour 9) has 1.6x multiplier vs lunch (hour 12) with 1.0x
    print(f"  Midday avg: {avg_midday:.1f}ms, Open avg: {avg_open:.1f}ms")
    assert avg_open > avg_midday * 1.2, "Market open should have ~50% higher latency on average"
    print("  ✅ PASS: Time of day correctly affects latency")

    print("\n✅ Latency Model: ALL TESTS PASSED")


def test_integration():
    """Integration test: All components together"""
    print("\n" + "=" * 70)
    print("TEST 4: Integration Test")
    print("=" * 70)

    print("\n📊 Simulating realistic trade execution...")

    # Setup
    impact_model = MarketImpactModel()
    order_manager = OrderManager(account_balance=100000, enable_partial_fills=True)
    latency_model = LatencyProfile.get_profile('retail_average')

    # Trading scenario
    price = 50000
    available_capital = 100000
    avg_volume = 10

    # Step 1: Calculate optimal order size considering market impact
    print("\n1. Calculate optimal order size:")
    optimal = impact_model.estimate_optimal_order_size(
        available_capital=available_capital,
        price=price,
        avg_volume=avg_volume,
        volatility=0.02,
        max_impact_pct=0.005
    )
    print(f"   Optimal size: {optimal['optimal_size']:.4f} BTC")
    print(f"   Expected impact: {optimal['expected_impact_pct']:.4%}")

    # Step 2: Calculate execution price with impact
    print("\n2. Calculate execution price with market impact:")
    impact = impact_model.calculate_impact(
        order_size=optimal['optimal_size'],
        price=price,
        avg_volume=avg_volume,
        volatility=0.02,
        bid_ask_spread=50,
        time_of_day=14
    )
    exec_price_with_impact = impact_model.calculate_execution_price(
        side='buy',
        price=price,
        impact_pct=impact['total_impact_pct']
    )
    print(f"   Market price: ${price:,.2f}")
    print(f"   Execution price: ${exec_price_with_impact:,.2f}")
    print(f"   Impact cost: ${impact['total_impact_dollars']:.2f}")

    # Step 3: Calculate latency delay
    print("\n3. Calculate execution latency:")
    latency_ms = latency_model.calculate_total_latency(
        order_type='market',
        market_volatility=1.0,
        time_of_day=14
    )
    print(f"   Latency: {latency_ms:.1f}ms")
    bars_delayed = max(1, int(latency_ms / 60000))  # Assume 1min bars
    print(f"   Bars delayed: {bars_delayed}")

    # Step 4: Create and execute order
    print("\n4. Execute order through order manager:")
    order = order_manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=optimal['optimal_size']
    )
    print(f"   Order created: {order.order_id}")

    # Simulate execution after latency delay with impact-adjusted price
    order_manager.process_orders(
        current_price=exec_price_with_impact,
        current_timestamp=datetime.now() + timedelta(milliseconds=latency_ms),
        high=exec_price_with_impact + 50,
        low=exec_price_with_impact - 50,
        volume=100,
        avg_volume=avg_volume
    )

    # Step 5: Calculate total costs
    print("\n5. Total execution costs:")
    print(f"   Order value: ${optimal['optimal_value']:,.2f}")
    print(f"   Market impact: ${impact['total_impact_dollars']:.2f}")
    
    # Latency slippage (assume 0.01% per 100ms)
    latency_slippage_pct = (latency_ms / 100) * 0.0001
    latency_cost = optimal['optimal_value'] * latency_slippage_pct
    print(f"   Latency slippage: ${latency_cost:.2f}")
    
    total_cost = impact['total_impact_dollars'] + latency_cost
    total_cost_pct = total_cost / optimal['optimal_value']
    print(f"   TOTAL COST: ${total_cost:.2f} ({total_cost_pct:.4%})")

    # Validation
    assert order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL], "Order should be filled"
    assert total_cost > 0, "Should have positive execution costs"
    assert total_cost_pct < 0.02, "Total costs should be < 2%"

    print("\n✅ Integration Test: PASSED")
    print(f"\n💡 Key Insight: Realistic execution adds ~${total_cost:.2f} ({total_cost_pct:.4%}) cost")
    print("   This is what was missing in original backtests!")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 70)
    print("FASE 1 - REALISTIC EXECUTION MODELS TEST SUITE")
    print("=" * 70)
    print("\nTesting components:")
    print("  1. Market Impact Model")
    print("  2. Order Manager")
    print("  3. Latency Model")
    print("  4. Integration Test")

    try:
        test_market_impact()
        test_order_manager()
        test_latency_model()
        test_integration()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\nFase 1 components are ready for integration into backtester.")
        print("\nNext steps:")
        print("  1. Integrate market impact into backtester_core.py")
        print("  2. Replace simple order execution with order_manager.py")
        print("  3. Add latency simulation to signal processing")
        print("  4. Re-run backtests and compare results")
        print("\nExpected impact on metrics:")
        print("  - Sharpe Ratio: -20% to -30%")
        print("  - Total Return: -30% to -40%")
        print("  - Win Rate: -5% to -10%")
        print("\nThis is GOOD - it means backtests will be realistic!")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
