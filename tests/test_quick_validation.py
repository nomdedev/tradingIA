"""
Test de validación rápida de todos los componentes FASE 1
"""

import sys
sys.path.append('.')

print("=" * 60)
print("VALIDACIÓN RÁPIDA - FASE 1 REALISTIC EXECUTION")
print("=" * 60)

# Test 1: Imports de componentes core
print("\n1. Verificando imports de componentes...")
try:
    from src.execution.market_impact import MarketImpactModel, VolumeProfileAnalyzer
    print("   ✓ market_impact.py")
except Exception as e:
    print(f"   ✗ market_impact.py: {e}")
    sys.exit(1)

try:
    from src.execution.order_manager import OrderManager, Order, OrderType, OrderStatus
    print("   ✓ order_manager.py")
except Exception as e:
    print(f"   ✗ order_manager.py: {e}")
    sys.exit(1)

try:
    from src.execution.latency_model import LatencyModel, LatencyProfile
    print("   ✓ latency_model.py")
except Exception as e:
    print(f"   ✗ latency_model.py: {e}")
    sys.exit(1)

# Test 2: Inicialización de BacktesterCore
print("\n2. Verificando BacktesterCore...")
try:
    from core.execution.backtester_core import BacktesterCore
    
    # Sin realistic execution
    b1 = BacktesterCore(10000, enable_realistic_execution=False)
    assert b1.enable_realistic_execution == False
    print("   ✓ Inicialización sin realistic execution")
    
    # Con realistic execution
    b2 = BacktesterCore(10000, enable_realistic_execution=True, latency_profile='retail_average')
    assert b2.enable_realistic_execution == True
    assert hasattr(b2, 'market_impact_model')
    assert hasattr(b2, 'volume_analyzer')
    assert hasattr(b2, 'latency_model')
    print("   ✓ Inicialización con realistic execution")
    print(f"     - market_impact_model: {type(b2.market_impact_model).__name__}")
    print(f"     - volume_analyzer: {type(b2.volume_analyzer).__name__}")
    print(f"     - latency_model: {type(b2.latency_model).__name__}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verificar método _calculate_realistic_execution_price
print("\n3. Verificando método _calculate_realistic_execution_price...")
try:
    backtester = BacktesterCore(10000, enable_realistic_execution=True)
    
    result = backtester._calculate_realistic_execution_price(
        base_price=50000,
        order_size=1.0,
        avg_volume=10.0,
        volatility=0.02,
        side='buy',
        timestamp=None
    )
    
    assert 'execution_price' in result
    assert 'impact_cost' in result
    assert 'latency_ms' in result
    assert result['execution_price'] > 50000  # Buy should increase price
    
    print("   ✓ Método _calculate_realistic_execution_price funcional")
    print(f"     - Base price: $50,000")
    print(f"     - Execution price: ${result['execution_price']:,.2f}")
    print(f"     - Impact cost: ${result['impact_cost']:,.2f}")
    print(f"     - Latency: {result['latency_ms']:.2f}ms")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verificar perfiles de latencia
print("\n4. Verificando perfiles de latencia...")
try:
    profiles = ['co-located', 'institutional', 'retail_fast', 'retail_average', 'retail_slow', 'mobile']
    
    for profile_name in profiles:
        profile = LatencyProfile.get_profile(profile_name)
        assert profile is not None
        latency = profile.calculate_total_latency()
        print(f"   ✓ {profile_name:20s} ~{latency:.1f}ms")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verificar market impact con diferentes tamaños
print("\n5. Verificando market impact model...")
try:
    model = MarketImpactModel()
    
    test_cases = [
        (0.1, 10, "Small order"),
        (1.0, 10, "Medium order"),
        (5.0, 10, "Large order"),
    ]
    
    for order_size, avg_volume, desc in test_cases:
        impact = model.calculate_impact(
            order_size=order_size,
            avg_volume=avg_volume,
            price=50000,
            volatility=0.02,
            bid_ask_spread=10  # $10 spread
        )
        impact_pct = impact['total_impact_pct'] * 100  # Convert to percentage
        print(f"   ✓ {desc:15s}: {impact_pct:.3f}% impact")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verificar order manager
print("\n6. Verificando order manager...")
try:
    from src.execution.order_manager import OrderSide
    from datetime import datetime
    
    manager = OrderManager()
    
    # Create a simple market order
    order1 = Order(
        order_id="TEST1",
        symbol="BTCUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0
    )
    
    # Verify order creation
    assert order1.status == OrderStatus.PENDING
    assert order1.quantity == 1.0
    assert order1.order_type == OrderType.MARKET
    
    print("   ✓ Order creation funcional")
    print(f"     - Order ID: {order1.order_id}")
    print(f"     - Type: {order1.order_type.value}")
    print(f"     - Status: {order1.status.value}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TODAS LAS VALIDACIONES PASARON")
print("=" * 60)
print("\nComponentes FASE 1 están correctamente integrados y funcionales.")
