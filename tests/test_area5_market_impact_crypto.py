"""
Test ÁREA 5: Market Impact Model para Crypto.

Valida que el modelo de impacto específico para crypto tenga:
1. Liquidez variable por hora (24h)
2. Estimación de volumen global (no single exchange)
3. Penalización por ventas (sell_penalty)
4. get_best_execution_hours() funcional
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest


class TestArea5MarketImpactCrypto(unittest.TestCase):
    """Tests para MarketImpactModelCrypto."""
    
    def setUp(self):
        """Setup para cada test."""
        # Import dentro del test para evitar problemas de import
        from src.execution.market_impact import MarketImpactModelCrypto
        self.model = MarketImpactModelCrypto(symbol="BTC")
    
    def test_liquidity_by_hour_exists(self):
        """Test que exista liquidez para las 24 horas."""
        # Debe tener 24 entradas (0-23)
        self.assertEqual(len(self.model.liquidity_by_hour), 24)
        
        # Todas las horas deben estar presentes
        for hour in range(24):
            self.assertIn(hour, self.model.liquidity_by_hour)
        
        # Valores deben estar entre 0 y 1
        for hour, liq in self.model.liquidity_by_hour.items():
            self.assertGreater(liq, 0, f"Hora {hour} tiene liquidez <= 0")
            self.assertLessEqual(liq, 1.0, f"Hora {hour} tiene liquidez > 1")
    
    def test_global_volume_estimation(self):
        """Test que use volumen global, no de un solo exchange."""
        # BTC debe tener volumen realista (> $1B/día)
        btc_volume = self.model.global_daily_volume.get('BTC', 0)
        self.assertGreater(btc_volume, 1_000_000_000, 
                          "Volumen BTC debe ser > $1B para ser global")
        
        # ETH también
        eth_volume = self.model.global_daily_volume.get('ETH', 0)
        self.assertGreater(eth_volume, 500_000_000,
                          "Volumen ETH debe ser > $500M para ser global")
        
        # Debe haber un default para otros símbolos
        self.assertIn('default', self.model.global_daily_volume)
    
    def test_sell_penalty_exists(self):
        """Test que exista penalización por ventas."""
        self.assertTrue(hasattr(self.model, 'sell_penalty'))
        self.assertGreater(self.model.sell_penalty, 1.0,
                          "Sell penalty debe ser > 1.0 (penalización)")
        # Típicamente 20-50% más
        self.assertLess(self.model.sell_penalty, 2.0,
                       "Sell penalty no debe ser > 2.0 (excesivo)")
    
    def test_sell_vs_buy_impact(self):
        """Test que sells tengan más impacto que buys."""
        price = 50000
        order_size = 100000  # $100k
        hour = 14  # Alta liquidez
        
        buy_impact = self.model.calculate_impact(
            order_size_usd=order_size,
            price=price,
            hour_utc=hour,
            is_buy=True
        )
        
        sell_impact = self.model.calculate_impact(
            order_size_usd=order_size,
            price=price,
            hour_utc=hour,
            is_buy=False
        )
        
        self.assertGreater(sell_impact['total_impact_pct'], 
                          buy_impact['total_impact_pct'],
                          "Sells deben tener más impacto que buys")
        
        # La diferencia debe ser aproximadamente sell_penalty
        ratio = sell_impact['total_impact_pct'] / buy_impact['total_impact_pct']
        self.assertAlmostEqual(ratio, self.model.sell_penalty, delta=0.1,
                              msg="Ratio debe ser cercano a sell_penalty")
    
    def test_get_best_execution_hours(self):
        """Test que get_best_execution_hours funcione correctamente."""
        best_hours = self.model.get_best_execution_hours(top_n=5)
        
        # Debe retornar 5 tuplas
        self.assertEqual(len(best_hours), 5)
        
        # Cada elemento debe ser (hora, liquidez)
        for hour, liq in best_hours:
            self.assertIsInstance(hour, int)
            self.assertIsInstance(liq, float)
            self.assertIn(hour, range(24))
        
        # Deben estar ordenadas de mayor a menor liquidez
        liquidities = [liq for _, liq in best_hours]
        self.assertEqual(liquidities, sorted(liquidities, reverse=True),
                        "Horas deben estar ordenadas por liquidez descendente")
        
        # La mejor hora debe tener liquidez alta
        best_hour, best_liq = best_hours[0]
        self.assertGreater(best_liq, 0.9, 
                          "La mejor hora debe tener liquidez > 0.9")
    
    def test_hour_affects_impact(self):
        """Test que la hora afecte el impacto."""
        price = 50000
        order_size = 500000  # $500k para que sea notable
        
        # Hora de alta liquidez
        high_liq_impact = self.model.calculate_impact(
            order_size_usd=order_size,
            price=price,
            hour_utc=14,  # US market peak
        )
        
        # Hora de baja liquidez
        low_liq_impact = self.model.calculate_impact(
            order_size_usd=order_size,
            price=price,
            hour_utc=3,  # Asia noche
        )
        
        self.assertGreater(low_liq_impact['total_impact_pct'],
                          high_liq_impact['total_impact_pct'],
                          "Baja liquidez debe tener más impacto")
        
        # Verificar que el factor de liquidez se reporte correctamente
        self.assertGreater(high_liq_impact['liquidity_factor'],
                          low_liq_impact['liquidity_factor'])
    
    def test_estimate_optimal_order_size(self):
        """Test estimación de tamaño óptimo."""
        result = self.model.estimate_optimal_order_size(
            available_usd=1_000_000,
            price=50000,
            hour_utc=14,
            max_impact_pct=0.003,  # 30 bps
        )
        
        # Debe retornar las claves esperadas
        expected_keys = [
            'optimal_size_usd', 'optimal_size_units', 'expected_impact_pct',
            'expected_impact_usd', 'capital_utilization', 'liquidity_factor',
            'recommendation'
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Falta clave: {key}")
        
        # El impacto esperado debe estar bajo el máximo
        self.assertLessEqual(result['expected_impact_pct'], 0.003,
                            "Impacto debe estar bajo el límite")
        
        # Capital utilization debe ser razonable
        self.assertGreater(result['capital_utilization'], 0)
        self.assertLessEqual(result['capital_utilization'], 1.0)
    
    def test_zero_order_returns_zero_impact(self):
        """Test que orden de 0 retorne impacto 0."""
        result = self.model.calculate_impact(
            order_size_usd=0,
            price=50000,
            hour_utc=14,
        )
        
        self.assertEqual(result['total_impact_pct'], 0.0)
        self.assertEqual(result['total_impact_usd'], 0.0)
    
    def test_execution_price_calculation(self):
        """Test cálculo de precio de ejecución."""
        price = 50000
        impact = 0.01  # 1%
        
        buy_price = self.model.calculate_execution_price('buy', price, impact)
        sell_price = self.model.calculate_execution_price('sell', price, impact)
        
        # Buy debe ser más alto (pagamos más)
        self.assertEqual(buy_price, 50500)  # 50000 * 1.01
        
        # Sell debe ser más bajo (recibimos menos)
        self.assertEqual(sell_price, 49500)  # 50000 * 0.99


def run_tests():
    """Ejecutar tests y mostrar resumen."""
    print("=" * 60)
    print("ÁREA 5: Market Impact Model Crypto Tests")
    print("=" * 60)
    
    # Crear suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestArea5MarketImpactCrypto)
    
    # Ejecutar con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN:")
    print(f"  Tests ejecutados: {result.testsRun}")
    print(f"  Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Fallidos: {len(result.failures)}")
    print(f"  Errores: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ÁREA 5: TODOS LOS TESTS PASARON")
    else:
        print("\n❌ ÁREA 5: HAY TESTS FALLIDOS")
        for test, trace in result.failures + result.errors:
            print(f"\n  - {test}: {trace.split(chr(10))[0]}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
