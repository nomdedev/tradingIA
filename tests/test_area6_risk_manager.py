"""
Test ÁREA 6: Risk Manager Mejorado.

Valida que el Risk Manager tenga:
1. Total drawdown desde high water mark (no solo diario)
2. Tracking de pérdidas consecutivas
3. VaR/CVaR básico
4. Correlation-adjusted risk
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from datetime import date
import numpy as np


class TestArea6RiskManager(unittest.TestCase):
    """Tests para RiskManager mejorado."""
    
    def setUp(self):
        """Setup para cada test."""
        from core.risk.risk_manager import RiskManager
        self.rm = RiskManager({
            'max_daily_drawdown': 0.05,
            'max_total_drawdown': 0.15,
            'max_consecutive_losses': 5,
        })
        self.rm.initialize(initial_equity=100000)
    
    def test_high_water_mark_tracking(self):
        """Test que se trackee el high water mark correctamente."""
        # Inicialmente HWM = equity inicial
        self.assertEqual(self.rm.high_water_mark, 100000)
        
        # Subir equity -> HWM sube
        self.rm.update_state(110000, date(2024, 1, 1))
        self.assertEqual(self.rm.high_water_mark, 110000)
        
        # Bajar equity -> HWM no baja
        self.rm.update_state(105000, date(2024, 1, 1))
        self.assertEqual(self.rm.high_water_mark, 110000)
        
        # Subir más -> HWM sube
        self.rm.update_state(115000, date(2024, 1, 2))
        self.assertEqual(self.rm.high_water_mark, 115000)
    
    def test_total_drawdown_calculation(self):
        """Test cálculo de total drawdown desde HWM."""
        # Simular subida y luego caída
        self.rm.update_state(120000, date(2024, 1, 1))  # HWM = 120k
        self.rm.update_state(100000, date(2024, 1, 2))  # Caída 20k
        
        total_dd = self.rm.get_total_drawdown()
        expected_dd = (120000 - 100000) / 120000  # 16.67%
        
        self.assertAlmostEqual(total_dd, expected_dd, places=4)
        self.assertAlmostEqual(total_dd, 0.1667, places=3)
    
    def test_total_drawdown_triggers_halt(self):
        """Test que total drawdown excesivo active halt."""
        # HWM = 100k, caer a 80k = 20% drawdown
        self.rm.update_state(80000, date(2024, 1, 1))
        
        result = self.rm.check_order({})
        
        self.assertFalse(result['allowed'])
        self.assertIn('Total Drawdown', result['reason'])
        self.assertTrue(self.rm.is_halted)
    
    def test_consecutive_losses_tracking(self):
        """Test tracking de pérdidas consecutivas."""
        # Inicialmente 0
        self.assertEqual(self.rm.consecutive_losses, 0)
        
        # 3 pérdidas
        self.rm.record_trade_result(-100)
        self.rm.record_trade_result(-50)
        self.rm.record_trade_result(-75)
        self.assertEqual(self.rm.consecutive_losses, 3)
        
        # 1 ganancia resetea contador
        self.rm.record_trade_result(200)
        self.assertEqual(self.rm.consecutive_losses, 0)
        
        # Más pérdidas
        self.rm.record_trade_result(-100)
        self.rm.record_trade_result(-100)
        self.assertEqual(self.rm.consecutive_losses, 2)
    
    def test_consecutive_losses_blocks_order(self):
        """Test que muchas pérdidas consecutivas bloqueen órdenes."""
        # 5 pérdidas consecutivas (límite)
        for _ in range(5):
            self.rm.record_trade_result(-100)
        
        result = self.rm.check_order({})
        
        self.assertFalse(result['allowed'])
        self.assertIn('consecutive losses', result['reason'])
    
    def test_var_calculation(self):
        """Test cálculo de VaR."""
        # Agregar returns históricos
        returns = [-0.02, 0.01, -0.03, 0.02, -0.01, 0.015, -0.025, 0.01,
                   -0.015, 0.02, -0.01, 0.005, -0.02, 0.01, -0.01, 0.015,
                   -0.02, 0.01, -0.015, 0.02, -0.025, 0.01, -0.01, 0.015]
        
        for r in returns:
            self.rm.returns_history.append(r)
        
        # Calcular VaR para $10,000
        var = self.rm.calculate_var(10000)
        
        # VaR debe ser positivo y razonable
        self.assertGreater(var, 0)
        self.assertLess(var, 1000)  # < 10% de la posición
    
    def test_cvar_greater_than_var(self):
        """Test que CVaR >= VaR (por definición)."""
        # Agregar returns con cola pesada
        returns = [-0.05, -0.03, -0.02, 0.01, 0.02, -0.01, 0.015, -0.025,
                   -0.04, 0.01, -0.015, 0.02, -0.01, 0.005, -0.02, 0.01,
                   -0.015, 0.02, -0.03, 0.01, -0.02, 0.015, -0.01, 0.02]
        
        for r in returns:
            self.rm.returns_history.append(r)
        
        var = self.rm.calculate_var(10000)
        cvar = self.rm.calculate_cvar(10000)
        
        self.assertGreaterEqual(cvar, var,
                               "CVaR debe ser >= VaR")
    
    def test_correlated_risk_calculation(self):
        """Test cálculo de riesgo correlacionado."""
        # Agregar posición existente
        self.rm.add_position('BTC', {'value': 10000, 'direction': 'long'})
        
        # Calcular riesgo de nueva posición
        new_pos = {'symbol': 'ETH', 'value': 5000, 'direction': 'long'}
        risk = self.rm.calculate_correlated_risk(new_pos)
        
        # Debe tener las claves esperadas
        self.assertIn('correlated_exposure', risk)
        self.assertIn('marginal_risk', risk)
        self.assertIn('recommendation', risk)
        
        # marginal_risk entre 0 y 1
        self.assertGreater(risk['marginal_risk'], 0)
        self.assertLessEqual(risk['marginal_risk'], 1.0)
    
    def test_risk_metrics_returned(self):
        """Test que check_order retorne métricas de riesgo."""
        result = self.rm.check_order({})
        
        self.assertIn('risk_metrics', result)
        metrics = result['risk_metrics']
        
        # Verificar claves importantes
        expected_keys = [
            'current_equity', 'high_water_mark', 'total_drawdown',
            'daily_drawdown', 'consecutive_losses', 'is_halted'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Falta métrica: {key}")
    
    def test_position_size_adjustment(self):
        """Test ajuste de tamaño por riesgo."""
        # Sin drawdown ni pérdidas -> ajuste 1.0
        adjustment = self.rm.get_position_size_adjustment()
        self.assertEqual(adjustment, 1.0)
        
        # Con drawdown 10% 
        self.rm.update_state(120000, date(2024, 1, 1))  # HWM = 120k
        self.rm.update_state(108000, date(2024, 1, 1))  # 10% DD
        
        adjustment = self.rm.get_position_size_adjustment()
        self.assertLess(adjustment, 1.0, "Con drawdown, ajuste debe ser < 1.0")
        self.assertGreater(adjustment, 0.2, "Ajuste mínimo es 0.2")
    
    def test_position_size_adjustment_consecutive_losses(self):
        """Test que pérdidas consecutivas reduzcan tamaño."""
        adjustment_0 = self.rm.get_position_size_adjustment()
        
        # 3 pérdidas consecutivas
        self.rm.record_trade_result(-100)
        self.rm.record_trade_result(-100)
        self.rm.record_trade_result(-100)
        
        adjustment_3 = self.rm.get_position_size_adjustment()
        
        self.assertLess(adjustment_3, adjustment_0,
                       "Con pérdidas consecutivas, tamaño debe reducirse")
    
    def test_initialize_method(self):
        """Test que initialize() configure el estado correctamente."""
        from core.risk.risk_manager import RiskManager
        rm = RiskManager()
        
        rm.initialize(50000)
        
        self.assertEqual(rm.initial_equity, 50000)
        self.assertEqual(rm.high_water_mark, 50000)
        self.assertEqual(rm.current_equity, 50000)
        self.assertEqual(rm.consecutive_losses, 0)
    
    def test_reset_method(self):
        """Test que reset() limpie el estado."""
        # Modificar estado
        self.rm.update_state(120000, date(2024, 1, 1))
        self.rm.record_trade_result(-100)
        self.rm.record_trade_result(-100)
        self.rm.add_position('BTC', {'value': 10000})
        self.rm.is_halted = True
        
        # Reset
        self.rm.reset()
        
        # Verificar estado limpio
        self.assertFalse(self.rm.is_halted)
        self.assertEqual(self.rm.consecutive_losses, 0)
        self.assertEqual(len(self.rm.open_positions), 0)


def run_tests():
    """Ejecutar tests y mostrar resumen."""
    print("=" * 60)
    print("ÁREA 6: Risk Manager Mejorado Tests")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestArea6RiskManager)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("RESUMEN:")
    print(f"  Tests ejecutados: {result.testsRun}")
    print(f"  Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Fallidos: {len(result.failures)}")
    print(f"  Errores: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ÁREA 6: TODOS LOS TESTS PASARON")
    else:
        print("\n❌ ÁREA 6: HAY TESTS FALLIDOS")
        for test, trace in result.failures + result.errors:
            print(f"\n  - {test}")
            print(f"    {trace.split(chr(10))[0]}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
