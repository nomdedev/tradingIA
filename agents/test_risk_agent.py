#!/usr/bin/env python3
"""
TEST SUITE: Moon Dev Risk Agent
==============================

Suite de pruebas para validar el funcionamiento del MoonDevRiskAgent
y SafeTradingWrapper en el contexto de trading de acciones (SPY).

Pruebas incluidas:
- Validación de parámetros de riesgo
- Evaluación de trades con diferentes escenarios
- Position sizing dinámico
- Stop losses adaptativos
- Integración con SafeTradingWrapper
- Manejo de errores y edge cases

Autor: Moon Dev AI Agents (Testing Suite)
"""

import sys
import os
import unittest
from unittest.mock import Mock

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from moondev_risk_agent import MoonDevRiskAgent
from safe_trading_wrapper import SafeTradingWrapper


class TestMoonDevRiskAgent(unittest.TestCase):
    """Pruebas unitarias para MoonDevRiskAgent"""

    def setUp(self):
        """Configurar entorno de pruebas"""
        self.risk_config = {
            'max_drawdown': 0.15,
            'max_portfolio_heat': 0.30,
            'max_risk_per_trade': 0.05,
            'volatility_lookback': 10,
            'initial_balance': 10000.0
        }
        self.risk_agent = MoonDevRiskAgent(**self.risk_config)

        # Datos de mercado de ejemplo
        self.market_data = {
            'Close': 450.0,
            'High': 455.0,
            'Low': 445.0,
            'ATR': 5.0
        }

    def test_initialization(self):
        """Probar inicialización correcta del agente"""
        self.assertEqual(self.risk_agent.max_drawdown, 0.15)
        self.assertEqual(self.risk_agent.max_portfolio_heat, 0.30)
        self.assertEqual(self.risk_agent.max_risk_per_trade, 0.05)
        self.assertEqual(len(self.risk_agent.price_history), 0)
        self.assertEqual(self.risk_agent.total_evaluations, 0)

    def test_hold_approval(self):
        """Probar que HOLD siempre se aprueba"""
        portfolio_state = {
            'balance': 5000.0,
            'shares': 10,
            'net_worth': 9500.0,
            'current_drawdown': 0.05
        }

        trade_proposal = {
            'action': 0,  # HOLD
            'price': 450.0,
            'size': 0,
            'confidence': 0.5
        }

        result = self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        self.assertTrue(result['approved'])
        self.assertEqual(result['adjusted_size'], 0)
        self.assertIsNone(result['stop_loss'])
        self.assertEqual(result['risk_score'], 0.0)

    def test_sell_approval(self):
        """Probar que SELL siempre se aprueba"""
        portfolio_state = {
            'balance': 5000.0,
            'shares': 10,
            'net_worth': 9500.0,
            'current_drawdown': 0.05
        }

        trade_proposal = {
            'action': 2,  # SELL
            'price': 450.0,
            'size': 5,
            'confidence': 0.5
        }

        result = self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        self.assertTrue(result['approved'])
        self.assertEqual(result['adjusted_size'], 5)  # Mantiene el size solicitado

    def test_buy_with_low_risk(self):
        """Probar BUY con bajo riesgo"""
        portfolio_state = {
            'balance': 9000.0,
            'shares': 0,
            'net_worth': 9000.0,
            'current_drawdown': 0.02
        }

        trade_proposal = {
            'action': 1,  # BUY
            'price': 450.0,
            'size': 10,
            'confidence': 0.8
        }

        result = self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        self.assertTrue(result['approved'])
        self.assertGreater(result['adjusted_size'], 0)
        self.assertIsNotNone(result['stop_loss'])
        self.assertLess(result['risk_score'], 0.7)

    def test_buy_blocked_by_drawdown(self):
        """Probar que BUY se bloquea con alto drawdown"""
        portfolio_state = {
            'balance': 9000.0,
            'shares': 0,
            'net_worth': 8500.0,  # 15% drawdown
            'current_drawdown': 0.15
        }

        trade_proposal = {
            'action': 1,  # BUY
            'price': 450.0,
            'size': 10,
            'confidence': 0.5
        }

        result = self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        self.assertFalse(result['approved'])
        self.assertEqual(result['adjusted_size'], 0)
        self.assertIn('Drawdown', result['reason'])

    def test_position_sizing(self):
        """Probar cálculo de position sizing"""
        portfolio_state = {
            'balance': 10000.0,
            'shares': 0,
            'net_worth': 10000.0,
            'current_drawdown': 0.0
        }

        # Agregar datos de volatilidad
        self.risk_agent.price_history = [440.0, 445.0, 450.0, 448.0, 452.0]

        trade_proposal = {
            'action': 1,
            'price': 450.0,
            'size': 100,  # Size grande
            'confidence': 0.7
        }

        result = self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        # Debería ajustar el size hacia abajo por riesgo
        self.assertLess(result['adjusted_size'], 100)
        self.assertGreater(result['adjusted_size'], 0)

    def test_stop_loss_calculation(self):
        """Probar cálculo de stop loss dinámico"""
        portfolio_state = {
            'balance': 10000.0,
            'shares': 0,
            'net_worth': 10000.0,
            'current_drawdown': 0.0
        }

        trade_proposal = {
            'action': 1,
            'price': 450.0,
            'size': 10,
            'confidence': 0.5
        }

        result = self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        self.assertIsNotNone(result['stop_loss'])
        self.assertLess(result['stop_loss'], 450.0)  # Stop loss por debajo del precio de entrada

    def test_portfolio_heat_calculation(self):
        """Probar cálculo de portfolio heat"""
        portfolio_state = {
            'balance': 5000.0,
            'shares': 10,
            'net_worth': 9500.0,
            'current_drawdown': 0.0
        }

        heat = self.risk_agent._calculate_portfolio_heat(portfolio_state, 450.0)
        self.assertGreater(heat, 0)
        self.assertLess(heat, 1)  # Debería ser un porcentaje

    def test_risk_stats_tracking(self):
        """Probar seguimiento de estadísticas de riesgo"""
        # Ejecutar algunas evaluaciones
        portfolio_state = {
            'balance': 10000.0,
            'shares': 0,
            'net_worth': 10000.0,
            'current_drawdown': 0.0
        }

        # Trade aprobado
        trade_proposal = {'action': 1, 'price': 450.0, 'size': 5, 'confidence': 0.5}
        self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        # Trade bloqueado
        portfolio_state['current_drawdown'] = 0.20  # Alto drawdown
        self.risk_agent.evaluate_trade_risk(trade_proposal, portfolio_state, self.market_data)

        final_stats = self.risk_agent.get_risk_stats()

        self.assertEqual(final_stats['total_evaluations'], 2)
        self.assertEqual(final_stats['trades_blocked'], 1)
        self.assertGreater(final_stats['block_rate'], 0)


class TestSafeTradingWrapper(unittest.TestCase):
    """Pruebas unitarias para SafeTradingWrapper"""

    def setUp(self):
        """Configurar entorno de pruebas"""
        # Mock trading agent
        self.mock_agent = Mock()
        self.mock_agent.__class__.__name__ = "MockTradingAgent"

        # Configuración de riesgo
        self.risk_config = {
            'max_drawdown': 0.10,
            'max_portfolio_heat': 0.20,
            'max_risk_per_trade': 0.03,
            'volatility_lookback': 5,
            'initial_balance': 10000.0
        }

        self.wrapper = SafeTradingWrapper(
            trading_agent=self.mock_agent,
            risk_config=self.risk_config,
            fallback_mode=True
        )

    def test_initialization(self):
        """Probar inicialización del wrapper"""
        self.assertEqual(self.wrapper.trading_agent, self.mock_agent)
        self.assertIsInstance(self.wrapper.risk_agent, MoonDevRiskAgent)
        self.assertTrue(self.wrapper.fallback_mode)
        self.assertEqual(self.wrapper.total_trades_processed, 0)

    def test_action_normalization(self):
        """Probar normalización de formatos de acción"""
        # Formato dict
        action_dict = {'action': 1, 'size': 10, 'confidence': 0.8}
        normalized = self.wrapper._normalize_action_format(action_dict)
        self.assertEqual(normalized, action_dict)

        # Formato tuple
        action_tuple = (1, 10, 0.8)
        normalized = self.wrapper._normalize_action_format(action_tuple)
        self.assertEqual(normalized['action'], 1)
        self.assertEqual(normalized['size'], 10)
        self.assertEqual(normalized['confidence'], 0.8)

        # Formato int
        action_int = 2
        normalized = self.wrapper._normalize_action_format(action_int)
        self.assertEqual(normalized['action'], 2)
        self.assertEqual(normalized['size'], 0)

    def test_get_action_with_risk_approval(self):
        """Probar get_action con aprobación de riesgo"""
        # Configurar mock para devolver acción BUY
        self.mock_agent.get_action.return_value = {'action': 1, 'size': 10, 'confidence': 0.7}

        portfolio_state = {
            'balance': 10000.0,
            'shares': 0,
            'net_worth': 10000.0,
            'current_drawdown': 0.0
        }

        market_data = {'Close': 450.0, 'High': 455.0, 'Low': 445.0}

        result = self.wrapper.get_action(portfolio_state, market_data)

        self.assertIn('risk_approved', result)
        self.assertIn('risk_reason', result)
        self.assertEqual(result['action'], 1)  # BUY

    def test_get_action_with_risk_block(self):
        """Probar get_action con bloqueo de riesgo"""
        # Configurar mock para devolver acción BUY
        self.mock_agent.get_action.return_value = {'action': 1, 'size': 10, 'confidence': 0.5}

        portfolio_state = {
            'balance': 10000.0,
            'shares': 0,
            'net_worth': 8500.0,  # 15% drawdown
            'current_drawdown': 0.15
        }

        market_data = {'Close': 450.0}

        result = self.wrapper.get_action(portfolio_state, market_data)

        self.assertEqual(result['action'], 0)  # HOLD (bloqueado)
        self.assertFalse(result['risk_approved'])
        self.assertIn('Drawdown', result['risk_reason'])

    def test_fallback_mode(self):
        """Probar modo fallback"""
        # Configurar wrapper sin fallback
        wrapper_no_fallback = SafeTradingWrapper(
            trading_agent=self.mock_agent,
            risk_config=self.risk_config,
            fallback_mode=False
        )

        # Hacer que el mock lance una excepción
        self.mock_agent.get_action.side_effect = Exception("Mock error")

        portfolio_state = {'balance': 10000.0, 'shares': 0, 'net_worth': 10000.0}

        # Debería lanzar excepción sin fallback
        with self.assertRaises(Exception):
            wrapper_no_fallback.get_action(portfolio_state)

        # Con fallback debería devolver HOLD
        result = self.wrapper.get_action(portfolio_state)
        self.assertEqual(result['action'], 0)  # HOLD
        self.assertTrue(result.get('fallback', False))

    def test_wrapper_stats(self):
        """Probar obtención de estadísticas del wrapper"""
        # Ejecutar algunas acciones
        self.mock_agent.get_action.return_value = {'action': 0, 'size': 0, 'confidence': 0.5}

        portfolio_state = {'balance': 10000.0, 'shares': 0, 'net_worth': 10000.0}

        for _ in range(3):
            self.wrapper.get_action(portfolio_state)

        stats = self.wrapper.get_wrapper_stats()

        self.assertEqual(stats['total_trades_processed'], 3)
        self.assertIn('block_rate', stats)
        self.assertIn('error_rate', stats)

    def test_reset_functionality(self):
        """Probar funcionalidad de reset"""
        # Ejecutar algunas acciones
        self.mock_agent.get_action.return_value = {'action': 1, 'size': 5, 'confidence': 0.5}
        portfolio_state = {'balance': 10000.0, 'shares': 0, 'net_worth': 10000.0}

        self.wrapper.get_action(portfolio_state)

        # Verificar que hay estadísticas
        self.assertGreater(self.wrapper.total_trades_processed, 0)

        # Reset
        self.wrapper.reset()

        # Verificar reset
        self.assertEqual(self.wrapper.total_trades_processed, 0)
        stats = self.wrapper.get_wrapper_stats()
        self.assertEqual(stats['total_trades_processed'], 0)


class TestIntegration(unittest.TestCase):
    """Pruebas de integración entre componentes"""

    def test_full_pipeline(self):
        """Probar pipeline completo de trading con riesgo"""
        # Crear un mock agent simple
        class SimpleTradingAgent:
            def __init__(self):
                self.call_count = 0

            def get_action(self, state, market_data=None):
                self.call_count += 1
                # Estrategia simple: BUY en la primera llamada, HOLD después
                if self.call_count == 1:
                    return {'action': 1, 'size': 10, 'confidence': 0.8}
                else:
                    return {'action': 0, 'size': 0, 'confidence': 0.5}

        trading_agent = SimpleTradingAgent()

        # Envolver con SafeTradingWrapper
        wrapper = SafeTradingWrapper(
            trading_agent=trading_agent,
            risk_config={'max_drawdown': 0.10, 'max_portfolio_heat': 0.20,
                        'max_risk_per_trade': 0.05, 'initial_balance': 10000.0}
        )

        # Simular datos de mercado
        market_data = {'Close': 450.0, 'High': 455.0, 'Low': 445.0, 'ATR': 3.0}

        # Estado inicial del portfolio
        portfolio_state = {
            'balance': 10000.0,
            'shares': 0,
            'net_worth': 10000.0,
            'current_drawdown': 0.0
        }

        # Ejecutar primera acción (debería ser BUY aprobado)
        result1 = wrapper.get_action(portfolio_state, market_data)
        self.assertEqual(result1['action'], 1)  # BUY
        self.assertTrue(result1['risk_approved'])

        # Actualizar portfolio state simulado
        portfolio_state['shares'] = result1['size']
        portfolio_state['balance'] -= result1['size'] * 450.0 * 1.001  # Costo con spread
        portfolio_state['net_worth'] = portfolio_state['balance'] + portfolio_state['shares'] * 450.0

        # Ejecutar segunda acción (debería ser HOLD)
        result2 = wrapper.get_action(portfolio_state, market_data)
        self.assertEqual(result2['action'], 0)  # HOLD

        # Verificar estadísticas
        stats = wrapper.get_wrapper_stats()
        self.assertEqual(stats['total_trades_processed'], 2)
        self.assertGreaterEqual(stats['trades_blocked_by_risk'], 0)


def run_tests():
    """Ejecutar todas las pruebas"""
    print("\n" + "="*70)
    print("🧪 EJECUTANDO TEST SUITE: Moon Dev Risk Agent")
    print("="*70)

    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar clases de prueba
    suite.addTests(loader.loadTestsFromTestCase(TestMoonDevRiskAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestSafeTradingWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resultados
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ TODAS LAS PRUEBAS PASARON")
        print(f"   Tests ejecutados: {result.testsRun}")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
        print(f"   Tests ejecutados: {result.testsRun}")
        print(f"   Fallos: {len(result.failures)}")
        print(f"   Errores: {len(result.errors)}")

    print("="*70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)