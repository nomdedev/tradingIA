#!/usr/bin/env python3
"""
Tests para el sistema de backtesting
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtester_core import BacktesterCore


class TestBacktesterCore:
    """Tests para BacktesterCore"""

    def setup_method(self):
        """Configurar datos de prueba"""
        self.backtester = BacktesterCore()

        # Generar datos de prueba
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)

        self.test_data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'Volume': np.random.randint(1000, 10000, 1000),
            'High': 100 + np.cumsum(np.random.randn(1000) * 0.5) + 1,
            'Low': 100 + np.cumsum(np.random.randn(1000) * 0.5) - 1,
            'Open': 100 + np.cumsum(np.random.randn(1000) * 0.5) + 0.1
        }, index=dates)

    def test_initialization(self):
        """Test inicialización del backtester"""
        assert isinstance(self.backtester, BacktesterCore)

    def test_run_backtest_basic(self):
        """Test ejecución básica de backtest"""
        def dummy_strategy(data):
            # Estrategia simple: comprar si precio > media
            mean_price = data['Close'].mean()
            signals = []
            for price in data['Close']:
                if price > mean_price:
                    signals.append(1)  # Buy
                else:
                    signals.append(-1)  # Sell
            return pd.Series(signals, index=data.index)

        # Ejecutar backtest
        results = self.backtester.run_backtest(
            self.test_data,
            dummy_strategy,
            initial_capital=10000
        )

        # Verificar que se obtuvieron resultados
        assert results is not None
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results

    def test_backtest_with_invalid_data(self):
        """Test manejo de datos inválidos"""
        invalid_data = pd.DataFrame()  # DataFrame vacío

        def dummy_strategy(data):
            return pd.Series([1] * len(data), index=data.index)

        # Debería manejar el error gracefully
        with pytest.raises((ValueError, KeyError)):
            self.backtester.run_backtest(invalid_data, dummy_strategy)

    def test_backtest_metrics_calculation(self):
        """Test cálculo correcto de métricas"""
        def constant_return_strategy(data):
            # Estrategia que siempre retorna 1 (buy)
            return pd.Series([1] * len(data), index=data.index)

        results = self.backtester.run_backtest(
            self.test_data,
            constant_return_strategy,
            initial_capital=10000
        )

        # Verificar métricas básicas
        assert isinstance(results['total_return'], (int, float))
        assert isinstance(results['sharpe_ratio'], (int, float))
        assert results['total_return'] >= -1  # Puede ser negativo
        assert results['max_drawdown'] >= 0  # Drawdown siempre positivo

    def test_multiple_runs_consistency(self):
        """Test consistencia en múltiples ejecuciones"""
        def deterministic_strategy(data):
            # Estrategia determinística basada en índice
            signals = []
            for i, row in enumerate(data.iterrows()):
                if i % 2 == 0:
                    signals.append(1)
                else:
                    signals.append(-1)
            return pd.Series(signals, index=data.index)

        # Ejecutar múltiples veces
        results1 = self.backtester.run_backtest(self.test_data, deterministic_strategy)
        results2 = self.backtester.run_backtest(self.test_data, deterministic_strategy)

        # Los resultados deberían ser consistentes (mismos datos de entrada)
        assert abs(results1['total_return'] - results2['total_return']) < 0.001

        self.assertIsInstance(results, dict)
        self.assertIn('best_params', results)
        self.assertIn('performance', results)
        self.assertIn('stability_score', results)

    def test_stability_analysis(self):
        """Test análisis de estabilidad"""
        # Parámetros simulados por período
        period_params = [
            {'threshold': 1.0},
            {'threshold': 1.1},
            {'threshold': 0.9},
            {'threshold': 1.2}
        ]

        stability = self.optimizer._analyze_stability(period_params)
        self.assertIsInstance(stability, float)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)

class TestMonteCarloSimulator(unittest.TestCase):
    """Tests para MonteCarloSimulator"""

    def setUp(self):
        """Configurar datos de prueba"""
        self.simulator = MonteCarloSimulator()

        # Generar retornos de prueba
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)  # 1 año de retornos

    def test_initialization(self):
        """Test inicialización del simulador"""
        self.assertIsInstance(self.simulator, MonteCarloSimulator)
        self.assertEqual(self.simulator.n_simulations, 1000)
        self.assertEqual(self.simulator.confidence_level, 0.95)

    def test_monte_carlo_simulation(self):
        """Test simulación Monte Carlo"""
        results = self.simulator.simulate(self.returns, n_simulations=100)

        self.assertIsInstance(results, dict)
        self.assertIn('VaR_95', results)
        self.assertIn('CVaR_95', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('simulations', results)

        # Verificar que VaR y CVaR sean negativos (pérdidas)
        self.assertLess(results['VaR_95'], 0)
        self.assertLess(results['CVaR_95'], 0)

    def test_risk_metrics_calculation(self):
        """Test cálculo de métricas de riesgo"""
        portfolio_returns = np.array([0.01, -0.02, 0.005, -0.015, 0.008])

        metrics = self.simulator._calculate_risk_metrics(portfolio_returns)

        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('volatility', metrics)

        # Sharpe ratio debería ser un número
        self.assertIsInstance(metrics['sharpe_ratio'], (int, float))

    def test_distribution_analysis(self):
        """Test análisis de distribución"""
        simulated_returns = np.random.normal(0.001, 0.02, (100, 252))

        analysis = self.simulator._analyze_distribution(simulated_returns)

        self.assertIn('mean', analysis)
        self.assertIn('std', analysis)
        self.assertIn('skewness', analysis)
        self.assertIn('kurtosis', analysis)
        self.assertIn('worst_case', analysis)
        self.assertIn('best_case', analysis)

class TestBacktestEngine(unittest.TestCase):
    """Tests para BacktestEngine"""

    def setUp(self):
        """Configurar datos de prueba"""
        self.engine = BacktestEngine()

        # Generar datos de mercado sintéticos
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)

        base_price = 100
        prices = []
        for i in range(500):
            change = np.random.normal(0.001, 0.02)
            base_price *= (1 + change)
            prices.append(base_price)

        self.market_data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 500),
            'High': np.array(prices) * 1.01,
            'Low': np.array(prices) * 0.99,
            'Open': np.array(prices) * 1.002
        }, index=dates)

    def test_initialization(self):
        """Test inicialización del engine"""
        self.assertIsInstance(self.engine, BacktestEngine)
        self.assertEqual(self.engine.initial_capital, 10000)
        self.assertEqual(self.engine.commission, 0.001)

    def test_synthetic_data_generation(self):
        """Test generación de datos sintéticos"""
        synthetic = self.engine._generate_synthetic_data('BTC/USD', periods=100)

        self.assertIsInstance(synthetic, pd.DataFrame)
        self.assertEqual(len(synthetic), 100)
        self.assertIn('Close', synthetic.columns)
        self.assertIn('Volume', synthetic.columns)

    def test_backtest_execution(self):
        """Test ejecución de backtest"""
        def simple_strategy(data):
            """Estrategia simple: comprar si precio > media"""
            ma = data['Close'].rolling(20).mean()
            if len(ma) < 20:
                return 0  # HOLD

            if data['Close'].iloc[-1] > ma.iloc[-1]:
                return 1  # BUY
            else:
                return 2  # SELL

        results = self.engine.run_backtest(
            self.market_data,
            simple_strategy,
            symbol='TEST/USD'
        )

        self.assertIsInstance(results, dict)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('win_rate', results)
        self.assertIn('trades', results)

    def test_multi_asset_backtest(self):
        """Test backtest multi-activo"""
        symbols = ['BTC/USD', 'ETH/USD']

        def multi_strategy(data, symbol):
            return 1 if symbol == 'BTC/USD' else 2  # BUY BTC, SELL ETH

        results = self.engine.run_multi_asset_backtest(
            symbols,
            multi_strategy,
            periods=200
        )

        self.assertIsInstance(results, dict)
        self.assertIn('portfolio_return', results)
        self.assertIn('asset_returns', results)
        for symbol in symbols:
            self.assertIn(symbol, results['asset_returns'])

    def test_performance_calculation(self):
        """Test cálculo de rendimiento"""
        # Simular trades
        trades = [
            {'entry_price': 100, 'exit_price': 110, 'size': 1, 'pnl': 10},
            {'entry_price': 110, 'exit_price': 105, 'size': 1, 'pnl': -5},
            {'entry_price': 105, 'exit_price': 115, 'size': 1, 'pnl': 10}
        ]

        performance = self.engine._calculate_performance(trades, 1000)

        self.assertIn('total_return', performance)
        self.assertIn('win_rate', performance)
        self.assertIn('avg_win', performance)
        self.assertIn('avg_loss', performance)
        self.assertIn('profit_factor', performance)

        # Verificar cálculos básicos
        self.assertEqual(performance['total_trades'], 3)
        self.assertEqual(performance['winning_trades'], 2)
        self.assertEqual(performance['win_rate'], 2/3)

if __name__ == '__main__':
    unittest.main()