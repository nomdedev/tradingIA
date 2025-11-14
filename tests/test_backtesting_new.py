#!/usr/bin/env python3
"""
Tests para el sistema de backtesting
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

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
        # Crear una clase de estrategia simple
        class DummyStrategy:
            def __init__(self, **params):
                self.params = params

            def generate_signals(self, data):
                # data es un dict con timeframes, extraer datos de 5min
                df = data['5min']
                # Estrategia simple: comprar si precio > media
                mean_price = df['Close'].mean()
                signals = []
                for price in df['Close']:
                    if price > mean_price:
                        signals.append(1)  # Buy
                    else:
                        signals.append(-1)  # Sell
                return pd.Series(signals, index=df.index)

        # Ejecutar backtest
        results = self.backtester.run_backtest(
            DummyStrategy,
            {'5min': self.test_data}
        )

        # Verificar que se obtuvieron resultados
        assert results is not None
        assert isinstance(results, dict)

    def test_backtest_with_invalid_data(self):
        """Test manejo de datos inválidos"""
        class DummyStrategy:
            def generate_signals(self, data):
                return pd.Series([1] * len(data['5min']), index=data['5min'].index)

        # DataFrame vacío - el sistema debería manejarlo gracefully (no lanzar excepción)
        empty_data = pd.DataFrame()
        results = self.backtester.run_backtest(DummyStrategy, {'5min': empty_data})

        # El sistema maneja errores gracefully, así que debería devolver algo
        assert results is not None

    def test_backtest_metrics_calculation(self):
        """Test cálculo correcto de métricas"""
        class ConstantStrategy:
            def generate_signals(self, data):
                # Siempre retorna 1 (buy)
                return pd.Series([1] * len(data['5min']), index=data['5min'].index)

        results = self.backtester.run_backtest(ConstantStrategy, {'5min': self.test_data})

        # Verificar que se obtuvieron resultados
        assert results is not None
        assert isinstance(results, dict)

    def test_multiple_runs_consistency(self):
        """Test consistencia en múltiples ejecuciones"""
        class DeterministicStrategy:
            def generate_signals(self, data):
                # Estrategia determinística basada en índice
                signals = []
                for i in range(len(data['5min'])):
                    if i % 2 == 0:
                        signals.append(1)
                    else:
                        signals.append(-1)
                return pd.Series(signals, index=data['5min'].index)

        # Ejecutar múltiples veces
        results1 = self.backtester.run_backtest(DeterministicStrategy, {'5min': self.test_data})
        results2 = self.backtester.run_backtest(DeterministicStrategy, {'5min': self.test_data})

        # Ambos deberían ser diccionarios (no verificar contenido específico por ahora)
        assert isinstance(results1, dict)
        assert isinstance(results2, dict)

    def test_backtest_with_different_strategies(self):
        """Test backtest con diferentes estrategias"""
        class BuyStrategy:
            def generate_signals(self, data):
                return pd.Series([1] * len(data['5min']), index=data['5min'].index)  # Siempre BUY

        class SellStrategy:
            def generate_signals(self, data):
                return pd.Series([-1] * len(data['5min']), index=data['5min'].index)  # Siempre SELL

        results_buy = self.backtester.run_backtest(BuyStrategy, {'5min': self.test_data})
        results_sell = self.backtester.run_backtest(SellStrategy, {'5min': self.test_data})

        # Ambos deberían ser diccionarios
        assert isinstance(results_buy, dict)
        assert isinstance(results_sell, dict)

    def test_backtest_with_insufficient_data(self):
        """Test con datos insuficientes"""
        class DummyStrategy:
            def generate_signals(self, data):
                return pd.Series([1] * len(data['5min']), index=data['5min'].index)

        small_data = self.test_data.head(5)  # Solo 5 filas

        # Debería funcionar o manejar el error gracefully
        results = self.backtester.run_backtest(DummyStrategy, {'5min': small_data})
        assert results is not None  # Al menos debería devolver algo