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

from core.execution.backtester_core import BacktesterCore


class TestBacktesterCore:
    """Tests para BacktesterCore"""

    def setup_method(self):
        """Configurar datos de prueba"""
        self.backtester = BacktesterCore()

        # Generar datos de prueba
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)

        self.test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'volume': np.random.randint(1000, 10000, 1000),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.5) + 1,
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.5) - 1,
            'open': 100 + np.cumsum(np.random.randn(1000) * 0.5) + 0.1
        }, index=dates)

    def test_initialization(self):
        """Test inicialización del backtester"""
        assert isinstance(self.backtester, BacktesterCore)

    def test_run_backtest_basic(self):
        """Test ejecución básica de backtest"""
        class DummyStrategy:
            def generate_signals(self, df_multi_tf):
                # Estrategia simple: comprar si precio > media
                data = df_multi_tf['5min']
                signals = pd.DataFrame(index=data.index)
                signals['entries'] = False
                signals['exits'] = False
                signals['signals'] = 0
                
                mean_price = data['close'].mean()
                buy_mask = data['close'] > mean_price
                sell_mask = data['close'] < mean_price
                
                signals.loc[buy_mask, 'entries'] = True
                signals.loc[buy_mask, 'signals'] = 1
                signals.loc[sell_mask, 'exits'] = True
                signals.loc[sell_mask, 'signals'] = -1
                
                return signals
            
            def get_parameters(self):
                return {}

        # Ejecutar backtest
        results = self.backtester.run_backtest(
            DummyStrategy,
            self.test_data,
            {}
        )

        # Verificar que se obtuvieron resultados
        assert results is not None
        assert 'metrics' in results
        assert 'total_return' in results['metrics']
        assert 'sharpe' in results['metrics']
        assert 'max_dd' in results['metrics']

    def test_backtest_with_invalid_data(self):
        """Test manejo de datos inválidos"""
        class DummyStrategy:
            def generate_signals(self, df_multi_tf):
                data = df_multi_tf['5min']
                signals = pd.DataFrame(index=data.index)
                signals['entries'] = True  # Siempre comprar
                signals['exits'] = False
                return signals
            
            def get_parameters(self):
                return {}

        invalid_data = pd.DataFrame()  # DataFrame vacío

        results = self.backtester.run_backtest(DummyStrategy, invalid_data, {})

        # Debería manejar el error gracefully y devolver un dict con error
        assert isinstance(results, dict)
        assert 'error' in results
        assert 'Empty dataset' in results['error']

    def test_backtest_metrics_calculation(self):
        """Test cálculo correcto de métricas"""
        class ConstantReturnStrategy:
            def generate_signals(self, df_multi_tf):
                # Estrategia que siempre retorna 1 (buy)
                data = df_multi_tf['5min']
                signals = pd.DataFrame(index=data.index)
                signals['entries'] = True  # Siempre comprar
                signals['exits'] = False
                return signals
            
            def get_parameters(self):
                return {}

        results = self.backtester.run_backtest(
            ConstantReturnStrategy,
            self.test_data,
            {}
        )

        # Verificar métricas básicas
        assert isinstance(results['metrics']['total_return'], (int, float))
        assert isinstance(results['metrics']['sharpe'], (int, float))
        assert results['metrics']['total_return'] >= -1  # Puede ser negativo
        assert results['metrics']['max_dd'] >= 0  # Drawdown siempre positivo

    def test_multiple_runs_consistency(self):
        """Test consistencia en múltiples ejecuciones"""
        class DeterministicStrategy:
            def generate_signals(self, df_multi_tf):
                # Estrategia determinística basada en índice
                data = df_multi_tf['5min']
                signals = pd.DataFrame(index=data.index)
                signals['entries'] = False
                signals['exits'] = False
                
                for i in range(len(data)):
                    if i % 2 == 0:
                        signals.iloc[i, signals.columns.get_loc('entries')] = True
                    else:
                        signals.iloc[i, signals.columns.get_loc('exits')] = True
                
                return signals
            
            def get_parameters(self):
                return {}

        # Ejecutar múltiples veces
        results1 = self.backtester.run_backtest(DeterministicStrategy, self.test_data, {})
        results2 = self.backtester.run_backtest(DeterministicStrategy, self.test_data, {})

        # Los resultados deberían ser consistentes (mismos datos de entrada)
        assert abs(results1['metrics']['total_return'] - results2['metrics']['total_return']) < 0.001

if __name__ == '__main__':
    pytest.main()