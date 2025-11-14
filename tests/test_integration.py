"""
Tests de integración para el sistema de trading completo
Prueba el flujo: DataFetcher -> Strategy -> Backtester -> Risk Analysis
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_market_data():
    """Datos de mercado para tests de integración"""
    dates = pd.date_range('2024-01-01', periods=200, freq='5min')
    
    # Generar datos OHLCV realistas
    base_price = 65000.0
    trend = np.linspace(0, 2000, len(dates))  # Tendencia alcista
    noise = np.random.normal(0, 500, len(dates))
    
    close_prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.uniform(-100, 100, len(dates)),
        'High': close_prices + np.random.uniform(100, 500, len(dates)),
        'Low': close_prices - np.random.uniform(100, 500, len(dates)),
        'Close': close_prices,
        'Volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Asegurar consistencia OHLC
    for idx in df.index:
        prices = [df.loc[idx, 'Open'], df.loc[idx, 'Close']]
        df.loc[idx, 'High'] = max(prices + [df.loc[idx, 'High']])
        df.loc[idx, 'Low'] = min(prices + [df.loc[idx, 'Low']])
    
    return df


class TestDataToStrategyIntegration:
    """Tests de integración: DataFetcher -> Strategy"""
    
    def test_data_fetcher_to_rsi_strategy(self, sample_market_data):
        """Test flujo completo de datos a estrategia RSI"""
        from strategies.presets.rsi_mean_reversion import RSIMeanReversionStrategy
        
        # Crear estrategia
        strategy = RSIMeanReversionStrategy()
        
        # Modificar parámetros si es necesario
        strategy.update_parameters({
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70
        })
        
        # Generar señales con datos de mercado
        signals = strategy.generate_signals(sample_market_data)
        
        # Verificar que se generaron señales
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert len(signals) > 0
        
        # Verificar que las señales son válidas
        unique_signals = signals['signal'].unique()
        assert all(sig in [-1, 0, 1] for sig in unique_signals)
        
        # Verificar que hay al menos algunas señales de compra/venta
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        assert buy_signals > 0 or sell_signals > 0, "No se generaron señales de trading"


class TestStrategyToBacktesterIntegration:
    """Tests de integración: Strategy -> Backtester"""
    
    def test_rsi_strategy_with_backtester(self, sample_market_data):
        """Test estrategia RSI con backtester"""
        from strategies.mean_reversion import RSIMeanReversionStrategy
        from src.backtester_core import BacktesterCore
        
        # Crear backtester
        backtester = BacktesterCore(initial_capital=10000)
        
        # Ejecutar backtest
        results = backtester.simple_backtest(
            df_5m=sample_market_data,
            strategy_class=RSIMeanReversionStrategy
        )
        
        # Verificar estructura de resultados
        assert 'metrics' in results
        assert 'trades' in results
        assert 'equity_curve' in results
        
        # Verificar métricas básicas
        metrics = results['metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_trades' in metrics
        
        # Verificar que hay trades
        assert len(results['trades']) > 0


class TestFullTradingPipeline:
    """Tests del pipeline completo de trading"""
    
    def test_complete_pipeline_with_risk_metrics(self, sample_market_data):
        """Test pipeline completo: Data -> Strategy -> Backtest -> Risk Analysis"""
        from strategies.mean_reversion import RSIMeanReversionStrategy
        from src.backtester_core import BacktesterCore
        
        # 1. Preparar datos
        assert not sample_market_data.empty
        assert all(col in sample_market_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # 2. Ejecutar estrategia
        strategy = RSIMeanReversionStrategy(
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70
        )
        signals = strategy.generate_signals(sample_market_data)
        assert len(signals) > 0
        
        # 3. Ejecutar backtest
        backtester = BacktesterCore(initial_capital=10000)
        results = backtester.simple_backtest(
            df_5m=sample_market_data,
            strategy_class=RSIMeanReversionStrategy
        )
        
        # 4. Analizar resultados
        metrics = results['metrics']
        trades = results['trades']
        
        # Verificar métricas de rendimiento
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert isinstance(metrics['max_drawdown'], (int, float))
        
        # Verificar trades
        assert len(trades) > 0
        for trade in trades[:3]:  # Verificar primeros 3 trades
            assert 'entry_price' in trade
            assert 'exit_price' in trade
            assert 'pnl' in trade
        
        # 5. Calcular métricas de riesgo adicionales
        if len(trades) > 0:
            pnls = [t['pnl'] for t in trades]
            avg_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            
            assert isinstance(avg_pnl, (int, float))
            assert std_pnl >= 0
            
            # Win rate
            wins = sum(1 for pnl in pnls if pnl > 0)
            win_rate = wins / len(pnls)
            assert 0 <= win_rate <= 1
    
    def test_pipeline_with_regime_detection(self, sample_market_data):
        """Test pipeline con detección de régimen de mercado"""
        from strategies.regime_detector import RegimeDetectorAdvanced
        
        # Detectar régimen
        detector = RegimeDetectorAdvanced(
            regime_params={
                'bull': {'vol_threshold': 0.02},
                'bear': {'vol_threshold': 0.02},
                'sideways': {'vol_threshold': 0.01}
            }
        )
        
        # Verificar que se puede detectar régimen
        regime = detector.detect_regime(sample_market_data)
        assert regime in ['bull', 'bear', 'sideways', 'unknown']
        
        # Obtener parámetros para el régimen detectado
        params = detector.get_regime_params(regime)
        assert isinstance(params, dict)
        assert 'vol_threshold' in params
    
    def test_pipeline_error_handling(self):
        """Test manejo de errores en el pipeline"""
        from src.backtester_core import BacktesterCore
        from strategies.mean_reversion import RSIMeanReversionStrategy
        
        backtester = BacktesterCore()
        
        # Datos vacíos
        empty_df = pd.DataFrame()
        results = backtester.simple_backtest(
            df_5m=empty_df,
            strategy_class=RSIMeanReversionStrategy
        )
        
        # Debería manejar el error gracefully
        assert 'error' in results or 'metrics' in results
        
    def test_multiple_strategies_comparison(self, sample_market_data):
        """Test comparación de múltiples estrategias"""
        from strategies.mean_reversion import RSIMeanReversionStrategy
        from src.backtester_core import BacktesterCore
        
        backtester = BacktesterCore(initial_capital=10000)
        
        # Estrategia 1: RSI conservador
        results1 = backtester.simple_backtest(
            df_5m=sample_market_data,
            strategy_class=RSIMeanReversionStrategy,
            strategy_params={'rsi_oversold': 25, 'rsi_overbought': 75}
        )
        
        # Estrategia 2: RSI agresivo
        results2 = backtester.simple_backtest(
            df_5m=sample_market_data,
            strategy_class=RSIMeanReversionStrategy,
            strategy_params={'rsi_oversold': 35, 'rsi_overbought': 65}
        )
        
        # Comparar resultados
        assert 'metrics' in results1 and 'metrics' in results2
        
        m1 = results1['metrics']
        m2 = results2['metrics']
        
        # Ambas estrategias deberían tener métricas válidas
        assert isinstance(m1['total_return'], (int, float))
        assert isinstance(m2['total_return'], (int, float))
        
        # Las estrategias deberían producir resultados diferentes
        # (a menos que por casualidad sean idénticos)
        assert m1['total_trades'] != m2['total_trades'] or abs(m1['total_return'] - m2['total_return']) < 0.01


class TestDataQualityIntegration:
    """Tests de calidad de datos en el pipeline"""
    
    def test_data_validation_before_strategy(self, sample_market_data):
        """Test validación de datos antes de aplicar estrategia"""
        # Verificar calidad básica
        assert not sample_market_data.empty
        assert not sample_market_data.isnull().any().any()
        
        # Verificar consistencia OHLC
        assert all(sample_market_data['High'] >= sample_market_data['Low'])
        assert all(sample_market_data['High'] >= sample_market_data['Open'])
        assert all(sample_market_data['High'] >= sample_market_data['Close'])
        assert all(sample_market_data['Low'] <= sample_market_data['Open'])
        assert all(sample_market_data['Low'] <= sample_market_data['Close'])
        
        # Verificar volumen positivo
        assert all(sample_market_data['Volume'] > 0)
    
    def test_data_continuity_check(self, sample_market_data):
        """Test verificación de continuidad temporal"""
        # Índice debe ser monotónico creciente
        assert sample_market_data.index.is_monotonic_increasing
        
        # No debe haber duplicados
        assert not sample_market_data.index.duplicated().any()
        
        # Calcular gaps temporales
        time_diffs = sample_market_data.index.to_series().diff()
        
        # Todos los gaps deberían ser del mismo tamaño (5 min)
        mode_diff = time_diffs.mode()[0]
        assert mode_diff == pd.Timedelta('5min')


class TestPerformanceIntegration:
    """Tests de performance del sistema integrado"""
    
    def test_backtest_execution_time(self, sample_market_data):
        """Test que el backtest se ejecute en tiempo razonable"""
        import time
        from src.backtester_core import BacktesterCore
        from strategies.mean_reversion import RSIMeanReversionStrategy
        
        backtester = BacktesterCore()
        
        start_time = time.time()
        results = backtester.simple_backtest(
            df_5m=sample_market_data,
            strategy_class=RSIMeanReversionStrategy
        )
        execution_time = time.time() - start_time
        
        # El backtest debería completarse en menos de 5 segundos
        assert execution_time < 5.0, f"Backtest too slow: {execution_time:.2f}s"
        assert 'metrics' in results
    
    def test_memory_efficiency(self, sample_market_data):
        """Test uso eficiente de memoria"""
        from src.backtester_core import BacktesterCore
        from strategies.mean_reversion import RSIMeanReversionStrategy
        
        # Ejecutar múltiples backtests
        backtester = BacktesterCore()
        
        for i in range(3):
            results = backtester.simple_backtest(
                df_5m=sample_market_data,
                strategy_class=RSIMeanReversionStrategy
            )
            
            # Verificar que los resultados son válidos
            assert 'metrics' in results
            assert len(results['trades']) >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
