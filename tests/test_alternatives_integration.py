"""
Test para alternatives_integration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np

try:
    from src.alternatives_integration import AlternativesIntegration
except ImportError:
    pytest.skip("AlternativesIntegration not available", allow_module_level=True)


class TestAlternativesIntegration:
    """Test suite para AlternativesIntegration."""

    @pytest.fixture
    def sample_data(self):
        """Crear datos de ejemplo para testing."""
        dates = pd.date_range('2024-01-01', periods=200, freq='5min')
        np.random.seed(42)

        # Crear datos OHLCV realistas
        close = 45000 + np.random.randn(200).cumsum() * 50
        high = close + np.random.uniform(0, 200, 200)
        low = close - np.random.uniform(0, 200, 200)
        open_price = close + np.random.randn(200) * 10
        volume = np.random.randint(100, 1000, 200)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        # Asegurar high >= max(close, open) y low <= min(close, open)
        df['high'] = df[['high', 'close', 'open']].max(axis=1)
        df['low'] = df[['low', 'close', 'open']].min(axis=1)

        return df

    @pytest.fixture
    def integrator(self):
        """Crear instancia de AlternativesIntegration."""
        return AlternativesIntegration()

    def test_generate_alternative_signals(self, integrator, sample_data):
        """Test generación de señales alternativas."""
        result = integrator.generate_alternative_signals(sample_data)

        # Verificar que se añadieron las columnas esperadas
        expected_columns = [
            'rsi_bb_signal', 'vwap_delta_signal', 'macd_adx_signal',
            'ichimoku_aroon_signal', 'stochastic_funding_signal',
            'alternative_score', 'alternative_score_norm', 'hybrid_signal'
        ]

        for col in expected_columns:
            assert col in result.columns, f"Falta columna: {col}"

        # Verificar que las señales son 0 o 1
        signal_cols = [col for col in expected_columns if 'signal' in col]
        for col in signal_cols:
            assert result[col].isin([0, 1]).all(), f"Señal {col} no es binaria"

        # Verificar que el score está en rango 0-5
        assert (result['alternative_score_norm'] >= 0).all()
        assert (result['alternative_score_norm'] <= 5).all()

    def test_rsi_bb_signal(self, integrator, sample_data):
        """Test señal RSI + Bollinger Bands."""
        result = integrator._add_rsi_bb_signal(sample_data)

        assert 'rsi_bb_signal' in result.columns
        assert result['rsi_bb_signal'].isin([0, 1]).all()

        # Verificar lógica básica: si hay señal, debería cumplir condiciones
        signal_mask = result['rsi_bb_signal'] == 1
        if signal_mask.any():
            assert (result.loc[signal_mask, 'rsi_14'] < 30).all()
            assert (result.loc[signal_mask, 'close'] < result.loc[signal_mask, 'bb_lower']).all()
            assert (result.loc[signal_mask, 'volume'] > 1.5 * result.loc[signal_mask, 'vol_sma_20']).all()

    def test_vwap_delta_signal(self, integrator, sample_data):
        """Test señal VWAP + delta volume."""
        result = integrator._add_vwap_delta_signal(sample_data)

        assert 'vwap_delta_signal' in result.columns
        assert result['vwap_delta_signal'].isin([0, 1]).all()

        # Verificar que VWAP se calcula
        assert 'vwap' in result.columns
        assert not result['vwap'].isna().all()

    def test_macd_adx_signal(self, integrator, sample_data):
        """Test señal MACD + ADX."""
        result = integrator._add_macd_adx_signal(sample_data)

        assert 'macd_adx_signal' in result.columns
        assert result['macd_adx_signal'].isin([0, 1]).all()

        # Verificar que ADX se calcula
        assert 'adx' in result.columns
        assert not result['adx'].isna().all()

    def test_ichimoku_aroon_signal(self, integrator, sample_data):
        """Test señal Ichimoku + Aroon."""
        result = integrator._add_ichimoku_aroon_signal(sample_data)

        assert 'ichimoku_aroon_signal' in result.columns
        assert result['ichimoku_aroon_signal'].isin([0, 1]).all()

        # Verificar componentes Ichimoku
        ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']
        for col in ichimoku_cols:
            assert col in result.columns

    def test_stochastic_funding_signal(self, integrator, sample_data):
        """Test señal Stochastic + Funding."""
        result = integrator._add_stochastic_funding_signal(sample_data)

        assert 'stochastic_funding_signal' in result.columns
        assert result['stochastic_funding_signal'].isin([0, 1]).all()

        # Verificar que se añade funding rate simulado
        assert 'funding_rate' in result.columns

    def test_backtest_alternatives(self, integrator, sample_data):
        """Test backtest de señales alternativas."""
        # Generar señales primero
        df_signals = integrator.generate_alternative_signals(sample_data)

        # Ejecutar backtest
        results = integrator.backtest_alternatives(df_signals)

        # Verificar métricas
        required_metrics = ['total_return', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'total_trades']
        for metric in required_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))

        # Verificar rangos razonables
        assert -1 <= results['total_return'] <= 5  # Retorno entre -100% y +500%
        assert 0 <= results['win_rate'] <= 1  # Win rate entre 0 y 1
        assert results['total_trades'] >= 0

    def test_compare_with_fvg(self, integrator, sample_data):
        """Test comparación con FVG."""
        # Crear datos mock con señal FVG
        df_fvg = sample_data.copy()
        df_fvg['fvg_signal'] = np.random.choice([0, 1], len(df_fvg))

        df_alt = integrator.generate_alternative_signals(sample_data)

        results = integrator.compare_with_fvg(df_fvg, df_alt)

        assert 'fvg' in results
        assert 'alternatives' in results
        assert 'improvement' in results

        # Verificar métricas de mejora
        improvement = results['improvement']
        required_improvements = ['win_rate_diff', 'sharpe_diff', 'return_diff']
        for imp in required_improvements:
            assert imp in improvement

    def test_missing_columns_error(self, integrator):
        """Test error cuando faltan columnas requeridas."""
        df_incomplete = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(ValueError, match="debe contener columnas"):
            integrator.generate_alternative_signals(df_incomplete)

    def test_calculate_composite_score(self, integrator, sample_data):
        """Test cálculo de score compuesto."""
        # Añadir señales individuales primero
        df = sample_data.copy()
        df['rsi_bb_signal'] = 1
        df['vwap_delta_signal'] = 0
        df['macd_adx_signal'] = 1
        df['ichimoku_aroon_signal'] = 0
        df['stochastic_funding_signal'] = 1

        result = integrator._calculate_composite_score(df)

        assert 'alternative_score' in result.columns
        assert 'alternative_score_norm' in result.columns

        # Verificar que el score es positivo cuando hay señales
        assert (result['alternative_score'] >= 0).all()
        assert (result['alternative_score_norm'] >= 0).all()
        assert (result['alternative_score_norm'] <= 5).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])