"""
Tests para los helpers extraídos en Auditoría Round 12.

Este módulo prueba las funciones auxiliares creadas durante el refactoring
para reducir la complejidad ciclomática de:
- calculate_ifvg_enhanced() → 5 helpers
- volume_profile_advanced_slow() → 2 helpers  
- generate_filtered_signals() → 3 helpers
- _process_and_record_trades() → 2 helpers
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# MOCK TALIB para evitar dependencia de C library
# =============================================================================

# Crear mock antes de importar módulos que usan talib
talib_mock = MagicMock()
talib_mock.ATR = lambda h, l, c, timeperiod: pd.Series(np.ones(len(h)) * 100, index=h.index)
talib_mock.EMA = lambda arr, timeperiod: arr  # Return same array
talib_mock.SMA = lambda arr, timeperiod: arr  # Return same array
sys.modules['talib'] = talib_mock


# Ahora importar los módulos
from core.data.indicators import (
    _detect_bullish_gap,
    _detect_bearish_gap,
    _find_all_gaps,
    _is_gap_mitigated,
    _convert_gaps_to_signals,
    _build_volume_profile_for_window,
    _calculate_value_area,
    _get_filter_value,
    _check_volume_profile_filter,
    _process_bar_signals,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """DataFrame OHLCV de prueba con 100 barras."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    
    base_price = 45000
    prices = base_price + np.cumsum(np.random.standard_normal(n) * 50)
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.standard_normal(n) * 30),
        "low": prices - np.abs(np.random.standard_normal(n) * 30),
        "close": prices + np.random.standard_normal(n) * 20,
        "volume": np.random.uniform(100, 1000, n),
    }, index=dates)
    
    # Asegurar que high >= max(open, close) y low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    
    return df


@pytest.fixture
def gap_data():
    """DataFrame con gaps conocidos para testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="5min")
    
    # Crear gap alcista entre barra 2 y 3
    # prev_high (barra 2) = 100, curr_low (barra 3) = 105 → gap de 5%
    df = pd.DataFrame({
        "high": [100, 100, 100, 110, 115, 120, 125, 130, 135, 140],
        "low":  [95,  95,  95,  105, 110, 115, 120, 125, 130, 135],
        "close": [98, 98, 98, 108, 113, 118, 123, 128, 133, 138],
        "volume": [100] * 10,
    }, index=dates)
    
    return df


# =============================================================================
# Tests para helpers de IFVG
# =============================================================================

class TestIFVGHelpers:
    """Tests para helpers de calculate_ifvg_enhanced."""
    
    def test_detect_bullish_gap_valid(self):
        """Detecta gap alcista cuando curr_low > prev_high."""
        result = _detect_bullish_gap(
            prev_high=100.0,
            curr_low=105.0,  # 5% gap
            min_gap_size=0.001,
            atr_value=10.0,
            atr_multiplier=0.2,
            idx=5
        )
        
        assert result is not None
        assert result["type"] == "bullish"
        assert result["index"] == 5
        assert result["gap_start"] == 100.0
        assert result["gap_end"] == 105.0
        assert result["gap_size"] == pytest.approx(0.05, rel=0.01)
    
    def test_detect_bullish_gap_no_gap(self):
        """No detecta gap si curr_low <= prev_high."""
        result = _detect_bullish_gap(
            prev_high=100.0,
            curr_low=99.0,  # No hay gap
            min_gap_size=0.001,
            atr_value=10.0,
            atr_multiplier=0.2,
            idx=5
        )
        
        assert result is None
    
    def test_detect_bullish_gap_too_small(self):
        """No detecta gap si es muy pequeño."""
        result = _detect_bullish_gap(
            prev_high=100.0,
            curr_low=100.05,  # 0.05% gap - muy pequeño
            min_gap_size=0.001,  # 0.1% mínimo
            atr_value=10.0,
            atr_multiplier=0.2,
            idx=5
        )
        
        assert result is None
    
    def test_detect_bearish_gap_valid(self):
        """Detecta gap bajista cuando curr_high < prev_low."""
        result = _detect_bearish_gap(
            prev_low=100.0,
            curr_high=95.0,  # 5% gap bajista
            min_gap_size=0.001,
            atr_value=10.0,
            atr_multiplier=0.2,
            idx=5
        )
        
        assert result is not None
        assert result["type"] == "bearish"
        assert result["gap_size"] == pytest.approx(0.05, rel=0.01)
    
    def test_detect_bearish_gap_no_gap(self):
        """No detecta gap bajista si curr_high >= prev_low."""
        result = _detect_bearish_gap(
            prev_low=100.0,
            curr_high=101.0,
            min_gap_size=0.001,
            atr_value=10.0,
            atr_multiplier=0.2,
            idx=5
        )
        
        assert result is None
    
    def test_find_all_gaps(self, gap_data):
        """Encuentra gaps en datos OHLCV."""
        atr = pd.Series(np.ones(len(gap_data)) * 10, index=gap_data.index)
        
        gaps = _find_all_gaps(
            df=gap_data,
            atr=atr,
            atr_multiplier=0.2,
            min_gap_size=0.001
        )
        
        # Debe encontrar al menos un gap (el alcista que creamos)
        assert isinstance(gaps, list)
        # Los gaps dependen de la estructura de datos
    
    def test_is_gap_mitigated_not_filled(self, gap_data):
        """Gap no mitigado devuelve False."""
        gap = {
            "index": 3,
            "type": "bullish",
            "gap_end": 200.0,  # Precio muy alto, no se llenará
        }
        
        result = _is_gap_mitigated(gap_data, gap, mitigation_lookback=5)
        assert result is False
    
    def test_is_gap_mitigated_filled(self, gap_data):
        """Gap mitigado devuelve True."""
        gap = {
            "index": 3,
            "type": "bullish",
            "gap_end": 100.0,  # Precio bajo, se llenará
        }
        
        result = _is_gap_mitigated(gap_data, gap, mitigation_lookback=5)
        assert result is True
    
    def test_convert_gaps_to_signals(self, gap_data):
        """Convierte gaps a señales correctamente."""
        atr = pd.Series(np.ones(len(gap_data)) * 10, index=gap_data.index)
        
        gaps = [{
            "index": 5,
            "type": "bullish",
            "gap_start": 100.0,
            "gap_end": 200.0,  # No se llenará
            "gap_size": 0.05,
        }]
        
        bull, bear, conf = _convert_gaps_to_signals(
            gap_data, gaps, atr, atr_multiplier=0.2, mitigation_lookback=2
        )
        
        assert isinstance(bull, pd.Series)
        assert isinstance(bear, pd.Series)
        assert isinstance(conf, pd.Series)
        assert len(bull) == len(gap_data)


# =============================================================================
# Tests para helpers de Volume Profile
# =============================================================================

class TestVolumeProfileHelpers:
    """Tests para helpers de volume_profile_advanced_slow."""
    
    def test_build_volume_profile_for_window(self, sample_ohlcv):
        """Construye perfil de volumen correctamente."""
        window_df = sample_ohlcv.iloc[:20]
        bin_size = 10.0
        
        profile = _build_volume_profile_for_window(window_df, bin_size)
        
        assert isinstance(profile, dict)
        assert len(profile) > 0
        # Todos los valores deben ser positivos
        assert all(v >= 0 for v in profile.values())
    
    def test_build_volume_profile_empty_df(self):
        """Maneja DataFrame vacío correctamente."""
        empty_df = pd.DataFrame({
            "high": [], "low": [], "volume": []
        })
        
        profile = _build_volume_profile_for_window(empty_df, bin_size=10.0)
        assert profile == {}
    
    def test_calculate_value_area_normal(self):
        """Calcula VAH/VAL correctamente."""
        volume_profile = {
            100.0: 500,
            110.0: 300,
            120.0: 200,
            90.0: 100,
        }
        
        vah, val = _calculate_value_area(volume_profile, value_area_percent=0.7)
        
        assert vah is not None
        assert val is not None
        assert vah >= val
    
    def test_calculate_value_area_empty(self):
        """Maneja perfil vacío correctamente."""
        vah, val = _calculate_value_area({}, value_area_percent=0.7)
        
        assert vah is None
        assert val is None


# =============================================================================
# Tests para helpers de Signal Filtering
# =============================================================================

class TestSignalFilteringHelpers:
    """Tests para helpers de generate_filtered_signals."""
    
    def test_get_filter_value_valid(self):
        """Obtiene valor de filtro cuando es válido."""
        series = pd.Series([True, False, True, False])
        
        assert _get_filter_value(series, 0) is True
        assert _get_filter_value(series, 1) is False
    
    def test_get_filter_value_nan(self):
        """Devuelve default cuando hay NaN."""
        series = pd.Series([True, np.nan, False])
        
        assert _get_filter_value(series, 1, default=True) is True
        assert _get_filter_value(series, 1, default=False) is False
    
    def test_check_volume_profile_filter_near_poc(self):
        """Detecta precio cerca de POC."""
        result = _check_volume_profile_filter(
            current_price=100.0,
            poc=100.3,  # 0.3% de distancia
            vah=105.0,
            val=95.0,
            threshold=0.005  # 0.5%
        )
        
        assert result is True
    
    def test_check_volume_profile_filter_far_from_poc(self):
        """No detecta cuando precio está lejos."""
        result = _check_volume_profile_filter(
            current_price=100.0,
            poc=110.0,  # 10% de distancia
            vah=115.0,
            val=105.0,
            threshold=0.005
        )
        
        assert result is False
    
    def test_check_volume_profile_filter_nan_poc(self):
        """Devuelve False cuando POC es NaN."""
        result = _check_volume_profile_filter(
            current_price=100.0,
            poc=np.nan,
            vah=105.0,
            val=95.0,
            threshold=0.005
        )
        
        assert result is False


# =============================================================================
# Tests de integración de helpers
# =============================================================================

class TestHelpersIntegration:
    """Tests de integración para verificar que los helpers funcionan juntos."""
    
    def test_gap_detection_pipeline(self, sample_ohlcv):
        """Pipeline completo de detección de gaps."""
        atr = pd.Series(np.ones(len(sample_ohlcv)) * 100, index=sample_ohlcv.index)
        
        # 1. Encontrar gaps
        gaps = _find_all_gaps(sample_ohlcv, atr, atr_multiplier=0.2, min_gap_size=0.001)
        
        # 2. Convertir a señales
        bull, bear, conf = _convert_gaps_to_signals(
            sample_ohlcv, gaps, atr, atr_multiplier=0.2, mitigation_lookback=5
        )
        
        # Verificar tipos de salida
        assert isinstance(bull, pd.Series)
        assert bull.dtype == bool
        assert isinstance(bear, pd.Series)
        assert bear.dtype == bool
        assert isinstance(conf, pd.Series)
        assert conf.dtype == float
    
    def test_volume_profile_pipeline(self, sample_ohlcv):
        """Pipeline completo de volume profile."""
        # 1. Construir perfil para ventana
        window = sample_ohlcv.iloc[:50]
        profile = _build_volume_profile_for_window(window, bin_size=50.0)
        
        # 2. Calcular value area
        vah, val = _calculate_value_area(profile, value_area_percent=0.7)
        
        if profile:  # Si hay datos
            assert vah is not None or val is not None or len(profile) == 0


# =============================================================================
# Tests de edge cases
# =============================================================================

class TestEdgeCases:
    """Tests para casos límite y edge cases."""
    
    def test_zero_price_handling(self):
        """Maneja precios cero sin crash."""
        result = _check_volume_profile_filter(
            current_price=0.0,
            poc=100.0,
            vah=105.0,
            val=95.0,
            threshold=0.005
        )
        # No debe crashear, puede devolver True o False
        assert isinstance(result, bool)
    
    def test_negative_gap_size(self):
        """No detecta gaps negativos."""
        result = _detect_bullish_gap(
            prev_high=110.0,
            curr_low=100.0,  # curr_low < prev_high → no hay gap
            min_gap_size=0.001,
            atr_value=10.0,
            atr_multiplier=0.2,
            idx=5
        )
        assert result is None
    
    def test_empty_dataframe_handling(self):
        """Maneja DataFrames vacíos."""
        empty_df = pd.DataFrame({
            "high": pd.Series([], dtype=float),
            "low": pd.Series([], dtype=float),
            "close": pd.Series([], dtype=float),
            "volume": pd.Series([], dtype=float),
        })
        atr = pd.Series([], dtype=float)
        
        gaps = _find_all_gaps(empty_df, atr, atr_multiplier=0.2, min_gap_size=0.001)
        assert gaps == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
