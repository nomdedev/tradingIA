"""
Tests para helpers de BacktesterCore extraídos en Auditoría Round 12.

Prueba las funciones:
- _extract_trade_info()
- _calculate_mae_mfe()

Nota: Usa implementación standalone de los helpers para evitar
dependencias problemáticas (sklearn, skopt, talib).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Implementación standalone de los helpers para testing
# (Copia de BacktesterCore para evitar importar sklearn/skopt)
# =============================================================================

class MockBacktesterHelpers:
    """Implementación standalone de helpers para testing."""
    
    def __init__(self):
        self.logger = MagicMock()
    
    def _extract_trade_info(self, trade) -> dict:
        """Extrae información de un trade del array de VectorBT."""
        try:
            entry_idx = int(trade['entry_idx'])
            exit_idx = int(trade['exit_idx'])
            entry_price = float(trade['entry_price'])
            exit_price = float(trade['exit_price'])
            size = float(trade['size'])
            
            if 'direction' in trade.dtype.names:
                direction = int(trade['direction'])
                side = "buy" if direction == 0 else "sell"
            else:
                side = "buy"
                self.logger.debug("Trade direction not available, assuming long trade")
            
            return {
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'side': side
            }
        except (ValueError, IndexError, KeyError):
            self.logger.warning("Using legacy index access for trade records")
            return {
                'entry_idx': int(trade[2]),
                'exit_idx': int(trade[3]),
                'entry_price': float(trade[5]),
                'exit_price': float(trade[6]),
                'size': float(trade[4]),
                'side': "buy"
            }

    def _calculate_mae_mfe(self, df_5m, entry_idx: int, exit_idx: int, 
                           entry_price: float, side: str) -> tuple:
        """Calcula MAE y MFE."""
        high_series = df_5m["high"].iloc[entry_idx:exit_idx + 1]
        low_series = df_5m["low"].iloc[entry_idx:exit_idx + 1]
        max_price = high_series.max()
        min_price = low_series.min()

        if entry_price == 0:
            return 0.0, 0.0

        if side == "buy":
            mae = (entry_price - min_price) / entry_price if min_price < entry_price else 0.0
            mfe = (max_price - entry_price) / entry_price if max_price > entry_price else 0.0
        else:
            mae = (max_price - entry_price) / entry_price if max_price > entry_price else 0.0
            mfe = (entry_price - min_price) / entry_price if min_price < entry_price else 0.0

        return mae, mfe


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_backtester():
    """Mock de BacktesterCore con los métodos helper."""
    return MockBacktesterHelpers()


@pytest.fixture
def sample_trade_record():
    """Trade record simulado de VectorBT."""
    # Simular numpy structured array con dtype.names
    dtype = np.dtype([
        ('entry_idx', 'i4'),
        ('exit_idx', 'i4'),
        ('entry_price', 'f8'),
        ('exit_price', 'f8'),
        ('size', 'f8'),
        ('direction', 'i4'),
    ])
    
    trade = np.array([(10, 20, 45000.0, 46000.0, 0.1, 0)], dtype=dtype)[0]
    return trade


@pytest.fixture
def sample_df_5m():
    """DataFrame 5min para cálculos de MAE/MFE."""
    dates = pd.date_range("2024-01-01", periods=30, freq="5min")
    
    # Crear movimiento de precios conocido
    prices = np.array([
        45000, 44900, 44800, 44700, 44600,  # Baja inicial
        44500, 44400, 44300, 44200, 44100,  # Sigue bajando
        44000, 44200, 44400, 44600, 44800,  # Empieza a subir
        45000, 45200, 45400, 45600, 45800,  # Sigue subiendo
        46000, 46200, 46400, 46600, 46800,  # Más subida
        47000, 47200, 47400, 47600, 47800,  # Máximo
    ])
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 50,
        "low": prices - 50,
        "close": prices,
        "volume": np.random.uniform(100, 1000, 30),
    }, index=dates)
    
    return df


# =============================================================================
# Tests para _extract_trade_info
# =============================================================================

class TestExtractTradeInfo:
    """Tests para el método _extract_trade_info."""
    
    def test_extract_long_trade(self, mock_backtester):
        """Extrae información correcta de trade long."""
        # Crear trade record con direction=0 (long)
        dtype = np.dtype([
            ('entry_idx', 'i4'),
            ('exit_idx', 'i4'),
            ('entry_price', 'f8'),
            ('exit_price', 'f8'),
            ('size', 'f8'),
            ('direction', 'i4'),
        ])
        trade = np.array([(10, 20, 45000.0, 46000.0, 0.1, 0)], dtype=dtype)[0]
        
        result = mock_backtester._extract_trade_info(trade)
        
        assert result['entry_idx'] == 10
        assert result['exit_idx'] == 20
        assert result['entry_price'] == pytest.approx(45000.0)
        assert result['exit_price'] == pytest.approx(46000.0)
        assert result['size'] == pytest.approx(0.1)
        assert result['side'] == "buy"
    
    def test_extract_short_trade(self, mock_backtester):
        """Extrae información correcta de trade short."""
        dtype = np.dtype([
            ('entry_idx', 'i4'),
            ('exit_idx', 'i4'),
            ('entry_price', 'f8'),
            ('exit_price', 'f8'),
            ('size', 'f8'),
            ('direction', 'i4'),
        ])
        trade = np.array([(10, 20, 46000.0, 45000.0, 0.1, 1)], dtype=dtype)[0]
        
        result = mock_backtester._extract_trade_info(trade)
        
        assert result['side'] == "sell"
    
    def test_extract_without_direction_field(self, mock_backtester):
        """Maneja trades sin campo direction (asume long)."""
        dtype = np.dtype([
            ('entry_idx', 'i4'),
            ('exit_idx', 'i4'),
            ('entry_price', 'f8'),
            ('exit_price', 'f8'),
            ('size', 'f8'),
        ])
        trade = np.array([(10, 20, 45000.0, 46000.0, 0.1)], dtype=dtype)[0]
        
        result = mock_backtester._extract_trade_info(trade)
        
        assert result['side'] == "buy"  # Default a long


# =============================================================================
# Tests para _calculate_mae_mfe
# =============================================================================

class TestCalculateMaeMfe:
    """Tests para el método _calculate_mae_mfe."""
    
    def test_mae_mfe_long_profitable(self, mock_backtester, sample_df_5m):
        """Calcula MAE/MFE para trade long profitable."""
        # Trade de índice 10 a 25 (entra en 44000, sale en 47000)
        entry_idx = 10
        exit_idx = 25
        entry_price = 44000.0
        side = "buy"
        
        mae, mfe = mock_backtester._calculate_mae_mfe(
            sample_df_5m, entry_idx, exit_idx, entry_price, side
        )
        
        # MAE: cuánto bajó desde entrada (entry - min_low) / entry
        # MFE: cuánto subió desde entrada (max_high - entry) / entry
        assert mae >= 0.0
        assert mfe >= 0.0
        assert mfe > mae  # Trade profitable debería tener MFE > MAE
    
    def test_mae_mfe_long_losing(self, mock_backtester, sample_df_5m):
        """Calcula MAE/MFE para trade long perdedor."""
        # Trade de índice 0 a 10 (entra en 45000, sale en 44000 - pierde)
        entry_idx = 0
        exit_idx = 10
        entry_price = 45000.0
        side = "buy"
        
        mae, mfe = mock_backtester._calculate_mae_mfe(
            sample_df_5m, entry_idx, exit_idx, entry_price, side
        )
        
        assert mae >= 0.0
        assert mfe >= 0.0
    
    def test_mae_mfe_short_profitable(self, mock_backtester, sample_df_5m):
        """Calcula MAE/MFE para trade short profitable."""
        # Short de índice 25 a 29 (entra en 47000, sale más bajo)
        # Para short: MAE = cuánto subió, MFE = cuánto bajó
        entry_idx = 0
        exit_idx = 10
        entry_price = 45000.0
        side = "sell"
        
        mae, mfe = mock_backtester._calculate_mae_mfe(
            sample_df_5m, entry_idx, exit_idx, entry_price, side
        )
        
        assert mae >= 0.0
        assert mfe >= 0.0
    
    def test_mae_mfe_single_bar(self, mock_backtester, sample_df_5m):
        """Calcula MAE/MFE para trade de una sola barra."""
        entry_idx = 5
        exit_idx = 5  # Mismo índice
        entry_price = 44500.0
        side = "buy"
        
        mae, mfe = mock_backtester._calculate_mae_mfe(
            sample_df_5m, entry_idx, exit_idx, entry_price, side
        )
        
        # Debe funcionar sin crash
        assert mae >= 0.0
        assert mfe >= 0.0


# =============================================================================
# Tests de integración
# =============================================================================

class TestBacktesterHelpersIntegration:
    """Tests de integración de los helpers."""
    
    def test_full_trade_processing_pipeline(self, mock_backtester, sample_df_5m):
        """Pipeline completo: extract → calculate MAE/MFE."""
        # Crear trade
        dtype = np.dtype([
            ('entry_idx', 'i4'),
            ('exit_idx', 'i4'),
            ('entry_price', 'f8'),
            ('exit_price', 'f8'),
            ('size', 'f8'),
            ('direction', 'i4'),
        ])
        trade = np.array([(5, 20, 44500.0, 46000.0, 0.1, 0)], dtype=dtype)[0]
        
        # 1. Extraer info
        trade_info = mock_backtester._extract_trade_info(trade)
        
        # 2. Calcular MAE/MFE
        mae, mfe = mock_backtester._calculate_mae_mfe(
            sample_df_5m,
            trade_info['entry_idx'],
            trade_info['exit_idx'],
            trade_info['entry_price'],
            trade_info['side']
        )
        
        # Verificar resultados coherentes
        assert trade_info['side'] == "buy"
        assert mae >= 0.0
        assert mfe >= 0.0


# =============================================================================
# Tests de edge cases
# =============================================================================

class TestBacktesterEdgeCases:
    """Tests para casos límite."""
    
    def test_mae_mfe_with_zero_entry_price(self, mock_backtester, sample_df_5m):
        """Maneja entry_price=0 sin división por cero."""
        entry_idx = 5
        exit_idx = 10
        entry_price = 0.0  # Edge case peligroso
        side = "buy"
        
        # No debe crashear
        try:
            mae, mfe = mock_backtester._calculate_mae_mfe(
                sample_df_5m, entry_idx, exit_idx, entry_price, side
            )
            # Si no crashea, verificar que devuelve valores válidos
            assert isinstance(mae, (int, float))
            assert isinstance(mfe, (int, float))
        except ZeroDivisionError:
            pytest.fail("No debe haber ZeroDivisionError con entry_price=0")
    
    def test_extract_trade_malformed_record(self, mock_backtester):
        """Maneja records malformados con fallback."""
        # Simular record con estructura diferente (fallback a índices)
        dtype = np.dtype([
            ('col0', 'i4'),
            ('col1', 'i4'),
            ('col2', 'i4'),  # entry_idx
            ('col3', 'i4'),  # exit_idx
            ('col4', 'f8'),  # size
            ('col5', 'f8'),  # entry_price
            ('col6', 'f8'),  # exit_price
        ])
        trade = np.array([(0, 0, 10, 20, 0.1, 45000.0, 46000.0)], dtype=dtype)[0]
        
        # Debe usar fallback
        result = mock_backtester._extract_trade_info(trade)
        
        # Verificar que extrajo algo (puede usar fallback)
        assert 'entry_idx' in result
        assert 'exit_idx' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
