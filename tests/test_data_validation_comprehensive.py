"""
Test Suite: Data Validation & Integrity
Tests críticos para validación de datos y prevención de errores comunes

Author: TradingIA Team
Created: 2025-11-13
Priority: CRITICAL
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from unittest.mock import Mock, patch

from core.backend_core import DataManager


class TestOHLCValidation:
    """Tests para validación de relaciones OHLC"""

    def test_valid_ohlc_relationships(self):
        """Valida que OHLC relationships son correctas"""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })

        dm = DataManager()
        # No debe levantar excepción
        validated = dm.validate_ohlc(data)
        assert len(validated) == 3

    def test_invalid_high_low_relationship(self):
        """Detecta cuando High < Low"""
        invalid_data = pd.DataFrame({
            'Open': [100],
            'High': [95],  # High < Low ❌
            'Low': [105],
            'Close': [103],
            'Volume': [1000]
        })

        dm = DataManager()
        with pytest.raises(ValueError, match="High must be >= Low"):
            dm.validate_ohlc(invalid_data)

    def test_invalid_close_above_high(self):
        """Detecta cuando Close > High"""
        invalid_data = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [110],  # Close > High ❌
            'Volume': [1000]
        })

        dm = DataManager()
        with pytest.raises(ValueError, match="Close must be <= High"):
            dm.validate_ohlc(invalid_data)

    def test_invalid_close_below_low(self):
        """Detecta cuando Close < Low"""
        invalid_data = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [90],  # Close < Low ❌
            'Volume': [1000]
        })

        dm = DataManager()
        with pytest.raises(ValueError, match="Close must be >= Low"):
            dm.validate_ohlc(invalid_data)

    def test_auto_correction_of_ohlc(self):
        """Auto-corrección de OHLC cuando es posible"""
        data = pd.DataFrame({
            'Open': [100, 101],
            'High': [104, 105],  # Será ajustado
            'Low': [96, 97],      # Será ajustado
            'Close': [110, 95],   # Valores problemáticos
            'Volume': [1000, 1100]
        })

        dm = DataManager()
        corrected = dm.validate_ohlc(data, auto_correct=True)

        # High debe ser >= max(Open, Close)
        assert (corrected['High'] >= corrected['Close']).all()
        assert (corrected['High'] >= corrected['Open']).all()

        # Low debe ser <= min(Open, Close)
        assert (corrected['Low'] <= corrected['Close']).all()
        assert (corrected['Low'] <= corrected['Open']).all()


class TestDataGapsHandling:
    """Tests para manejo de gaps en datos"""

    def test_detect_small_gaps(self):
        """Detecta gaps pequeños en datos"""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        # Eliminar algunos timestamps (crear gap)
        dates = dates.delete([4, 5, 6])  # Gap de 15 minutos

        data = pd.DataFrame({
            'Date': dates,
            'Close': [100] * 7
        })

        dm = DataManager()
        gaps = dm.detect_data_gaps(data, expected_freq='5min')

        assert len(gaps) == 1
        assert gaps[0]['duration_minutes'] == 20

    def test_handle_gaps_forward_fill(self):
        """Maneja gaps con forward fill"""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        dates = dates.delete([4, 5])  # Gap

        data = pd.DataFrame({
            'Date': dates,
            'Close': [100, 101, 102, 103, 104, 105, 106, 107]
        }).set_index('Date')

        dm = DataManager()
        filled = dm.handle_gaps(data, method='ffill', expected_freq='5min')

        # Debe tener 10 filas (rellenó 2 gaps)
        assert len(filled) == 10
        # Valores rellenados deben ser forward fill
        assert filled.loc['2024-01-01 00:20:00', 'Close'] == 103  # Valor anterior

    def test_raise_error_on_large_gaps(self):
        """Levanta error si gap excede threshold"""
        # Crear suficientes datos para tener un gap de 24 horas
        dates = pd.date_range('2024-01-01', periods=400, freq='5min')
        # Gap de 24 horas (eliminar 288 barras de 5min cada una)
        # Mantener primeras 50 barras, saltar 288, mantener las últimas
        keep_indices = list(range(50)) + list(range(50 + 288, 400))
        dates = dates[keep_indices]

        data = pd.DataFrame({
            'Date': dates,
            'Close': [100] * len(dates)
        })

        dm = DataManager()
        with pytest.raises(ValueError, match="Data gap exceeds maximum"):
            dm.handle_gaps(data, max_gap_hours=12, expected_freq='5min')

    def test_weekend_gaps_ignored(self):
        """Ignora gaps de fin de semana en datos diarios"""
        # Datos con sábado y domingo faltantes
        dates = pd.bdate_range('2024-01-01', periods=20)  # Solo business days

        data = pd.DataFrame({
            'Date': dates,
            'Close': [100] * 20
        })

        dm = DataManager()
        gaps = dm.detect_data_gaps(data, expected_freq='1D', ignore_weekends=True)

        # No debe detectar gaps de fin de semana
        assert len(gaps) == 0


class TestTimezoneHandling:
    """Tests para manejo correcto de timezones"""

    def test_convert_to_utc(self):
        """Convierte timestamps a UTC correctamente"""
        # Datos en EST
        dates_est = pd.date_range(
            '2024-01-01', periods=10, freq='1h',
            tz='America/New_York'
        )

        data = pd.DataFrame({
            'Date': dates_est,
            'Close': [100] * 10
        })

        dm = DataManager()
        utc_data = dm.normalize_timezone(data)

        # Todos los timestamps deben estar en UTC
        assert str(utc_data['Date'].dt.tz) == str(pytz.UTC)

        # Verificar conversión correcta (EST es UTC-5)
        original_time = dates_est[0]
        converted_time = utc_data['Date'].iloc[0]

        # EST -> UTC debe sumar 5 horas
        assert (converted_time - original_time.tz_convert('UTC')).total_seconds() == 0

    def test_detect_timezone_mismatch(self):
        """Detecta cuando hay timestamps con timezones mezcladas"""
        utc_dates = pd.date_range('2024-01-01', periods=5, freq='1h', tz='UTC')
        est_dates = pd.date_range('2024-01-01 06:00', periods=5, freq='1h', tz='America/New_York')

        # Mezclar timezones (problema común)
        mixed_dates = pd.Index(list(utc_dates) + list(est_dates))

        data = pd.DataFrame({
            'Date': mixed_dates,
            'Close': [100] * 10
        })

        dm = DataManager()
        with pytest.raises(ValueError, match="Mixed timezones detected"):
            dm.validate_timezone_consistency(data)

    def test_naive_datetime_warning(self):
        """Advierte cuando timestamps no tienen timezone"""
        naive_dates = pd.date_range('2024-01-01', periods=10, freq='1h')  # Sin tz

        data = pd.DataFrame({
            'Date': naive_dates,
            'Close': [100] * 10
        })

        dm = DataManager()
        with pytest.warns(UserWarning, match="Naive datetime detected"):
            dm.normalize_timezone(data, assume_utc=False)


class TestFutureDataDetection:
    """Tests para detectar look-ahead bias"""

    def test_detect_future_data(self):
        """Detecta datos con timestamps en el futuro"""
        dates = pd.date_range('2024-01-01', periods=10, freq='1D')
        # Agregar dato futuro
        future_date = pd.Timestamp.now() + timedelta(days=30)
        dates = dates.insert(len(dates), future_date)

        data = pd.DataFrame({
            'Date': dates,
            'Close': [100] * 11
        })

        dm = DataManager()
        with pytest.raises(ValueError, match="Future data detected"):
            dm.validate_no_future_data(data)

    def test_allow_recent_data_with_tolerance(self):
        """Permite datos recientes con tolerancia de tiempo"""
        # Datos hasta hace 1 minuto (puede haber delay de API)
        dates = pd.date_range(
            pd.Timestamp.now() - timedelta(hours=1),
            periods=10,
            freq='5min'
        )

        data = pd.DataFrame({
            'Date': dates,
            'Close': [100] * 10
        })

        dm = DataManager()
        # No debe levantar error con tolerancia de 5 minutos
        dm.validate_no_future_data(data, tolerance_minutes=5)


class TestVolumeValidation:
    """Tests para validación de volumen"""

    def test_zero_volume_detection(self):
        """Detecta barras con volumen cero"""
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 0, 1200]  # Volumen 0 en medio
        })

        dm = DataManager()
        zero_volume_bars = dm.detect_zero_volume(data)

        assert len(zero_volume_bars) == 1
        assert zero_volume_bars[0] == 1  # Índice de barra con vol=0

    def test_negative_volume_error(self):
        """Error en volumen negativo"""
        data = pd.DataFrame({
            'Close': [100, 101],
            'Volume': [1000, -500]  # Volumen negativo ❌
        })

        dm = DataManager()
        with pytest.raises(ValueError, match="Negative volume detected"):
            dm.validate_volume(data)

    def test_handle_zero_volume_interpolation(self):
        """Interpola volumen cero con promedio de vecinos"""
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103],
            'Volume': [1000, 0, 0, 1200]
        })

        dm = DataManager()
        fixed = dm.handle_zero_volume(data, method='interpolate')

        # Volumen debe estar interpolado (no cero)
        assert (fixed['Volume'] > 0).all()
        # Debe ser promedio razonable
        assert 500 < fixed.iloc[1]['Volume'] < 1500


class TestDuplicateTimestamps:
    """Tests para manejo de timestamps duplicados"""

    def test_detect_duplicates(self):
        """Detecta timestamps duplicados"""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        # Duplicar un timestamp
        dates = dates.insert(5, dates[5])

        data = pd.DataFrame({
            'Date': dates,
            'Close': [100] * 11
        })

        dm = DataManager()
        duplicates = dm.detect_duplicate_timestamps(data)

        assert len(duplicates) == 1

    def test_remove_duplicates_keep_first(self):
        """Remueve duplicados manteniendo primero"""
        dates = pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:00', '2024-01-01 10:05'])
        data = pd.DataFrame({
            'Date': dates,
            'Close': [100, 101, 102]
        })

        dm = DataManager()
        cleaned = dm.remove_duplicate_timestamps(data, keep='first')

        assert len(cleaned) == 2
        assert cleaned.iloc[0]['Close'] == 100  # Mantuvo primero

    def test_remove_duplicates_average(self):
        """Promedia valores en timestamps duplicados"""
        dates = pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:00', '2024-01-01 10:05'])
        data = pd.DataFrame({
            'Date': dates,
            'Close': [100, 102, 105]
        })

        dm = DataManager()
        cleaned = dm.remove_duplicate_timestamps(data, keep='average')

        assert len(cleaned) == 2
        assert cleaned.iloc[0]['Close'] == 101  # Promedio de 100 y 102


class TestLargeDatasetHandling:
    """Tests para manejo de datasets grandes"""

    @pytest.mark.slow
    def test_process_million_bars(self):
        """Procesa 1M+ barras sin crash de memoria"""
        # Simular 1 millón de barras
        dates = pd.date_range('2020-01-01', periods=1_000_000, freq='1min')

        # Generar datos sintéticos
        np.random.seed(42)
        data = pd.DataFrame({
            'Date': dates,
            'Close': 45000 + np.random.randn(1_000_000) * 1000,
            'Volume': np.random.randint(100, 10000, 1_000_000)
        })

        dm = DataManager()
        # Debe procesar en chunks
        result = dm.process_large_dataset(data, chunk_size=100_000)

        assert len(result) == 1_000_000
        # Verificar que no hay memory leak
        # (implementación específica depende de profiler)

    def test_chunked_indicator_calculation(self):
        """Calcula indicadores en chunks para datos grandes"""
        large_data = pd.DataFrame({
            'Close': np.random.randn(500_000) + 45000
        })

        dm = DataManager()
        # SMA debe calcularse en chunks
        sma = dm.calculate_sma_chunked(large_data, period=200, chunk_size=50_000)

        assert len(sma) == 500_000
        assert not np.isnan(sma['SMA_200'].iloc[-1])  # Último valor válido


class TestDataIntegrityE2E:
    """Tests end-to-end de integridad de datos"""

    def test_complete_validation_pipeline(self):
        """Pipeline completo de validación"""
        # Datos con múltiples problemas
        problematic_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'Open': [100] * 100,
            'High': [105] * 100,
            'Low': [95] * 100,
            'Close': [103] * 100,
            'Volume': [1000] * 100
        })

        # Inyectar problemas
        problematic_data.loc[10, 'High'] = 90  # High < Low
        problematic_data.loc[20, 'Volume'] = 0  # Volumen cero
        problematic_data.loc[30, 'Volume'] = -100  # Volumen negativo

        dm = DataManager()

        # Pipeline debe detectar y corregir/alertar todos los problemas
        with pytest.raises(ValueError) as exc_info:
            dm.validate_complete(problematic_data)

        errors = str(exc_info.value)
        assert "High must be >= Low" in errors

    def test_validation_performance(self):
        """Validación debe ser rápida incluso con datos grandes"""
        import time

        data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100_000, freq='1min'),
            'Open': [100] * 100_000,
            'High': [105] * 100_000,
            'Low': [95] * 100_000,
            'Close': [103] * 100_000,
            'Volume': [1000] * 100_000
        })

        dm = DataManager()

        start = time.time()
        dm.validate_complete(data)
        elapsed = time.time() - start

        # Validación debe tomar < 5 segundos para 100K barras
        assert elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
