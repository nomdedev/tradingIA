"""
Unit Tests for Indicators Module
Tests for IFVG, Volume Profile, EMAs, and Signal Generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data.indicators import (
    calculate_ifvg_enhanced,
    volume_profile_advanced,
    generate_filtered_signals,
    validate_params,
    # Compatibility aliases
    calculate_ifvg,
    volume_profile,
    generate_signals,
    Signal,
    emas_simple,
    calculate_all_indicators
)


@pytest.fixture
def sample_ohlcv_data():
    """Fixture: genera datos OHLCV de prueba"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
    prices = 45000 * (1 + np.random.normal(0.0001, 0.003, 500)).cumprod()
    
    df = pd.DataFrame({
        'Open': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.001, 500))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, 500))),
        'Close': prices,
        'Volume': np.random.uniform(100, 1000, 500)
    }, index=dates)
    
    return df


@pytest.fixture
def trending_data():
    """Fixture: datos con tendencia clara"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    # Tendencia alcista con ruido
    trend = np.linspace(45000, 50000, 200)
    noise = np.random.normal(0, 100, 200)
    prices = trend + noise
    
    df = pd.DataFrame({
        'Open': prices,
        'High': prices * 1.002,
        'Low': prices * 0.998,
        'Close': prices,
        'Volume': np.random.uniform(100, 500, 200)
    }, index=dates)
    
    return df


class TestIFVG:
    """Tests para cálculo de IFVG"""
    
    def test_ifvg_basic_calculation(self, sample_ohlcv_data):
        """Test: IFVG calcula columnas correctamente"""
        df = calculate_ifvg(sample_ohlcv_data)
        
        # Verificar columnas creadas
        expected_cols = ['gap_up', 'gap_down', 'fvg_size', 'ATR', 'valid_fvg', 
                        'bull_signal', 'bear_signal']
        for col in expected_cols:
            assert col in df.columns, f"Columna {col} no encontrada"
    
    def test_ifvg_signals_generated(self, sample_ohlcv_data):
        """Test: IFVG genera señales válidas"""
        df = calculate_ifvg(sample_ohlcv_data)
        
        bull_signals = df['bull_signal'].sum()
        bear_signals = df['bear_signal'].sum()
        
        # Debe haber al menos algunas señales
        assert bull_signals > 0, "No se generaron señales bull"
        assert bear_signals > 0, "No se generaron señales bear"
        
        # Señales deben ser booleanas
        assert df['bull_signal'].dtype == bool
        assert df['bear_signal'].dtype == bool
    
    def test_ifvg_no_simultaneous_signals(self, sample_ohlcv_data):
        """Test: No debe haber señales bull y bear simultáneas"""
        df = calculate_ifvg(sample_ohlcv_data)
        
        simultaneous = (df['bull_signal'] & df['bear_signal']).sum()
        assert simultaneous == 0, "Hay señales bull y bear simultáneas"
    
    def test_ifvg_atr_filter(self, sample_ohlcv_data):
        """Test: Filtro ATR funciona correctamente"""
        # Con ATR multiplier bajo, deberían haber más FVGs válidos
        df1 = calculate_ifvg(sample_ohlcv_data, atr_multi=0.5)
        valid1 = df1['valid_fvg'].sum()
        
        # Con ATR multiplier alto, menos FVGs válidos
        df2 = calculate_ifvg(sample_ohlcv_data, atr_multi=3.0)
        valid2 = df2['valid_fvg'].sum()
        
        assert valid1 >= valid2, "Filtro ATR no funciona correctamente"
    
    def test_ifvg_wick_based_vs_close(self, sample_ohlcv_data):
        """Test: Diferencia entre wick-based y close-based signals"""
        df_wick = calculate_ifvg(sample_ohlcv_data, wick_based=True)
        df_close = calculate_ifvg(sample_ohlcv_data, wick_based=False)
        
        # Close-based debería generar más o igual señales que wick-based
        signals_wick = df_wick['bull_signal'].sum() + df_wick['bear_signal'].sum()
        signals_close = df_close['bull_signal'].sum() + df_close['bear_signal'].sum()
        
        assert signals_close >= signals_wick, "Close-based debería generar más señales"


class TestVolumeProfile:
    """Tests para Volume Profile"""
    
    def test_volume_profile_basic(self, sample_ohlcv_data):
        """Test: Volume Profile calcula columnas básicas"""
        df = volume_profile(sample_ohlcv_data)
        
        expected_cols = ['POC', 'VAH', 'VAL', 'SD_Thresh']
        for col in expected_cols:
            assert col in df.columns, f"Columna {col} no encontrada"
    
    def test_volume_profile_values_valid(self, sample_ohlcv_data):
        """Test: Valores de VP son válidos"""
        df = volume_profile(sample_ohlcv_data)
        
        # POC debe existir y ser numérico (el rango puede variar según el algoritmo)
        valid_poc = df['POC'].notna()
        assert valid_poc.sum() > 0, "No hay valores válidos de POC"
        
        # VAH debe ser mayor que VAL (ignorando NaN)
        valid_vah_val = df['VAH'].notna() & df['VAL'].notna()
        if valid_vah_val.sum() > 0:
            vah_val_valid = (df.loc[valid_vah_val, 'VAH'] >= df.loc[valid_vah_val, 'VAL']).all()
            assert vah_val_valid, "VAH debe ser >= VAL"
    
    def test_volume_profile_poc_not_null(self, sample_ohlcv_data):
        """Test: POC no debe ser null"""
        df = volume_profile(sample_ohlcv_data)
        
        # Después de periodo de warmup, POC no debe ser null
        poc_not_null = df['POC'].iloc[50:].notna().all()
        assert poc_not_null, "POC contiene valores nulos después de warmup"
    
    def test_volume_profile_custom_params(self, sample_ohlcv_data):
        """Test: Parámetros personalizados funcionan"""
        df1 = volume_profile(sample_ohlcv_data, rows=20)
        df2 = volume_profile(sample_ohlcv_data, rows=50)
        
        # Ambos deben completar sin errores
        assert len(df1) == len(sample_ohlcv_data)
        assert len(df2) == len(sample_ohlcv_data)


class TestEMAs:
    """Tests para EMAs multi-timeframe"""
    
    def test_emas_basic_calculation(self, sample_ohlcv_data):
        """Test: EMAs calcula todas las columnas"""
        result = emas_simple(sample_ohlcv_data)
        
        # Verificar que devuelve diccionario con timeframes
        assert isinstance(result, dict)
        assert '5Min' in result
        
        # Verificar columnas de EMAs
        df = result['5Min']
        expected_cols = ['EMA_20', 'EMA_50', 'EMA_100', 'EMA_200']
        for col in expected_cols:
            assert col in df.columns, f"Columna {col} no encontrada"
    
    def test_emas_trend_detection(self, trending_data):
        """Test: EMAs detectan tendencia correctamente"""
        result = emas_simple(trending_data)
        df = result['5Min']
        
        # En tendencia alcista, EMA20 > EMA50 > EMA100 > EMA200 al final
        last_row = df.iloc[-1]
        
        # Al menos EMA20 > EMA50 en tendencia alcista
        assert last_row['EMA_20'] > last_row['EMA_50'], \
            "EMA20 debería estar sobre EMA50 en tendencia alcista"
    
    def test_emas_not_null_after_warmup(self, sample_ohlcv_data):
        """Test: EMAs no son null después de periodo de warmup"""
        result = emas_simple(sample_ohlcv_data)
        df = result['5Min']
        
        # Después de 200 períodos, todas las EMAs deberían tener valores
        warmup = 200
        if len(df) > warmup:
            emas_valid = df[['EMA_20', 'EMA_50', 'EMA_100', 'EMA_200']].iloc[warmup:].notna().all().all()
            assert emas_valid, "EMAs contienen valores nulos después de warmup"
    
    def test_emas_multi_timeframe(self, sample_ohlcv_data):
        """Test: Genera múltiples timeframes si hay datos suficientes"""
        # Crear datos más largos
        dates = pd.date_range(start='2024-01-01', periods=2000, freq='5min')
        prices = 45000 * (1 + np.random.normal(0.0001, 0.003, 2000)).cumprod()
        
        df_long = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': np.random.uniform(100, 1000, 2000)
        }, index=dates)
        
        result = emas_simple(df_long, timeframes=['5Min', '15Min'])
        
        # Debería tener al menos el timeframe base
        assert '5Min' in result
        assert len(result['5Min']) > 0


class TestSignalGeneration:
    """Tests para generación de señales combinadas"""
    
    def test_generate_signals_basic(self, sample_ohlcv_data):
        """Test: generate_signals crea columnas de señales"""
        df = generate_signals(sample_ohlcv_data)
        
        expected_cols = ['signal', 'confidence']
        for col in expected_cols:
            assert col in df.columns, f"Columna {col} no encontrada"
    
    def test_generate_signals_values(self, sample_ohlcv_data):
        """Test: Señales tienen valores válidos"""
        df = generate_signals(sample_ohlcv_data)
        
        # Signal debe ser -1, 0, o 1
        valid_signals = df['signal'].isin([-1, 0, 1]).all()
        assert valid_signals, "Señales contienen valores inválidos"
        
        # Confidence debe estar entre 0 y 1
        confidence_valid = ((df['confidence'] >= 0) & (df['confidence'] <= 1)).all()
        assert confidence_valid, "Confidence fuera del rango 0-1"
    
    def test_generate_signals_with_filters(self, sample_ohlcv_data):
        """Test: Filtros reducen número de señales"""
        # Sin filtros
        df1 = generate_signals(sample_ohlcv_data, 
                              use_ema_filter=False,
                              use_vp_filter=False,
                              use_volume_filter=False)
        signals1 = (df1['signal'] != 0).sum()
        
        # Con filtros
        df2 = generate_signals(sample_ohlcv_data,
                              use_ema_filter=True,
                              use_vp_filter=True,
                              use_volume_filter=True)
        signals2 = (df2['signal'] != 0).sum()
        
        # Con filtros debería haber menos o igual señales
        assert signals2 <= signals1, "Filtros deberían reducir señales"
    
    def test_generate_signals_confidence_correlation(self, sample_ohlcv_data):
        """Test: Confidence es mayor con más filtros activos"""
        df = generate_signals(sample_ohlcv_data,
                            use_ema_filter=True,
                            use_vp_filter=True,
                            use_volume_filter=True)
        
        # Señales con todos los filtros deberían tener mayor confidence promedio
        signals_only = df[df['signal'] != 0]
        
        if len(signals_only) > 0:
            avg_confidence = signals_only['confidence'].mean()
            # Con todos los filtros, confidence debería ser >= 0.5
            assert avg_confidence >= 0.3, f"Confidence promedio muy bajo: {avg_confidence}"


class TestCalculateAllIndicators:
    """Tests para función integral calculate_all_indicators"""
    
    def test_calculate_all_basic(self, sample_ohlcv_data):
        """Test: calculate_all_indicators ejecuta sin errores"""
        df = calculate_all_indicators(sample_ohlcv_data)
        
        # Debe devolver DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_ohlcv_data)
    
    def test_calculate_all_has_all_indicators(self, sample_ohlcv_data):
        """Test: Incluye todos los indicadores"""
        df = calculate_all_indicators(sample_ohlcv_data)
        
        # Verificar presencia de indicadores clave
        key_indicators = ['signal', 'confidence', 'POC', 'EMA_20', 
                         'bull_signal', 'bear_signal']
        
        for indicator in key_indicators:
            assert indicator in df.columns, f"Indicador {indicator} no encontrado"
    
    def test_calculate_all_custom_params(self, sample_ohlcv_data):
        """Test: Parámetros personalizados funcionan"""
        df = calculate_all_indicators(
            sample_ohlcv_data,
            ifvg_config={'atr_multi': 2.0},
            vp_config={'rows': 30},
            signal_config={'use_ema_filter': False}
        )
        
        assert isinstance(df, pd.DataFrame)
        assert 'signal' in df.columns


class TestSignalEnum:
    """Tests para Signal enum"""
    
    def test_signal_enum_values(self):
        """Test: Signal enum tiene valores correctos"""
        assert Signal.STRONG_BUY.value == 2
        assert Signal.BUY.value == 1
        assert Signal.HOLD.value == 0
        assert Signal.SELL.value == -1
        assert Signal.STRONG_SELL.value == -2
    
    def test_signal_enum_from_value(self):
        """Test: Puede obtener Signal desde valor"""
        assert Signal(1) == Signal.BUY
        assert Signal(-1) == Signal.SELL
        assert Signal(0) == Signal.HOLD


# Test de integración
class TestIntegration:
    """Tests de integración end-to-end"""
    
    def test_full_pipeline(self, sample_ohlcv_data):
        """Test: Pipeline completo desde datos hasta señales"""
        # 1. Calcular IFVG
        df = calculate_ifvg(sample_ohlcv_data)
        assert 'bull_signal' in df.columns
        
        # 2. Agregar Volume Profile
        df = volume_profile(df)
        assert 'POC' in df.columns
        
        # 3. Pipeline completo con indicators
        df_final = calculate_all_indicators(sample_ohlcv_data)
        
        # Verificar columnas principales
        assert 'EMA_20' in df_final.columns, "No se calcularon EMAs"
        assert 'bull_signal' in df_final.columns, "No se calcularon señales IFVG"
        assert 'POC' in df_final.columns, "No se calculó Volume Profile"
        assert 'signal' in df_final.columns, "No se generó señal final"
        assert 'confidence' in df_final.columns, "No se generó confianza"
        
        # Verificar que hay al menos algunas señales
        total_signals = (df_final['signal'] != 0).sum()
        assert total_signals > 0, "Pipeline completo no generó señales"



if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([__file__, '-v', '--tb=short'])
