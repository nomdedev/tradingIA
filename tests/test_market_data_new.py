"""
Tests para DataFetcher y obtención de datos de mercado
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class MockAlpacaAPI:
    """Mock de Alpaca API para testing"""
    
    def __init__(self, api_key=None, secret_key=None, base_url=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
    
    def get_crypto_bars(self, symbol, timeframe, start, end):
        """Mock de get_crypto_bars - devuelve columnas en mayúsculas como Alpaca real"""
        # Generar datos mock con OHLCV válido
        dates = pd.date_range(start=start, end=end, freq='5min')[:100]
        
        # Generar datos OHLCV válidos donde high >= low y open/close están dentro del rango
        base_price = np.random.uniform(60000, 70000, len(dates))
        price_range = np.random.uniform(100, 1000, len(dates))
        
        df = pd.DataFrame({
            'Open': base_price,
            'High': base_price + price_range,
            'Low': base_price - price_range,
            'Close': base_price + np.random.uniform(-price_range/2, price_range/2, len(dates)),
            'Volume': np.random.uniform(1000, 10000, len(dates)),
        }, index=dates)
        
        # Crear objeto mock con atributo df
        result = Mock()
        result.df = df
        return result


# Fixtures globales
@pytest.fixture
def good_quality_data():
    """Fixture con datos OHLCV de buena calidad para tests de cache"""
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    
    # Generar datos OHLCV válidos y consistentes
    base_price = 65000.0
    
    df = pd.DataFrame({
        'Open': base_price + np.random.normal(0, 500, 100),
        'High': base_price + np.random.normal(500, 300, 100),
        'Low': base_price + np.random.normal(-500, 300, 100),
        'Close': base_price + np.random.normal(0, 500, 100),
        'Volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Asegurar consistencia OHLC
    for idx in df.index:
        prices = [df.loc[idx, 'Open'], df.loc[idx, 'Close']]
        df.loc[idx, 'High'] = max(prices + [df.loc[idx, 'High']])
        df.loc[idx, 'Low'] = min(prices + [df.loc[idx, 'Low']])
    
    return df


class TestDataFetcher:
    """Tests para DataFetcher"""
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Fixture con DataFetcher mockeado"""
        with patch('api.data_fetcher.tradeapi') as mock_api:
            mock_api.REST.return_value = MockAlpacaAPI()
            
            # Import después del patch
            from api.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            return fetcher
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Fixture con datos OHLCV de ejemplo válidos"""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        # Generar datos OHLCV válidos
        base_price = np.random.uniform(60000, 70000, 100)
        price_range = np.random.uniform(100, 1000, 100)
        
        df = pd.DataFrame({
            'Open': base_price,
            'High': base_price + price_range,
            'Low': base_price - price_range,
            'Close': base_price + np.random.uniform(-price_range/2, price_range/2, 100),
            'Volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        return df
    
    def test_data_fetcher_initialization(self, mock_data_fetcher):

        """Test inicialización de DataFetcher"""
        assert mock_data_fetcher is not None
        assert hasattr(mock_data_fetcher, 'api')
        assert hasattr(mock_data_fetcher, 'rate_limit_delay')
        assert hasattr(mock_data_fetcher, 'max_retries')
    
    def test_parse_timeframe(self, mock_data_fetcher):
        """Test parseo de timeframes"""
        # Verificar que el método existe
        assert hasattr(mock_data_fetcher, '_parse_timeframe')
        
        # Test timeframes comunes
        timeframes = ['1Min', '5Min', '15Min', '1H', '1D']
        for tf in timeframes:
            result = mock_data_fetcher._parse_timeframe(tf)
            assert result is not None
    
    def test_get_historical_data_basic(self, mock_data_fetcher):
        """Test obtención básica de datos históricos"""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        df = mock_data_fetcher.get_historical_data(
            symbol='BTCUSD',
            timeframe='5Min',
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Verificar columnas esperadas
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_cols:
            assert col in df.columns
    
    def test_get_historical_data_with_string_dates(self, mock_data_fetcher):
        """Test con fechas como strings"""
        df = mock_data_fetcher.get_historical_data(
            symbol='BTCUSD',
            timeframe='5Min',
            start_date='2024-01-01',
            end_date='2024-01-07'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_data_validation(self, sample_ohlcv_data):
        """Test validación de datos OHLCV"""
        df = sample_ohlcv_data
        
        # Verificar que high >= low
        assert all(df['High'] >= df['Low'])
        
        # Verificar que todos los valores son positivos
        assert all(df['Volume'] >= 0)
        assert all(df['Close'] > 0)
        
        # Verificar que no hay NaN
        assert not df.isnull().any().any()
    
    def test_data_consistency(self, sample_ohlcv_data):
        """Test consistencia de datos"""
        df = sample_ohlcv_data
        
        # Verificar que open y close están entre low y high
        assert all((df['Open'] >= df['Low']) & (df['Open'] <= df['High']))
        assert all((df['Close'] >= df['Low']) & (df['Close'] <= df['High']))
    
    def test_empty_data_handling(self, mock_data_fetcher):
        """Test manejo de datos vacíos"""
        with patch.object(mock_data_fetcher.api, 'get_crypto_bars') as mock_bars:
            # Configurar para retornar DataFrame vacío
            result = Mock()
            result.df = pd.DataFrame()
            mock_bars.return_value = result
            
            df = mock_data_fetcher.get_historical_data(
                symbol='INVALID',
                timeframe='5Min'
            )
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0


class TestMarketDataQuality:
    """Tests para calidad de datos de mercado"""
    
    @pytest.fixture
    def good_quality_data_local(self):
        """Datos de buena calidad"""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        base_price = 65000
        df = pd.DataFrame({
            'Open': [base_price + np.random.normal(0, 100) for _ in range(100)],
            'High': [base_price + np.random.normal(200, 100) for _ in range(100)],
            'Low': [base_price + np.random.normal(-200, 100) for _ in range(100)],
            'Close': [base_price + np.random.normal(0, 100) for _ in range(100)],
            'Volume': np.random.uniform(1000, 10000, 100),
        }, index=dates)
        
        # Asegurar consistencia OHLC
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return df
    
    @pytest.fixture
    def poor_quality_data(self):
        """Datos de mala calidad con inconsistencias"""
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')
        
        df = pd.DataFrame({
            'Open': np.random.uniform(60000, 70000, 50),
            'High': np.random.uniform(60000, 70000, 50),
            'Low': np.random.uniform(60000, 70000, 50),
            'Close': np.random.uniform(60000, 70000, 50),
            'Volume': np.random.uniform(0, 10000, 50),
        }, index=dates)
        
        # Introducir inconsistencias
        df.loc[df.index[10], 'High'] = float(df.loc[df.index[10], 'Low']) - 100  # High < Low
        df.loc[df.index[20], 'Volume'] = -100  # Volume negativo
        
        return df
    
    def test_ohlc_consistency(self, good_quality_data_local):
        """Test consistencia OHLC"""
        df = good_quality_data_local
        
        # High debe ser el máximo
        assert all(df['High'] >= df['Open'])
        assert all(df['High'] >= df['Close'])
        assert all(df['High'] >= df['Low'])
        
        # Low debe ser el mínimo
        assert all(df['Low'] <= df['Open'])
        assert all(df['Low'] <= df['Close'])
        assert all(df['Low'] <= df['High'])
    
    def test_detect_inconsistent_data(self, poor_quality_data):
        """Test detección de datos inconsistentes"""
        df = poor_quality_data
        
        # Detectar high < low
        inconsistent_hl = df[df['High'] < df['Low']]
        assert len(inconsistent_hl) > 0  # Debería detectar la inconsistencia
        
        # Detectar volumen negativo
        negative_volume = df[df['Volume'] < 0]
        assert len(negative_volume) > 0
    
    def test_data_completeness(self, good_quality_data_local):
        """Test completitud de datos"""
        df = good_quality_data_local
        
        # No debería haber valores NaN
        assert not df.isnull().any().any()
        
        # Todas las filas deberían tener datos
        assert len(df) > 0
        assert all(df.count() == len(df))
    
    def test_timestamp_continuity(self, good_quality_data):
        """Test continuidad de timestamps"""
        df = good_quality_data
        
        # Los timestamps deberían estar ordenados
        assert df.index.is_monotonic_increasing
        
        # No debería haber duplicados
        assert not df.index.duplicated().any()
    
    def test_price_range_validity(self, good_quality_data):
        """Test validez de rangos de precios"""
        df = good_quality_data
        
        # Los precios deberían ser positivos
        assert all(df['Open'] > 0)
        assert all(df['High'] > 0)
        assert all(df['Low'] > 0)
        assert all(df['Close'] > 0)
        
        # El volumen debería ser no negativo
        assert all(df['Volume'] >= 0)
    
    def test_volatility_metrics(self, good_quality_data):
        """Test cálculo de métricas de volatilidad"""
        df = good_quality_data
        
        # Calcular rango (high - low)
        df['range'] = df['High'] - df['Low']
        
        # El rango debería ser positivo o cero
        assert all(df['range'] >= 0)
        
        # Calcular volatilidad simple (std de returns)
        df['returns'] = df['Close'].pct_change()
        volatility = df['returns'].std()
        
        # La volatilidad debería ser un número válido
        assert not np.isnan(volatility)
        assert volatility >= 0


class TestDataCaching:
    """Tests para sistema de caché de datos"""
    
    def test_cache_directory_creation(self, tmp_path):
        """Test creación de directorio de caché"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        assert cache_dir.exists()
        assert cache_dir.is_dir()
    
    def test_cache_file_save_load(self, tmp_path, good_quality_data):
        """Test guardar y cargar desde caché"""
        cache_file = tmp_path / "test_data.csv"
        
        # Guardar
        good_quality_data.to_csv(cache_file)
        assert cache_file.exists()
        
        # Cargar
        loaded_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Verificar que los datos son iguales
        pd.testing.assert_frame_equal(
            good_quality_data.reset_index(drop=True),
            loaded_data.reset_index(drop=True)
        )
    
    def test_cache_expiration(self, tmp_path):
        """Test expiración de caché"""
        import time
        
        cache_file = tmp_path / "test_data.csv"
        
        # Crear archivo
        pd.DataFrame({'test': [1, 2, 3]}).to_csv(cache_file)
        
        # Obtener tiempo de modificación
        mtime = cache_file.stat().st_mtime
        current_time = time.time()
        
        # El archivo debería ser reciente
        age_seconds = current_time - mtime
        assert age_seconds < 10  # Menos de 10 segundos
