import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from datetime import datetime
from src.backend_core import DataManager, StrategyEngine

# Global fixtures
@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_data():
    """Create sample OHLCV DataFrame"""
    dates = pd.date_range('2023-01-01', periods=100, freq='5min')
    np.random.seed(42)
    data = {
        'Date': dates,
        'Open': 45000 + np.random.randn(100) * 1000,
        'High': 45500 + np.random.randn(100) * 1000,
        'Low': 44500 + np.random.randn(100) * 1000,
        'Close': 45000 + np.random.randn(100) * 1000,
        'Volume': np.random.randint(100, 1000, 100)
    }
    df = pd.DataFrame(data)
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    return df

class TestDataManager:
    """Test suite for DataManager class"""

    @patch('alpaca_trade_api.REST')
    def test_load_alpaca_data_success(self, mock_rest, temp_cache_dir, sample_data):
        """Test successful data loading from Alpaca API"""
        # Setup mock
        mock_api_instance = Mock()
        mock_rest.return_value = mock_api_instance
        
        # Mock the get_crypto_bars method to return a mock with df attribute
        mock_bars = Mock()
        # Create proper format for Alpaca response - should have DatetimeIndex
        alpaca_format = sample_data.copy()
        alpaca_format = alpaca_format.set_index('Date')
        alpaca_format.columns = ['open', 'high', 'low', 'close', 'volume']
        mock_bars.df = alpaca_format
        mock_api_instance.get_crypto_bars.return_value = mock_bars

        # Create DataManager with API credentials
        dm = DataManager(api_key='test_key', secret_key='test_secret', cache_dir=temp_cache_dir)
        
        result = dm.load_alpaca_data('BTCUSD', '2023-01-01', '2023-01-02', '5Min')
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'SMA_Vol']
        mock_api_instance.get_crypto_bars.assert_called_once()

    @patch('alpaca_trade_api.REST')
    def test_load_alpaca_data_fallback_to_cache(self, mock_rest, temp_cache_dir, sample_data):
        """Test fallback to cached data when API fails"""
        from src.backend_core import DataManager
        # Setup mock to fail
        mock_api_instance = Mock()
        mock_rest.return_value = mock_api_instance
        mock_api_instance.get_crypto_bars.side_effect = Exception("API Error")

        # Save sample data to cache
        cache_file = os.path.join(temp_cache_dir, 'BTCUSD_5Min.csv')
        sample_data.to_csv(cache_file, index=False)

        # Create DataManager with API credentials
        dm = DataManager(api_key='test_key', secret_key='test_secret', cache_dir=temp_cache_dir)

        # Test loading data (should use cache)
        result = dm.load_alpaca_data('BTCUSD', '2023-01-01', '2023-01-02', '5Min')

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

        # Test loading data (should use cache)
        result = dm.load_alpaca_data('BTCUSD', '2023-01-01', '2023-01-02', '5Min')

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_resample_multi_tf(self, sample_data):
        """Test multi-timeframe resampling"""
        dm = DataManager()

        # Add required columns for resampling
        sample_data = sample_data.set_index('Date')

        result = dm.resample_multi_tf(sample_data)

        # Assertions
        assert isinstance(result, dict)
        assert '5min' in result
        assert '15min' in result
        assert '1h' in result

        # Check shapes (15min should have ~1/3 the rows, 1h should have ~1/12)
        assert len(result['15min']) < len(result['5min'])
        assert len(result['1h']) < len(result['15min'])

        # Check all required columns exist
        for tf_data in result.values():
            assert all(col in tf_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_get_data_info(self, temp_cache_dir, sample_data):
        """Test get_data_info method"""
        # Save sample data to cache
        cache_file = os.path.join(temp_cache_dir, 'BTCUSD_5Min.csv')
        sample_data.to_csv(cache_file, index=False)

        dm = DataManager(cache_dir=temp_cache_dir)

        result = dm.get_data_info()

        # Assertions
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'n_bars' in result
        assert 'start_date' in result
        assert 'end_date' in result
        assert 'last_update' in result
        assert 'status' in result
        assert result['status'] == 'OK'
        assert result['n_bars'] == 100

    def test_save_cache(self, temp_cache_dir, sample_data):
        """Test cache saving functionality"""
        dm = DataManager(cache_dir=temp_cache_dir)

        # Save data
        dm.save_cache(sample_data, 'BTCUSD', '5Min')

        # Check file exists
        cache_file = os.path.join(temp_cache_dir, 'BTCUSD_5Min.csv')
        assert os.path.exists(cache_file)

        # Load and verify
        loaded_data = pd.read_csv(cache_file)
        assert len(loaded_data) == len(sample_data)


class TestStrategyEngine:
    """Test suite for StrategyEngine class"""

    @pytest.fixture
    def sample_config(self, temp_cache_dir):
        """Create sample strategies config file"""
        config = [
            {
                "name": "IBS_BB",
                "module": "mean_reversion_ibs_bb_crypto",
                "class": "IBSBBStrategy",
                "params": {
                    "atr_multi": {"default": 0.3, "min": 0.1, "max": 0.5, "step": 0.05},
                    "vol_thresh": {"default": 1.2, "min": 0.8, "max": 2.0, "step": 0.1}
                }
            },
            {
                "name": "MACD_ADX",
                "module": "momentum_macd_adx_crypto",
                "class": "MACDADXStrategy",
                "params": {
                    "fast_period": {"default": 12, "min": 8, "max": 20, "step": 1},
                    "slow_period": {"default": 26, "min": 20, "max": 40, "step": 1}
                }
            },
            {
                "name": "PAIRS_TRADING",
                "module": "pairs_trading_cointegration_crypto",
                "class": "PairsTradingStrategy",
                "params": {
                    "lookback": {"default": 100, "min": 50, "max": 200, "step": 10}
                }
            },
            {
                "name": "HFT_VMA",
                "module": "hft_momentum_vma",
                "class": "HFTVMAStrategy",
                "params": {
                    "vma_period": {"default": 20, "min": 10, "max": 50, "step": 5}
                }
            },
            {
                "name": "LSTM_ML",
                "module": "lstm_ml_reversion",
                "class": "LSTMMLStrategy",
                "params": {
                    "sequence_length": {"default": 60, "min": 30, "max": 120, "step": 10}
                }
            }
        ]

        config_file = os.path.join(temp_cache_dir, 'strategies_registry.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return config_file

    def test_list_available_strategies(self, sample_config):
        """Test listing available strategies"""
        se = StrategyEngine(strategies_config_file=sample_config)

        strategies = se.list_available_strategies()

        # Assertions
        assert isinstance(strategies, list)
        assert len(strategies) == 5
        assert "IBS_BB" in strategies
        assert "MACD_ADX" in strategies
        assert "PAIRS_TRADING" in strategies
        assert "HFT_VMA" in strategies
        assert "LSTM_ML" in strategies

    def test_get_strategy_params(self, sample_config):
        """Test getting strategy parameters"""
        se = StrategyEngine(strategies_config_file=sample_config)

        params = se.get_strategy_params("IBS_BB")

        # Assertions
        assert isinstance(params, dict)
        assert "atr_multi" in params
        assert "vol_thresh" in params

        # Check parameter structure
        atr_param = params["atr_multi"]
        assert "default" in atr_param
        assert "min" in atr_param
        assert "max" in atr_param
        assert "step" in atr_param
        assert atr_param["default"] == 0.3
        assert atr_param["min"] == 0.1
        assert atr_param["max"] == 0.5

    def test_validate_params_valid(self, sample_config):
        """Test parameter validation with valid parameters"""
        se = StrategyEngine(strategies_config_file=sample_config)

        valid_params = {
            "atr_multi": 0.3,
            "vol_thresh": 1.2
        }

        is_valid, message = se.validate_params("IBS_BB", valid_params)

        assert is_valid is True
        assert message == "Parameters valid"

    def test_validate_params_invalid_range(self, sample_config):
        """Test parameter validation with out-of-range parameters"""
        se = StrategyEngine(strategies_config_file=sample_config)

        invalid_params = {
            "atr_multi": 1.0,  # Above max 0.5
            "vol_thresh": 1.2
        }

        is_valid, message = se.validate_params("IBS_BB", invalid_params)

        assert is_valid is False
        assert "atr_multi" in message.lower()

    def test_validate_params_missing_param(self, sample_config):
        """Test parameter validation with missing parameters"""
        se = StrategyEngine(strategies_config_file=sample_config)

        incomplete_params = {
            "atr_multi": 0.3
            # Missing vol_thresh
        }

        is_valid, message = se.validate_params("IBS_BB", incomplete_params)

        assert is_valid is False
        assert "missing" in message.lower() or "required" in message.lower()

    @patch('importlib.import_module')
    def test_load_strategy_module(self, mock_import, sample_config):
        """Test loading strategy module"""
        # Setup mock
        mock_module = Mock()
        mock_strategy_class = Mock()
        mock_module.IBSBBStrategy = mock_strategy_class
        mock_import.return_value = mock_module

        se = StrategyEngine(strategies_config_file=sample_config)

        strategy_class = se.load_strategy_module("IBS_BB")

        # Assertions
        assert strategy_class == mock_strategy_class
        mock_import.assert_called_with("mean_reversion_ibs_bb_crypto")


if __name__ == "__main__":
    pytest.main([__file__])