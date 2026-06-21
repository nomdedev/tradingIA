"""
Tests for corrupted data handling

Validates that the system properly handles:
- Negative prices
- NaN values
- Missing columns
- Inconsistent OHLC data
- Empty datasets
"""

import pytest
import pandas as pd
import numpy as np


class TestCorruptedDataHandling:
    """Test corrupted data detection and handling"""
    
    @pytest.fixture
    def valid_df(self):
        """Create valid OHLCV DataFrame"""
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 105, 100),
            'high': np.random.uniform(105, 110, 100),
            'low': np.random.uniform(95, 100, 100),
            'close': np.random.uniform(100, 105, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }).set_index('timestamp')
    
    def test_negative_prices_detection(self, valid_df):
        """Test detection of negative price values"""
        from core.backend_core import StrategyEngine
        
        # Inject negative prices
        df_corrupt = valid_df.copy()
        df_corrupt.iloc[10, df_corrupt.columns.get_loc('close')] = -50.0
        
        is_valid, issues = StrategyEngine.validate_price_data(df_corrupt)
        
        assert not is_valid, "Should detect negative prices"
        assert any('non-positive' in issue for issue in issues), "Should report non-positive values"
    
    def test_zero_prices_detection(self, valid_df):
        """Test detection of zero price values"""
        from core.backend_core import StrategyEngine
        
        # Inject zero prices
        df_corrupt = valid_df.copy()
        df_corrupt.iloc[5, df_corrupt.columns.get_loc('open')] = 0.0
        
        is_valid, issues = StrategyEngine.validate_price_data(df_corrupt)
        
        assert not is_valid, "Should detect zero prices"
        assert any('non-positive' in issue for issue in issues)
    
    def test_nan_values_detection(self, valid_df):
        """Test detection of NaN values"""
        from core.backend_core import StrategyEngine
        
        # Inject NaN values
        df_corrupt = valid_df.copy()
        df_corrupt.iloc[20:25, df_corrupt.columns.get_loc('high')] = np.nan
        
        is_valid, issues = StrategyEngine.validate_price_data(df_corrupt)
        
        assert not is_valid, "Should detect NaN values"
        assert any('NaN' in issue for issue in issues)
    
    def test_high_less_than_low_detection(self, valid_df):
        """Test detection of high < low anomaly"""
        from core.backend_core import StrategyEngine
        
        # Create invalid OHLC
        df_corrupt = valid_df.copy()
        df_corrupt.iloc[30, df_corrupt.columns.get_loc('high')] = 90.0  # Less than low
        df_corrupt.iloc[30, df_corrupt.columns.get_loc('low')] = 100.0
        
        is_valid, issues = StrategyEngine.validate_price_data(df_corrupt)
        
        assert not is_valid, "Should detect high < low"
        assert any('high < low' in issue for issue in issues)
    
    def test_close_outside_range_detection(self, valid_df):
        """Test detection of close outside high-low range"""
        from core.backend_core import StrategyEngine
        
        # Create close above high
        df_corrupt = valid_df.copy()
        df_corrupt.iloc[40, df_corrupt.columns.get_loc('close')] = 200.0  # Way above high
        
        is_valid, issues = StrategyEngine.validate_price_data(df_corrupt)
        
        assert not is_valid, "Should detect close outside range"
        assert any('close outside' in issue for issue in issues)
    
    def test_missing_columns_detection(self):
        """Test detection of missing required columns"""
        from core.backend_core import StrategyEngine
        
        # DataFrame missing 'close' column
        df_incomplete = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97]
            # 'close' is missing
        })
        
        is_valid, issues = StrategyEngine.validate_price_data(df_incomplete)
        
        assert not is_valid, "Should detect missing columns"
        assert any('Missing columns' in issue for issue in issues)
    
    def test_valid_data_passes(self, valid_df):
        """Test that valid data passes validation"""
        from core.backend_core import StrategyEngine
        
        # Fix the valid_df to ensure OHLC consistency
        df = valid_df.copy()
        for i in range(len(df)):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            df.iloc[i, df.columns.get_loc('open')] = np.random.uniform(low, high)
            df.iloc[i, df.columns.get_loc('close')] = np.random.uniform(low, high)
        
        is_valid, issues = StrategyEngine.validate_price_data(df)
        
        assert is_valid, f"Valid data should pass validation. Issues: {issues}"
        assert len(issues) == 0, "Should have no issues"


class TestDataSufficiencyValidation:
    """Test data sufficiency checks"""
    
    def test_insufficient_data_for_backtest(self):
        """Test that insufficient data is properly rejected"""
        from core.execution.backtester_core import BacktesterCore
        
        backtester = BacktesterCore.__new__(BacktesterCore)
        backtester.logger = __import__('logging').getLogger('test')
        
        # Create DataFrame with only 10 bars (should fail for min_bars=50)
        small_df = pd.DataFrame({
            'open': range(10),
            'high': range(1, 11),
            'low': range(-1, 9),
            'close': range(10),
            'volume': [100] * 10
        }, index=pd.date_range('2023-01-01', periods=10, freq='5min'))
        
        df_multi = {'5Min': small_df}
        
        with pytest.raises(ValueError, match="Insufficient data"):
            backtester.validate_data_sufficiency(df_multi, min_bars=50)
    
    def test_empty_dataframe_rejected(self):
        """Test that empty DataFrames are rejected"""
        from core.execution.backtester_core import BacktesterCore
        
        backtester = BacktesterCore.__new__(BacktesterCore)
        backtester.logger = __import__('logging').getLogger('test')
        
        empty_df = pd.DataFrame()
        df_multi = {'5Min': empty_df}
        
        with pytest.raises(ValueError, match="Empty dataset"):
            backtester.validate_data_sufficiency(df_multi)


class TestDataRecovery:
    """Test data recovery and cleaning capabilities"""
    
    def test_fillna_forward_fill(self):
        """Test forward fill for NaN handling"""
        df = pd.DataFrame({
            'close': [100.0, np.nan, np.nan, 103.0, 104.0]
        })
        
        # Forward fill
        df_filled = df.ffill()
        
        assert not df_filled['close'].isna().any(), "All NaN should be filled"
        assert np.isclose(df_filled['close'].iloc[1], 100.0), "Should forward fill"
        assert np.isclose(df_filled['close'].iloc[2], 100.0), "Should forward fill"
    
    def test_dropna_removes_invalid_rows(self):
        """Test that dropna properly removes invalid rows"""
        df = pd.DataFrame({
            'open': [100, 101, np.nan, 103],
            'close': [100, 101, 102, 103]
        })
        
        df_clean = df.dropna()
        
        assert len(df_clean) == 3, "Should remove one row"
        assert not df_clean.isna().any().any(), "No NaN should remain"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
