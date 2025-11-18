"""
Tests for Data Validator Module
"""

import pytest
import pandas as pd
import numpy as np
from core.data.data_validator import DataValidator, ValidationSeverity, validate_dataframe


class TestDataValidator:
    """Test suite for DataValidator"""

    def test_ohlc_valid(self):
        """Test valid OHLC relationships"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105]
        })

        validator = DataValidator()
        result = validator.validate_ohlc_relationships(df)

        assert result.passed == True
        assert result.severity == ValidationSeverity.INFO

    def test_ohlc_invalid(self):
        """Test invalid OHLC relationships"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [95, 106, 107],  # Open > High
            'low': [105, 96, 97],   # Low > High
            'close': [103, 104, 105]
        })

        validator = DataValidator()
        result = validator.validate_ohlc_relationships(df)

        assert result.passed == False
        assert result.severity == ValidationSeverity.WARNING
        assert "invalid OHLC relationships" in result.message

    def test_time_gaps_detection(self):
        """Test time gaps detection"""
        # Create data with 5min frequency but missing some bars
        timestamps = pd.date_range('2023-01-01 09:00:00', periods=5, freq='5min')
        # Remove one timestamp to create gap
        timestamps = timestamps.delete(2)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100, 101, 103, 104]
        })

        validator = DataValidator()
        result = validator.detect_time_gaps(df, '5min')

        assert result.passed == False
        assert result.severity == ValidationSeverity.ERROR  # 1 gap out of 4 rows = 25% > 10%
        assert "time gaps" in result.message

    def test_duplicate_timestamps(self):
        """Test duplicate timestamps detection"""
        timestamps = ['2023-01-01 09:00:00', '2023-01-01 09:00:00', '2023-01-01 09:05:00']
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'open': [100, 101, 102]
        })

        validator = DataValidator()
        result = validator.check_duplicate_timestamps(df)

        assert result.passed == False
        assert result.severity == ValidationSeverity.ERROR
        assert "duplicate timestamps" in result.message

    def test_timezone_validation(self):
        """Test timezone validation"""
        # Timezone-naive timestamps
        timestamps = pd.date_range('2023-01-01', periods=3, freq='1h')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100, 101, 102]
        })

        validator = DataValidator()
        result = validator.validate_timezone(df, 'UTC')

        assert result.passed == False
        assert result.severity == ValidationSeverity.WARNING
        assert "timezone-naive" in result.message

    def test_look_ahead_bias(self):
        """Test look-ahead bias detection"""
        # Unsorted timestamps
        timestamps = ['2023-01-01 09:05:00', '2023-01-01 09:00:00', '2023-01-01 09:10:00']
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'open': [100, 101, 102]
        })

        validator = DataValidator()
        result = validator.detect_look_ahead_bias(df)

        assert result.passed == False
        assert result.severity == ValidationSeverity.ERROR
        assert "not sorted chronologically" in result.message

    def test_volume_validation(self):
        """Test volume validation"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1h'),
            'open': [100, 101, 102, 103, 104],
            'volume': [1000, 0, 0, 0, 0]  # Mostly zero volume
        })

        validator = DataValidator()
        result = validator.validate_volume(df)

        assert result.passed == False
        assert result.severity == ValidationSeverity.ERROR
        assert "zero volume" in result.message

    def test_large_dataset(self):
        """Test large dataset detection"""
        # Create large dataset
        timestamps = pd.date_range('2023-01-01', periods=150000, freq='1min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.random.randn(150000) + 100
        })

        validator = DataValidator()
        result = validator.check_large_dataset(df, max_rows=100000)

        assert result.passed == False
        assert result.severity == ValidationSeverity.WARNING
        assert "Dataset has" in result.message and "rows" in result.message

    def test_run_all_validations(self):
        """Test running all validations"""
        # Create valid dataset
        timestamps = pd.date_range('2023-01-01', periods=10, freq='5min', tz='UTC')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000 + i*10 for i in range(10)]
        })

        validator = DataValidator()
        results = validator.run_all_validations(df)

        assert len(results) == 7  # All validation checks

        # Check that all passed
        passed_results = [r for r in results if r.passed]
        assert len(passed_results) == 7

    def test_validation_summary(self):
        """Test validation summary"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='5min', tz='UTC'),
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105]
        })

        summary = validate_dataframe(df)

        assert summary['status'] == 'passed'
        assert summary['total_checks'] == 7
        assert summary['passed'] == 7
        assert summary['errors'] == 0


if __name__ == '__main__':
    pytest.main([__file__])