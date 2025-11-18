"""
Tests for Optimized Backtester
"""

import pytest
import pandas as pd
import numpy as np
from core.execution.optimized_backtester import OptimizedBacktester, BacktestResult

def sample_strategy(data, threshold=0.01):
    """Sample strategy for testing"""
    signals = pd.Series(0, index=data.index)
    returns = data['close'].pct_change()

    # Simple momentum strategy
    signals[returns > threshold] = 1
    signals[returns < -threshold] = -1

    return signals

class TestOptimizedBacktester:
    """Test optimized backtester"""

    def test_single_backtest(self):
        """Test single backtest execution"""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        backtester = OptimizedBacktester()
        signals = sample_strategy(data)
        result = backtester._run_backtest(data, signals)

        assert isinstance(result, BacktestResult)
        assert 'total_return' in result.metrics
        assert 'sharpe_ratio' in result.metrics
        assert isinstance(result.equity_curve, pd.Series)
        assert result.execution_time > 0

    def test_parallel_backtests(self):
        """Test parallel backtest execution"""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100
        }, index=dates)

        backtester = OptimizedBacktester(max_workers=2)

        # Test parameter sets
        parameter_sets = [
            {'threshold': 0.01},
            {'threshold': 0.02},
            {'threshold': 0.005}
        ]

        results = backtester.run_parallel_backtests(data, sample_strategy, parameter_sets)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, BacktestResult)
            assert result.parameter_set is not None

    def test_backtest_metrics(self):
        """Test that metrics are calculated correctly"""
        # Create predictable data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        data = pd.DataFrame({'close': prices}, index=dates)

        # Create signals that should generate profit
        signals = pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=dates)

        backtester = OptimizedBacktester()
        result = backtester._run_backtest(data, signals)

        assert result.metrics['total_return'] > 0
        assert len(result.trades) > 0

        # Test advanced metrics are present
        expected_metrics = [
            'total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'recovery_factor', 'k_ratio', 'max_drawdown', 'hurst_exponent',
            'bootstrap_confidence', 'win_rate', 'profit_factor'
        ]

        for metric in expected_metrics:
            assert metric in result.metrics, f"Missing metric: {metric}"
            assert isinstance(result.metrics[metric], (int, float)), f"Invalid type for {metric}"

    def test_advanced_metrics_calculation(self):
        """Test advanced metrics calculation with known data"""
        # Create data with known characteristics
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # Create equity curve with steady growth and some drawdown
        equity = np.cumsum(np.random.normal(0.001, 0.01, 100)) + 10000
        data = pd.DataFrame({'close': equity}, index=dates)

        signals = pd.Series(0, index=dates)
        signals.iloc[10] = 1  # Single trade

        backtester = OptimizedBacktester()
        result = backtester._run_backtest(data, signals)

        # Verify metrics are reasonable
        assert -1 <= result.metrics['total_return'] <= 1  # Reasonable return range
        assert result.metrics['max_drawdown'] <= 0  # Drawdown should be negative or zero
        assert 0 <= result.metrics['hurst_exponent'] <= 1  # Hurst should be in [0,1]
        assert result.metrics['win_rate'] >= 0  # Win rate should be non-negative
        assert result.metrics['profit_factor'] >= 0  # Profit factor should be non-negative

if __name__ == '__main__':
    pytest.main([__file__])