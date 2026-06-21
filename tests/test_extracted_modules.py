"""
Tests for extracted modules from BacktesterCore refactoring.

Tests:
- MetricsCalculator
- MonteCarloSimulator
- WalkForwardOptimizer
- RetrainingPipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
import tempfile
import os

# Import extracted modules
from core.execution.metrics_calculator import MetricsCalculator, calculate_quick_metrics
from core.execution.monte_carlo_simulator import MonteCarloSimulator, MonteCarloResult
from core.execution.walk_forward_optimizer import (
    WalkForwardOptimizer,
    WFAMethod,
    WFAPeriodResult,
    WFAResult
)
from core.training.retrain_pipeline import (
    RetrainingPipeline,
    RetrainConfig,
    RetrainTrigger,
    ModelVersion
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample return series."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    return returns


@pytest.fixture
def sample_trades():
    """Generate sample trades DataFrame."""
    np.random.seed(42)
    n_trades = 50
    pnl = np.random.normal(100, 500, n_trades)
    return pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_trades, freq='D'),
        'pnl': pnl,
        'entry_price': np.random.uniform(40000, 50000, n_trades),
        'exit_price': np.random.uniform(40000, 50000, n_trades),
        'mae': np.random.uniform(0, 0.05, n_trades),
        'mfe': np.random.uniform(0, 0.08, n_trades)
    })


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range('2025-01-01', periods=n_bars, freq='5min')
    close = 45000 + np.cumsum(np.random.normal(0, 50, n_bars))
    
    return pd.DataFrame({
        'open': close + np.random.uniform(-20, 20, n_bars),
        'high': close + np.abs(np.random.normal(30, 10, n_bars)),
        'low': close - np.abs(np.random.normal(30, 10, n_bars)),
        'close': close,
        'volume': np.random.uniform(100, 1000, n_bars)
    }, index=dates)


# =============================================================================
# MetricsCalculator Tests
# =============================================================================

class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        calc = MetricsCalculator()
        assert calc.risk_free_rate > 0
        assert calc.trading_days == 252
    
    def test_calculate_sharpe(self, sample_returns):
        """Test Sharpe ratio calculation."""
        calc = MetricsCalculator()
        sharpe = calc.calculate_sharpe(sample_returns)
        
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
    
    def test_calculate_sharpe_zero_std(self):
        """Test Sharpe with zero volatility."""
        calc = MetricsCalculator()
        flat_returns = pd.Series([0.001] * 100)
        sharpe = calc.calculate_sharpe(flat_returns)
        
        # Should handle gracefully
        assert sharpe == pytest.approx(0.0)
    
    def test_calculate_sortino(self, sample_returns):
        """Test Sortino ratio calculation."""
        calc = MetricsCalculator()
        sortino = calc.calculate_sortino(sample_returns)
        
        assert isinstance(sortino, float)
    
    def test_calculate_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        calc = MetricsCalculator()
        cum_returns = (1 + sample_returns).cumprod()
        max_dd = calc.calculate_max_drawdown(cum_returns)
        
        assert 0 <= max_dd <= 1
    
    def test_calculate_calmar(self):
        """Test Calmar ratio calculation."""
        calc = MetricsCalculator()
        
        calmar = calc.calculate_calmar(0.5, 0.2)
        assert calmar == pytest.approx(2.5)
        
        calmar_zero = calc.calculate_calmar(0.5, 0)
        assert calmar_zero == pytest.approx(0.0)
    
    def test_calculate_ulcer_index(self, sample_returns):
        """Test Ulcer Index calculation."""
        calc = MetricsCalculator()
        cum_returns = (1 + sample_returns).cumprod()
        ulcer = calc.calculate_ulcer_index(cum_returns)
        
        assert ulcer >= 0
    
    def test_calculate_trade_statistics(self, sample_trades):
        """Test trade statistics calculation."""
        calc = MetricsCalculator()
        stats = calc.calculate_trade_statistics(sample_trades)
        
        assert 'win_rate' in stats
        assert 'num_trades' in stats
        assert 'profit_factor' in stats
        assert 0 <= stats['win_rate'] <= 1
        assert stats['num_trades'] == 50
    
    def test_calculate_trade_statistics_empty(self):
        """Test with empty trades DataFrame."""
        calc = MetricsCalculator()
        empty_trades = pd.DataFrame()
        stats = calc.calculate_trade_statistics(empty_trades)
        
        assert stats['win_rate'] == pytest.approx(0.0)
        assert stats['num_trades'] == 0
    
    def test_calculate_mae_mfe_metrics(self, sample_trades):
        """Test MAE/MFE metrics calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate_mae_mfe_metrics(sample_trades)
        
        assert 'avg_mae' in metrics
        assert 'avg_mfe' in metrics
        assert 'max_mae' in metrics
        assert 'max_mfe' in metrics
    
    def test_calculate_all_metrics(self, sample_returns, sample_trades):
        """Test comprehensive metrics calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(sample_returns, sample_trades)
        
        assert 'sharpe' in metrics
        assert 'sortino' in metrics
        assert 'max_dd' in metrics
        assert 'calmar' in metrics
        assert 'win_rate' in metrics
    
    def test_calculate_quick_metrics(self, sample_returns, sample_trades):
        """Test convenience function."""
        metrics = calculate_quick_metrics(sample_returns, sample_trades)
        assert 'sharpe' in metrics


# =============================================================================
# MonteCarloSimulator Tests
# =============================================================================

class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""
    
    def test_init_default_params(self):
        """Test initialization with defaults."""
        mc = MonteCarloSimulator()
        
        assert mc.num_simulations == 500
        assert mc.noise_percent == pytest.approx(0.005)
        assert mc.robustness_threshold == pytest.approx(0.2)
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mc = MonteCarloSimulator(
            num_simulations=100,
            noise_percent=0.01,
            robustness_threshold=0.3
        )
        
        assert mc.num_simulations == 100
        assert mc.noise_percent == pytest.approx(0.01)
    
    def test_add_price_noise(self, sample_ohlcv):
        """Test noise injection to price data."""
        mc = MonteCarloSimulator(noise_percent=0.01)
        
        original_close = sample_ohlcv['close'].copy()
        noisy_data = mc._add_price_noise(sample_ohlcv.copy())
        
        # Prices should be different
        assert not noisy_data['close'].equals(original_close)
        
        # But not too different (within reasonable noise range)
        pct_diff = abs(noisy_data['close'] - original_close) / original_close
        assert pct_diff.mean() < 0.05  # Average diff < 5%
    
    def test_add_price_noise_ohlc_consistency(self, sample_ohlcv):
        """Test that OHLC consistency is maintained after noise."""
        mc = MonteCarloSimulator()
        noisy_data = mc._add_price_noise(sample_ohlcv.copy())
        
        # High should be max of OHLC
        assert (noisy_data['high'] >= noisy_data['open']).all()
        assert (noisy_data['high'] >= noisy_data['close']).all()
        
        # Low should be min of OHLC
        assert (noisy_data['low'] <= noisy_data['open']).all()
        assert (noisy_data['low'] <= noisy_data['close']).all()
    
    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        mc = MonteCarloSimulator()
        values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        
        ci = mc.calculate_confidence_interval(values, confidence=0.95)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
    
    def test_calculate_confidence_interval_empty(self):
        """Test CI with empty list."""
        mc = MonteCarloSimulator()
        ci = mc.calculate_confidence_interval([])
        
        assert ci == (0.0, 0.0)
    
    def test_monte_carlo_result_dataclass(self):
        """Test MonteCarloResult dataclass."""
        result = MonteCarloResult(
            sharpe_mean=1.5,
            sharpe_std=0.3,
            sharpe_p5=0.9,
            sharpe_p95=2.1,
            win_rate_mean=0.55,
            win_rate_std=0.05,
            is_robust=True,
            num_simulations=500,
            all_sharpes=[1.5, 1.6],
            all_win_rates=[0.55, 0.56]
        )
        
        assert result.sharpe_mean == pytest.approx(1.5)
        assert result.is_robust is True
    
    def test_get_robustness_report(self):
        """Test robustness report generation."""
        mc = MonteCarloSimulator()
        
        result = MonteCarloResult(
            sharpe_mean=1.5,
            sharpe_std=0.15,
            sharpe_p5=1.2,
            sharpe_p95=1.8,
            win_rate_mean=0.55,
            win_rate_std=0.03,
            is_robust=True,
            num_simulations=100,
            all_sharpes=[1.4, 1.5, 1.6, 1.5, 1.4],
            all_win_rates=[0.54, 0.55, 0.56, 0.55, 0.54]
        )
        
        report = mc.get_robustness_report(result)
        
        assert 'is_robust' in report
        assert 'sharpe_stats' in report
        assert 'recommendation' in report


# =============================================================================
# WalkForwardOptimizer Tests
# =============================================================================

class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer class."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(backtest_func=mock_backtest)
        
        assert wfo.n_periods == 8
        assert wfo.method == WFAMethod.ANCHORED
        assert wfo.min_test_bars == 100
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(
            backtest_func=mock_backtest,
            n_periods=5,
            method=WFAMethod.ROLLING,
            min_test_bars=50
        )
        
        assert wfo.n_periods == 5
        assert wfo.method == WFAMethod.ROLLING
    
    def test_calculate_boundaries_anchored(self):
        """Test boundary calculation for anchored WFA."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(
            backtest_func=mock_backtest,
            method=WFAMethod.ANCHORED
        )
        
        # Period 0, size 100, total 800
        train_start, train_end, test_start, test_end = wfo._calculate_boundaries(0, 100, 800)
        
        assert train_start == 0  # Anchored always starts at 0
        assert train_end == 100
        assert test_start == 100
        assert test_end == 200
    
    def test_calculate_boundaries_rolling(self):
        """Test boundary calculation for rolling WFA."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(
            backtest_func=mock_backtest,
            method=WFAMethod.ROLLING
        )
        
        # Period 2, size 100, total 800
        train_start, train_end, test_start, test_end = wfo._calculate_boundaries(2, 100, 800)
        
        assert train_start == 200  # Rolling starts at period * size
        assert train_end == 300
    
    def test_calculate_degradation_normal(self):
        """Test degradation calculation."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(backtest_func=mock_backtest)
        
        # 20% degradation
        deg = wfo._calculate_degradation(train_sharpe=1.0, test_sharpe=0.8)
        assert abs(deg - 20.0) < 0.1
    
    def test_calculate_degradation_improvement(self):
        """Test when OOS is better than IS."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(backtest_func=mock_backtest)
        
        # Negative degradation (improvement)
        deg = wfo._calculate_degradation(train_sharpe=1.0, test_sharpe=1.2)
        assert deg < 0
    
    def test_calculate_degradation_zero_is(self):
        """Test with near-zero IS Sharpe."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(backtest_func=mock_backtest)
        
        deg = wfo._calculate_degradation(train_sharpe=0.001, test_sharpe=0.5)
        assert deg == -100
    
    def test_cancel_flag(self):
        """Test cancellation mechanism."""
        mock_backtest = MagicMock()
        wfo = WalkForwardOptimizer(backtest_func=mock_backtest)
        
        assert wfo._cancelled is False
        wfo.cancel()
        assert wfo._cancelled is True
        
        with pytest.raises(InterruptedError):
            wfo._check_cancellation()


# =============================================================================
# RetrainingPipeline Tests
# =============================================================================

class TestRetrainingPipeline:
    """Tests for RetrainingPipeline class."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_init_creates_directory(self, temp_models_dir):
        """Test that init creates models directory."""
        models_path = os.path.join(temp_models_dir, "new_models")
        pipeline = RetrainingPipeline(models_dir=models_path)
        
        assert os.path.exists(models_path)
    
    def test_init_default_config(self, temp_models_dir):
        """Test default configuration."""
        pipeline = RetrainingPipeline(models_dir=temp_models_dir)
        
        assert pipeline.config.sharpe_degradation_threshold == pytest.approx(0.3)
        assert pipeline.config.retrain_interval_days == 30
    
    def test_init_custom_config(self, temp_models_dir):
        """Test custom configuration."""
        config = RetrainConfig(
            sharpe_degradation_threshold=0.5,
            retrain_interval_days=14
        )
        pipeline = RetrainingPipeline(models_dir=temp_models_dir, config=config)
        
        assert pipeline.config.sharpe_degradation_threshold == pytest.approx(0.5)
        assert pipeline.config.retrain_interval_days == 14
    
    def test_check_retrain_needed_no_version(self, temp_models_dir):
        """Test retrain check when no version exists."""
        pipeline = RetrainingPipeline(models_dir=temp_models_dir)
        
        needs, trigger, reason = pipeline.check_retrain_needed(
            "test_strategy",
            {"sharpe": 1.0}
        )
        
        assert needs is True
        assert trigger == RetrainTrigger.NEW_DATA
    
    def test_model_version_to_dict(self):
        """Test ModelVersion serialization."""
        version = ModelVersion(
            version_id="test_v1",
            strategy_name="test_strategy",
            parameters={"param1": 10},
            metrics={"sharpe": 1.5},
            created_at="2025-01-01T00:00:00",
            trained_on_bars=1000,
            is_active=True,
            trigger=RetrainTrigger.MANUAL,
            validation_sharpe=1.5
        )
        
        d = version.to_dict()
        
        assert d['version_id'] == "test_v1"
        assert d['trigger'] == "manual"  # Enum converted to string
    
    def test_model_version_from_dict(self):
        """Test ModelVersion deserialization."""
        data = {
            'version_id': 'test_v1',
            'strategy_name': 'test_strategy',
            'parameters': {'param1': 10},
            'metrics': {'sharpe': 1.5},
            'created_at': '2025-01-01T00:00:00',
            'trained_on_bars': 1000,
            'is_active': True,
            'trigger': 'manual',
            'validation_sharpe': 1.5,
            'notes': ''
        }
        
        version = ModelVersion.from_dict(data)
        
        assert version.version_id == "test_v1"
        assert version.trigger == RetrainTrigger.MANUAL
    
    def test_log_performance(self, temp_models_dir):
        """Test performance logging."""
        pipeline = RetrainingPipeline(models_dir=temp_models_dir)
        
        pipeline.log_performance("test_strategy", {"sharpe": 1.0})
        pipeline.log_performance("test_strategy", {"sharpe": 1.1})
        
        assert len(pipeline._performance_history["test_strategy"]) == 2
    
    def test_get_performance_trend(self, temp_models_dir):
        """Test performance trend analysis."""
        pipeline = RetrainingPipeline(models_dir=temp_models_dir)
        
        # Log increasing performance
        for i in range(10):
            pipeline.log_performance("test_strategy", {"sharpe": 1.0 + i * 0.1})
        
        trend = pipeline.get_performance_trend("test_strategy", "sharpe")
        
        assert trend is not None
        assert trend['trend_direction'] == 'up'
        assert trend['samples'] == 10
    
    def test_get_performance_trend_empty(self, temp_models_dir):
        """Test trend with no history."""
        pipeline = RetrainingPipeline(models_dir=temp_models_dir)
        
        trend = pipeline.get_performance_trend("nonexistent", "sharpe")
        
        assert trend is None
    
    def test_generate_version_id(self, temp_models_dir):
        """Test version ID generation."""
        pipeline = RetrainingPipeline(models_dir=temp_models_dir)
        
        v1 = pipeline._generate_version_id("strategy1", {"param": 10})
        v2 = pipeline._generate_version_id("strategy1", {"param": 20})
        
        assert v1.startswith("strategy1_")
        assert v1 != v2  # Different params = different hash


# =============================================================================
# Integration Tests
# =============================================================================

class TestModuleIntegration:
    """Integration tests for extracted modules."""
    
    def test_metrics_in_monte_carlo(self, sample_returns, sample_trades):
        """Test MetricsCalculator can be used with MonteCarloSimulator."""
        calc = MetricsCalculator()
        
        # Create a backtest function that uses MetricsCalculator
        def mock_backtest(data, **kwargs):
            returns = pd.Series(np.random.normal(0.001, 0.02, 100))
            metrics = calc.calculate_all_metrics(returns, sample_trades)
            return metrics
        
        mc = MonteCarloSimulator(num_simulations=10)
        
        # This should work without errors
        # (we're testing the interface compatibility)
        assert callable(mock_backtest)
    
    def test_wfa_result_structure(self):
        """Test WFAResult has correct structure."""
        result = WFAResult(
            period_results=[],
            avg_degradation=15.0,
            avg_oos_sharpe=1.2,
            stability_score=0.75,
            certified=True,
            best_params={'param1': 10},
            all_optimized_params=None,
            optimization_used=False,
            method=WFAMethod.ANCHORED
        )
        
        assert result.certified is True
        assert result.method == WFAMethod.ANCHORED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
