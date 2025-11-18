"""
Tests for Strategy Framework
"""

import pytest
import pandas as pd
import numpy as np
from core.strategies.base_strategy import BaseStrategy, StrategyConfig
from core.strategies.strategy_registry import StrategyRegistry
from core.strategies.momentum_strategy import MomentumStrategy
from core.strategies.mean_reversion_strategy import MeanReversionStrategy
from core.strategies.breakout_strategy import BreakoutStrategy

class TestBaseStrategy:
    """Test base strategy functionality"""

    def test_strategy_config_validation(self):
        """Test strategy configuration validation"""
        # Valid config
        config = StrategyConfig(
            name="TestStrategy",
            description="Test strategy",
            parameters={"param1": 1}
        )
        assert config.name == "TestStrategy"

        # Invalid config - empty name
        with pytest.raises(ValueError):
            StrategyConfig(name="", description="Test")

        # Invalid config - empty description
        with pytest.raises(ValueError):
            StrategyConfig(name="Test", description="")

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented"""
        config = StrategyConfig(
            name="TestStrategy",
            description="Test strategy"
        )

        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseStrategy(config)

class TestMomentumStrategy:
    """Test momentum strategy"""

    def test_initialization(self):
        """Test momentum strategy initialization"""
        config = StrategyConfig(
            name="MomentumStrategy",
            description="Test momentum strategy",
            parameters={
                'fast_period': 10,
                'slow_period': 20,
                'momentum_threshold': 0.02
            }
        )

        strategy = MomentumStrategy(config)
        assert strategy.config.name == "MomentumStrategy"
        assert strategy.get_required_parameters() == ['fast_period', 'slow_period', 'momentum_threshold']

    def test_calculate_indicators(self):
        """Test indicator calculation"""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 100
        }, index=dates)

        config = StrategyConfig(
            name="MomentumStrategy",
            description="Test momentum strategy",
            parameters={
                'fast_period': 10,
                'slow_period': 20,
                'momentum_threshold': 0.02
            }
        )

        strategy = MomentumStrategy(config)
        result = strategy.calculate_indicators(data)

        # Check that indicators were added
        assert 'fast_ma' in result.columns
        assert 'slow_ma' in result.columns
        assert 'momentum' in result.columns
        assert 'rsi' in result.columns

    def test_generate_signals(self):
        """Test signal generation"""
        # Create trending data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.linspace(100, 120, 50) + np.random.normal(0, 1, 50)
        data = pd.DataFrame({
            'open': prices - 1,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices
        }, index=dates)

        config = StrategyConfig(
            name="MomentumStrategy",
            description="Test momentum strategy",
            parameters={
                'fast_period': 5,
                'slow_period': 10,
                'momentum_threshold': 0.01
            }
        )

        strategy = MomentumStrategy(config)
        data_with_indicators = strategy.calculate_indicators(data)
        signals = strategy.generate_signals(data_with_indicators)

        assert len(signals) == len(data)
        assert signals.dtype in [np.int64, np.int32, int]
        # Check that signals are valid (-1, 0, 1)
        assert all(signal in [-1, 0, 1] for signal in signals)

class TestMeanReversionStrategy:
    """Test mean reversion strategy"""

    def test_initialization(self):
        """Test mean reversion strategy initialization"""
        config = StrategyConfig(
            name="MeanReversionStrategy",
            description="Test mean reversion strategy",
            parameters={
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5
            }
        )

        strategy = MeanReversionStrategy(config)
        assert strategy.config.name == "MeanReversionStrategy"
        assert strategy.get_required_parameters() == ['lookback_period', 'entry_threshold', 'exit_threshold']

    def test_calculate_indicators(self):
        """Test indicator calculation"""
        # Create oscillating data around a mean
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_price = 100
        oscillation = np.sin(np.linspace(0, 4*np.pi, 100)) * 5
        prices = base_price + oscillation + np.random.normal(0, 1, 100)

        data = pd.DataFrame({
            'open': prices - 0.5,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices
        }, index=dates)

        config = StrategyConfig(
            name="MeanReversionStrategy",
            description="Test mean reversion strategy",
            parameters={
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5
            }
        )

        strategy = MeanReversionStrategy(config)
        result = strategy.calculate_indicators(data)

        # Check that indicators were added
        assert 'rolling_mean' in result.columns
        assert 'rolling_std' in result.columns
        assert 'z_score' in result.columns
        assert 'bb_upper' in result.columns
        assert 'bb_lower' in result.columns

class TestBreakoutStrategy:
    """Test breakout strategy"""

    def test_initialization(self):
        """Test breakout strategy initialization"""
        config = StrategyConfig(
            name="BreakoutStrategy",
            description="Test breakout strategy",
            parameters={
                'lookback_period': 20,
                'breakout_threshold': 0.05,
                'consolidation_period': 10
            }
        )

        strategy = BreakoutStrategy(config)
        assert strategy.config.name == "BreakoutStrategy"
        assert strategy.get_required_parameters() == ['lookback_period', 'breakout_threshold', 'consolidation_period']

class TestStrategyRegistry:
    """Test strategy registry functionality"""

    def test_registry_operations(self):
        """Test basic registry operations"""
        registry = StrategyRegistry()

        # Register strategies
        registry.register_strategy(MomentumStrategy)
        registry.register_strategy(MeanReversionStrategy)
        registry.register_strategy(BreakoutStrategy)

        # Check listing
        strategies = registry.list_strategies()
        assert 'MomentumStrategy' in strategies
        assert 'MeanReversionStrategy' in strategies
        assert 'BreakoutStrategy' in strategies

        # Get strategy class
        momentum_class = registry.get_strategy_class('MomentumStrategy')
        assert momentum_class == MomentumStrategy

        # Test invalid strategy
        with pytest.raises(ValueError):
            registry.get_strategy_class('InvalidStrategy')

    def test_strategy_creation(self):
        """Test strategy creation from config"""
        registry = StrategyRegistry()
        registry.register_strategy(MomentumStrategy)

        config = StrategyConfig(
            name="MomentumStrategy",
            description="Test momentum strategy",
            parameters={
                'fast_period': 10,
                'slow_period': 20,
                'momentum_threshold': 0.02
            }
        )

        strategy = registry.create_strategy(config)
        assert isinstance(strategy, MomentumStrategy)
        assert strategy.config.name == "MomentumStrategy"

    def test_strategy_info(self):
        """Test strategy information retrieval"""
        registry = StrategyRegistry()
        registry.register_strategy(MomentumStrategy)

        info = registry.get_strategy_info('MomentumStrategy')
        assert info['name'] == 'MomentumStrategy'
        assert info['class'] == 'MomentumStrategy'
        assert 'required_parameters' in info
        assert 'fast_period' in info['required_parameters']

if __name__ == '__main__':
    pytest.main([__file__])