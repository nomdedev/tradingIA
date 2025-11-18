"""
Base Strategy Classes for Extensible Trading Strategy Framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    filters: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")
        if not self.description:
            raise ValueError("Strategy description cannot be empty")

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    This class defines the interface that all trading strategies must implement,
    providing a consistent framework for strategy development and execution.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration object
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._validate_config()

    def _validate_config(self):
        """Validate strategy configuration"""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.config.parameters:
                raise ValueError(f"Required parameter '{param}' not found in strategy config")

    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """
        Return list of required parameter names for this strategy.

        Returns:
            List of parameter names that must be provided in config
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market data.

        Args:
            data: DataFrame with OHLCV data and technical indicators

        Returns:
            Series of signals: 1 (buy), -1 (sell), 0 (hold)
        """
        pass

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators required by the strategy.

        This method can be overridden by subclasses to add custom indicators.
        By default, returns the input data unchanged.

        Args:
            data: Raw market data

        Returns:
            DataFrame with additional calculated indicators
        """
        return data.copy()

    def apply_filters(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply additional filters to trading signals.

        Args:
            data: Market data with indicators
            signals: Raw trading signals

        Returns:
            Filtered trading signals
        """
        filtered_signals = signals.copy()

        # Apply risk management filters
        if 'max_drawdown_filter' in self.config.filters:
            filtered_signals = self._apply_max_drawdown_filter(data, filtered_signals)

        if 'volatility_filter' in self.config.filters:
            filtered_signals = self._apply_volatility_filter(data, filtered_signals)

        if 'trend_filter' in self.config.filters:
            filtered_signals = self._apply_trend_filter(data, filtered_signals)

        return filtered_signals

    def _apply_max_drawdown_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply maximum drawdown filter"""
        max_dd_threshold = self.config.risk_management.get('max_drawdown_threshold', 0.1)

        # Calculate rolling drawdown
        rolling_max = data['close'].expanding().max()
        drawdown = (data['close'] - rolling_max) / rolling_max

        # Filter signals when drawdown is too large
        mask = drawdown.abs() <= max_dd_threshold
        return signals.where(mask, 0)

    def _apply_volatility_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply volatility filter"""
        vol_threshold = self.config.risk_management.get('volatility_threshold', 0.02)

        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()

        # Filter signals when volatility is too high
        mask = volatility <= vol_threshold
        return signals.where(mask, 0)

    def _apply_trend_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply trend filter"""
        trend_period = self.config.risk_management.get('trend_period', 50)

        # Simple trend filter using moving averages
        short_ma = data['close'].rolling(20).mean()
        long_ma = data['close'].rolling(trend_period).mean()

        # Only allow signals in uptrend
        trend_up = short_ma > long_ma
        return signals.where(trend_up, 0)

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the strategy.

        Returns:
            Dictionary with strategy metadata
        """
        return {
            'name': self.config.name,
            'description': self.config.description,
            'type': self.__class__.__name__,
            'parameters': self.config.parameters,
            'risk_management': self.config.risk_management,
            'filters': self.config.filters
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data meets strategy requirements.

        Args:
            data: Market data to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close']

        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False

        if len(data) < 50:  # Minimum data length
            self.logger.error("Insufficient data length for strategy")
            return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}')"