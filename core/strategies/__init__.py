"""
Strategies Module - Extensible Trading Strategy Framework
"""

from .base_strategy import BaseStrategy, StrategyConfig
from .strategy_registry import StrategyRegistry
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy

__all__ = [
    'BaseStrategy',
    'StrategyConfig',
    'StrategyRegistry',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy'
]