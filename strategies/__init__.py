"""
Strategies Package - Pre-configured Trading Strategies

This package contains ready-to-use trading strategies that can be loaded
and used directly in the trading platform.

Available Strategies:
- RSI Mean Reversion: Buy oversold, sell overbought
- MACD Momentum: Trade MACD crossovers
- Bollinger Bands: Mean reversion on band touches
- MA Crossover: Classic moving average crossovers
- Volume Breakout: Volume-confirmed breakouts

Usage:
    from strategies import load_strategy, list_available_strategies
    
    # List all strategies
    strategies = list_available_strategies()
    
    # Load a strategy
    strategy = load_strategy('rsi_mean_reversion')
    
    # Load with preset
    strategy = load_strategy('macd_momentum', preset='aggressive')
    
    # Generate signals
    signals = strategy.generate_signals(df)

Custom Strategies:
    To add your own strategy:
    1. Create a new .py file in strategies/presets/
    2. Inherit from BaseStrategy
    3. Implement generate_signals(), get_parameters(), set_parameters()
    4. Optionally define PRESETS dict
    5. Strategy will be auto-discovered by StrategyLoader
"""

from .strategy_loader import (
    load_strategy,
    list_available_strategies,
    get_loader,
    StrategyLoader
)

from .base_strategy import BaseStrategy

__all__ = [
    'load_strategy',
    'list_available_strategies',
    'get_loader',
    'StrategyLoader',
    'BaseStrategy'
]

__version__ = '1.0.0'

# Strategy catalog
STRATEGY_CATALOG = {
    'rsi_mean_reversion': {
        'name': 'RSI Mean Reversion',
        'description': 'Buy when RSI oversold, sell when overbought',
        'best_for': 'Ranging markets',
        'timeframes': ['1h', '4h', '1d']
    },
    'macd_momentum': {
        'name': 'MACD Momentum',
        'description': 'Trade MACD line crossovers with signal line',
        'best_for': 'Trending markets',
        'timeframes': ['15m', '1h', '4h']
    },
    'bollinger_bands': {
        'name': 'Bollinger Bands',
        'description': 'Mean reversion on band touches',
        'best_for': 'Medium volatility markets',
        'timeframes': ['1h', '4h']
    },
    'ma_crossover': {
        'name': 'Moving Average Crossover',
        'description': 'Classic dual MA crossover strategy',
        'best_for': 'Strong trends',
        'timeframes': ['1h', '4h', '1d']
    },
    'volume_breakout': {
        'name': 'Volume Breakout',
        'description': 'Volume-confirmed price breakouts',
        'best_for': 'High volume assets',
        'timeframes': ['15m', '1h']
    }
}
