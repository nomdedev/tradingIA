"""
Configuration package for BTC IFVG Trading System

USAGE:
    # Standard config (recommended)
    from config import ALPACA_CONFIG, TRADING_CONFIG, BACKTEST_CONFIG
    
    # For MTF strategies (advanced)
    from config.mtf_config import MTF_CONFIG, INDICATOR_PARAMS

CONFIG FILES:
    - config.py: Base configuration (API, trading, backtest, strategies)
    - mtf_config.py: Multi-timeframe specific config (advanced use)
    - app_config.json: Application settings (GUI preferences)
    - user_preferences.json: User-specific overrides
    - strategies_registry.json: Strategy definitions
    - costs_params.json: Transaction cost parameters
    - training_config.yaml: ML training configuration
    
PRIORITY (highest to lowest):
    1. Environment variables (.env)
    2. User preferences (user_preferences.json)
    3. config.py / mtf_config.py

NOTE: config.py is the canonical source. mtf_config.py extends it for MTF strategies.
"""

from .config import (
    ALPACA_CONFIG,
    TRADING_CONFIG,
    BACKTEST_CONFIG,
    IFVG_CONFIG,
    VP_CONFIG,
    EMA_CONFIG,
    SIGNAL_CONFIG,
    PAPER_TRADING_CONFIG,
    OPTIMIZATION_CONFIG,
    get_config,
    validate_config,
)

__all__ = [
    'ALPACA_CONFIG',
    'TRADING_CONFIG',
    'BACKTEST_CONFIG',
    'IFVG_CONFIG',
    'VP_CONFIG',
    'EMA_CONFIG',
    'SIGNAL_CONFIG',
    'PAPER_TRADING_CONFIG',
    'OPTIMIZATION_CONFIG',
    'get_config',
    'validate_config',
]
