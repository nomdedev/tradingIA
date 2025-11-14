"""
Trading Platform Package
Main entry point for the BTC Trading Strategy Platform
"""

__version__ = "1.0.0"
__author__ = "TradingIA Team"
__description__ = "Complete BTC Trading Strategy Platform with GUI"

# Legacy components (for backward compatibility)
from .data_fetcher import DataFetcher, get_historical_data, get_multi_tf_data
from .indicators import (
    calculate_ifvg_enhanced,
    volume_profile_advanced,
    emas_multi_tf,
    generate_filtered_signals,
    validate_params
)

# New platform components - import GUI component lazily to avoid PyQt6 dependency in tests
from .backend_core import DataManager, StrategyEngine
from .backtester_core import BacktesterCore
from .analysis_engines import AnalysisEngines
from .settings_manager import SettingsManager
from .reporters_engine import ReportersEngine

# Lazy import for main GUI to avoid PyQt6 requirement in non-GUI contexts


def get_main():
    """Lazy import of main GUI function"""
    from .main_platform import main
    return main


__all__ = [
    # Legacy
    'DataFetcher',
    'get_historical_data',
    'get_multi_tf_data',
    'calculate_ifvg_enhanced',
    'volume_profile_advanced',
    'emas_multi_tf',
    'generate_filtered_signals',
    'validate_params',
    # New platform
    'get_main',
    'DataManager',
    'StrategyEngine',
    'BacktesterCore',
    'AnalysisEngines',
    'SettingsManager',
    'ReportersEngine',
]
