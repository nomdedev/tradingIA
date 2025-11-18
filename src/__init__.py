"""
Trading Platform Package
Main entry point for the BTC Trading Strategy Platform
"""

__version__ = "2.0.0"
__author__ = "TradingIA Team"
__description__ = "Complete BTC Trading Strategy Platform with GUI"

# Simplified imports to avoid path issues
# Components will be imported directly in the modules that need them

# Lazy import for main GUI to avoid PyQt6 requirement in non-GUI contexts
def get_main():
    """Lazy import of main GUI function"""
    from .main_platform import main
    return main

__all__ = [
    'get_main',
]
