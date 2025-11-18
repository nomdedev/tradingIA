"""
Risk Management Module for Trading IA

This module provides advanced risk management tools including:
- Kelly Criterion position sizing
- MAE/MFE analysis
- VaR and CVaR calculations
- Risk-adjusted portfolio optimization
"""

from .kelly_sizer import KellyPositionSizer

__all__ = [
    'KellyPositionSizer'
]