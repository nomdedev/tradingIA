"""
UI Module - Interactive Dashboard for Trading Strategy Management
"""

from .main_window import MainWindow
from .dashboard_controller import DashboardController
from .strategy_panel import StrategyPanel
from .backtest_panel import BacktestPanel
from .results_panel import ResultsPanel
from .charts_widget import ChartsWidget

__all__ = [
    'MainWindow',
    'DashboardController',
    'StrategyPanel',
    'BacktestPanel',
    'ResultsPanel',
    'ChartsWidget'
]