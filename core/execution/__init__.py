# Execution Engine Module
"""
Execution module providing backtesting and analysis capabilities.

Modules:
- backtester_core: Main backtesting engine
- metrics_calculator: Trading metrics calculation
- monte_carlo_simulator: Monte Carlo robustness testing
- walk_forward_optimizer: Walk-Forward Analysis
"""

# Import lightweight modules first (no heavy dependencies)
from core.execution.metrics_calculator import MetricsCalculator
from core.execution.monte_carlo_simulator import MonteCarloSimulator, MonteCarloResult
from core.execution.walk_forward_optimizer import (
    WalkForwardOptimizer,
    WFAMethod,
    WFAPeriodResult,
    WFAResult
)

# Lazy import for BacktesterCore (has heavy dependencies like skopt, vectorbt)
def get_backtester_core():
    """Lazy import for BacktesterCore to avoid loading heavy dependencies."""
    from core.execution.backtester_core import BacktesterCore
    return BacktesterCore

__all__ = [
    'get_backtester_core',
    'MetricsCalculator',
    'MonteCarloSimulator',
    'MonteCarloResult',
    'WalkForwardOptimizer',
    'WFAMethod',
    'WFAPeriodResult',
    'WFAResult'
]
