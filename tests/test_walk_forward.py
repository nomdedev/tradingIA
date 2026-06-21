import pytest
import pandas as pd
import numpy as np
from core.execution.backtester_core import BacktesterCore
from core.strategies.base_strategy import BaseStrategy, StrategyConfig
from core.optimization.walk_forward import WalkForwardOptimizer
from core.optimization.genetic_optimizer import OptimizationConfig

class DummyStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        config = StrategyConfig(name="Dummy", description="Test", parameters=kwargs)
        super().__init__(config)
        self.ma_period = kwargs.get('ma_period', 10)
        
    def get_required_parameters(self):
        return ['ma_period']
        
    def get_parameters(self):
        return self.config.parameters
        
    def generate_signals(self, data_dict):
        df = data_dict['5min']
        close = df['close']
        ma = close.rolling(window=int(self.ma_period)).mean()
        
        entries = (close > ma) & (close.shift(1) <= ma.shift(1))
        exits = (close < ma) & (close.shift(1) >= ma.shift(1))
        
        return {'entries': entries, 'exits': exits}

def test_walk_forward_optimization():
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
    # Random walk
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices,
        'high': prices,
        'low': prices,
        'volume': 1000
    }, index=dates)
    
    data_dict = {'5min': df}
    
    backtester = BacktesterCore(initial_capital=10000)
    optimizer = WalkForwardOptimizer(backtester)
    
    param_ranges = {
        'ma_period': (5, 50)
    }
    
    # Fast config for testing
    opt_config = OptimizationConfig(
        population_size=5,
        generations=2,
        max_workers=1
    )
    
    result = optimizer.run_wfa(
        data_dict=data_dict,
        strategy_class=DummyStrategy,
        param_ranges=param_ranges,
        n_periods=3,
        optimization_config=opt_config
    )
    
    assert result is not None
    assert len(result.period_results) == 3
    assert len(result.oos_equity_curve) > 0
    
    print(f"Stability Score: {result.stability_score}")
    for p in result.period_results:
        print(f"Period {p['period']}: Best Params {p['best_params']}, OOS Sharpe {p['oos_metrics'].get('sharpe', 0):.2f}")
