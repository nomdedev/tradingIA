import pytest
import pandas as pd
import numpy as np
from core.execution.backtester_core import BacktesterCore
from core.strategies.base_strategy import BaseStrategy, StrategyConfig

class LosingStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        # Backtester passes kwargs. BaseStrategy needs StrategyConfig.
        config = StrategyConfig(
            name="Losing", 
            description="Losing Strategy", 
            parameters=kwargs
        )
        super().__init__(config)
        
    def get_required_parameters(self):
        return {}
        
    def get_parameters(self):
        return {}
        
    def generate_signals(self, data_dict):
        df = data_dict['5min']
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        
        # Enter at index 10
        entries.iloc[10] = True
        # Hold until end (or let VBT handle it)
        # To ensure we lose money as price drops, we stay long.
        # Exit at the end
        exits.iloc[-1] = True
        
        return {'entries': entries, 'exits': exits}

def test_kill_switch_activation():
    # Create dummy data: Price dropping consistently
    # 1 day of data at 5min intervals = 288 candles
    # Let's create 2 days to test daily reset logic if needed, but for simple DD check 1 day is enough.
    dates = pd.date_range(start='2023-01-01', periods=500, freq='5min')
    
    # Price drops from 100 to 90 (10% drop)
    prices = np.linspace(100, 90, 500) 
    df = pd.DataFrame({
        'close': prices, 
        'open': prices, 
        'high': prices, 
        'low': prices, 
        'volume': 1000
    }, index=dates)
    
    data_dict = {'5min': df}
    
    # Initialize Backtester with Kill Switch
    # enable_realistic_execution=False to use VBT default sizing (Full Equity)
    backtester = BacktesterCore(initial_capital=10000, enable_realistic_execution=False)
    
    # Configure RiskManager: 2% Max Daily Drawdown
    # Initial Equity = 10000. 2% = 200.
    # If price drops from 100 to 98, we lose 2% (if fully invested).
    # We need to ensure we are invested.
    # VBT default size is usually inf or 100% equity? 
    # BacktesterCore uses simple execution if realistic is False, but we set it True.
    # If realistic is True, it calculates order size.
    
    # Let's force a simpler setup or ensure order size is large enough.
    # BacktesterCore._calculate_order_size_for_execution uses Kelly or simple.
    # If Kelly is off, it might use fixed size?
    
    # Let's just check if RiskManager is present
    assert backtester.risk_manager is not None
    backtester.risk_manager.max_daily_drawdown = 0.02 
    
    # Run backtest
    result = backtester.run_simple_backtest(data_dict, LosingStrategy, {})
    
    equity_curve = result['equity_curve']
    
    # We expect the backtest to stop when equity drops ~2%.
    # 100 -> 90 is 10% drop.
    # If we are fully invested, we should stop around 20% of the way (index ~100).
    
    print(f"Final Equity: {equity_curve[-1]}")
    print(f"Length: {len(equity_curve)} / {len(dates)}")
    
    # Assert that we stopped early
    assert len(equity_curve) < 500
    assert len(equity_curve) > 10 # Should run for a bit
    
    # Check that the last equity is roughly -2% from start (or slightly more due to gap)
    # Start: 10000. End should be around 9800.
    assert equity_curve[-1] < 9850
