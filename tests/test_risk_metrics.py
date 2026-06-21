import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from core.execution.backtester_core import BacktesterCore

class TestRiskMetrics:
    
    def test_mae_mfe_calculation_long(self):
        """Test MAE/MFE calculation for a Long trade"""
        backtester = BacktesterCore()
        
        # Mock DataFrame with price data
        # Trade: Entry at 100, Exit at 110
        # Path: 100 -> 95 (Low) -> 115 (High) -> 110
        dates = pd.date_range(start="2023-01-01", periods=5, freq="5min")
        df = pd.DataFrame({
            "open": [100, 98, 105, 112, 110],
            "high": [102, 100, 115, 114, 112], # Max High = 115
            "low":  [99,  95, 102, 108, 109],  # Min Low = 95
            "close":[100, 96, 110, 112, 110]
        }, index=dates)
        
        # Mock VectorBT Portfolio trade record
        # We need to mimic the structure expected by _process_and_record_trades
        # But since we are testing the logic, we might need to mock the method or extract the logic
        # The current implementation uses hardcoded indices which is brittle.
        
        # Let's verify the fix by creating a mock portfolio that behaves like VBT
        mock_portfolio = MagicMock()
        
        # Create a structured array for trades
        # VBT fields: id, col, size, entry_idx, entry_price, entry_fees, exit_idx, exit_price, exit_fees, pnl, return, direction, status, parent_id
        dtype = [
            ('id', 'i8'), ('col', 'i8'), ('size', 'f8'), 
            ('entry_idx', 'i8'), ('entry_price', 'f8'), ('entry_fees', 'f8'),
            ('exit_idx', 'i8'), ('exit_price', 'f8'), ('exit_fees', 'f8'),
            ('pnl', 'f8'), ('return', 'f8'), ('direction', 'i8'), 
            ('status', 'i8'), ('parent_id', 'i8')
        ]
        
        # Trade: Long (direction=0), Entry Idx=0, Exit Idx=4
        trade_data = (
            0, 0, 1.0,      # id, col, size
            0, 100.0, 0.0,  # entry_idx, entry_price, fees
            4, 110.0, 0.0,  # exit_idx, exit_price, fees
            10.0, 0.1, 0,   # pnl, return, direction (0=Long)
            1, 0            # status, parent_id
        )
        
        trades_arr = np.array([trade_data], dtype=dtype)
        mock_portfolio.trades.records = trades_arr
        mock_portfolio.trades.count.return_value = 1
        
        # Run the method
        # Note: We need to patch the method to use named fields if we change it, 
        # or ensure the test matches the current implementation's expectations.
        # The current implementation uses indices: 
        # entry_idx=2 (wrong in my dtype above?), let's check the code again.
        
        # Code says:
        # entry_idx = int(trade[2]) 
        # exit_idx = int(trade[3])
        # entry_price = float(trade[5])
        # exit_price = float(trade[6])
        # size = float(trade[4])
        
        # This implies the code expects a specific column order which might NOT match standard VBT.
        # If I fix the code to use named fields, this test will be robust.
        
        backtester._process_and_record_trades(mock_portfolio, df)
        
        # Verify results
        assert len(backtester.trade_history) == 1
        trade = backtester.trade_history.iloc[0]
        
        # MAE: (Entry - MinLow) / Entry = (100 - 95) / 100 = 0.05 (5%)
        assert trade["mae"] == pytest.approx(0.05)
        
        # MFE: (MaxHigh - Entry) / Entry = (115 - 100) / 100 = 0.15 (15%)
        assert trade["mfe"] == pytest.approx(0.15)

    def test_mae_mfe_calculation_short(self):
        """Test MAE/MFE calculation for a Short trade"""
        backtester = BacktesterCore()
        
        # Mock DataFrame
        # Trade: Short Entry at 100, Exit at 90
        # Path: 100 -> 105 (High) -> 85 (Low) -> 90
        dates = pd.date_range(start="2023-01-01", periods=5, freq="5min")
        df = pd.DataFrame({
            "open": [100, 102, 95, 88, 90],
            "high": [100, 105, 98, 92, 91], # Max High = 105 (Adverse)
            "low":  [98,  99,  90, 85, 89], # Min Low = 85 (Favorable)
            "close":[100, 104, 92, 88, 90]
        }, index=dates)
        
        mock_portfolio = MagicMock()
        dtype = [
            ('id', 'i8'), ('col', 'i8'), ('size', 'f8'), 
            ('entry_idx', 'i8'), ('entry_price', 'f8'), ('entry_fees', 'f8'),
            ('exit_idx', 'i8'), ('exit_price', 'f8'), ('exit_fees', 'f8'),
            ('pnl', 'f8'), ('return', 'f8'), ('direction', 'i8'), 
            ('status', 'i8'), ('parent_id', 'i8')
        ]
        
        # Trade: Short (direction=1), Entry Idx=0, Exit Idx=4
        trade_data = (
            0, 0, 1.0,      # id, col, size
            0, 100.0, 0.0,  # entry_idx, entry_price, fees
            4, 90.0, 0.0,   # exit_idx, exit_price, fees
            10.0, 0.1, 1,   # pnl, return, direction (1=Short)
            1, 0            # status, parent_id
        )
        
        trades_arr = np.array([trade_data], dtype=dtype)
        mock_portfolio.trades.records = trades_arr
        mock_portfolio.trades.count.return_value = 1
        
        backtester._process_and_record_trades(mock_portfolio, df)
        
        trade = backtester.trade_history.iloc[0]
        
        # MAE (Short): (MaxHigh - Entry) / Entry = (105 - 100) / 100 = 0.05
        assert trade["mae"] == pytest.approx(0.05)
        
        # MFE (Short): (Entry - MinLow) / Entry = (100 - 85) / 100 = 0.15
        assert trade["mfe"] == pytest.approx(0.15)
