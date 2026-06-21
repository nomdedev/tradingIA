"""
Test de integración del Council y Risk Manager en el Backtester.
"""

import unittest
import pandas as pd
import os
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.council import Council
from core.risk.risk_manager import RiskManager

# We need to patch MultiTFDataHandler BEFORE importing AdvancedBacktester if it's imported at module level
# But here it is imported inside the test file.
# Let's patch it where it is used.


class TestCouncilIntegration(unittest.TestCase):
    def setUp(self):
        # Setup paths relative to project root
        self.rules_dir = str(PROJECT_ROOT / "core" / "rules")
        self.data_path = str(PROJECT_ROOT / "data" / "btc_15Min.csv")

        # Create dummy data if not exists
        if not os.path.exists(self.data_path):
            os.makedirs(PROJECT_ROOT / "data", exist_ok=True)
            dates = pd.date_range(start="2024-01-01", periods=100, freq="15T")
            df = pd.DataFrame(
                {
                    "open": [100] * 100,
                    "high": [105] * 100,
                    "low": [95] * 100,
                    "close": [102] * 100,
                    "volume": [1000] * 100,
                },
                index=dates,
            )
            df.to_csv(self.data_path)

    @patch("src.backtester.MultiTFDataHandler")
    def test_backtest_with_council_and_risk(self, MockDataHandler):
        from src.backtester import AdvancedBacktester

        # 1. Initialize Components
        council = Council(rules_dir=self.rules_dir)
        risk_manager = RiskManager({"max_daily_drawdown": 0.05})

        backtester = AdvancedBacktester(capital=10000, council=council, risk_manager=risk_manager)

        # 2. Create Data In-Memory
        dates = pd.date_range(start="2024-01-01", periods=200, freq="15T")
        df = pd.DataFrame(
            {
                "open": [100.0] * 200,
                "high": [105.0] * 200,
                "low": [95.0] * 200,
                "close": [102.0] * 200,
                "volume": [1000.0] * 200,
            },
            index=dates,
        )

        # Add some volatility to generate signals
        import numpy as np

        np.random.seed(42)
        df["close"] = df["close"] + np.random.normal(0, 2, 200)
        df["high"] = df["close"] + 2
        df["low"] = df["close"] - 2

        dfs = {"5m": df, "15m": df, "1h": df}  # Mock multi-tf

        # 3. Run Backtest (using dummy params)
        params = {"atr_period": 14, "atr_multi": 1.5, "vp_rows": 10, "va_percent": 0.7, "vol_thresh": 1.0}

        # We expect it to run without errors
        try:
            result = backtester.run_optimized_backtest(dfs, params)
            print("Backtest finished successfully")
            print("Metrics:", result["metrics"])
        except Exception as e:
            self.fail(f"Backtest failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
