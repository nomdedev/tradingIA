"""
Script to run backtest using data from SQLite database.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtester import AdvancedBacktester
from core.data.sql_data_handler import SQLDataHandler
from config.mtf_config import MTF_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Starting SQL Backtest...")

    # 1. Initialize Data Handler
    data_handler = SQLDataHandler()

    # 2. Fetch Data
    # Using a small range for testing
    start_date = "2023-01-01"
    end_date = "2023-02-01"

    # Note: DB has 'BTC', not 'BTCUSD'
    dfs = data_handler.get_multi_tf_data(symbol="BTC", start_date=start_date, end_date=end_date)

    if not dfs:
        logger.error("No data loaded.")
        return

    # 3. Map keys for Backtester
    # Backtester expects '5m', '15m', '1h'
    # SQLDataHandler returns 'entry', 'momentum', 'trend' (based on MTF_CONFIG)

    # Check what keys we have
    logger.info(f"Loaded keys: {list(dfs.keys())}")

    # Create the mapping based on MTF_CONFIG
    # MTF_CONFIG['timeframes'] = {'entry': '5Min', 'momentum': '15Min', 'trend': '1H'}
    # We want: '5m' -> dfs['entry'], '15m' -> dfs['momentum'], '1h' -> dfs['trend']

    backtest_dfs = {}

    # Mapping logic:
    # entry (5Min) -> 5m
    # momentum (15Min) -> 15m
    # trend (1H) -> 1h

    if "entry" in dfs:
        backtest_dfs["5m"] = dfs["entry"]
    if "momentum" in dfs:
        backtest_dfs["15m"] = dfs["momentum"]
    if "trend" in dfs:
        backtest_dfs["1h"] = dfs["trend"]

    # 4. Initialize Backtester
    backtester = AdvancedBacktester()

    # 5. Run Backtest
    # Define some dummy params
    params = {
        "ema_fast": 9,
        "ema_slow": 21,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "min_confidence": 0.5,
    }

    try:
        logger.info("Running optimized backtest...")
        results = backtester.run_optimized_backtest(backtest_dfs, params)

        metrics = results["metrics"]
        logger.info("✅ Backtest Complete")
        logger.info(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
