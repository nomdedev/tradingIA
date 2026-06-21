"""
Optimization with Council
=========================
Runs Bayesian Optimization on strategy parameters, guided by the Council's decisions.
The Council acts as a gatekeeper during the backtest, ensuring that only robust trades are taken.
The optimizer then finds the parameters that yield the best performance *within* the Council's safety rules.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtester import AdvancedBacktester
from core.council import Council
from core.risk.risk_manager import RiskManager
from core.data.sql_data_handler import SQLDataHandler
from config.mtf_config import MTF_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Starting Optimization with Council...")

    # 1. Initialize Components
    # Council with rules
    council = Council(rules_dir=str(project_root / "core" / "rules"))

    # Risk Manager
    risk_config = {"max_daily_drawdown": 0.05, "max_total_drawdown": 0.15, "max_exposure": 1.0}
    risk_manager = RiskManager(config=risk_config)

    # Data Handler
    data_handler = SQLDataHandler()

    # 2. Load Data
    # Using a 6-month period for optimization to be faster than full history
    start_date = "2024-01-01"
    end_date = "2024-06-01"

    logger.info(f"Loading data from {start_date} to {end_date}...")
    dfs = data_handler.get_multi_tf_data(symbol="BTC", start_date=start_date, end_date=end_date)

    if not dfs:
        logger.error("No data loaded.")
        return

    # Map keys for Backtester
    backtest_dfs = {}
    if "entry" in dfs:
        backtest_dfs["5m"] = dfs["entry"]
    if "momentum" in dfs:
        backtest_dfs["15m"] = dfs["momentum"]
    if "trend" in dfs:
        backtest_dfs["1h"] = dfs["trend"]

    # 3. Initialize Backtester with Council
    # We use 'next_open' execution for realism
    backtester = AdvancedBacktester(
        capital=10000.0, council=council, risk_manager=risk_manager, execution_mode="next_open", dynamic_slippage=True
    )

    # 4. Run Walk-Forward Optimization
    # This will use _bayesian_optimization internally
    # We use 3 periods for the demo
    logger.info("Running Walk-Forward Optimization...")
    try:
        wf_results = backtester.walk_forward_optimization(backtest_dfs, n_periods=3, train_split=0.7)

        # 5. Report Results
        print("\n🏆 Optimization Results (Guided by Council)")
        print("=" * 50)
        print(f"Avg Test Calmar: {wf_results['avg_test_calmar']:.3f}")
        print(f"Robustness Score: {wf_results['robustness_score']:.3f}")
        print("\nBest Parameters Overall:")
        for k, v in wf_results["best_params_overall"].items():
            print(f"  {k}: {v}")

        # Save to file
        results_file = project_root / "results" / "optimization_council_results.txt"
        with open(results_file, "w") as f:
            f.write("Optimization Results (Guided by Council)\n")
            f.write("========================================\n")
            f.write(f"Avg Test Calmar: {wf_results['avg_test_calmar']:.3f}\n")
            f.write(f"Robustness Score: {wf_results['robustness_score']:.3f}\n")
            f.write("\nBest Parameters Overall:\n")
            for k, v in wf_results["best_params_overall"].items():
                f.write(f"  {k}: {v}\n")

        # Save to JSON for Dashboard
        import json

        json_file = project_root / "results" / "optimization_council_results.json"

        # Convert numpy types to native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        with open(json_file, "w") as f:
            json.dump(wf_results, f, default=convert_numpy, indent=4)

        logger.info(f"Results saved to {results_file} and {json_file}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
