"""
Walk-Forward Analysis (WFA) Module.

Implements robust validation by simulating the re-optimization process over time.
Prevents overfitting by testing on unseen (Out-of-Sample) data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta

from core.execution.backtester_core import BacktesterCore
from core.optimization.genetic_optimizer import GeneticOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class WFAResult:
    """Results of a Walk-Forward Analysis"""
    oos_equity_curve: List[float]
    oos_trades: List[Dict]
    period_results: List[Dict]  # Details for each period (IS params, OOS metrics)
    stability_score: float
    overall_metrics: Dict[str, float]

class WalkForwardOptimizer:
    def __init__(self, backtester: BacktesterCore):
        self.backtester = backtester
        self.logger = logging.getLogger(__name__)

    def run_wfa(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy_class: Any,
        param_ranges: Dict[str, Any],
        n_periods: int = 5,
        train_ratio: float = 0.7,
        optimization_config: OptimizationConfig = None
    ) -> WFAResult:
        """
        Run Walk-Forward Analysis.
        
        Args:
            data_dict: Dictionary of DataFrames (must include '5min' or similar)
            strategy_class: Strategy class to optimize
            param_ranges: Dictionary of parameter ranges for optimization
            n_periods: Number of Walk-Forward periods
            train_ratio: Ratio of data used for training (IS) vs testing (OOS) in each window
            optimization_config: Config for the genetic optimizer
            
        Returns:
            WFAResult object
        """
        self.logger.info(f"🚀 Starting Walk-Forward Analysis ({n_periods} periods)")
        
        # 1. Prepare Data
        # We assume '5min' is the base timeframe for slicing
        df = data_dict.get('5min')
        if df is None:
            # Fallback to first available
            df = next(iter(data_dict.values()))
            
        total_duration = df.index[-1] - df.index[0]
        period_duration = total_duration / n_periods
        
        self.logger.info(f"   Total Duration: {total_duration}")
        self.logger.info(f"   Period Duration: {period_duration}")
        
        oos_trades_all = []
        oos_equity_pieces = []
        period_results = []
        
        current_capital = self.backtester.initial_capital
        
        # 2. Iterate through periods
        # We use an Expanding Window or Rolling Window?
        # Let's use Rolling Window for simplicity and robustness:
        # Window N: Train on [Start_N, End_IS_N], Test on [End_IS_N, End_OOS_N]
        # Where End_OOS_N is roughly Start_N + period_duration
        
        # Actually, standard WFA usually does:
        # Divide data into N segments.
        # Step 1: Train on Seg 1, Test on Seg 2? No, that's too small IS.
        # Usually: Train on Seg 1..k, Test on Seg k+1.
        
        # Let's implement "Anchored Walk-Forward" (Expanding IS)
        # Period 1: Train [0..1], Test [1..2] -> Wait, we need initial data.
        # Let's divide into N+1 segments.
        # Period i (1 to N):
        #   IS: Start to Segment i
        #   OOS: Segment i to Segment i+1
        
        # Better approach for fixed number of periods:
        # Split data into N equal chunks.
        # We need at least 1 chunk for initial training.
        # So we will have N-1 OOS periods.
        
        # Let's refine:
        # We want N OOS periods. So we need N+1 chunks?
        # Or we use a sliding window of size W, step S.
        
        # Implementation:
        # 1. Divide total time into N equal OOS blocks.
        # 2. For each OOS block, define an IS block immediately preceding it.
        #    IS block size = OOS block size * (train_ratio / (1-train_ratio))?
        #    Or just fixed ratio.
        
        start_time = df.index[0]
        end_time = df.index[-1]
        
        # Calculate OOS duration
        # We reserve some initial data for the first IS.
        # Let's say we want N OOS periods covering the last X% of data?
        # Or just cover the whole dataset?
        # Standard: Cover whole dataset? No, need initial history.
        
        # Simple approach:
        # Divide data into N+1 segments.
        # Segment 0 is initial IS.
        # Segment 1..N are OOS.
        # For OOS i (1..N):
        #   IS = Segment 0..i-1 (Anchored) OR Segment i-1 (Rolling)
        #   OOS = Segment i
        
        # Let's use Rolling Window with overlap if needed, but simpler:
        # Split into N equal segments.
        # Loop i from 0 to N-2:
        #   IS = Segment i
        #   OOS = Segment i+1
        # This gives N-1 tests.
        
        # Let's use the user parameter n_periods as the number of OOS tests.
        # So we split data into n_periods + 1 segments.
        
        segments = np.array_split(df.index, n_periods + 1)
        
        cumulative_equity = [self.backtester.initial_capital]
        
        for i in range(n_periods):
            # Define IS and OOS ranges
            # IS: Segment i
            # OOS: Segment i+1
            
            is_idx = segments[i]
            oos_idx = segments[i+1]
            
            is_start, is_end = is_idx[0], is_idx[-1]
            oos_start, oos_end = oos_idx[0], oos_idx[-1]
            
            self.logger.info(f"🔄 Period {i+1}/{n_periods}")
            self.logger.info(f"   IS:  {is_start} -> {is_end} ({len(is_idx)} bars)")
            self.logger.info(f"   OOS: {oos_start} -> {oos_end} ({len(oos_idx)} bars)")
            
            # Slice Data for IS
            is_data = {k: v.loc[is_start:is_end] for k, v in data_dict.items()}
            
            # 1. Optimize on IS
            self.logger.info("   🧬 Optimizing on In-Sample data...")
            
            # Create wrapper for backtest function
            def backtest_wrapper(**params):
                # Backtester expects strategy_params as a dict
                # GeneticOptimizer passes params as kwargs
                try:
                    res = self.backtester.run_simple_backtest(is_data, strategy_class, params)
                    # GeneticOptimizer expects a list of objects with .metrics attribute
                    class ResultWrapper:
                        def __init__(self, metrics):
                            self.metrics = metrics
                    return [ResultWrapper(res['metrics'])]
                except Exception as e:
                    self.logger.error(f"Backtest failed in optimization: {e}")
                    return []

            # Configure Optimizer
            gen_opt = GeneticOptimizer(config=optimization_config or OptimizationConfig())
            
            # Set bounds
            # param_ranges is {name: (min, max)} or {name: [options]}
            bounds = {}
            param_types = {}
            categorical_values = {}
            
            for name, range_val in param_ranges.items():
                if isinstance(range_val, list):
                    # Categorical
                    param_types[name] = "categorical"
                    categorical_values[name] = range_val
                    bounds[name] = (0, len(range_val)-1) # Dummy bounds
                elif isinstance(range_val, tuple):
                    bounds[name] = range_val
                    # Infer type from values
                    if isinstance(range_val[0], int) and isinstance(range_val[1], int):
                        param_types[name] = "int"
                    else:
                        param_types[name] = "float"
            
            gen_opt.set_parameter_bounds(bounds, param_types, categorical_values)
            
            # Run Optimization
            opt_results = gen_opt.optimize(backtest_wrapper)
            best_params = opt_results['best_parameters']
            best_metrics = opt_results['best_metrics']
            
            self.logger.info(f"   ✅ Best Params: {best_params}")
            self.logger.info(f"   Training Sharpe: {best_metrics['sharpe_ratio']:.2f}")
            
            # 2. Validate on OOS
            self.logger.info("   🧪 Testing on Out-of-Sample data...")
            oos_data = {k: v.loc[oos_start:oos_end] for k, v in data_dict.items()}
            
            # Run backtest on OOS
            # We need to update backtester capital? 
            # Ideally WFA chains the equity.
            self.backtester.initial_capital = current_capital
            
            result = self.backtester.run_simple_backtest(
                oos_data,
                strategy_class,
                best_params
            )
            
            # Extract results
            metrics = result['metrics']
            trades = result['trades']
            equity = result['equity_curve']
            
            # Update current capital for next round
            if equity:
                current_capital = equity[-1]
            
            # Store results
            period_res = {
                "period": i + 1,
                "is_range": (is_start, is_end),
                "oos_range": (oos_start, oos_end),
                "best_params": best_params,
                "is_metrics": {
                    "sharpe": best_metrics.get('sharpe_ratio', 0.0),
                    "return": best_metrics.get('total_return', 0.0)
                },
                "oos_metrics": metrics
            }
            period_results.append(period_res)
            
            # Append trades (with offset if needed, but trades usually have timestamps)
            oos_trades_all.extend(trades)
            
            # Append equity (stitching)
            # Equity curve from backtester starts at initial_capital.
            # We need to append it to cumulative.
            # But cumulative already has start point.
            # We take equity[1:] to avoid double counting start?
            # Actually, backtester returns absolute equity values.
            # We can just extend.
            if i == 0:
                oos_equity_pieces.extend(equity)
            else:
                # Remove first point if it duplicates?
                # Backtest starts at current_capital.
                oos_equity_pieces.extend(equity[1:])
                
            self.logger.info(f"   OOS Sharpe: {metrics.get('sharpe', 0):.2f}")
            self.logger.info(f"   OOS Return: {metrics.get('total_return', 0):.2%}")

        # 3. Compile Final Results
        self.logger.info("🏁 Walk-Forward Analysis Complete")
        
        # Calculate stability score
        # 1. Consistency: % of periods with positive OOS return
        positive_oos = sum(1 for p in period_results if p['oos_metrics'].get('total_return', 0) > 0)
        consistency_score = positive_oos / len(period_results) if period_results else 0.0
        
        # 2. Degradation: Average ratio of OOS Sharpe / IS Sharpe
        degradation_ratios = []
        for p in period_results:
            is_s = p['is_metrics'].get('sharpe', 0)
            oos_s = p['oos_metrics'].get('sharpe_ratio', 0)
            
            if abs(is_s) < 0.01:
                ratio = 1.0 if abs(oos_s) < 0.01 else 0.0
            else:
                ratio = oos_s / is_s
            
            # Map ratio to a 0-1 score.
            # If ratio >= 0.8 -> 1.0 (Good preservation of performance)
            # If ratio <= 0.0 -> 0.0 (Complete failure)
            score = min(max(ratio, 0), 1.0)
            degradation_ratios.append(score)
            
        avg_degradation_score = np.mean(degradation_ratios) if degradation_ratios else 0.0
        
        # Weighted Score
        stability_score = (consistency_score * 0.5) + (avg_degradation_score * 0.5)
        
        # Overall metrics
        # We should recalculate metrics on the stitched equity curve
        # But for now, let's just sum up or average
        
        return WFAResult(
            oos_equity_curve=oos_equity_pieces,
            oos_trades=oos_trades_all,
            period_results=period_results,
            stability_score=stability_score,
            overall_metrics={} # To be calculated
        )
