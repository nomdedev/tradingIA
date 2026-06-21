"""
Walk-Forward Optimizer Module

Extracted from BacktesterCore for better modularity.
Implements Walk-Forward Analysis (WFA) for strategy validation:
- Anchored and rolling window approaches
- Bayesian optimization integration
- Stability scoring and certification
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum


class WFAMethod(Enum):
    """Walk-Forward Analysis methods."""
    ANCHORED = "anchored"  # IS starts from beginning
    ROLLING = "rolling"    # IS window slides
    

@dataclass
class WFAPeriodResult:
    """Result from a single WFA period."""
    period: int
    train_bars: int
    test_bars: int
    train_sharpe: float
    test_sharpe: float
    degradation_pct: float
    best_params: Optional[Dict]
    train_metrics: Dict
    test_metrics: Dict


@dataclass 
class WFAResult:
    """Complete Walk-Forward Analysis result."""
    period_results: List[WFAPeriodResult]
    avg_degradation: float
    avg_oos_sharpe: float
    stability_score: float
    certified: bool
    best_params: Dict
    all_optimized_params: Optional[List[Dict]]
    optimization_used: bool
    method: WFAMethod


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis for strategy validation.
    
    Implements:
    - Anchored WFA (expanding IS window)
    - Rolling WFA (fixed IS window)
    - Bayesian parameter optimization
    - Stability scoring and certification
    
    Certification criteria:
    - Average degradation < 30%
    - OOS Sharpe > 0.5
    - Stability Score > 0.5
    """
    
    def __init__(
        self,
        backtest_func: Callable,
        optimize_func: Optional[Callable] = None,
        n_periods: int = 8,
        method: WFAMethod = WFAMethod.ANCHORED,
        min_test_bars: int = 100,
        certification_criteria: Optional[Dict] = None
    ) -> None:
        """
        Initialize Walk-Forward Optimizer.
        
        Args:
            backtest_func: Function to run single backtest
            optimize_func: Function for parameter optimization
            n_periods: Number of WFA periods
            method: ANCHORED or ROLLING window
            min_test_bars: Minimum bars for test period
            certification_criteria: Custom certification thresholds
        """
        self.backtest_func = backtest_func
        self.optimize_func = optimize_func
        self.n_periods = n_periods
        self.method = method
        self.min_test_bars = min_test_bars
        self.logger = logging.getLogger(__name__)
        
        # Certification criteria
        self.cert_criteria = certification_criteria or {
            'max_degradation': 30,    # Max % degradation allowed
            'min_oos_sharpe': 0.5,    # Min OOS Sharpe required
            'min_stability': 0.5      # Min stability score
        }
        
        # Cancellation flag
        self._cancelled = False
    
    def cancel(self) -> None:
        """Cancel running analysis."""
        self._cancelled = True
    
    def _check_cancellation(self) -> None:
        """Check if analysis should be cancelled."""
        if self._cancelled:
            raise InterruptedError("Walk-forward analysis cancelled")
    
    def run(
        self,
        df_multi_tf: Dict[str, pd.DataFrame],
        strategy_class: Any,
        strategy_params: Optional[Dict] = None,
        param_ranges: Optional[Dict] = None,
    ) -> WFAResult:
        """
        Run Walk-Forward Analysis.
        
        Args:
            df_multi_tf: Dict of DataFrames by timeframe
            strategy_class: Strategy class to optimize
            strategy_params: Initial parameters (if no optimization)
            param_ranges: Parameter ranges for optimization
            
        Returns:
            WFAResult with all period results and statistics
        """
        self._cancelled = False
        
        # Get primary timeframe data
        primary_tf = '5min' if '5min' in df_multi_tf else list(df_multi_tf.keys())[0]
        df_primary = df_multi_tf[primary_tf].copy()
        total_bars = len(df_primary)
        period_size = total_bars // self.n_periods
        
        # Determine if optimization is enabled
        use_optimization = (
            self.optimize_func is not None and 
            param_ranges is not None
        )
        
        if use_optimization:
            self.logger.info(f"🧬 WFA with Bayesian optimization ({self.n_periods} periods)")
        else:
            self.logger.info(f"📊 WFA without optimization ({self.n_periods} periods)")
            if param_ranges is None and strategy_params is None:
                raise ValueError("Must provide strategy_params or param_ranges")
        
        # Initialize tracking
        period_results: List[WFAPeriodResult] = []
        all_train_sharpes: List[float] = []
        all_test_sharpes: List[float] = []
        all_degradations: List[float] = []
        all_optimized_params: List[Dict] = []
        best_params = strategy_params or {}
        
        for i in range(self.n_periods):
            self._check_cancellation()
            
            # Calculate period boundaries
            train_start, train_end, test_start, test_end = self._calculate_boundaries(
                i, period_size, total_bars
            )
            
            if test_end - test_start < self.min_test_bars:
                self.logger.warning(
                    f"Period {i+1}: OOS too small "
                    f"({test_end - test_start} < {self.min_test_bars}), skipping"
                )
                break
            
            # Split data
            train_data = {tf: df.iloc[train_start:train_end] for tf, df in df_multi_tf.items()}
            test_data = {tf: df.iloc[test_start:test_end] for tf, df in df_multi_tf.items()}
            
            self.logger.info(
                f"📈 Period {i+1}/{self.n_periods}: "
                f"IS[{train_start}:{train_end}] -> OOS[{test_start}:{test_end}]"
            )
            
            # Optimize parameters if enabled
            if use_optimization:
                self.logger.info("   🔍 Optimizing parameters in IS...")
                best_params = self.optimize_func(strategy_class, train_data, param_ranges)
                all_optimized_params.append(best_params.copy())
                self.logger.info(f"   ✅ Params period {i+1}: {best_params}")
            elif strategy_params:
                best_params = strategy_params
            
            # Run backtests
            train_result = self.backtest_func(strategy_class, train_data, best_params)
            test_result = self.backtest_func(strategy_class, test_data, best_params)
            
            if "error" not in train_result and "error" not in test_result:
                train_sharpe = train_result["metrics"]["sharpe"]
                test_sharpe = test_result["metrics"]["sharpe"]
                
                # Calculate degradation
                degradation_pct = self._calculate_degradation(train_sharpe, test_sharpe)
                
                period_result = WFAPeriodResult(
                    period=i + 1,
                    train_bars=train_end - train_start,
                    test_bars=test_end - test_start,
                    train_sharpe=train_sharpe,
                    test_sharpe=test_sharpe,
                    degradation_pct=degradation_pct,
                    best_params=best_params.copy() if use_optimization else None,
                    train_metrics=train_result["metrics"],
                    test_metrics=test_result["metrics"]
                )
                period_results.append(period_result)
                
                all_train_sharpes.append(train_sharpe)
                all_test_sharpes.append(test_sharpe)
                all_degradations.append(degradation_pct)
                
                self.logger.info(
                    f"   📊 IS Sharpe: {train_sharpe:.2f} -> "
                    f"OOS Sharpe: {test_sharpe:.2f} "
                    f"(Degradation: {degradation_pct:.1f}%)"
                )
            else:
                error = train_result.get("error", "") or test_result.get("error", "")
                self.logger.warning(f"   ⚠️ Period {i+1} failed: {error}")
        
        # Calculate final statistics
        return self._calculate_final_results(
            period_results,
            all_train_sharpes,
            all_test_sharpes,
            all_degradations,
            best_params,
            all_optimized_params,
            use_optimization
        )
    
    def _calculate_boundaries(
        self,
        period_idx: int,
        period_size: int,
        total_bars: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate train/test boundaries for a period.
        
        Args:
            period_idx: Current period index (0-based)
            period_size: Size of each period
            total_bars: Total number of bars
            
        Returns:
            Tuple of (train_start, train_end, test_start, test_end)
        """
        if self.method == WFAMethod.ANCHORED:
            # Anchored: IS always starts from beginning
            train_start = 0
            train_end = (period_idx + 1) * period_size
        else:
            # Rolling: fixed window size
            train_start = period_idx * period_size
            train_end = train_start + period_size
        
        test_start = train_end
        test_end = min((period_idx + 2) * period_size, total_bars)
        
        return train_start, train_end, test_start, test_end
    
    def _calculate_degradation(
        self,
        train_sharpe: float,
        test_sharpe: float
    ) -> float:
        """
        Calculate performance degradation percentage.
        
        Degradation = (IS - OOS) / |IS| * 100
        Positive = OOS worse than IS (expected)
        Negative = OOS better than IS (unusual)
        
        Args:
            train_sharpe: In-sample Sharpe ratio
            test_sharpe: Out-of-sample Sharpe ratio
            
        Returns:
            Degradation percentage
        """
        if abs(train_sharpe) > 0.01:
            return ((train_sharpe - test_sharpe) / abs(train_sharpe)) * 100
        else:
            return 0 if abs(test_sharpe) < 0.01 else -100
    
    def _calculate_final_results(
        self,
        period_results: List[WFAPeriodResult],
        all_train_sharpes: List[float],
        all_test_sharpes: List[float],
        all_degradations: List[float],
        best_params: Dict,
        all_optimized_params: List[Dict],
        use_optimization: bool
    ) -> WFAResult:
        """Calculate final WFA statistics and certification."""
        
        if not period_results:
            return WFAResult(
                period_results=[],
                avg_degradation=0,
                avg_oos_sharpe=0,
                stability_score=0,
                certified=False,
                best_params=best_params,
                all_optimized_params=None,
                optimization_used=use_optimization,
                method=self.method
            )
        
        avg_degradation = np.mean(all_degradations)
        std_degradation = np.std(all_degradations)
        avg_oos_sharpe = np.mean(all_test_sharpes)
        
        # Calculate stability score (0-1)
        degradation_penalty = min(abs(avg_degradation) / 100, 1.0)
        variability_penalty = min(std_degradation / 50, 0.5)
        stability_score = max(0, 1.0 - degradation_penalty - variability_penalty)
        
        # Certification check
        certified = (
            abs(avg_degradation) < self.cert_criteria['max_degradation'] and
            avg_oos_sharpe > self.cert_criteria['min_oos_sharpe'] and
            stability_score > self.cert_criteria['min_stability']
        )
        
        self.logger.info("\n🏁 WFA Complete:")
        self.logger.info(f"   📉 Avg Degradation: {avg_degradation:.1f}%")
        self.logger.info(f"   📊 Avg OOS Sharpe: {avg_oos_sharpe:.2f}")
        self.logger.info(f"   🎯 Stability Score: {stability_score:.2f}")
        self.logger.info(f"   {'✅ CERTIFIED' if certified else '❌ NOT CERTIFIED'}")
        
        # Convert to dict for backward compatibility
        period_results_dicts = [
            {
                "period": r.period,
                "train_bars": r.train_bars,
                "test_bars": r.test_bars,
                "train_metrics": r.train_metrics,
                "test_metrics": r.test_metrics,
                "best_params": r.best_params,
                "degradation_pct": r.degradation_pct
            }
            for r in period_results
        ]
        
        return WFAResult(
            period_results=period_results_dicts,
            avg_degradation=avg_degradation,
            avg_oos_sharpe=avg_oos_sharpe,
            stability_score=stability_score,
            certified=certified,
            best_params=best_params,
            all_optimized_params=all_optimized_params if use_optimization else None,
            optimization_used=use_optimization,
            method=self.method
        )
    
    def get_parameter_stability(
        self, 
        result: WFAResult
    ) -> Optional[Dict]:
        """
        Analyze parameter stability across WFA periods.
        
        Args:
            result: WFAResult from run()
            
        Returns:
            Dict with parameter statistics, or None if no optimization
        """
        if not result.all_optimized_params:
            return None
        
        params_df = pd.DataFrame(result.all_optimized_params)
        
        stability_report = {}
        for col in params_df.columns:
            values = params_df[col].values
            stability_report[col] = {
                'mean': round(float(np.mean(values)), 4),
                'std': round(float(np.std(values)), 4),
                'min': round(float(np.min(values)), 4),
                'max': round(float(np.max(values)), 4),
                'cv': round(float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0, 4)
            }
        
        return stability_report
