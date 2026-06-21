"""
Monte Carlo Simulator Module

Extracted from BacktesterCore for better modularity.
Performs Monte Carlo simulations on trading strategies to:
- Test robustness under noise/perturbation
- Generate confidence intervals
- Estimate strategy reliability
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    sharpe_mean: float
    sharpe_std: float
    sharpe_p5: float
    sharpe_p95: float
    win_rate_mean: float
    win_rate_std: float
    is_robust: bool
    num_simulations: int
    all_sharpes: List[float]
    all_win_rates: List[float]


class MonteCarloSimulator:
    """
    Monte Carlo simulation for trading strategy robustness testing.
    
    Performs repeated simulations with:
    - Price noise injection
    - Parameter perturbation
    - Bootstrapped trade sequences
    
    Helps assess:
    - Strategy robustness to market noise
    - Confidence intervals for metrics
    - Over-fitting detection
    """
    
    def __init__(
        self,
        num_simulations: int = 500,
        noise_percent: float = 0.005,
        robustness_threshold: float = 0.2,
        confidence_level: float = 0.95
    ) -> None:
        """
        Initialize Monte Carlo simulator.
        
        Args:
            num_simulations: Number of simulation runs
            noise_percent: Noise level to add (5bps default)
            robustness_threshold: Max acceptable std for robustness
            confidence_level: Confidence level for intervals
        """
        self.num_simulations = num_simulations
        self.noise_percent = noise_percent
        self.robustness_threshold = robustness_threshold
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def run_simulation(
        self,
        backtest_func: Callable,
        data: pd.DataFrame,
        **backtest_kwargs
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation with noise injection.
        
        Args:
            backtest_func: Function to run single backtest
            data: Original OHLCV data
            **backtest_kwargs: Additional args for backtest function
            
        Returns:
            MonteCarloResult with statistics
        """
        sharpes: List[float] = []
        win_rates: List[float] = []
        
        self.logger.info(
            f"Starting Monte Carlo: {self.num_simulations} simulations, "
            f"{self.noise_percent*100:.2f}% noise"
        )
        
        for i in range(self.num_simulations):
            try:
                # Add noise to data
                noisy_data = self._add_price_noise(data.copy())
                
                # Run backtest
                result = backtest_func(noisy_data, **backtest_kwargs)
                
                if result is not None and 'sharpe' in result:
                    sharpes.append(result['sharpe'])
                    win_rates.append(result.get('win_rate', 0))
                    
            except Exception as e:
                self.logger.debug(f"Simulation {i} failed: {e}")
                continue
        
        if not sharpes:
            self.logger.warning("No successful simulations")
            return MonteCarloResult(
                sharpe_mean=0, sharpe_std=1, sharpe_p5=0, sharpe_p95=0,
                win_rate_mean=0, win_rate_std=1, is_robust=False,
                num_simulations=0, all_sharpes=[], all_win_rates=[]
            )
        
        sharpe_array = np.array(sharpes)
        win_rate_array = np.array(win_rates)
        
        sharpe_std = sharpe_array.std()
        is_robust = sharpe_std < self.robustness_threshold
        
        return MonteCarloResult(
            sharpe_mean=round(sharpe_array.mean(), 4),
            sharpe_std=round(sharpe_std, 4),
            sharpe_p5=round(np.percentile(sharpe_array, 5), 4),
            sharpe_p95=round(np.percentile(sharpe_array, 95), 4),
            win_rate_mean=round(win_rate_array.mean(), 4),
            win_rate_std=round(win_rate_array.std(), 4),
            is_robust=is_robust,
            num_simulations=len(sharpes),
            all_sharpes=sharpes,
            all_win_rates=win_rates
        )
    
    def _add_price_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add random noise to OHLCV data.
        
        Args:
            data: Original DataFrame
            
        Returns:
            DataFrame with noise added
        """
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in data.columns:
                noise = np.random.normal(
                    1, 
                    self.noise_percent, 
                    len(data)
                )
                data[col] = data[col] * noise
        
        # Ensure OHLC consistency
        if all(c in data.columns for c in price_cols):
            data['high'] = data[['open', 'high', 'close']].max(axis=1)
            data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def run_bootstrap_trades(
        self,
        trades: pd.DataFrame,
        metrics_func: Callable,
        sample_size: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Bootstrap simulation on trade sequence.
        
        Resamples trades with replacement to estimate
        metric distribution without price noise.
        
        Args:
            trades: DataFrame of trades with 'pnl' column
            metrics_func: Function to calculate metrics from trades
            sample_size: Number of trades per sample (default: original size)
            
        Returns:
            MonteCarloResult from bootstrapped trades
        """
        if trades.empty:
            return MonteCarloResult(
                sharpe_mean=0, sharpe_std=1, sharpe_p5=0, sharpe_p95=0,
                win_rate_mean=0, win_rate_std=1, is_robust=False,
                num_simulations=0, all_sharpes=[], all_win_rates=[]
            )
        
        n = sample_size or len(trades)
        sharpes: List[float] = []
        win_rates: List[float] = []
        
        for _ in range(self.num_simulations):
            # Sample with replacement
            sample = trades.sample(n=n, replace=True)
            
            try:
                metrics = metrics_func(sample)
                sharpes.append(metrics.get('sharpe', 0))
                win_rates.append(metrics.get('win_rate', 0))
            except Exception:
                continue
        
        if not sharpes:
            return MonteCarloResult(
                sharpe_mean=0, sharpe_std=1, sharpe_p5=0, sharpe_p95=0,
                win_rate_mean=0, win_rate_std=1, is_robust=False,
                num_simulations=0, all_sharpes=[], all_win_rates=[]
            )
        
        sharpe_array = np.array(sharpes)
        win_rate_array = np.array(win_rates)
        
        return MonteCarloResult(
            sharpe_mean=round(sharpe_array.mean(), 4),
            sharpe_std=round(sharpe_array.std(), 4),
            sharpe_p5=round(np.percentile(sharpe_array, 5), 4),
            sharpe_p95=round(np.percentile(sharpe_array, 95), 4),
            win_rate_mean=round(win_rate_array.mean(), 4),
            win_rate_std=round(win_rate_array.std(), 4),
            is_robust=sharpe_array.std() < self.robustness_threshold,
            num_simulations=len(sharpes),
            all_sharpes=sharpes,
            all_win_rates=win_rates
        )
    
    def calculate_confidence_interval(
        self,
        values: List[float],
        confidence: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for values.
        
        Args:
            values: List of metric values
            confidence: Confidence level (default: self.confidence_level)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not values:
            return (0.0, 0.0)
        
        conf = confidence or self.confidence_level
        alpha = (1 - conf) / 2
        
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        
        return (round(lower, 4), round(upper, 4))
    
    def get_robustness_report(self, result: MonteCarloResult) -> Dict:
        """
        Generate detailed robustness report.
        
        Args:
            result: MonteCarloResult from simulation
            
        Returns:
            Dictionary with robustness analysis
        """
        if not result.all_sharpes:
            return {"error": "No simulation data"}
        
        sharpes = np.array(result.all_sharpes)
        
        return {
            'is_robust': result.is_robust,
            'sharpe_stats': {
                'mean': result.sharpe_mean,
                'std': result.sharpe_std,
                'median': round(np.median(sharpes), 4),
                'ci_95': self.calculate_confidence_interval(result.all_sharpes),
                'positive_pct': round((sharpes > 0).mean() * 100, 1)
            },
            'win_rate_stats': {
                'mean': result.win_rate_mean,
                'std': result.win_rate_std,
                'ci_95': self.calculate_confidence_interval(result.all_win_rates)
            },
            'simulation_info': {
                'total_runs': self.num_simulations,
                'successful_runs': result.num_simulations,
                'success_rate': round(result.num_simulations / self.num_simulations * 100, 1)
            },
            'recommendation': self._get_recommendation(result)
        }
    
    def _get_recommendation(self, result: MonteCarloResult) -> str:
        """Generate recommendation based on results."""
        if result.sharpe_std > 0.5:
            return "HIGH RISK: Very unstable results - consider different strategy"
        elif result.sharpe_std > self.robustness_threshold:
            return "MODERATE RISK: Results vary significantly - reduce position sizes"
        elif result.sharpe_p5 < 0:
            return "CAUTION: 5th percentile Sharpe is negative - monitor closely"
        elif result.is_robust and result.sharpe_mean > 0.5:
            return "GOOD: Strategy shows robust performance"
        else:
            return "ACCEPTABLE: Strategy is stable but returns are modest"
