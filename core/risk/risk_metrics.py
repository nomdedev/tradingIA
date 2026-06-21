"""
Risk Metrics Calculator
Implements advanced risk metrics including VaR, CVaR, and Stress Testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class RiskMetricsCalculator:
    """
    Calculates advanced risk metrics for trading strategies.
    """
    
    def __init__(self, returns_series: Union[pd.Series, np.ndarray] = None):
        """
        Initialize with a series of returns (percentage).
        
        Args:
            returns_series: Series of returns (e.g., daily returns)
        """
        self.returns = None
        if returns_series is not None:
            self.set_returns(returns_series)
            
    def set_returns(self, returns_series: Union[pd.Series, np.ndarray]):
        """Set the returns data for analysis"""
        if isinstance(returns_series, list):
            self.returns = np.array(returns_series)
        elif isinstance(returns_series, pd.Series):
            self.returns = returns_series.values
        else:
            self.returns = returns_series
            
        # Remove NaNs
        self.returns = self.returns[~np.isnan(self.returns)]
        
    def calculate_var(self, confidence_level: float = 0.95, method: str = "historical") -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR value (positive float representing loss %)
        """
        if self.returns is None or len(self.returns) == 0:
            return 0.0
            
        if method == "historical":
            # Historical VaR is the percentile of the return distribution
            # For 95% confidence, we look at the 5% worst returns
            percentile = (1.0 - confidence_level) * 100
            var = -np.percentile(self.returns, percentile)
            return max(0.0, var)
            
        elif method == "parametric":
            # Parametric VaR assumes normal distribution
            mu = np.mean(self.returns)
            sigma = np.std(self.returns)
            # Z-score for the confidence level
            from scipy.stats import norm
            z_score = norm.ppf(1.0 - confidence_level)
            var = -(mu + z_score * sigma)
            return max(0.0, var)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        Average of losses exceeding VaR.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            CVaR value (positive float representing loss %)
        """
        if self.returns is None or len(self.returns) == 0:
            return 0.0
            
        # Calculate VaR threshold (negative return value)
        percentile = (1.0 - confidence_level) * 100
        var_threshold = np.percentile(self.returns, percentile)
        
        # Filter returns worse than threshold
        tail_losses = self.returns[self.returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
            
        # CVaR is the average of these tail losses (negated to be positive)
        cvar = -np.mean(tail_losses)
        return max(0.0, cvar)
        
    def calculate_max_drawdown(self, equity_curve: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Maximum Drawdown from equity curve.
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Max Drawdown % (positive float)
        """
        if isinstance(equity_curve, list):
            equity_curve = np.array(equity_curve)
            
        if len(equity_curve) == 0:
            return 0.0
            
        # Calculate peaks
        peaks = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdowns
        drawdowns = (peaks - equity_curve) / peaks
        
        # Max drawdown
        return np.max(drawdowns)
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe Ratio (annualized assuming daily returns)"""
        if self.returns is None or len(self.returns) == 0:
            return 0.0
            
        excess_returns = self.returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
            
        # Annualize (assuming 252 trading days, or 365 for crypto)
        # Let's assume these are per-trade returns or daily returns passed in.
        # If daily: sqrt(365) for crypto
        annualization_factor = np.sqrt(365) 
        
        return np.mean(excess_returns) / np.std(excess_returns) * annualization_factor

    def calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """Calculate Sortino Ratio (downside risk only)"""
        if self.returns is None or len(self.returns) == 0:
            return 0.0
            
        excess_returns = self.returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
            
        annualization_factor = np.sqrt(365)
        return np.mean(excess_returns) / np.std(downside_returns) * annualization_factor

    def monte_carlo_simulation(self, num_simulations: int = 1000, horizon: int = 30) -> Dict:
        """
        Run Monte Carlo simulation to project future equity.
        
        Args:
            num_simulations: Number of paths to simulate
            horizon: Number of steps (days/trades) to project
            
        Returns:
            Dictionary with simulation results (percentiles)
        """
        if self.returns is None or len(self.returns) < 10:
            return {}
            
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        
        # Simulate paths
        # Shape: (num_simulations, horizon)
        simulated_returns = np.random.normal(mu, sigma, (num_simulations, horizon))
        
        # Calculate cumulative returns path (starting at 1.0)
        simulated_paths = np.cumprod(1 + simulated_returns, axis=1)
        
        # Get final values
        final_values = simulated_paths[:, -1]
        
        return {
            "mean_final": np.mean(final_values),
            "median_final": np.median(final_values),
            "p95_final": np.percentile(final_values, 95), # Best case
            "p05_final": np.percentile(final_values, 5),  # Worst case
            "paths": simulated_paths # Full paths for plotting
        }
