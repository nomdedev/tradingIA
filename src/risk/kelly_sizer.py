"""
Kelly Criterion Position Sizing for Optimal Capital Allocation

This module implements the Kelly Criterion formula for optimal position sizing
in trading strategies. The Kelly Criterion maximizes long-term growth by
balancing expected returns against risk.

Formula: f = (bp - q) / b
Where:
- f = fraction of capital to risk
- b = odds (win_loss_ratio - 1)
- p = probability of winning
- q = probability of losing (1 - p)

Author: AI Assistant
Date: 16 de Noviembre, 2025
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Result of Kelly calculation"""
    kelly_fraction: float
    kelly_full: float
    kelly_half: float  # Conservative (50% of full Kelly)
    kelly_quarter: float  # Very conservative (25% of full Kelly)
    optimal_position_size: float
    expected_growth_rate: float
    confidence_interval: Tuple[float, float]


class KellyPositionSizer:
    """
    Kelly Criterion position sizing calculator.

    Provides optimal position sizing based on win rate and win/loss ratio.
    Includes risk adjustments and market impact considerations.
    """

    def __init__(self,
                 kelly_fraction: float = 0.5,
                 max_position_pct: float = 0.10,  # Max 10% of capital
                 min_position_pct: float = 0.001,  # Min 0.1% of capital
                 volatility_adjustment: bool = True):
        """
        Initialize Kelly position sizer.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25-1.0, default 0.5)
            max_position_pct: Maximum position size as % of capital
            min_position_pct: Minimum position size as % of capital
            volatility_adjustment: Adjust for market volatility
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.volatility_adjustment = volatility_adjustment

        # Validation
        if not 0.1 <= kelly_fraction <= 1.0:
            raise ValueError("Kelly fraction must be between 0.1 and 1.0")

        logger.info(f"Kelly Position Sizer initialized: fraction={kelly_fraction}, "
                   f"max_pos={max_position_pct:.1%}")

    def calculate_kelly_fraction(self,
                               win_rate: float,
                               win_loss_ratio: float,
                               market_impact_pct: float = 0.0) -> KellyResult:
        """
        Calculate optimal Kelly fraction.

        Args:
            win_rate: Probability of winning (0.0-1.0)
            win_loss_ratio: Average win / Average loss ratio
            market_impact_pct: Estimated market impact cost (0.0-1.0)

        Returns:
            KellyResult with all sizing calculations
        """
        # Input validation
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError("Win rate must be between 0.0 and 1.0")

        if win_loss_ratio <= 0:
            # No edge - return zero position
            return KellyResult(
                kelly_fraction=0.0,
                kelly_full=0.0,
                kelly_half=0.0,
                kelly_quarter=0.0,
                optimal_position_size=0.0,
                expected_growth_rate=0.0,
                confidence_interval=(0.0, 0.0)
            )

        # Adjust win/loss ratio for market impact costs
        adjusted_win_loss_ratio = win_loss_ratio * (1.0 - market_impact_pct)

        if adjusted_win_loss_ratio <= 1.0:
            # Expected value is negative after costs
            return KellyResult(
                kelly_fraction=0.0,
                kelly_full=0.0,
                kelly_half=0.0,
                kelly_quarter=0.0,
                optimal_position_size=0.0,
                expected_growth_rate=0.0,
                confidence_interval=(0.0, 0.0)
            )

        # Kelly formula: f = (bp - q) / b
        # Where: b = win_loss_ratio (odds), p = win_rate, q = 1 - p
        # Simplifica a: f = p - q/b = p - (1-p)/b
        p = win_rate
        q = 1.0 - win_rate
        b = adjusted_win_loss_ratio

        # Fórmula correcta de Kelly
        kelly_full = (p * b - q) / b
        # Equivalente: kelly_full = p - q/b

        # Handle edge cases
        if not np.isfinite(kelly_full) or kelly_full < 0:
            kelly_full = 0.0

        # Conservative fractions
        kelly_half = kelly_full * 0.5
        kelly_quarter = kelly_full * 0.25

        # Apply user-specified fraction
        kelly_fraction = kelly_full * self.kelly_fraction

        # Calculate expected growth rate
        expected_growth_rate = self._calculate_expected_growth(
            kelly_fraction, win_rate, adjusted_win_loss_ratio
        )

        # Calculate confidence interval (simplified approximation)
        confidence_interval = self._calculate_confidence_interval(
            kelly_fraction, win_rate, adjusted_win_loss_ratio
        )

        return KellyResult(
            kelly_fraction=kelly_fraction,
            kelly_full=kelly_full,
            kelly_half=kelly_half,
            kelly_quarter=kelly_quarter,
            optimal_position_size=kelly_fraction,  # Will be scaled by capital later
            expected_growth_rate=expected_growth_rate,
            confidence_interval=confidence_interval
        )

    def calculate_position_size(self,
                              capital: float,
                              win_rate: float,
                              win_loss_ratio: float,
                              current_volatility: float = 0.0,
                              market_impact_pct: float = 0.0) -> Dict:
        """
        Calculate actual position size for given capital.

        Args:
            capital: Available capital
            win_rate: Probability of winning
            win_loss_ratio: Average win / Average loss ratio
            current_volatility: Current market volatility (0.0-1.0)
            market_impact_pct: Estimated market impact cost

        Returns:
            Dictionary with position sizing details (includes non-float values)
        """
        # Get Kelly calculation
        kelly_result = self.calculate_kelly_fraction(
            win_rate, win_loss_ratio, market_impact_pct
        )

        # Adjust for volatility (reduce position in high volatility)
        volatility_multiplier = 1.0
        if self.volatility_adjustment and current_volatility > 0:
            # Use exponential decay for smoother adjustment
            # High volatility (0.5+) reduces position significantly
            # Low volatility (<0.1) has minimal impact
            volatility_multiplier = np.exp(-2.0 * current_volatility)
            # Ensure minimum of 0.3 (never reduce more than 70%)
            volatility_multiplier = max(0.3, min(1.0, volatility_multiplier))

        # Calculate base position size
        base_position_size = kelly_result.kelly_fraction * capital * volatility_multiplier

        # Apply bounds
        max_position = capital * self.max_position_pct
        min_position = capital * self.min_position_pct

        optimal_position_size = np.clip(base_position_size, min_position, max_position)

        # Calculate risk metrics
        position_pct = optimal_position_size / capital
        risk_per_trade = position_pct * (1.0 / win_loss_ratio)  # Risk = position / win_loss_ratio

        return {
            'position_size': optimal_position_size,
            'position_pct': position_pct,
            'risk_per_trade_pct': risk_per_trade,
            'kelly_fraction': kelly_result.kelly_fraction,
            'kelly_full': kelly_result.kelly_full,
            'expected_growth_rate': kelly_result.expected_growth_rate,
            'confidence_interval': kelly_result.confidence_interval,
            'volatility_adjustment': volatility_multiplier,
            'market_impact_adjusted': market_impact_pct > 0
        }

    def optimize_kelly_fraction(self,
                              historical_trades: pd.DataFrame,
                              optimization_metric: str = 'sharpe') -> float:
        """
        Optimize Kelly fraction based on historical performance.

        Args:
            historical_trades: DataFrame with trade results
            optimization_metric: 'sharpe', 'sortino', 'max_dd', 'total_return'

        Returns:
            Optimal Kelly fraction
        """
        if historical_trades.empty:
            return self.kelly_fraction  # Return default

        # Calculate win rate and win/loss ratio from historical data
        wins = historical_trades[historical_trades['pnl'] > 0]
        losses = historical_trades[historical_trades['pnl'] < 0]

        if len(losses) == 0:
            win_rate = 1.0
            win_loss_ratio = float('inf')
        else:
            win_rate = len(wins) / len(historical_trades)
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses['pnl'].mean())
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Test different Kelly fractions
        kelly_fractions = np.linspace(0.1, 1.0, 10)
        best_metric = -float('inf')
        best_fraction = 0.5

        for fraction in kelly_fractions:
            temp_sizer = KellyPositionSizer(kelly_fraction=fraction)

            # Simulate portfolio with this Kelly fraction
            portfolio_values = self._simulate_portfolio(
                historical_trades, temp_sizer, initial_capital=10000
            )

            # Calculate optimization metric (con risk-free rate)
            if optimization_metric == 'sharpe':
                returns = pd.Series(portfolio_values).pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    rf_daily = 0.04 / 252
                    excess_returns = returns - rf_daily
                    metric = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
                else:
                    metric = 0
            elif optimization_metric == 'sortino':
                returns = pd.Series(portfolio_values).pct_change().dropna()
                rf_daily = 0.04 / 252
                excess_returns = returns - rf_daily
                downside_returns = excess_returns[excess_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    metric = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
                else:
                    metric = excess_returns.mean() * np.sqrt(252) if excess_returns.mean() > 0 else 0
            elif optimization_metric == 'max_dd':
                peak = pd.Series(portfolio_values).expanding().max()
                drawdown = (pd.Series(portfolio_values) - peak) / peak
                metric = -drawdown.min()  # Negative because we want to minimize drawdown
            elif optimization_metric == 'total_return':
                metric = (portfolio_values[-1] / portfolio_values[0]) - 1
            else:
                metric = 0

            if metric > best_metric:
                best_metric = metric
                best_fraction = fraction

        logger.info(f"Optimized Kelly fraction: {best_fraction:.2f} "
                   f"(metric: {optimization_metric} = {best_metric:.3f})")

        return best_fraction

    def _calculate_expected_growth(self,
                                 kelly_fraction: float,
                                 win_rate: float,
                                 win_loss_ratio: float) -> float:
        """Calculate expected growth rate with Kelly sizing"""
        if kelly_fraction <= 0:
            return 0.0

        # Expected growth rate formula
        growth_rate = win_rate * np.log(1 + kelly_fraction * (win_loss_ratio - 1)) + \
                     (1 - win_rate) * np.log(1 - kelly_fraction)

        return growth_rate

    def _calculate_confidence_interval(self,
                                     kelly_fraction: float,
                                     win_rate: float,
                                     win_loss_ratio: float,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for Kelly fraction (simplified)"""
        if kelly_fraction <= 0:
            return (0.0, 0.0)

        # Simplified confidence interval based on binomial distribution
        n = 100  # Assume 100 trades for confidence calculation
        variance = (win_rate * (1 - win_rate)) / n

        # Standard error of Kelly fraction
        se = np.sqrt(variance) * 2  # Approximation

        margin = se * 1.96  # 95% confidence
        lower = max(0, kelly_fraction - margin)
        upper = kelly_fraction + margin

        return (lower, upper)

    def _simulate_portfolio(self,
                          trades: pd.DataFrame,
                          sizer: 'KellyPositionSizer',
                          initial_capital: float = 10000) -> List[float]:
        """Simulate portfolio growth with Kelly sizing"""
        capital = initial_capital
        portfolio_values = [capital]

        for _, trade in trades.iterrows():
            # Calculate position size using Kelly
            sizing_result = sizer.calculate_position_size(
                capital=capital,
                win_rate=0.5,  # Simplified assumption
                win_loss_ratio=2.0,  # Simplified assumption
                current_volatility=0.2
            )

            position_size = sizing_result['position_size']
            pnl = trade['pnl']

            # Update capital
            capital += pnl
            portfolio_values.append(capital)

        return portfolio_values

    def get_risk_warnings(self, kelly_result: KellyResult) -> List[str]:
        """Get risk warnings based on Kelly calculation"""
        warnings = []

        if kelly_result.kelly_full > 1.0:
            warnings.append("⚠️ Kelly fraction > 100% - High risk of ruin")

        if kelly_result.kelly_full > 0.5:
            warnings.append("⚠️ Kelly fraction > 50% - Aggressive sizing")

        if kelly_result.kelly_fraction < 0.1:
            warnings.append("⚠️ Kelly fraction < 10% - Very conservative")

        if kelly_result.expected_growth_rate < 0:
            warnings.append("❌ Negative expected growth - Avoid trading")

        return warnings