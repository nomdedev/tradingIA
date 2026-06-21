"""
Metrics Calculator Module

Extracted from BacktesterCore for better modularity.
Handles all trading metrics calculations including:
- Sharpe, Sortino, Calmar ratios
- Drawdown metrics
- Trade statistics (win rate, profit factor)
- MAE/MFE analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

# Import constants
try:
    from core.constants import (
        RISK_FREE_RATE_DAILY,
        TRADING_DAYS_PER_YEAR,
    )
except ImportError:
    RISK_FREE_RATE_DAILY = 0.04 / 252
    TRADING_DAYS_PER_YEAR = 252

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates comprehensive trading performance metrics.
    
    Provides methods for:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis (Max DD, Ulcer Index)
    - Trade statistics (Win Rate, Profit Factor)
    - Information Ratio vs benchmark
    - MAE/MFE trade quality metrics
    """
    
    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE_DAILY,
        trading_days: int = TRADING_DAYS_PER_YEAR
    ) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Daily risk-free rate for excess return calculations
            trading_days: Trading days per year for annualization
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        trades_records: pd.DataFrame,
        close: Optional[pd.Series] = None,
        trade_history: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calculate comprehensive trading metrics.
        
        Args:
            returns: Series of periodic returns
            trades_records: DataFrame of trade records with 'pnl' column
            close: Price series for benchmark comparison (optional)
            trade_history: DataFrame with MAE/MFE columns (optional)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        try:
            metrics = {}
            
            # Basic returns metrics
            cumulative_returns = (1 + returns).cumprod()
            metrics['total_return'] = round(cumulative_returns.iloc[-1] - 1, 3)
            
            # Sharpe Ratio
            metrics['sharpe'] = self.calculate_sharpe(returns)
            
            # Sortino Ratio
            metrics['sortino'] = self.calculate_sortino(returns)
            
            # Max Drawdown and Calmar
            metrics['max_dd'] = self.calculate_max_drawdown(cumulative_returns)
            metrics['calmar'] = self.calculate_calmar(
                metrics['total_return'], 
                metrics['max_dd']
            )
            
            # Ulcer Index
            metrics['ulcer'] = self.calculate_ulcer_index(cumulative_returns)
            
            # Trade statistics
            trade_stats = self.calculate_trade_statistics(trades_records)
            metrics.update(trade_stats)
            
            # Information Ratio
            metrics['ir'] = self.calculate_information_ratio(returns, close)
            
            # MAE/MFE metrics
            mae_mfe = self.calculate_mae_mfe_metrics(trade_history)
            metrics.update(mae_mfe)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}
    
    def calculate_sharpe(self, returns: pd.Series) -> float:
        """
        Calculate annualized Sharpe Ratio.
        
        Args:
            returns: Series of periodic returns
            
        Returns:
            Annualized Sharpe Ratio
        """
        excess_returns = returns - self.risk_free_rate
        std = excess_returns.std()
        # Use threshold to avoid division by very small numbers
        if std > 1e-10:
            sharpe = (excess_returns.mean() / std 
                     * np.sqrt(self.trading_days))
        else:
            sharpe = 0.0
        return round(sharpe, 3)
    
    def calculate_sortino(self, returns: pd.Series) -> float:
        """
        Calculate annualized Sortino Ratio.
        
        Uses only downside deviation for risk measurement.
        
        Args:
            returns: Series of periodic returns
            
        Returns:
            Annualized Sortino Ratio
        """
        excess_returns = returns - self.risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino = (excess_returns.mean() / downside_std 
                          * np.sqrt(self.trading_days))
            else:
                sortino = 0.0
        else:
            sortino = 0.0
        
        return round(sortino, 3)
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from cumulative returns.
        
        Args:
            cumulative_returns: Cumulative return series
            
        Returns:
            Maximum drawdown as positive decimal
        """
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return round(abs(drawdown.min()), 3)
    
    def calculate_calmar(self, total_return: float, max_dd: float) -> float:
        """
        Calculate Calmar Ratio.
        
        Args:
            total_return: Total return over period
            max_dd: Maximum drawdown
            
        Returns:
            Calmar Ratio
        """
        if max_dd > 0:
            return round(total_return / max_dd, 3)
        return 0.0
    
    def calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate Ulcer Index (measure of downside risk).
        
        The Ulcer Index measures the depth and duration of drawdowns.
        Lower values indicate less painful investment experience.
        
        Args:
            cumulative_returns: Cumulative return series
            
        Returns:
            Ulcer Index value
        """
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return round(np.sqrt((drawdown ** 2).mean()), 3)
    
    def calculate_trade_statistics(self, trades_records: pd.DataFrame) -> Dict:
        """
        Calculate trade-level statistics.
        
        Args:
            trades_records: DataFrame with 'pnl' column
            
        Returns:
            Dictionary with win_rate, num_trades, profit_factor
        """
        if trades_records.empty or 'pnl' not in trades_records.columns:
            return {
                'win_rate': 0.0,
                'num_trades': 0,
                'profit_factor': 0.0
            }
        
        wins = trades_records['pnl'] > 0
        win_rate = wins.mean()
        num_trades = len(trades_records)
        
        gross_profit = trades_records[wins]['pnl'].sum()
        gross_loss = abs(trades_records[~wins]['pnl'].sum())
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        
        return {
            'win_rate': round(win_rate, 3),
            'num_trades': num_trades,
            'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999.99
        }
    
    def calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_close: Optional[pd.Series]
    ) -> float:
        """
        Calculate Information Ratio vs buy-and-hold benchmark.
        
        Args:
            returns: Strategy returns
            benchmark_close: Price series for benchmark
            
        Returns:
            Annualized Information Ratio
        """
        if benchmark_close is None or len(benchmark_close) < 2:
            return 0.0
        
        try:
            bh_returns = benchmark_close.pct_change().dropna()
            
            # Align lengths
            min_len = min(len(returns), len(bh_returns))
            aligned_returns = returns.iloc[-min_len:]
            aligned_bh = bh_returns.iloc[-min_len:]
            
            active_return = aligned_returns - aligned_bh
            tracking_error = active_return.std()
            
            if tracking_error > 0:
                ir = (active_return.mean() / tracking_error 
                     * np.sqrt(self.trading_days))
            else:
                ir = 0.0
            
            return round(ir, 3)
            
        except Exception:
            return 0.0
    
    def calculate_mae_mfe_metrics(
        self, 
        trade_history: Optional[pd.DataFrame]
    ) -> Dict:
        """
        Calculate Maximum Adverse/Favorable Excursion metrics.
        
        MAE: How far against the position moved before exit
        MFE: How far in favor the position moved before exit
        
        Args:
            trade_history: DataFrame with 'mae' and 'mfe' columns
            
        Returns:
            Dictionary with avg/max MAE and MFE values
        """
        default = {
            'avg_mae': 0.0,
            'avg_mfe': 0.0,
            'max_mae': 0.0,
            'max_mfe': 0.0
        }
        
        if trade_history is None or trade_history.empty:
            return default
        
        result = {}
        
        if 'mae' in trade_history.columns:
            result['avg_mae'] = round(trade_history['mae'].mean(), 4)
            result['max_mae'] = round(trade_history['mae'].max(), 4)
        else:
            result['avg_mae'] = 0.0
            result['max_mae'] = 0.0
        
        if 'mfe' in trade_history.columns:
            result['avg_mfe'] = round(trade_history['mfe'].mean(), 4)
            result['max_mfe'] = round(trade_history['mfe'].max(), 4)
        else:
            result['avg_mfe'] = 0.0
            result['max_mfe'] = 0.0
        
        return result
    
    def calculate_drawdown_series(self, cumulative_returns: pd.Series) -> pd.Series:
        """
        Calculate full drawdown series.
        
        Args:
            cumulative_returns: Cumulative return series
            
        Returns:
            Series of drawdowns at each point
        """
        peak = cumulative_returns.expanding().max()
        return (cumulative_returns - peak) / peak
    
    def calculate_recovery_time(
        self, 
        cumulative_returns: pd.Series,
        threshold: float = 0.05
    ) -> int:
        """
        Calculate average time to recover from drawdowns.
        
        Args:
            cumulative_returns: Cumulative return series
            threshold: Minimum drawdown depth to consider
            
        Returns:
            Average recovery bars
        """
        drawdown = self.calculate_drawdown_series(cumulative_returns)
        
        in_dd = False
        dd_start = 0
        recovery_times = []
        
        for i, dd in enumerate(drawdown):
            if not in_dd and dd < -threshold:
                in_dd = True
                dd_start = i
            elif in_dd and dd >= 0:
                recovery_times.append(i - dd_start)
                in_dd = False
        
        return int(np.mean(recovery_times)) if recovery_times else 0


# Convenience function for quick metric calculation
def calculate_quick_metrics(
    returns: pd.Series,
    trades: pd.DataFrame
) -> Dict:
    """
    Quick metrics calculation using default settings.
    
    Args:
        returns: Return series
        trades: Trades DataFrame
        
    Returns:
        Metrics dictionary
    """
    calc = MetricsCalculator()
    return calc.calculate_all_metrics(returns, trades)
