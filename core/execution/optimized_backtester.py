"""
Optimized Backtesting Engine with Parallel Processing.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest result"""
    metrics: Dict[str, float]
    trades: List[Dict]
    equity_curve: pd.Series
    execution_time: float
    parameter_set: Optional[Dict] = None

class OptimizedBacktester:
    """High-performance backtesting with parallel processing"""

    def __init__(self, initial_capital: float = 10000, max_workers: int = 4):
        self.initial_capital = initial_capital
        self.max_workers = max_workers

    def run_parallel_backtests(self, data: pd.DataFrame, strategy_func: Callable,
                              parameter_sets: List[Dict]) -> List[BacktestResult]:
        """Run multiple backtests in parallel"""
        start_time = time.time()

        def run_single_backtest(params: Dict) -> BacktestResult:
            try:
                signals = strategy_func(data, **params)
                result = self._run_backtest(data, signals)
                result.parameter_set = params
                return result
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                return BacktestResult({}, [], pd.Series(), 0.0, params)

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_single_backtest, params) for params in parameter_sets]
            for future in as_completed(futures):
                results.append(future.result())

        total_time = time.time() - start_time
        logger.info(f"Parallel backtesting completed in {total_time:.2f}s")
        return results

    def _run_backtest(self, data: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """Run single backtest"""
        start_time = time.time()
        capital = self.initial_capital
        equity = [capital]
        trades = []
        position = 0

        for i in range(1, len(data)):
            price = data.iloc[i]['close']

            if signals.iloc[i] != 0 and position == 0:
                # Entry
                position_size = capital * 0.1
                shares = position_size / price
                position = shares if signals.iloc[i] > 0 else -shares
                capital -= position_size
                trades.append({'type': 'entry', 'time': data.index[i], 'price': price})

            elif signals.iloc[i] == 0 and position != 0:
                # Exit
                pnl = (price - trades[-1]['price']) * position
                capital += abs(position) * price + pnl
                trades.append({'type': 'exit', 'time': data.index[i], 'price': price, 'pnl': pnl})
                position = 0

            equity.append(capital + abs(position) * price if position != 0 else capital)

        equity_series = pd.Series(equity, index=data.index[:len(equity)])
        returns = equity_series.pct_change().fillna(0)

        # Calculate advanced metrics
        metrics = self._calculate_advanced_metrics(equity_series, returns)

        return BacktestResult(metrics, trades, equity_series, time.time() - start_time)

    def _calculate_advanced_metrics(self, equity_series: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive backtesting metrics"""
        # Basic metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        max_drawdown = (equity_series / equity_series.cummax() - 1).min()

        # Sharpe Ratio (annualized con risk-free rate)
        rf_daily = 0.04 / 252
        excess_returns = returns - rf_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0

        # Sortino Ratio (only downside volatility con risk-free rate)
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = (excess_returns.mean() / downside_returns.std() * np.sqrt(252)
                        if len(downside_returns) > 0 and downside_returns.std() > 0 else 0.0)

        # Calmar Ratio (annual return / max drawdown)
        annual_return = total_return * (252 / len(returns))  # Assuming daily data
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Recovery Factor (net profit / max drawdown)
        net_profit = equity_series.iloc[-1] - equity_series.iloc[0]
        recovery_factor = net_profit / abs(max_drawdown) if max_drawdown != 0 else 0

        # K-Ratio (slope of equity curve / standard error of slope)
        k_ratio = self._calculate_k_ratio(equity_series)

        # Statistical validation
        hurst_exponent = self._calculate_hurst_exponent(returns)
        bootstrap_confidence = self._bootstrap_analysis(returns)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'k_ratio': k_ratio,
            'max_drawdown': max_drawdown,
            'hurst_exponent': hurst_exponent,
            'bootstrap_confidence': bootstrap_confidence,
            'win_rate': len([t for t in self._get_trades_from_equity(equity_series) if t.get('pnl', 0) > 0]) /
                       max(len(self._get_trades_from_equity(equity_series)), 1),
            'profit_factor': self._calculate_profit_factor(equity_series)
        }

    def _calculate_k_ratio(self, equity_series: pd.Series) -> float:
        """Calculate K-Ratio (slope efficiency)"""
        try:
            # Linear regression on equity curve
            x = np.arange(len(equity_series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, equity_series.values)

            # K-Ratio = slope / standard error of slope
            k_ratio = slope / std_err if std_err > 0 else 0
            return k_ratio
        except Exception as e:
            logger.warning(f"Error calculating K-Ratio: {e}")
            return 0.0

    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for persistence analysis"""
        try:
            # Simplified Hurst exponent calculation
            lags = range(2, min(100, len(returns)//2))
            tau = []
            for lag in lags:
                # Calculate rescaled range
                rs = []
                for i in range(0, len(returns) - lag, lag):
                    chunk = returns[i:i+lag]
                    if len(chunk) > 0:
                        mean_adj = chunk - chunk.mean()
                        cumulative = mean_adj.cumsum()
                        r = cumulative.max() - cumulative.min()
                        s = chunk.std()
                        if s > 0:
                            rs.append(r / s)

                if rs:
                    tau.append(np.mean(rs))

            if len(tau) > 1:
                # Fit line to log-log plot
                x = np.log(lags[:len(tau)])
                y = np.log(tau)
                slope, _, _, _, _ = stats.linregress(x, y)
                hurst = slope / 2  # Hurst exponent is half the slope
                return max(0, min(1, hurst))  # Clamp to [0,1]
            return 0.5  # Random walk default
        except Exception as e:
            logger.warning(f"Error calculating Hurst exponent: {e}")
            return 0.5

    def _bootstrap_analysis(self, returns: pd.Series, n_bootstraps: int = 1000) -> float:
        """Bootstrap analysis for confidence intervals"""
        try:
            # Bootstrap Sharpe ratios
            sharpe_bootstraps = []
            for _ in range(n_bootstraps):
                sample = returns.sample(n=len(returns), replace=True)
                if sample.std() > 0:
                    sharpe = sample.mean() / sample.std() * np.sqrt(252)
                    sharpe_bootstraps.append(sharpe)

            if sharpe_bootstraps:
                # Return confidence interval width as measure of robustness
                lower = np.percentile(sharpe_bootstraps, 2.5)
                upper = np.percentile(sharpe_bootstraps, 97.5)
                return upper - lower  # Confidence interval width
            return 0.0
        except Exception as e:
            logger.warning(f"Error in bootstrap analysis: {e}")
            return 0.0

    def _get_trades_from_equity(self, equity_series: pd.Series) -> List[Dict]:
        """Extract trade information from equity curve (simplified)"""
        # This is a simplified version - in practice you'd track trades separately
        returns = equity_series.pct_change()
        trades = []
        in_trade = False
        entry_price = 0

        for i, ret in enumerate(returns):
            if not in_trade and ret != 0:
                in_trade = True
                entry_price = equity_series.iloc[i-1] if i > 0 else equity_series.iloc[0]
            elif in_trade and ret == 0:
                exit_price = equity_series.iloc[i]
                pnl = exit_price - entry_price
                trades.append({'pnl': pnl})
                in_trade = False

        return trades

    def _calculate_profit_factor(self, equity_series: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        trades = self._get_trades_from_equity(equity_series)
        if not trades:
            return 1.0

        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]

        total_profit = sum(profits)
        total_loss = sum(losses)

        return total_profit / total_loss if total_loss > 0 else float('inf')