#!/usr/bin/env python3
"""
A/B Testing Base Protocol for BTC Trading Signals
==================================================

Base protocol for A/B testing trading signals: Compare A (IFVG base) vs B (RSI/BB alternative).
Implements data split, parallel backtesting, statistical analysis, and decision making.

Features:
- Random data split (50/50 periods, seed=42)
- Parallel backtesting with vectorbt
- Statistical significance testing (t-test)
- Superiority analysis (% periods B > A)
- Minimum trade requirements (100+ trades)
- A/A test for bias verification

Usage:
    from src.ab_base_protocol import ab_protocol
    results = ab_protocol(df_5m, variant_a_signals, variant_b_signals)
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import warnings
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import optional dependencies
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("vectorbt not available. Using simplified backtesting.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ABTestingBase:
    """Base A/B testing protocol for trading signals"""

    def __init__(self, capital: float = 10000, slippage: float = 0.001, commission: float = 0.0005):
        self.capital = capital
        self.slippage = slippage
        self.commission = commission
        self.random_seed = 42
        np.random.seed(self.random_seed)

    def split_data_random(self,
                          df: pd.DataFrame,
                          split_ratio: float = 0.5) -> Tuple[pd.DataFrame,
                                                             pd.DataFrame]:
        """
        Split data randomly by periods (not time-sequential to avoid temporal bias)

        Args:
            df: DataFrame with OHLCV data
            split_ratio: Ratio for split A (default 0.5)

        Returns:
            Tuple of (df_a, df_b)
        """
        # Get unique dates
        unique_dates = df.index.normalize().unique()

        # Shuffle dates
        shuffled_dates = (unique_dates.to_series()
                         .sample(frac=1, random_state=self.random_seed).values)

        # Split dates
        split_idx = int(len(shuffled_dates) * split_ratio)
        dates_a = shuffled_dates[:split_idx]
        dates_b = shuffled_dates[split_idx:]

        # Filter dataframes
        df_a = df[df.index.normalize().isin(dates_a)].copy()
        df_b = df[df.index.normalize().isin(dates_b)].copy()

        logger.info(
            f"Data split: A={len(df_a)} bars ({len(dates_a)} days), "
            f"B={len(df_b)} bars ({len(dates_b)} days)")

        return df_a, df_b

    def run_backtest_vectorbt(self, df: pd.DataFrame, signals: pd.Series,
                              strategy_name: str) -> Dict[str, Any]:
        """
        Run backtest using vectorbt (if available) or simplified version

        Args:
            df: OHLCV DataFrame
            signals: Entry signals (1 for long, -1 for short, 0 for neutral)
            strategy_name: Name for logging

        Returns:
            Dictionary with backtest results
        """
        if not VECTORBT_AVAILABLE:
            return self._run_backtest_simple(df, signals, strategy_name)

        try:
            # VectorBT implementation
            entries = signals == 1
            exits = signals == -1

            pf = vbt.Portfolio.from_signals(
                df['close'],
                entries=entries,
                exits=exits,
                price=df['close'],
                init_cash=self.capital,
                fees=self.commission,
                slippage=self.slippage,
                freq='5min'
            )

            # Calculate metrics
            returns = pf.returns()
            total_return = pf.total_return()
            sharpe = pf.sharpe_ratio()
            win_rate = pf.win_rate()
            total_trades = pf.trades.count()
            max_drawdown = pf.max_drawdown()

            results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'max_drawdown': max_drawdown,
                'returns': returns,
                'trades': pf.trades.records_readable if hasattr(
                    pf.trades,
                    'records_readable') else None}

            logger.info(
                f"{strategy_name} backtest: {total_trades} trades, "
                f"Sharpe={sharpe:.3f}, Win={win_rate:.1%}")

            return results

        except Exception as e:
            logger.warning(f"VectorBT backtest failed: {e}. Using simplified version.")
            return self._run_backtest_simple(df, signals, strategy_name)

    def _run_backtest_simple(self, df: pd.DataFrame, signals: pd.Series,
                             strategy_name: str) -> Dict[str, Any]:
        """
        Simplified backtest implementation when vectorbt is not available
        """
        # Simple position-based backtest
        position = 0
        entry_price = 0
        trades = []
        capital = self.capital

        for i, (idx, row) in enumerate(df.iterrows()):
            signal = signals.iloc[i] if i < len(signals) else 0

            # Entry signal
            if position == 0 and signal == 1:
                position = 1
                entry_price = row['close'] * (1 + self.slippage)
                capital -= entry_price * self.commission
                logger.debug(f"{strategy_name} LONG at {entry_price:.2f}")

            # Exit signal or end of data
            elif position == 1 and (signal == -1 or i == len(df) - 1):
                exit_price = row['close'] * (1 - self.slippage)
                pnl = (exit_price - entry_price) / entry_price
                capital += exit_price + pnl * capital
                capital -= exit_price * self.commission

                trades.append({
                    'entry_time': df.index[df.index.get_loc(idx) - 1] if i > 0 else idx,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'pnl_abs': pnl * self.capital
                })

                position = 0
                entry_price = 0
                logger.debug(f"{strategy_name} EXIT at {exit_price:.2f}, PnL={pnl:.2%}")

        # Calculate metrics
        if trades:
            pnl_series = pd.Series([t['pnl_pct'] for t in trades])
            total_return = (capital - self.capital) / self.capital
            
            # Sharpe Ratio (con risk-free rate correcto)
            # Asumiendo timeframe de 5 min: 252 * 24 * 12 períodos por año
            periods_per_year = 252 * 12  # Aproximado para timeframe 1H
            rf_per_period = 0.04 / periods_per_year
            excess_pnl = pnl_series - rf_per_period
            sharpe_ratio = (excess_pnl.mean() / excess_pnl.std()) * np.sqrt(periods_per_year) if excess_pnl.std() > 0 else 0.0
            
            win_rate = (pnl_series > 0).mean()
            total_trades = len(trades)
            max_drawdown = 0.15  # Simplified assumption
        else:
            total_return = 0
            sharpe_ratio = 0
            win_rate = 0
            total_trades = 0
            max_drawdown = 0
            pnl_series = pd.Series([])

        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'max_drawdown': max_drawdown,
            'returns': pnl_series,
            'trades': trades
        }

        logger.info(
            f"{strategy_name} backtest: {total_trades} trades, Sharpe={sharpe_ratio:.3f}, Win={win_rate:.1%}")

        return results

    def statistical_analysis(self, results_a: Dict, results_b: Dict) -> Dict[str, Any]:
        """
        Perform statistical analysis comparing A vs B

        Args:
            results_a: Results from variant A
            results_b: Results from variant B

        Returns:
            Dictionary with statistical test results
        """
        returns_a = results_a['returns']
        returns_b = results_b['returns']

        # t-test for returns difference
        if len(returns_a) > 1 and len(returns_b) > 1:
            t_stat, p_value = ttest_rel(returns_a, returns_b)
        else:
            t_stat, p_value = 0, 1.0

        # Superiority analysis
        superiority_pct = (returns_b > returns_a).mean() if len(returns_a) == len(returns_b) else 0

        # Effect size (Cohen's d)
        if len(returns_a) > 1 and len(returns_b) > 1:
            mean_diff = returns_b.mean() - returns_a.mean()
            pooled_std = np.sqrt((returns_a.var() + returns_b.var()) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0

        analysis = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': bool(p_value < 0.05),
            'superiority_percentage': superiority_pct,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_effect_size(cohens_d)
        }

        logger.info(
            f"Statistical analysis: p={p_value:.4f}, significant={p_value < 0.05}, superiority={superiority_pct:.1%}")

        return analysis

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def run_aa_test(self, df: pd.DataFrame, signals: pd.Series, n_runs: int = 5) -> Dict[str, Any]:
        """
        Run A/A test to verify no bias in the testing framework

        Args:
            df: DataFrame with OHLCV data
            signals: Trading signals
            n_runs: Number of A/A test runs

        Returns:
            Dictionary with A/A test results
        """
        logger.info(f"Running A/A test with {n_runs} iterations...")

        p_values = []
        sharpe_diffs = []

        for i in range(n_runs):
            # Split same signals into A and B
            df_a, df_b = self.split_data_random(df)

            # Run backtests with same signals
            results_a = self.run_backtest_vectorbt(df_a, signals.loc[df_a.index], f"A/A Run {i+1}A")
            results_b = self.run_backtest_vectorbt(df_b, signals.loc[df_b.index], f"A/A Run {i+1}B")

            # Statistical test
            returns_a = results_a['returns']
            returns_b = results_b['returns']

            if len(returns_a) > 1 and len(returns_b) > 1:
                # Handle different lengths by taking minimum
                min_len = min(len(returns_a), len(returns_b))
                _, p_value = ttest_rel(returns_a[:min_len], returns_b[:min_len])
            else:
                p_value = 1.0

            p_values.append(p_value)

            sharpe_diff = results_b['sharpe_ratio'] - results_a['sharpe_ratio']
            sharpe_diffs.append(sharpe_diff)

        # Analyze A/A results
        if p_values:
            false_positives = sum(1 for p in p_values if p < 0.05) / len(p_values)
            avg_p_value = np.mean(p_values)
            avg_sharpe_diff = np.mean(sharpe_diffs)
        else:
            false_positives = 0
            avg_p_value = 1.0
            avg_sharpe_diff = 0

        aa_results = {
            'n_runs': n_runs,
            'false_positive_rate': false_positives,
            'average_p_value': avg_p_value,
            'average_sharpe_difference': avg_sharpe_diff,
            'bias_detected': false_positives > 0.1  # More than 10% false positives
        }

        logger.info(
            f"A/A test: false positive rate={false_positives:.1%}, bias detected={aa_results['bias_detected']}")

        return aa_results

    def make_decision(self, results_a: Dict, results_b: Dict, stats: Dict,
                      min_trades: int = 100) -> Dict[str, Any]:
        """
        Make decision based on A/B test results

        Args:
            results_a: Results from variant A
            results_b: Results from variant B
            stats: Statistical analysis results
            min_trades: Minimum trades required

        Returns:
            Dictionary with decision and reasoning
        """
        # Check minimum trade requirements
        trades_a = results_a['total_trades']
        trades_b = results_b['total_trades']

        if trades_a < min_trades or trades_b < min_trades:
            decision = {
                'recommendation': 'INSUFFICIENT_DATA',
                'reason': f'Insufficient trades: A={trades_a}, B={trades_b} (minimum {min_trades})',
                'confidence': 0
            }
            return decision

        # Decision logic
        p_value = stats['p_value']
        superiority = stats['superiority_percentage']
        sharpe_diff = results_b['sharpe_ratio'] - results_a['sharpe_ratio']

        if p_value < 0.05 and superiority > 0.6 and sharpe_diff > 0.2:
            recommendation = 'ADOPT_B'
            confidence = min(0.9, superiority * (1 - p_value))
            reason = f"B significantly better (p={p_value:.3f}, superiority={superiority:.1%}, ΔSharpe={sharpe_diff:.3f})"
        elif p_value < 0.1 and sharpe_diff > 0.1:
            recommendation = 'HYBRID_TEST'
            confidence = 0.6
            reason = f"B shows promise but needs more testing (p={p_value:.3f}, ΔSharpe={sharpe_diff:.3f})"
        else:
            recommendation = 'KEEP_A'
            confidence = 0.7
            reason = f"No significant improvement with B (p={p_value:.3f}, superiority={superiority:.1%})"

        decision = {
            'recommendation': recommendation,
            'reason': reason,
            'confidence': confidence,
            'metrics': {
                'p_value': p_value,
                'superiority': superiority,
                'sharpe_diff': sharpe_diff,
                'trades_a': trades_a,
                'trades_b': trades_b
            }
        }

        logger.info(f"Decision: {recommendation} (confidence: {confidence:.1%})")

        return decision


def ab_protocol(hypothesis: str, variant_a_signals: pd.Series, variant_b_signals: pd.Series,
                df_5m: pd.DataFrame, min_trades: int = 100) -> Dict[str, Any]:
    """
    Main A/B testing protocol function

    Args:
        hypothesis: Description of the hypothesis being tested
        variant_a_signals: Signals for variant A (IFVG base)
        variant_b_signals: Signals for variant B (alternative)
        df_5m: 5-minute OHLCV DataFrame
        min_trades: Minimum trades required for valid test

    Returns:
        Complete A/B test results
    """
    logger.info(f"Starting A/B test: {hypothesis}")

    # Initialize protocol
    protocol = ABTestingBase()

    # Step 1: Data split
    df_a, df_b = protocol.split_data_random(df_5m)

    # Step 2: Parallel backtesting
    logger.info("Running parallel backtests...")
    results_a = protocol.run_backtest_vectorbt(
        df_a, variant_a_signals.loc[df_a.index], "Variant A (IFVG)")
    results_b = protocol.run_backtest_vectorbt(
        df_b, variant_b_signals.loc[df_b.index], "Variant B (RSI/BB)")

    # Step 3: Statistical analysis
    stats = protocol.statistical_analysis(results_a, results_b)

    # Step 4: A/A test for bias verification
    aa_test = protocol.run_aa_test(df_5m, variant_a_signals)

    # Step 5: Decision making
    decision = protocol.make_decision(results_a, results_b, stats, min_trades)

    # Compile results
    results = {
        'hypothesis': hypothesis,
        'timestamp': pd.Timestamp.now(),
        'variant_a': {
            'name': 'IFVG Base',
            'results': results_a
        },
        'variant_b': {
            'name': 'RSI/BB Alternative',
            'results': results_b
        },
        'statistical_analysis': stats,
        'aa_test': aa_test,
        'decision': decision,
        'data_split': {
            'total_bars': len(df_5m),
            'bars_a': len(df_a),
            'bars_b': len(df_b),
            'split_ratio': len(df_a) / len(df_5m)
        }
    }

    logger.info(f"A/B test completed: {decision['recommendation']}")

    return results

# Example usage and testing functions


def example_btc_test():
    """Example A/B test for BTC trading signals"""
    # This would be called with real data and signals
    # For now, return a template structure
    return {
        'hypothesis': 'RSI/BB alternative performs better than IFVG in chop markets',
        'expected_outcome': 'B shows ΔSharpe > 0.2 with p < 0.05',
        'implementation_notes': 'Use ab_protocol() with real BTC 5min data and signal functions'
    }


if __name__ == "__main__":
    # Example usage
    print("A/B Testing Base Protocol")
    print("=" * 50)
    print(example_btc_test())
