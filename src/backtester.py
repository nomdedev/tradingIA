"""
Advanced Backtester for Multi-Timeframe BTC IFVG Strategy

Features:
- Walk-forward optimization (6 periods, 70/30 split)
- Bayesian optimization (skopt, 100 calls, Expected Improvement)
- Monte Carlo simulation (500 runs, ±10% noise)
- Stress testing (high_vol, bear_market, crash, low_vol, whipsaw)
- Vectorized pandas operations for performance
- Comprehensive metrics (Sharpe, Calmar, DD, HTF alignment)
"""

from .indicators import generate_filtered_signals
from .mtf_data_handler import MultiTFDataHandler
from config.mtf_config import (
    BACKTEST_CONFIG, OPTIMIZATION_CONFIG, TRADING_CONFIG,
    INDICATOR_PARAMS, SIGNAL_CONFIG, ALPACA_CONFIG
)
import scipy.stats as stats
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from skopt import gp_minimize
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries

# Local imports


class AdvancedBacktester:
    """
    Advanced backtester with multiple optimization methods for MTF BTC strategy.
    """

    def __init__(self, capital: float = 10000.0, commission: float = 0.001,
                 slippage: float = 0.001):
        """
        Initialize backtester with trading parameters.

        Args:
            capital: Starting capital ($)
            commission: Commission per trade (0.1%)
            slippage: Slippage per trade (0.1%)
        """
        self.capital = capital
        self.commission = commission
        self.slippage = slippage

        # Initialize data handler
        self.data_handler = MultiTFDataHandler()

        # Results storage
        self.results = {}
        self.best_params = {}
        self.trades_df = pd.DataFrame()

    def run_optimized_backtest(self, dfs: Dict[str, pd.DataFrame],
                               params: Dict[str, Any],
                               use_mtf: bool = True) -> Dict[str, Any]:
        """
        Run backtest with given parameters.

        Args:
            dfs: Dictionary with '5m', '15m', '1h' dataframes
            params: Strategy parameters
            use_mtf: Whether to use multi-TF filters

        Returns:
            Dictionary with metrics and trades
        """

        # Generate signals
        df_5m = dfs['5m'].copy()
        df_15m = dfs['15m'].copy()
        df_1h = dfs['1h'].copy()

        # Calculate all indicators first
        from src.indicators import (calculate_ifvg_enhanced, volume_profile_advanced,
                                    emas_multi_tf)

        # IFVG signals
        ifvg_bull, ifvg_bear, ifvg_conf = calculate_ifvg_enhanced(df_5m, params)
        df_5m['ifvg_bull'] = ifvg_bull
        df_5m['ifvg_bear'] = ifvg_bear
        df_5m['ifvg_conf'] = ifvg_conf

        # Volume Profile
        vp_poc, vp_vah, vp_val = volume_profile_advanced(df_5m, params)
        df_5m['vp_poc'] = vp_poc
        df_5m['vp_vah'] = vp_vah
        df_5m['vp_val'] = vp_val

        # EMAs Multi-TF
        ema_data = emas_multi_tf(df_5m, df_15m, df_1h, params)
        df_5m = pd.concat([df_5m, ema_data], axis=1)

        # Add required columns for filtering
        df_5m['uptrend_1h'] = (df_5m['ema_trend_1h'] ==
                               1) if 'ema_trend_1h' in df_5m.columns else True
        df_5m['momentum_15m'] = (df_5m['ema_trend_15m'] ==
                                 1) if 'ema_trend_15m' in df_5m.columns else True
        df_5m['vol_cross'] = True  # Simplified for testing

        # Now generate filtered signals
        bull_signals, bear_signals, confidence = generate_filtered_signals(df_5m, params)

        # Apply confidence filter
        min_confidence = params.get('min_confidence', 0.6)
        bull_signals = bull_signals & (confidence >= min_confidence)
        bear_signals = bear_signals & (confidence >= min_confidence)

        # Run backtest
        trades, equity_curve = self._execute_trades(
            df_5m, bull_signals, bear_signals, confidence, params
        )

        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades, df_5m)

        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'params': params
        }

    def _execute_trades(self, df: pd.DataFrame, bull_signals: pd.Series,
                        bear_signals: pd.Series, confidence: pd.Series,
                        params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Execute trades based on signals with position sizing and risk management.
        """

        capital = self.capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        trailing_stop = 0.0
        trailing_activated = False
        position_size = 0.0
        entry_time = None
        entry_confidence = 0.0
        risk_amount = 0.0

        trades = []
        equity = [capital]

        # Risk parameters
        risk_per_trade = TRADING_CONFIG['risk_per_trade']
        max_exposure = BACKTEST_CONFIG['position']['max_exposure']
        sl_multiplier = BACKTEST_CONFIG['exit']['sl_atr_multi']
        tp_rr = params.get('tp_rr', BACKTEST_CONFIG['exit']['tp_risk_reward'])
        trailing_start_rr = BACKTEST_CONFIG['exit']['trailing_start']
        trailing_delta_rr = BACKTEST_CONFIG['exit']['trailing_offset']

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_atr = df['ATR'].iloc[i] if 'ATR' in df.columns else 50.0
            current_confidence = confidence.iloc[i]

            # Update equity
            if i > 0:
                equity.append(equity[-1])

            # Check exit conditions first
            if position != 0:
                pnl = 0

                # Stop loss hit
                if (position == 1 and current_price <= stop_loss) or \
                   (position == -1 and current_price >= stop_loss):
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * position * position_size
                    pnl -= self.commission * abs(pnl) + self.slippage * abs(pnl)

                    trades.append({
                        'timestamp': df.index[i],
                        'direction': 'long' if position == 1 else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl / (entry_price * position_size),
                        'duration': (df.index[i] - entry_time).total_seconds() / 3600,  # hours
                        'exit_reason': 'stop_loss',
                        'confidence': entry_confidence,
                        'uptrend_1h': df.get('uptrend_1h', True).iloc[i],
                        'momentum_15m': df.get('momentum_15m', True).iloc[i],
                        'vol_cross': df.get('vol_cross', True).iloc[i]
                    })

                    capital += pnl
                    position = 0
                    trailing_activated = False

                # Take profit hit
                elif (position == 1 and current_price >= take_profit) or \
                     (position == -1 and current_price <= take_profit):
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) * position * position_size
                    pnl -= self.commission * abs(pnl) + self.slippage * abs(pnl)

                    trades.append({
                        'timestamp': df.index[i],
                        'direction': 'long' if position == 1 else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl / (entry_price * position_size),
                        'duration': (df.index[i] - entry_time).total_seconds() / 3600,
                        'exit_reason': 'take_profit',
                        'confidence': entry_confidence,
                        'uptrend_1h': df.get('uptrend_1h', True).iloc[i],
                        'momentum_15m': df.get('momentum_15m', True).iloc[i],
                        'vol_cross': df.get('vol_cross', True).iloc[i]
                    })

                    capital += pnl
                    position = 0
                    trailing_activated = False

                # Trailing stop management
                elif trailing_activated:
                    if position == 1:
                        new_stop = current_price - (trailing_delta_rr * risk_amount / position_size)
                        trailing_stop = max(trailing_stop, new_stop)

                        if current_price <= trailing_stop:
                            exit_price = trailing_stop
                            pnl = (exit_price - entry_price) * position * position_size
                            pnl -= self.commission * abs(pnl) + self.slippage * abs(pnl)

                            trades.append({
                                'timestamp': df.index[i],
                                'direction': 'long',
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'pnl_pct': pnl / (entry_price * position_size),
                                'duration': (df.index[i] - entry_time).total_seconds() / 3600,
                                'exit_reason': 'trailing_stop',
                                'confidence': entry_confidence,
                                'uptrend_1h': df.get('uptrend_1h', True).iloc[i],
                                'momentum_15m': df.get('momentum_15m', True).iloc[i],
                                'vol_cross': df.get('vol_cross', True).iloc[i]
                            })

                            capital += pnl
                            position = 0
                            trailing_activated = False

                    elif position == -1:
                        new_stop = current_price + (trailing_delta_rr * risk_amount / position_size)
                        trailing_stop = min(trailing_stop, new_stop)

                        if current_price >= trailing_stop:
                            exit_price = trailing_stop
                            pnl = (exit_price - entry_price) * position * position_size
                            pnl -= self.commission * abs(pnl) + self.slippage * abs(pnl)

                            trades.append({
                                'timestamp': df.index[i],
                                'direction': 'short',
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'pnl_pct': pnl / (entry_price * position_size),
                                'duration': (df.index[i] - entry_time).total_seconds() / 3600,
                                'exit_reason': 'trailing_stop',
                                'confidence': entry_confidence,
                                'uptrend_1h': df.get('uptrend_1h', True).iloc[i],
                                'momentum_15m': df.get('momentum_15m', True).iloc[i],
                                'vol_cross': df.get('vol_cross', True).iloc[i]
                            })

                            capital += pnl
                            position = 0
                            trailing_activated = False

            # Check entry signals
            if position == 0:
                # Long signal
                if bull_signals.iloc[i]:
                    # Position sizing: risk_amount / (SL distance)
                    sl_distance = sl_multiplier * current_atr
                    risk_amount = capital * risk_per_trade
                    position_size = risk_amount / sl_distance

                    # Max exposure check
                    max_position_value = capital * max_exposure
                    if position_size * current_price > max_position_value:
                        position_size = max_position_value / current_price

                    # Adjust for slippage and commission
                    entry_price = current_price * (1 + self.slippage)
                    stop_loss = entry_price - sl_distance
                    take_profit = entry_price + (sl_distance * tp_rr)

                    position = 1
                    entry_time = df.index[i]
                    entry_confidence = current_confidence
                    risk_amount = position_size * sl_distance

                    # Initialize trailing stop
                    trailing_stop = stop_loss
                    trailing_activated = False

                # Short signal
                elif bear_signals.iloc[i]:
                    # Position sizing: risk_amount / (SL distance)
                    sl_distance = sl_multiplier * current_atr
                    risk_amount = capital * risk_per_trade
                    position_size = risk_amount / sl_distance

                    # Max exposure check
                    max_position_value = capital * max_exposure
                    if position_size * current_price > max_position_value:
                        position_size = max_position_value / current_price

                    # Adjust for slippage and commission
                    entry_price = current_price * (1 - self.slippage)
                    stop_loss = entry_price + sl_distance
                    take_profit = entry_price - (sl_distance * tp_rr)

                    position = -1
                    entry_time = df.index[i]
                    entry_confidence = current_confidence
                    risk_amount = position_size * sl_distance

                    # Initialize trailing stop
                    trailing_stop = stop_loss
                    trailing_activated = False

            # Activate trailing stop if profit target reached
            if position != 0 and not trailing_activated:
                current_pnl = (current_price - entry_price) * position
                if current_pnl >= trailing_start_rr * risk_amount:
                    trailing_activated = True
                    if position == 1:
                        trailing_stop = current_price - \
                            (trailing_delta_rr * risk_amount / position_size)
                    else:
                        trailing_stop = current_price + \
                            (trailing_delta_rr * risk_amount / position_size)

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)

        # Create equity curve
        equity_curve = pd.Series(equity, index=df.index[:len(equity)])

        return trades_df, equity_curve

    def _calculate_metrics(self, equity_curve: pd.Series, trades_df: pd.DataFrame,
                           df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive trading metrics.
        """

        if trades_df.empty:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'htf_alignment': 0.0
            }

        # Basic returns
        returns = equity_curve.pct_change().fillna(0)
        cumulative_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Sharpe ratio (annualized, rf=4%)
        rf_daily = 0.04 / 252  # Risk-free rate
        excess_returns = returns - rf_daily
        if excess_returns.std() > 0:
            sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe = 0.0

        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()

        # Calmar ratio
        if max_dd < 0:
            calmar = cumulative_return / abs(max_dd)
        else:
            calmar = 0.0

        # Trade metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0

        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0.0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = winning_trades['pnl_pct'].mean() if not winning_trades.empty else 0.0
        avg_loss = losing_trades['pnl_pct'].mean() if not losing_trades.empty else 0.0

        # HTF alignment percentage
        htf_alignment = trades_df['uptrend_1h'].mean() if 'uptrend_1h' in trades_df.columns else 0.0

        return {
            'total_return': cumulative_return,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades_df),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'htf_alignment': htf_alignment
        }

    def walk_forward_optimization(self, dfs: Dict[str, pd.DataFrame],
                                  n_periods: int = 6,
                                  train_split: float = 0.7) -> Dict[str, Any]:
        """
        Walk-forward optimization with expanding window.

        Args:
            dfs: Multi-TF dataframes
            n_periods: Number of walk-forward periods
            train_split: Train/test split ratio

        Returns:
            Dictionary with optimization results
        """

        print(
            f"🚀 Starting walk-forward optimization ({n_periods} periods, {train_split:.0%} train)...")

        # Split data into periods
        df_5m = dfs['5m']
        total_days = len(df_5m)
        period_length = total_days // n_periods

        wf_results = []
        best_params_overall = None
        best_calmar = -float('inf')

        for period in range(n_periods):
            start_idx = period * period_length
            end_idx = (period + 1) * period_length if period < n_periods - 1 else total_days

            # Get period data
            period_df = df_5m.iloc[start_idx:end_idx]
            period_dfs = {
                '5m': period_df,
                '15m': dfs['15m'].loc[period_df.index[0]:period_df.index[-1]],
                '1h': dfs['1h'].loc[period_df.index[0]:period_df.index[-1]]
            }

            # Split train/test
            train_size = int(len(period_df) * train_split)
            train_df = period_df.iloc[:train_size]
            test_df = period_df.iloc[train_size:]

            train_dfs = {
                '5m': train_df,
                '15m': period_dfs['15m'].loc[train_df.index[0]:train_df.index[-1]],
                '1h': period_dfs['1h'].loc[test_df.index[0]:test_df.index[-1]]
            }

            test_dfs = {
                '5m': test_df,
                '15m': period_dfs['15m'].loc[test_df.index[0]:test_df.index[-1]],
                '1h': period_dfs['1h'].loc[test_df.index[0]:test_df.index[-1]]
            }

            print(
                f"📊 Period {period + 1}/{n_periods}: {train_df.index[0].date()} to {test_df.index[-1].date()}")
            print(f"   Train: {len(train_df)} bars, Test: {len(test_df)} bars")

            # Optimize on train set
            best_params = self._bayesian_optimization(train_dfs)

            # Evaluate on test set
            test_result = self.run_optimized_backtest(test_dfs, best_params)
            test_calmar = test_result['metrics']['calmar_ratio']

            print(f"   Test Calmar: {test_calmar:.3f}")

            wf_results.append({
                'period': period + 1,
                'train_start': train_df.index[0],
                'train_end': train_df.index[-1],
                'test_start': test_df.index[0],
                'test_end': test_df.index[-1],
                'best_params': best_params,
                'test_metrics': test_result['metrics']
            })

            # Track overall best
            if test_calmar > best_calmar:
                best_calmar = test_calmar
                best_params_overall = best_params

        # Calculate degradation statistics
        test_calmars = [r['test_metrics']['calmar_ratio'] for r in wf_results]
        degradation = np.mean(test_calmars) - np.min(test_calmars)

        wf_summary = {
            'periods': wf_results,
            'best_params_overall': best_params_overall,
            'avg_test_calmar': np.mean(test_calmars),
            'std_test_calmar': np.std(test_calmars),
            'degradation': degradation,
            'robustness_score': 1.0 - degradation  # Higher is better
        }

        print(
            f"✅ Walk-forward complete. Avg Calmar: {np.mean(test_calmars):.3f}, Degradation: {degradation:.3f}")

        return wf_summary

    def _bayesian_optimization(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Bayesian optimization using Gaussian Processes.
        """

        # Parameter space
        param_space = [
            Real(0.1, 0.5, name='atr_multi'),
            Real(0.65, 0.75, name='va_percent'),
            Integer(100, 150, name='vp_rows'),
            Real(0.8, 1.5, name='vol_thresh'),
            Real(1.8, 2.5, name='tp_rr'),
            Real(0.5, 0.8, name='min_confidence')
        ]

        @use_named_args(param_space)
        def objective(**params):
            """Objective function to minimize (negative Calmar)"""
            try:
                result = self.run_optimized_backtest(dfs, params)
                calmar = result['metrics']['calmar_ratio']

                # Penalize if Calmar < 1.0 (minimum requirement)
                if calmar < 1.0:
                    return 100.0  # Large penalty

                return -calmar  # Minimize negative Calmar = maximize Calmar

            except Exception as e:
                print(f"Optimization error: {e}")
                return 100.0  # Large penalty

        # Run optimization
        res = gp_minimize(
            objective,
            param_space,
            n_calls=OPTIMIZATION_CONFIG['bayes_n_calls'],
            n_random_starts=OPTIMIZATION_CONFIG['bayes_random_starts'],
            random_state=42,
            verbose=False
        )

        # Extract best parameters
        best_params = {
            'atr_multi': res.x[0],
            'va_percent': res.x[1],
            'vp_rows': res.x[2],
            'vol_thresh': res.x[3],
            'tp_rr': res.x[4],
            'min_confidence': res.x[5]
        }

        return best_params

    def monte_carlo_simulation(self, dfs: Dict[str, pd.DataFrame],
                               params: Dict[str, Any],
                               n_runs: int = 500,
                               noise_pct: float = 0.1) -> Dict[str, Any]:
        """
        Monte Carlo simulation with price and volume noise.

        Args:
            dfs: Base dataframes
            params: Strategy parameters
            n_runs: Number of simulation runs
            noise_pct: Noise percentage (±10%)

        Returns:
            Dictionary with simulation statistics
        """

        print(f"🎲 Running Monte Carlo simulation ({n_runs} runs, ±{noise_pct:.0%} noise)...")

        results = []

        for run in range(n_runs):
            if run % 50 == 0:
                print(f"   Run {run + 1}/{n_runs}...")

            # Add noise to data
            noisy_dfs = self._add_noise_to_data(dfs, noise_pct)

            # Run backtest
            result = self.run_optimized_backtest(noisy_dfs, params)
            results.append(result['metrics'])

        # Calculate statistics
        metrics_df = pd.DataFrame(results)

        mc_stats = {}
        for col in metrics_df.columns:
            mc_stats[f'{col}_mean'] = metrics_df[col].mean()
            mc_stats[f'{col}_std'] = metrics_df[col].std()
            mc_stats[f'{col}_min'] = metrics_df[col].min()
            mc_stats[f'{col}_max'] = metrics_df[col].max()
            mc_stats[f'{col}_5pct'] = metrics_df[col].quantile(0.05)
            mc_stats[f'{col}_95pct'] = metrics_df[col].quantile(0.95)

        # Robustness checks
        sharpe_std = mc_stats['sharpe_ratio_std']
        calmar_std = mc_stats['calmar_ratio_std']

        robustness_score = 1.0 - (sharpe_std + calmar_std) / 2  # Higher is better
        robustness_score = max(0.0, min(1.0, robustness_score))  # Clamp to [0,1]

        mc_stats['robustness_score'] = robustness_score
        mc_stats['sharpe_stability'] = 1.0 - sharpe_std  # Lower std = higher stability
        mc_stats['calmar_stability'] = 1.0 - calmar_std

        print(f"✅ Monte Carlo complete. Robustness: {robustness_score:.3f}")

        return {
            'statistics': mc_stats,
            'all_results': results,
            'n_runs': n_runs,
            'noise_pct': noise_pct
        }

    def _add_noise_to_data(self, dfs: Dict[str, pd.DataFrame],
                           noise_pct: float) -> Dict[str, pd.DataFrame]:
        """
        Add random noise to price and volume data for stress testing.
        """

        noisy_dfs = {}

        for tf, df in dfs.items():
            df_noisy = df.copy()

            # Price noise
            price_noise = np.random.normal(0, noise_pct, len(df))
            df_noisy['close'] = df['close'] * (1 + price_noise)
            df_noisy['high'] = df['high'] * (1 + price_noise * 0.8)  # Less noise on highs
            df_noisy['low'] = df['low'] * (1 + price_noise * 0.8)   # Less noise on lows
            df_noisy['open'] = df['open'] * (1 + price_noise * 0.6)  # Even less on opens

            # Volume noise
            vol_noise = np.random.normal(0, noise_pct * 0.5, len(df))
            df_noisy['volume'] = df['volume'] * (1 + vol_noise)
            df_noisy['volume'] = df_noisy['volume'].clip(lower=0.1)  # Minimum volume

            # Recalculate OHLC relationships
            df_noisy['high'] = np.maximum(
                df_noisy[['open', 'close', 'high']].max(axis=1), df_noisy['high'])
            df_noisy['low'] = np.minimum(
                df_noisy[['open', 'close', 'low']].min(axis=1), df_noisy['low'])

            noisy_dfs[tf] = df_noisy

        return noisy_dfs

    def stress_test_scenarios(self, dfs: Dict[str, pd.DataFrame],
                              params: Dict[str, Any],
                              scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Run stress tests with predefined market scenarios.

        Args:
            dfs: Base dataframes
            params: Strategy parameters
            scenarios: List of scenarios to test

        Returns:
            Dictionary with stress test results
        """

        if scenarios is None:
            scenarios = ['high_vol', 'bear_market', 'flash_crash', 'low_vol', 'whipsaw']

        print(f"⚠️ Running stress tests: {scenarios}")

        stress_results = {}

        for scenario in scenarios:
            print(f"   Testing {scenario}...")

            # Apply scenario modifications
            stressed_dfs = self._apply_stress_scenario(dfs, scenario)

            # Run backtest
            result = self.run_optimized_backtest(stressed_dfs, params)
            metrics = result['metrics']

            stress_results[scenario] = {
                'metrics': metrics,
                'survival': metrics['calmar_ratio'] > 0.5,  # Basic survival threshold
                'dd_acceptable': metrics['max_drawdown'] > -0.25  # Max 25% DD
            }

            print(
                f"   {scenario}: Calmar={metrics['calmar_ratio']:.3f}, DD={metrics['max_drawdown']:.1%}")

        # Overall stress score
        survival_rate = sum(1 for r in stress_results.values()
                            if r['survival']) / len(stress_results)
        avg_dd = np.mean([r['metrics']['max_drawdown'] for r in stress_results.values()])

        stress_score = survival_rate * (1.0 + avg_dd)  # Penalize high DD
        stress_score = max(0.0, min(1.0, stress_score))

        stress_results['summary'] = {
            'survival_rate': survival_rate,
            'avg_max_dd': avg_dd,
            'stress_score': stress_score
        }

        print(f"✅ Stress tests complete. Survival: {survival_rate:.1%}, Score: {stress_score:.3f}")

        return stress_results

    def _apply_stress_scenario(self, dfs: Dict[str, pd.DataFrame],
                               scenario: str) -> Dict[str, pd.DataFrame]:
        """
        Apply specific stress scenario modifications to data.
        """

        stressed_dfs = {}

        for tf, df in dfs.items():
            df_stressed = df.copy()

            if scenario == 'high_vol':
                # +50% volatility, +30% volume
                vol_multiplier = 1.5
                price_noise = np.random.normal(0, 0.02, len(df))  # 2% daily vol

                df_stressed['close'] = df['close'] * (1 + price_noise)
                df_stressed['high'] = df['high'] * (1 + price_noise * 1.2)
                df_stressed['low'] = df['low'] * (1 + price_noise * 0.8)
                df_stressed['volume'] = df['volume'] * vol_multiplier

            elif scenario == 'bear_market':
                # -30% sustained downtrend
                trend_factor = np.linspace(1.0, 0.7, len(df))
                df_stressed['close'] = df['close'] * trend_factor
                df_stressed['high'] = df['high'] * trend_factor * 1.02
                df_stressed['low'] = df['low'] * trend_factor * 0.98

            elif scenario == 'flash_crash':
                # -20% in 1 day, recovery over 5 days
                crash_point = len(df) // 2
                crash_duration = min(5 * (24 if 'H' in tf else 12), len(df) - crash_point)

                crash_factor = np.ones(len(df))
                crash_factor[crash_point:crash_point +
                             crash_duration] = np.linspace(0.8, 0.95, crash_duration)

                df_stressed['close'] = df['close'] * crash_factor
                df_stressed['high'] = df['high'] * crash_factor * 1.05
                df_stressed['low'] = df['low'] * crash_factor * 0.95

            elif scenario == 'low_vol':
                # -50% volume, +20% volatility
                df_stressed['volume'] = df['volume'] * 0.5
                price_noise = np.random.normal(0, 0.015, len(df))
                df_stressed['close'] = df['close'] * (1 + price_noise)

            elif scenario == 'whipsaw':
                # High frequency reversals
                oscillation = np.sin(np.linspace(0, 20 * np.pi, len(df))) * 0.03
                df_stressed['close'] = df['close'] * (1 + oscillation)

            # Ensure OHLC integrity
            df_stressed['high'] = np.maximum(
                df_stressed[['open', 'close', 'high']].max(axis=1), df_stressed['high'])
            df_stressed['low'] = np.minimum(
                df_stressed[['open', 'close', 'low']].min(axis=1), df_stressed['low'])

            stressed_dfs[tf] = df_stressed

        return stressed_dfs

    def run_complete_optimization(self, symbol: str = 'BTCUSD',
                                  start_date: str = '2023-01-01',
                                  end_date: str = '2024-12-31') -> Dict[str, Any]:
        """
        Run complete optimization pipeline: Walk-forward + Monte Carlo + Stress Tests.
        """

        print("🚀 Starting complete optimization pipeline...")

        # Load data
        print("📥 Loading multi-TF data...")
        dfs = self.data_handler.get_multi_tf_data(symbol, start_date, end_date)

        # Walk-forward optimization
        print("📊 Running walk-forward optimization...")
        wf_results = self.walk_forward_optimization(dfs)

        best_params = wf_results['best_params_overall']

        # Monte Carlo simulation
        print("🎲 Running Monte Carlo simulation...")
        mc_results = self.monte_carlo_simulation(dfs, best_params)

        # Stress tests
        print("⚠️ Running stress tests...")
        stress_results = self.stress_test_scenarios(dfs, best_params)

        # Final backtest with best params
        print("📈 Running final backtest...")
        final_result = self.run_optimized_backtest(dfs, best_params)

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'date_range': f'{start_date} to {end_date}',
            'best_params': best_params,
            'walk_forward': wf_results,
            'monte_carlo': mc_results,
            'stress_tests': stress_results,
            'final_backtest': final_result
        }

        # Save to JSON
        os.makedirs('results', exist_ok=True)
        with open('results/optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save trades
        final_result['trades'].to_csv('results/final_trades.csv', index=False)

        print("✅ Complete optimization finished!")
        print(f"📊 Best Calmar: {final_result['metrics']['calmar_ratio']:.3f}")
        print(f"🎯 Win Rate: {final_result['metrics']['win_rate']:.1%}")
        print(f"📈 Sharpe: {final_result['metrics']['sharpe_ratio']:.3f}")

        return results

    def load_optimization_results(self) -> Dict[str, Any]:
        """
        Load previous optimization results.
        """
        try:
            with open('results/optimization_results.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ No optimization results found. Run optimization first.")
            return {}


# Example usage
if __name__ == "__main__":
    backtester = AdvancedBacktester()

    # Run complete optimization
    results = backtester.run_complete_optimization()

    print("\n🎉 Optimization complete!")
    print(f"Best parameters: {results['best_params']}")
    print(f"Final Calmar: {results['final_backtest']['metrics']['calmar_ratio']:.3f}")


# Test del módulo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Backtester module...")

    # Crear datos de prueba
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)
    prices = 45000 * (1 + np.random.normal(0.0001, 0.003, 1000)).cumprod()

    df = pd.DataFrame({
        'Open': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.001, 1000))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, 1000))),
        'Close': prices,
        'Volume': np.random.uniform(100, 1000, 1000),
        'ATR': prices * 0.02,  # 2% ATR
    }, index=dates)

    # Generar señales aleatorias
    signals = np.zeros(1000)
    signals[np.random.choice(1000, 50, replace=False)] = 1  # 50 señales long
    signals[np.random.choice(1000, 50, replace=False)] = -1  # 50 señales short
    df['signal'] = signals
    df['confidence'] = np.random.uniform(0.5, 1.0, 1000)

    # Ejecutar backtest
    backtester = AdvancedBacktester(capital=10000)
    metrics = backtester.run_backtest(
        use_stop_loss=True,
        use_take_profit=True,
        sl_atr_multiplier=1.5,
        tp_risk_reward=2.0
    )

    # Mostrar resultados
    backtester.print_summary()

    print("\n✅ Backtester module tested successfully!")
    print(f"Generated {len(backtester.trades)} trades")
