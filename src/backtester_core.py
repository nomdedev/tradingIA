"""
Advanced Backtester Core for Multi-Timeframe Trading Strategies.

This module provides a comprehensive backtesting framework with:
- Walk-forward optimization
- Monte Carlo simulation
- Risk management and metrics calculation
- VectorBT integration for portfolio simulation
"""

import pandas as pd
import numpy as np
import logging
import traceback
import threading
from skopt import gp_minimize
from skopt.space import Real, Integer
import vectorbt as vbt
from typing import Dict, List, Optional, Union


class BacktesterCore:
    """
    Advanced backtesting engine for trading strategies.

    Provides comprehensive backtesting capabilities including:
    - Simple backtesting with metrics calculation
    - Walk-forward optimization
    - Monte Carlo simulation for robustness testing
    - Risk management and realistic cost modeling
    - VectorBT integration for portfolio simulation
    """

    def __init__(self, initial_capital=10000, commission=0.001, slippage_pct=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.logger = logging.getLogger(__name__)
        self._cancel_flag = threading.Event()
        self._current_thread = None

    def cancel_backtest(self):
        """Cancel ongoing backtest operation"""
        self._cancel_flag.set()
        self.logger.info("Backtest cancellation requested")

    def _check_cancellation(self):
        """Check if cancellation has been requested"""
        if self._cancel_flag.is_set():
            raise InterruptedError("Backtest cancelled by user")

    def validate_data_sufficiency(self, df_multi_tf: Dict[str, pd.DataFrame], min_bars: int = 50):
        """Validate that datasets have sufficient data for backtesting"""
        for tf, df in df_multi_tf.items():
            if df is None or df.empty:
                raise ValueError(f"Empty dataset for timeframe {tf}")

            if len(df) < min_bars:
                raise ValueError(
                    f"Insufficient data for timeframe {tf}: {len(df)} bars < {min_bars} minimum")

        return True

    def cap_extreme_metrics(self, metrics: Dict) -> Dict:
        """Cap extreme metric values to prevent unrealistic results"""
        capped_metrics = metrics.copy()

        # Cap Sharpe ratio to [-10, 10]
        if 'sharpe' in capped_metrics:
            original_sharpe = capped_metrics['sharpe']
            capped_metrics['sharpe'] = max(-10, min(10, capped_metrics['sharpe']))
            if original_sharpe != capped_metrics['sharpe']:
                self.logger.warning(f"Sharpe ratio capped from {original_sharpe} to "
                                    f"{capped_metrics['sharpe']}")

        # Cap Sortino ratio to [-10, 10]
        if 'sortino' in capped_metrics:
            original_sortino = capped_metrics['sortino']
            capped_metrics['sortino'] = max(-10, min(10, capped_metrics['sortino']))
            if original_sortino != capped_metrics['sortino']:
                self.logger.warning(f"Sortino ratio capped from {original_sortino} to "
                                    f"{capped_metrics['sortino']}")

        # Cap drawdown to [0, 1] (0-100%)
        if 'max_dd' in capped_metrics:
            capped_metrics['max_dd'] = max(0, min(1, capped_metrics['max_dd']))

        # Cap profit factor to [0, 100]
        if 'profit_factor' in capped_metrics:
            if capped_metrics['profit_factor'] == float('inf'):
                capped_metrics['profit_factor'] = 100
            else:
                capped_metrics['profit_factor'] = min(100, capped_metrics['profit_factor'])

        return capped_metrics

    def run_simple_backtest(self,
                            df_multi_tf: Dict[str,
                                              pd.DataFrame],
                            strategy_class,
                            strategy_params: Dict) -> Dict:
        try:
            # Reset cancellation flag
            self._cancel_flag.clear()

            # Validate data sufficiency
            self.validate_data_sufficiency(df_multi_tf)

            # Check for cancellation
            self._check_cancellation()

            # Extract 5min data for backtesting
            df_5m = df_multi_tf['5min'].copy()

            # Initialize strategy
            strategy = strategy_class(**strategy_params)

            # Check for cancellation
            self._check_cancellation()

            # Generate signals
            signals = strategy.generate_signals(df_multi_tf)

            # Check for cancellation
            self._check_cancellation()

            # Run backtest
            portfolio = vbt.Portfolio.from_signals(
                close=df_5m['Close'],
                entries=signals['entries'],
                exits=signals['exits'],
                price=df_5m['Close'],
                init_cash=self.initial_capital,
                fees=self.commission,
                slippage=self.slippage_pct
            )

            # Calculate metrics
            metrics = self.calculate_metrics(portfolio.returns(), portfolio.trades.records)

            # Cap extreme metrics
            metrics = self.cap_extreme_metrics(metrics)

            # Format trades
            trades = self._format_trades(portfolio.trades.records, df_5m.index)

            # Calculate realistic costs
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df = self.calculate_realistic_costs(trades_df)

            return {
                'metrics': metrics,
                'trades': trades,
                'equity_curve': portfolio.value().tolist(),
                'signals': signals['signals'].to_dict('records') if hasattr(
                    signals,
                    'to_dict') else []}

        except InterruptedError:
            self.logger.info("Backtest cancelled")
            return {'error': 'Backtest cancelled by user'}
        except Exception as e:
            error_msg = f"Error in simple backtest: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def run_backtest(self,
                     strategy_class,
                     df_multi_tf: Union[Dict[str,
                                             pd.DataFrame],
                                        pd.DataFrame],
                     strategy_params: Optional[Dict] = None) -> Dict:
        """Alias for run_simple_backtest with different parameter order for compatibility"""
        if strategy_params is None:
            strategy_params = {}

        # Convert DataFrame to dict format if needed
        if isinstance(df_multi_tf, pd.DataFrame):
            df_multi_tf = {'5min': df_multi_tf}

        return self.run_simple_backtest(df_multi_tf, strategy_class, strategy_params)

    def run_walk_forward(self,
                         df_multi_tf: Dict[str,
                                           pd.DataFrame],
                         strategy_class,
                         strategy_params: Dict,
                         n_periods: int = 8,
                         opt_method: str = 'bayes') -> Dict:
        try:
            df_5m = df_multi_tf['5min'].copy()
            total_bars = len(df_5m)
            period_size = total_bars // n_periods

            periods_results = []
            all_train_metrics = []
            all_test_metrics = []
            best_params = strategy_params  # Initialize with default params

            for i in range(n_periods):
                # Check for cancellation
                self._check_cancellation()

                train_start = i * period_size
                train_end = (i + 1) * period_size
                test_start = train_end
                test_end = min((i + 2) * period_size, total_bars)

                if test_end - test_start < 100:  # Minimum test size
                    break

                # Split data
                train_data = {tf: df.iloc[train_start:train_end] for tf, df in df_multi_tf.items()}
                test_data = {tf: df.iloc[test_start:test_end] for tf, df in df_multi_tf.items()}

                # Optimize on training data
                if opt_method == 'bayes':
                    best_params = self._bayesian_optimize(strategy_class, train_data,
                                                         strategy_params)
                else:
                    best_params = strategy_params  # Use provided params

                # Test on out-of-sample data
                train_result = self.run_simple_backtest(train_data, strategy_class,
                                                       best_params)
                test_result = self.run_simple_backtest(test_data, strategy_class,
                                                      best_params)

                if 'error' not in train_result and 'error' not in test_result:
                    train_sharpe = train_result['metrics']['sharpe']
                    test_sharpe = test_result['metrics']['sharpe']
                    if train_sharpe != 0:
                        degradation_pct = ((test_sharpe - train_sharpe) / abs(train_sharpe)) * 100
                    else:
                        degradation_pct = 0

                    period_result = {
                        'period': i + 1,
                        'train_metrics': train_result['metrics'],
                        'test_metrics': test_result['metrics'],
                        'degradation_pct': degradation_pct
                    }
                    periods_results.append(period_result)

                    all_train_metrics.append(train_result['metrics']['sharpe'])
                    all_test_metrics.append(test_result['metrics']['sharpe'])

            avg_degradation = np.mean([p['degradation_pct']
                                      for p in periods_results]) if periods_results else 0

            return {
                'period_results': periods_results,
                'avg_degradation': avg_degradation,
                'best_params': best_params
            }

        except InterruptedError:
            self.logger.info("Walk-forward analysis cancelled")
            return {'error': 'Walk-forward cancelled by user'}
        except Exception as e:
            error_msg = f"Error in walk-forward analysis: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def run_monte_carlo(self,
                        df_multi_tf: Dict[str,
                                          pd.DataFrame],
                        strategy_class,
                        strategy_params: Dict,
                        n_simulations: int = 500,
                        noise_pct: float = 10,
                        seed: int | None = None) -> Dict:
        try:
            # Set seed for reproducibility
            if seed is not None:
                np.random.seed(seed)
                self.logger.info(f"Monte Carlo using seed: {seed}")

            sharpe_results = []
            win_rate_results = []

            for i in range(n_simulations):
                # Check for cancellation
                self._check_cancellation()

                # Add noise to data
                noisy_data = {}
                for tf, df in df_multi_tf.items():
                    noise = np.random.normal(0, noise_pct / 100, len(df))
                    noisy_df = df.copy()
                    noisy_df['Close'] = df['Close'] * (1 + noise)
                    noisy_df['High'] = df['High'] * (1 + noise * 0.5)
                    noisy_df['Low'] = df['Low'] * (1 + noise * 0.5)
                    noisy_data[tf] = noisy_df

                # Run backtest
                result = self.run_simple_backtest(noisy_data, strategy_class, strategy_params)

                if 'error' not in result:
                    sharpe_results.append(result['metrics']['sharpe'])
                    win_rate_results.append(result['metrics']['win_rate'])

            if sharpe_results:
                sharpe_mean = np.mean(sharpe_results)
                sharpe_std = np.std(sharpe_results)
                robust = sharpe_std < 0.2  # Robust if std < 0.2

                # Create simulations list with individual results
                simulations = []
                for i, (sharpe, win_rate) in enumerate(zip(sharpe_results, win_rate_results)):
                    simulations.append({
                        'simulation_id': i,
                        'sharpe_ratio': sharpe,
                        'win_rate': win_rate
                    })

                return {
                    'simulations': simulations,
                    'summary_stats': {
                        'sharpe_mean': sharpe_mean,
                        'sharpe_std': sharpe_std,
                        'win_rate_mean': np.mean(win_rate_results),
                        'win_rate_std': np.std(win_rate_results),
                        'robust': robust
                    },
                    'sharpe_distribution': sharpe_results
                }
            else:
                return {'error': 'No valid Monte Carlo results'}

        except InterruptedError:
            self.logger.info("Monte Carlo cancelled")
            return {'error': 'Monte Carlo cancelled by user'}
        except Exception as e:
            error_msg = f"Error in Monte Carlo simulation: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def calculate_realistic_costs(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Commission: 0.1% round-trip
            trades_df['commission_cost'] = trades_df['pnl_pct'].abs() * 0.001

            # Slippage: base + vol_spike adjustment
            base_slippage = self.slippage_pct
            vol_spike_mult = 1.5  # Could be calculated from volatility
            trades_df['slippage_cost'] = trades_df['pnl_pct'].abs() * (base_slippage *
                                                                       vol_spike_mult)

            # Funding rate (if perpetual futures) - simplified
            funding_rate = 0.0001  # 0.01% per 8h, simplified to per trade
            trades_df['funding_cost'] = trades_df['pnl_pct'].abs() * funding_rate

            # Total cost
            trades_df['total_cost'] = (trades_df['commission_cost'] + trades_df['slippage_cost'] +
                                       trades_df['funding_cost'])

            return trades_df

        except Exception as e:
            self.logger.error(f"Error calculating realistic costs: {e}")
            return trades_df

    def calculate_metrics(self, returns: pd.Series, trades_records: pd.DataFrame) -> Dict:
        try:
            # Basic returns metrics
            cumulative_returns = (1 + returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1

            # Sharpe Ratio (annualized, assuming daily returns)
            risk_free_rate = 0.04 / 252  # 4% annual risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                     if excess_returns.std() > 0 else 0)

            # Calmar Ratio
            max_dd = self._calculate_max_drawdown(cumulative_returns)
            calmar = total_return / max_dd if max_dd > 0 else 0

            # Win Rate
            if not trades_records.empty:
                win_rate = (trades_records['pnl'] > 0).mean()
                num_trades = len(trades_records)
            else:
                win_rate = 0
                num_trades = 0

            # Information Ratio (vs buy-and-hold)
            bh_returns = returns  # Simplified, should be market returns
            ir = excess_returns.mean() / (returns - bh_returns).std() * \
                np.sqrt(252) if (returns - bh_returns).std() > 0 else 0

            # Ulcer Index
            ulcer = self._calculate_ulcer_index(cumulative_returns)

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            sortino = excess_returns.mean() / downside_returns.std() * \
                np.sqrt(252) if len(downside_returns) > 0 else 0

            # Profit Factor
            gross_profit = trades_records[trades_records['pnl'] >
                                          0]['pnl'].sum() if not trades_records.empty else 0
            gross_loss = abs(trades_records[trades_records['pnl'] < 0]
                             ['pnl'].sum()) if not trades_records.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            return {
                'sharpe': round(sharpe, 3),
                'calmar': round(calmar, 3),
                'win_rate': round(win_rate, 3),
                'max_dd': round(max_dd, 3),
                'num_trades': num_trades,
                'ir': round(ir, 3),
                'ulcer': round(ulcer, 3),
                'sortino': round(sortino, 3),
                'profit_factor': round(profit_factor, 3),
                'total_return': round(total_return, 3)
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {'error': str(e)}

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())

    def _calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return np.sqrt((drawdown ** 2).mean())

    def _bayesian_optimize(self, strategy_class, train_data: Dict, param_space: Dict) -> Dict:
        try:
            # Define parameter space for optimization
            space = []
            param_names = []

            for param_name, param_config in param_space.items():
                if param_config.get('type') == 'int':
                    space.append(Integer(param_config['min'], param_config['max'], name=param_name))
                else:
                    space.append(Real(param_config['min'], param_config['max'], name=param_name))
                param_names.append(param_name)

            def objective(params):
                param_dict = dict(zip(param_names, params))
                result = self.run_simple_backtest(train_data, strategy_class, param_dict)
                if 'error' in result:
                    return 0  # Return neutral score for errors
                return -result['metrics']['sharpe']  # Minimize negative Sharpe

            # Run optimization
            res = gp_minimize(objective, space, n_calls=50, random_state=42)

            # Return best parameters
            best_params = dict(zip(param_names, res.x))
            return best_params

        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {e}")
            return param_space  # Return original params on error

    def _format_trades(self, trades_records: pd.DataFrame,
                       df_index: pd.DatetimeIndex) -> List[Dict]:
        if trades_records.empty:
            return []

        trades = []
        for _, trade in trades_records.iterrows():
            # Map entry_idx to timestamp
            entry_idx = trade['entry_idx']
            entry_idx_int = int(entry_idx)  # Convert float to int
            entry_timestamp = df_index[entry_idx_int] if entry_idx_int < len(df_index) else None

            trades.append({
                'timestamp': entry_timestamp,
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl_pct': trade['return'],  # VectorBT return is already in decimal format
                'score': 4,  # Placeholder, should be calculated by strategy
                # 0=long, 1=short in VectorBT
                'entry_type': 'long' if trade['direction'] == 0 else 'short',
                'reason_exit': 'target'  # Placeholder
            })

        return trades
