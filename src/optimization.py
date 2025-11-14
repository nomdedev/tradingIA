"""
Parameter Optimization Module
==============================
Grid search, walk-forward analysis y genetic algorithm optimization
para encontrar los mejores parámetros de la estrategia IFVG.
"""

from config.config import TRADING_CONFIG
from src.data_fetcher import DataFetcher
from src.backtester import Backtester
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Optional
import json
import logging
from itertools import product
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Optimización de parámetros con grid search y walk-forward analysis"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize optimizer

        Args:
            data: Historical OHLCV data. If None, will fetch from Alpaca
        """
        self.data = data
        self.results = []
        self.best_params = None
        self.best_sharpe = -np.inf

        logger.info("✅ ParameterOptimizer initialized")

    def _load_data(self, start_date, end_date):
        """Load historical data if not provided"""
        if self.data is not None:
            # Filter existing data
            return self.data[
                (self.data.index >= start_date) &
                (self.data.index <= end_date)
            ].copy()

        # Fetch new data
        fetcher = DataFetcher()
        return fetcher.get_historical_data(
            symbol=TRADING_CONFIG['symbol'],
            timeframe=TRADING_CONFIG['timeframe'],
            start_date=start_date,
            end_date=end_date
        )

    def grid_search(
        self,
        param_grid: Dict[str, List],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        optimize_metric: str = 'sharpe_ratio',
        max_workers: int = 4
    ) -> pd.DataFrame:
        """
        Grid search optimization

        Args:
            param_grid: Dictionary of parameters to test
                Example: {
                    'sl_pct': [0.01, 0.015, 0.02],
                    'tp_pct': [0.03, 0.04, 0.05],
                    'volume_threshold': [1.5, 2.0, 2.5]
                }
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital for backtesting
            optimize_metric: Metric to optimize ('sharpe_ratio', 'profit_factor', 'total_return')
            max_workers: Number of parallel workers

        Returns:
            DataFrame with all tested combinations and results
        """
        logger.info("=" * 80)
        logger.info("🔍 Starting Grid Search Optimization")
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Optimizing: {optimize_metric}")
        logger.info("=" * 80)

        # Load data
        data = self._load_data(start_date, end_date)

        if data is None or len(data) == 0:
            logger.error("❌ No data available for optimization")
            return pd.DataFrame()

        logger.info(f"📊 Data loaded: {len(data)} bars")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        total_tests = len(combinations)
        logger.info(f"🔬 Testing {total_tests} parameter combinations...")

        # Reset results
        self.results = []
        self.best_sharpe = -np.inf

        # Test each combination
        if max_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for combo in combinations:
                    params = dict(zip(param_names, combo))
                    future = executor.submit(
                        self._test_params,
                        data.copy(),
                        params,
                        initial_capital
                    )
                    futures.append((future, params))

                # Collect results
                completed = 0
                for future, params in futures:
                    try:
                        metrics = future.result()
                        if metrics is not None:
                            result = {**params, **metrics}
                            self.results.append(result)

                            # Check if best
                            if metrics.get(optimize_metric, -np.inf) > self.best_sharpe:
                                self.best_sharpe = metrics[optimize_metric]
                                self.best_params = params

                        completed += 1
                        if completed % 10 == 0:
                            logger.info(
                                f"Progress: {completed}/{total_tests} ({completed/total_tests*100:.1f}%)")

                    except Exception as e:
                        logger.error(f"Error in combination {params}: {e}")

        else:
            # Sequential execution
            for i, combo in enumerate(combinations, 1):
                params = dict(zip(param_names, combo))

                try:
                    metrics = self._test_params(data.copy(), params, initial_capital)

                    if metrics is not None:
                        result = {**params, **metrics}
                        self.results.append(result)

                        # Check if best
                        if metrics.get(optimize_metric, -np.inf) > self.best_sharpe:
                            self.best_sharpe = metrics[optimize_metric]
                            self.best_params = params

                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{total_tests} ({i/total_tests*100:.1f}%)")

                except Exception as e:
                    logger.error(f"Error in combination {params}: {e}")

        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)

        if len(results_df) > 0:
            # Sort by optimization metric
            results_df = results_df.sort_values(optimize_metric, ascending=False)

            logger.info("=" * 80)
            logger.info("✅ Grid Search Complete")
            logger.info(f"Best {optimize_metric}: {self.best_sharpe:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            logger.info("=" * 80)
            logger.info("\nTop 5 Results:")
            logger.info(results_df.head(5).to_string())

        else:
            logger.warning("⚠️ No valid results from grid search")

        return results_df

    def _test_params(self, data: pd.DataFrame, params: Dict,
                     initial_capital: float) -> Optional[Dict]:
        """
        Test a single parameter combination

        Args:
            data: Historical data
            params: Parameter dictionary
            initial_capital: Initial capital

        Returns:
            Dictionary with performance metrics or None
        """
        try:
            # Extract parameters
            risk_per_trade = params.get('risk_per_trade', 0.02)
            commission = params.get('commission', 0.001)
            slippage = params.get('slippage', 0.0005)
            sl_atr_mult = params.get('sl_atr_multiplier', 1.5)
            tp_rr = params.get('tp_risk_reward', 2.0)

            # Run backtest
            backtester = Backtester(
                df=data,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                commission=commission,
                slippage=slippage
            )

            # Execute backtest
            metrics = backtester.run_backtest(
                use_stop_loss=True,
                use_take_profit=True,
                sl_atr_multiplier=sl_atr_mult,
                tp_risk_reward=tp_rr
            )

            return metrics

        except Exception as e:
            logger.error(f"Error testing params {params}: {e}")
            return None

    def walk_forward_analysis(
        self,
        param_grid: Dict[str, List],
        start_date: str,
        end_date: str,
        train_period_days: int = 90,
        test_period_days: int = 30,
        initial_capital: float = 10000.0,
        optimize_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Walk-forward optimization

        Divide el período total en ventanas de entrenamiento y test.
        Optimiza en train, valida en test, y avanza la ventana.

        Args:
            param_grid: Parameter grid for optimization
            start_date: Overall start date
            end_date: Overall end date
            train_period_days: Training window size in days
            test_period_days: Test window size in days
            initial_capital: Initial capital
            optimize_metric: Metric to optimize

        Returns:
            Dictionary with walk-forward results
        """
        logger.info("=" * 80)
        logger.info("📈 Starting Walk-Forward Analysis")
        logger.info(f"Train period: {train_period_days} days")
        logger.info(f"Test period: {test_period_days} days")
        logger.info("=" * 80)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate windows
        windows = []
        current_start = start_dt

        while current_start + timedelta(days=train_period_days + test_period_days) <= end_dt:
            train_end = current_start + timedelta(days=train_period_days)
            test_end = train_end + timedelta(days=test_period_days)

            windows.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })

            # Slide window forward
            current_start = train_end

        logger.info(f"Generated {len(windows)} walk-forward windows")

        if len(windows) == 0:
            logger.error("❌ Not enough data for walk-forward analysis")
            return {}

        # Results for each window
        window_results = []

        for i, window in enumerate(windows, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Window {i}/{len(windows)}")
            logger.info(f"Train: {window['train_start'].date()} to {window['train_end'].date()}")
            logger.info(f"Test:  {window['test_start'].date()} to {window['test_end'].date()}")
            logger.info('=' * 60)

            # Optimize on training period
            train_results = self.grid_search(
                param_grid=param_grid,
                start_date=window['train_start'].strftime('%Y-%m-%d'),
                end_date=window['train_end'].strftime('%Y-%m-%d'),
                initial_capital=initial_capital,
                optimize_metric=optimize_metric,
                max_workers=1  # Sequential for walk-forward
            )

            if len(train_results) == 0:
                logger.warning(f"⚠️ No results for window {i}")
                continue

            # Best params from training
            best_train_params = train_results.iloc[0].to_dict()

            # Extract only parameter values (remove metrics)
            param_keys = list(param_grid.keys())
            optimized_params = {k: best_train_params[k]
                                for k in param_keys if k in best_train_params}

            logger.info(f"Best train params: {optimized_params}")
            logger.info(f"Train {optimize_metric}: {best_train_params.get(optimize_metric, 0):.4f}")

            # Test on out-of-sample period
            test_data = self._load_data(
                window['test_start'].strftime('%Y-%m-%d'),
                window['test_end'].strftime('%Y-%m-%d')
            )

            if test_data is None or len(test_data) == 0:
                logger.warning(f"⚠️ No test data for window {i}")
                continue

            test_metrics = self._test_params(test_data, optimized_params, initial_capital)

            if test_metrics is not None:
                logger.info(f"✅ Test {optimize_metric}: {test_metrics.get(optimize_metric, 0):.4f}")

                window_results.append({
                    'window': i,
                    'train_start': window['train_start'],
                    'train_end': window['train_end'],
                    'test_start': window['test_start'],
                    'test_end': window['test_end'],
                    'train_metric': best_train_params.get(optimize_metric, 0),
                    'test_metric': test_metrics.get(optimize_metric, 0),
                    'params': optimized_params,
                    'test_metrics': test_metrics
                })

        # Summary statistics
        if len(window_results) > 0:
            test_metrics_values = [w['test_metric'] for w in window_results]

            summary = {
                'num_windows': len(window_results),
                'avg_test_metric': np.mean(test_metrics_values),
                'std_test_metric': np.std(test_metrics_values),
                'min_test_metric': np.min(test_metrics_values),
                'max_test_metric': np.max(test_metrics_values),
                'window_results': window_results
            }

            logger.info("\n" + "=" * 80)
            logger.info("✅ Walk-Forward Analysis Complete")
            logger.info(
                f"Average test {optimize_metric}: {summary['avg_test_metric']:.4f} ± {summary['std_test_metric']:.4f}")
            logger.info(
                f"Range: [{summary['min_test_metric']:.4f}, {summary['max_test_metric']:.4f}]")
            logger.info("=" * 80)

            return summary

        else:
            logger.warning("⚠️ No valid window results")
            return {}

    def monte_carlo_simulation(
        self,
        trades_df: pd.DataFrame,
        n_simulations: int = 1000,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation of trading results

        Randomly reorders historical trades to estimate distribution
        of possible outcomes.

        Args:
            trades_df: DataFrame with historical trades
            n_simulations: Number of simulations to run
            initial_capital: Starting capital

        Returns:
            Dictionary with simulation statistics
        """
        logger.info("=" * 80)
        logger.info(f"🎲 Running Monte Carlo Simulation ({n_simulations} iterations)")
        logger.info("=" * 80)

        if len(trades_df) == 0:
            logger.error("❌ No trades to simulate")
            return {}

        # Extract P&L from trades
        pnl_values = trades_df['pnl'].values.astype(float)

        # Run simulations
        final_capitals = []
        max_drawdowns = []
        sharpe_ratios = []

        for i in range(n_simulations):
            # Randomly shuffle trades
            shuffled_pnl = np.random.choice(pnl_values, size=len(pnl_values), replace=True)

            # Calculate equity curve
            equity = np.zeros(len(shuffled_pnl) + 1)
            equity[0] = initial_capital

            for j, pnl in enumerate(shuffled_pnl):
                equity[j + 1] = equity[j] + pnl

            # Calculate metrics
            final_capital = equity[-1]
            final_capitals.append(final_capital)

            # Drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100
            max_dd = abs(drawdown.min())
            max_drawdowns.append(max_dd)

            # Sharpe (daily returns)
            returns = np.diff(equity) / equity[:-1]
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

        # Calculate statistics
        final_capitals = np.array(final_capitals)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)

        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        capital_percentiles = np.percentile(final_capitals, percentiles)
        dd_percentiles = np.percentile(max_drawdowns, percentiles)
        sharpe_percentiles = np.percentile(sharpe_ratios, percentiles)

        results = {
            'n_simulations': n_simulations,
            'n_trades': len(trades_df),
            'initial_capital': initial_capital,
            'final_capital': {
                'mean': final_capitals.mean(),
                'std': final_capitals.std(),
                'min': final_capitals.min(),
                'max': final_capitals.max(),
                'percentiles': dict(zip(percentiles, capital_percentiles))
            },
            'max_drawdown': {
                'mean': max_drawdowns.mean(),
                'std': max_drawdowns.std(),
                'min': max_drawdowns.min(),
                'max': max_drawdowns.max(),
                'percentiles': dict(zip(percentiles, dd_percentiles))
            },
            'sharpe_ratio': {
                'mean': sharpe_ratios.mean(),
                'std': sharpe_ratios.std(),
                'min': sharpe_ratios.min(),
                'max': sharpe_ratios.max(),
                'percentiles': dict(zip(percentiles, sharpe_percentiles))
            },
            'probability_profit': (final_capitals > initial_capital).sum() / n_simulations * 100,
            'risk_of_ruin': (final_capitals < initial_capital * 0.5).sum() / n_simulations * 100
        }

        logger.info("\n📊 Monte Carlo Results:")
        logger.info(
            f"Final Capital (mean): ${results['final_capital']['mean']:.2f} ± ${results['final_capital']['std']:.2f}")
        logger.info(
            f"Max Drawdown (mean): {results['max_drawdown']['mean']:.2f}% ± {results['max_drawdown']['std']:.2f}%")
        logger.info(
            f"Sharpe Ratio (mean): {results['sharpe_ratio']['mean']:.3f} ± {results['sharpe_ratio']['std']:.3f}")
        logger.info(f"Probability of Profit: {results['probability_profit']:.1f}%")
        logger.info(f"Risk of Ruin (>50% loss): {results['risk_of_ruin']:.1f}%")
        logger.info("=" * 80)

        return results

    def save_results(self, filepath: str = 'results/optimization_results.json'):
        """Save optimization results to file"""
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_params': self.best_params,
            'best_sharpe': self.best_sharpe,
            'all_results': self.results
        }

        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"💾 Results saved to {file_path}")


def run_optimization_example():
    """Example usage of optimizer"""

    # Define parameter grid
    param_grid = {
        'risk_per_trade': [0.01, 0.015, 0.02],
        'sl_atr_multiplier': [1.0, 1.5, 2.0],
        'tp_risk_reward': [1.5, 2.0, 2.5],
        'commission': [0.0005, 0.001],
        'slippage': [0.0001, 0.0005]
    }

    # Create optimizer
    optimizer = ParameterOptimizer()

    # Grid search
    results = optimizer.grid_search(
        param_grid=param_grid,
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_capital=10000.0,
        optimize_metric='sharpe_ratio',
        max_workers=4
    )

    # Save results
    if len(results) > 0:
        results.to_csv('results/grid_search_results.csv', index=False)
        optimizer.save_results()

        print("\n" + "=" * 80)
        print("Top 10 Parameter Combinations:")
        print("=" * 80)
        print(results.head(10).to_string())


if __name__ == '__main__':
    run_optimization_example()
