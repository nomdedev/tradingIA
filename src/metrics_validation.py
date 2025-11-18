#!/usr/bin/env python3
"""
Metrics Validation Module for BTC Trading Strategy
==================================================

Comprehensive statistical validation metrics for trading strategies.

Features:
- Core metrics: win_rate, profit_factor, sharpe, calmar, sortino
- Risk metrics: volatility, VaR95%, information_ratio
- Statistical tests: statistical_power, deflated_sharpe
- Clustering analysis by score/HTF alignment
- Walk-forward validation
- Monte Carlo robustness testing
- Sensitivity analysis heatmaps
- System comparison with paired t-tests

Usage:
    from src.metrics_validation import calculate_metrics, run_walk_forward_validation
    metrics = calculate_metrics(trades_df)
    wf_results = run_walk_forward_validation(df_5m, strategy_func)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_1samp, ttest_rel
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MetricsValidator:
    """Comprehensive metrics validation for trading strategies"""

    def __init__(self):
        self.risk_free_rate = 0.04  # 4% annual BTC risk-free proxy
        self.trading_days = 252  # Annual trading days for crypto

    def calculate_core_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate core trading performance metrics

        Args:
            trades_df: DataFrame with columns ['PnL %', 'Entry Time', 'Exit Time']

        Returns:
            Dict with all calculated metrics
        """
        if trades_df.empty:
            return {metric: 0.0 for metric in [
                'win_rate', 'profit_factor', 'sharpe_ratio', 'calmar_ratio',
                'sortino_ratio', 'volatility', 'var_95', 'information_ratio',
                'statistical_power', 'deflated_sharpe', 'total_return', 'max_drawdown'
            ]}

        # Ensure PnL is in decimal format (not percentage)
        pnl = trades_df['PnL %'].copy()
        if pnl.max() > 10:  # Likely percentage format
            pnl = pnl / 100.0

        returns = pnl.values
        cumulative = np.cumprod(1 + returns) - 1

        # Win Rate
        win_rate = np.mean(returns > 0)

        # Profit Factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        profit_factor = (winning_trades.sum() / abs(losing_trades.sum())
                         ) if len(losing_trades) > 0 else float('inf')

        # Sharpe Ratio (annualized)
        excess_returns = returns - self.risk_free_rate / self.trading_days
        if len(returns) > 1:
            sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * \
                np.sqrt(self.trading_days)
        else:
            sharpe_ratio = 0.0

        # Calmar Ratio (sin inf)
        max_drawdown = self._calculate_max_drawdown(cumulative)
        total_return = cumulative[-1] if len(cumulative) > 0 else 0.0
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Sortino Ratio (downside deviation only, sin inf)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = (np.mean(returns) / np.std(downside_returns)) * \
                np.sqrt(self.trading_days)
        else:
            sortino_ratio = float('inf')

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(self.trading_days)

        # VaR 95%
        var_95 = np.percentile(returns, 5)

        # Information Ratio (vs buy-hold BTC approximation)
        btc_proxy_return = self.risk_free_rate / self.trading_days  # Simplified
        tracking_error = np.std(returns - btc_proxy_return)
        information_ratio = (np.mean(returns) - btc_proxy_return) / \
            tracking_error if tracking_error > 0 else 0.0

        # Statistical Power (t-test vs zero returns)
        if len(returns) > 1:
            t_stat, p_value = ttest_1samp(returns, 0)
            statistical_power = 1 - p_value  # Simplified power calculation
        else:
            statistical_power = 0.0

        # Deflated Sharpe (adjusted for number of trades)
        n_trades = len(returns)
        deflated_sharpe = sharpe_ratio / np.sqrt(n_trades) if n_trades > 0 else 0.0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            'var_95': var_95,
            'information_ratio': information_ratio,
            'statistical_power': statistical_power,
            'deflated_sharpe': deflated_sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': n_trades
        }

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = cumulative_returns[0]
        max_dd = 0.0

        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            dd = peak - ret
            max_dd = max(max_dd, dd)

        return max_dd

    def calculate_metrics_by_cluster(self, trades_df: pd.DataFrame,
                                     cluster_col: str = 'confluence_score') -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics segmented by cluster (e.g., confluence score, HTF alignment)

        Args:
            trades_df: DataFrame with trading results
            cluster_col: Column to group by (default: 'confluence_score')

        Returns:
            Dict of metrics by cluster
        """
        if cluster_col not in trades_df.columns:
            return {}

        results = {}
        for cluster_value, group in trades_df.groupby(cluster_col):
            results[f'cluster_{cluster_value}'] = self.calculate_core_metrics(group)

        return results

    def run_walk_forward_validation(self, df_5m: pd.DataFrame,
                                    strategy_func: callable,
                                    window_size: int = 8,
                                    test_periods: int = 3,
                                    train_ratio: float = 0.7) -> Dict[str, Any]:
        """
        Run walk-forward validation with rolling windows

        Args:
            df_5m: 5-minute BTC data
            strategy_func: Function that generates signals and returns trades
            window_size: Number of periods in each window
            test_periods: Number of periods to test out-of-sample
            train_ratio: Ratio of data for training vs testing

        Returns:
            Dict with validation results
        """
        results = {
            'in_sample_metrics': [],
            'out_of_sample_metrics': [],
            'sharpe_oos': [],
            'periods': []
        }

        # Split data into rolling windows
        total_periods = len(df_5m) // (window_size * 12 * 24)  # Approximate periods

        for i in range(max(1, total_periods - window_size)):
            start_idx = i * (window_size * 12 * 24)
            end_idx = (i + window_size) * (12 * 24)

            if end_idx >= len(df_5m):
                break

            # Split train/test
            split_idx = int(start_idx + (end_idx - start_idx) * train_ratio)
            train_data = df_5m.iloc[start_idx:split_idx]
            test_data = df_5m.iloc[split_idx:end_idx]

            try:
                # Run strategy on train data (for optimization)
                train_trades = strategy_func(train_data)
                train_metrics = self.calculate_core_metrics(train_trades)

                # Run strategy on test data (OOS validation)
                test_trades = strategy_func(test_data)
                test_metrics = self.calculate_core_metrics(test_trades)

                results['in_sample_metrics'].append(train_metrics)
                results['out_of_sample_metrics'].append(test_metrics)
                results['sharpe_oos'].append(test_metrics.get('sharpe_ratio', 0))
                results['periods'].append(f'Period_{i+1}')

            except Exception as e:
                print(f"Warning: Failed walk-forward period {i+1}: {e}")
                continue

        return results

    def run_monte_carlo_simulation(self, trades_df: pd.DataFrame,
                                   n_simulations: int = 500,
                                   noise_level: float = 0.1) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with noise injection

        Args:
            trades_df: Original trading results
            n_simulations: Number of Monte Carlo runs
            noise_level: Standard deviation of noise to inject

        Returns:
            Dict with simulation results
        """
        if trades_df.empty:
            return {'sharpe_std': 0.0, 'sharpe_mean': 0.0, 'sharpe_distribution': []}

        original_returns = (trades_df['PnL %'] / 100.0).values
        sharpe_ratios = []

        for _ in range(n_simulations):
            # Add noise to returns
            noise = np.random.normal(0, noise_level, len(original_returns))
            noisy_returns = original_returns + noise

            # Calculate metrics for noisy data
            noisy_df = pd.DataFrame({'PnL %': noisy_returns * 100})
            metrics = self.calculate_core_metrics(noisy_df)
            sharpe_ratios.append(metrics['sharpe_ratio'])

        return {
            'sharpe_std': np.std(sharpe_ratios),
            'sharpe_mean': np.mean(sharpe_ratios),
            'sharpe_distribution': sharpe_ratios,
            'robustness_score': 1.0 / (1.0 + np.std(sharpe_ratios))  # Higher is more robust
        }

    def run_sensitivity_analysis(self, strategy_func: callable,
                                 param_ranges: Dict[str, List[float]],
                                 base_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run sensitivity analysis for strategy parameters

        Args:
            strategy_func: Function that takes parameters and returns trades
            param_ranges: Dict of parameter names to list of values to test
            base_data: Base market data

        Returns:
            Dict with sensitivity results
        """
        results = {}
        param_names = list(param_ranges.keys())

        if len(param_names) == 2:
            # 2D heatmap
            param1, param2 = param_names
            p1_values = param_ranges[param1]
            p2_values = param_ranges[param2]

            heatmap_data = np.zeros((len(p1_values), len(p2_values)))

            for i, p1_val in enumerate(p1_values):
                for j, p2_val in enumerate(p2_values):
                    try:
                        params = {param1: p1_val, param2: p2_val}
                        trades = strategy_func(base_data, **params)
                        metrics = self.calculate_core_metrics(trades)
                        heatmap_data[i, j] = metrics.get('sharpe_ratio', 0)
                    except Exception:
                        heatmap_data[i, j] = 0

            results['heatmap'] = heatmap_data
            results['param1_values'] = p1_values
            results['param2_values'] = p2_values
            results['param_names'] = [param1, param2]

        else:
            # 1D sensitivity
            for param_name, param_values in param_ranges.items():
                sharpe_values = []
                for param_val in param_values:
                    try:
                        params = {param_name: param_val}
                        trades = strategy_func(base_data, **params)
                        metrics = self.calculate_core_metrics(trades)
                        sharpe_values.append(metrics.get('sharpe_ratio', 0))
                    except Exception:
                        sharpe_values.append(0)

                results[param_name] = {
                    'values': param_values,
                    'sharpe_ratios': sharpe_values
                }

        return results

    def compare_systems(self, system_a_trades: pd.DataFrame,
                        system_b_trades: pd.DataFrame,
                        system_c_trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compare multiple trading systems statistically

        Args:
            system_a_trades: Trades from system A (IFVG base)
            system_b_trades: Trades from system B (alternative)
            system_c_trades: Optional trades from system C

        Returns:
            Dict with comparison results
        """
        results = {}

        # Calculate metrics for each system
        systems = {'A': system_a_trades, 'B': system_b_trades}
        if system_c_trades is not None:
            systems['C'] = system_c_trades

        metrics_by_system = {}
        for name, trades in systems.items():
            metrics_by_system[name] = self.calculate_core_metrics(trades)

        results['metrics_by_system'] = metrics_by_system

        # Statistical comparison A vs B
        if not system_a_trades.empty and not system_b_trades.empty:
            returns_a = (system_a_trades['PnL %'] / 100.0).values
            returns_b = (system_b_trades['PnL %'] / 100.0).values

            # Paired t-test if same number of trades
            if len(returns_a) == len(returns_b):
                t_stat, p_value = ttest_rel(returns_a, returns_b)
                results['a_vs_b_paired_ttest'] = {'t_stat': t_stat, 'p_value': p_value}
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(returns_a, returns_b)
                results['a_vs_b_ttest'] = {'t_stat': t_stat, 'p_value': p_value}

            # Sharpe comparison
            sharpe_a = metrics_by_system['A']['sharpe_ratio']
            sharpe_b = metrics_by_system['B']['sharpe_ratio']
            results['sharpe_comparison'] = {
                'sharpe_a': sharpe_a,
                'sharpe_b': sharpe_b,
                'difference': sharpe_b - sharpe_a,
                'b_superior': sharpe_b > sharpe_a
            }

        return results

    def generate_validation_report(self, metrics: Dict[str, Any],
                                   output_path: str = "results/validation_report.csv") -> None:
        """
        Generate CSV report with validation metrics

        Args:
            metrics: Dict containing all validation results
            output_path: Path to save the report
        """
        # Flatten nested dicts for CSV
        flat_data = {}

        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    flat_data[new_key] = ','.join(map(str, value))
                else:
                    flat_data[new_key] = value

        flatten_dict(metrics)

        # Create DataFrame and save
        df = pd.DataFrame([flat_data])
        Path(output_path).parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Validation report saved to: {output_path}")

    def plot_validation_results(self, wf_results: Dict[str, Any],
                                mc_results: Dict[str, Any],
                                output_dir: str = "results/figures") -> None:
        """
        Generate validation plots

        Args:
            wf_results: Walk-forward validation results
            mc_results: Monte Carlo simulation results
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)

        # Walk-forward Sharpe plot
        if 'sharpe_oos' in wf_results and wf_results['sharpe_oos']:
            plt.figure(figsize=(12, 6))
            plt.plot(wf_results['sharpe_oos'], marker='o', linewidth=2)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Sharpe > 1.0')
            plt.title('Walk-Forward Out-of-Sample Sharpe Ratio')
            plt.xlabel('Period')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/walk_forward_sharpe.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Monte Carlo Sharpe distribution
        if 'sharpe_distribution' in mc_results and mc_results['sharpe_distribution']:
            plt.figure(figsize=(10, 6))
            plt.hist(mc_results['sharpe_distribution'], bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(
                x=np.mean(
                    mc_results['sharpe_distribution']),
                color='r',
                linestyle='--',
                linewidth=2,
                label=f'Mean: {np.mean(mc_results["sharpe_distribution"]):.2f}')
            plt.title('Monte Carlo Sharpe Ratio Distribution')
            plt.xlabel('Sharpe Ratio')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/monte_carlo_sharpe.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Validation plots saved to: {output_dir}")


# Convenience functions
def calculate_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Convenience function for core metrics calculation"""
    validator = MetricsValidator()
    return validator.calculate_core_metrics(trades_df)


def run_walk_forward_validation(df_5m: pd.DataFrame, strategy_func: callable,
                                window_size: int = 8, test_periods: int = 3) -> Dict[str, Any]:
    """Convenience function for walk-forward validation"""
    validator = MetricsValidator()
    return validator.run_walk_forward_validation(df_5m, strategy_func, window_size, test_periods)


def run_monte_carlo_simulation(trades_df: pd.DataFrame, n_simulations: int = 500) -> Dict[str, Any]:
    """Convenience function for Monte Carlo simulation"""
    validator = MetricsValidator()
    return validator.run_monte_carlo_simulation(trades_df, n_simulations)


def compare_systems(system_a_trades: pd.DataFrame, system_b_trades: pd.DataFrame,
                    system_c_trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Convenience function for system comparison"""
    validator = MetricsValidator()
    return validator.compare_systems(system_a_trades, system_b_trades, system_c_trades)


if __name__ == '__main__':
    # Example usage with mock data
    np.random.seed(42)

    # Generate mock trading results (BTC-like returns)
    n_trades = 200
    returns = np.random.normal(0.015, 0.08, n_trades)  # Mean 1.5%, std 8%
    returns[returns < -0.05] = returns[returns < -0.05] * 2  # Add some large losses

    mock_trades = pd.DataFrame({
        'PnL %': returns * 100,  # Convert to percentage
        'Entry Time': pd.date_range('2024-01-01', periods=n_trades, freq='4h'),
        'Exit Time': pd.date_range('2024-01-01', periods=n_trades, freq='4h') + pd.Timedelta(hours=4),
        'confluence_score': np.random.choice([3, 4, 5], n_trades)
    })

    # Calculate metrics
    validator = MetricsValidator()
    metrics = validator.calculate_core_metrics(mock_trades)

    print("📊 BTC Trading Strategy Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20}: {value:8.3f}")
        else:
            print(f"{key:20}: {value}")

    # Cluster analysis
    cluster_metrics = validator.calculate_metrics_by_cluster(mock_trades)
    print("\n📈 Metrics by Confluence Score:")
    print("=" * 50)
    for cluster, clust_metrics in cluster_metrics.items():
        sharpe = clust_metrics.get('sharpe_ratio', 0)
        win_rate = clust_metrics.get('win_rate', 0)
        print(f"{cluster:15}: Sharpe={sharpe:.2f}, Win Rate={win_rate:.1%}")

    # Monte Carlo simulation
    mc_results = validator.run_monte_carlo_simulation(mock_trades)
    print("\n🎲 Monte Carlo Robustness:")
    print(f"Sharpe Std: {mc_results['sharpe_std']:.3f}")
    print(f"Robustness Score: {mc_results['robustness_score']:.3f}")

    print("\n✅ Metrics validation completed!")
