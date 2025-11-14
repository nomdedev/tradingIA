#!/usr/bin/env python3
"""
Robustness and Anti-Snooping Analysis Module
=============================================

Comprehensive robustness testing and data snooping detection.

Features:
- Robustness metrics: Information Ratio, Sortino, VaR95%, Calmar, Ulcer Index
- Probabilistic Sharpe Ratio with bootstrap confidence intervals
- Anti-snooping: AIC/BIC penalization, White's Reality Check, Bootstrap CI
- Data mining detection: Multiple testing correction (Bonferroni)
- Walk-forward re-optimization to prevent overfitting
- Stress testing with noise injection and market condition simulation

Usage:
    from src.robustness_snooping import RobustnessAnalyzer
    analyzer = RobustnessAnalyzer()
    robustness = analyzer.calculate_robustness_metrics(returns)
    snooping_detected = analyzer.detect_snooping(opt_results)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RobustnessAnalyzer:
    """Comprehensive robustness and anti-snooping analysis"""

    def __init__(self):
        self.risk_free_rate = 0.04  # 4% annual
        self.trading_days = 252
        self.confidence_level = 0.95

    def calculate_robustness_metrics(
            self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive robustness metrics

        Args:
            returns: Strategy returns array
            benchmark_returns: Benchmark returns (default: buy-hold BTC proxy)

        Returns:
            Dict with all robustness metrics
        """
        if len(returns) == 0:
            return {metric: 0.0 for metric in [
                'information_ratio', 'sortino_ratio', 'var_95', 'calmar_ratio',
                'ulcer_index', 'probabilistic_sharpe', 'stability_score'
            ]}

        # Create benchmark if not provided (simplified BTC proxy)
        if benchmark_returns is None:
            benchmark_returns = np.full_like(returns, self.risk_free_rate / self.trading_days)

        # Information Ratio (IR)
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns, ddof=1)
        information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0

        # Sortino Ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns, ddof=1)
            sortino_ratio = np.mean(returns) / downside_deviation * np.sqrt(self.trading_days)
        else:
            sortino_ratio = float('inf')

        # Value at Risk 95%
        var_95 = np.percentile(returns, 5)

        # Calmar Ratio (annualized return / max drawdown)
        cumulative_returns = np.cumprod(1 + returns) - 1
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(cumulative_returns)

        # Probabilistic Sharpe Ratio
        probabilistic_sharpe = self._calculate_probabilistic_sharpe(returns)

        # Stability Score (lower variance in rolling Sharpe = more stable)
        stability_score = self._calculate_stability_score(returns)

        return {
            'information_ratio': information_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'calmar_ratio': calmar_ratio,
            'ulcer_index': ulcer_index,
            'probabilistic_sharpe': probabilistic_sharpe,
            'stability_score': stability_score
        }

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
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

    def _calculate_ulcer_index(self, cumulative_returns: np.ndarray) -> float:
        """Calculate Ulcer Index (average drawdown magnitude)"""
        if len(cumulative_returns) == 0:
            return 0.0

        # Calculate drawdowns
        peak = cumulative_returns[0]
        drawdowns = []

        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            dd = peak - ret
            drawdowns.append(dd)

        drawdowns = np.array(drawdowns)
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))

        return ulcer_index

    def _calculate_probabilistic_sharpe(self, returns: np.ndarray, n_boot: int = 1000) -> float:
        """
        Calculate Probabilistic Sharpe Ratio using bootstrap

        Returns: Percentage of bootstrap samples with Sharpe > 0
        """
        if len(returns) < 10:
            return 0.0

        # Bootstrap resampling
        sharpe_boot = []
        n = len(returns)

        for _ in range(n_boot):
            # Resample with replacement
            boot_returns = np.random.choice(returns, size=n, replace=True)
            boot_excess = boot_returns - self.risk_free_rate / self.trading_days

            if np.std(boot_excess, ddof=1) > 0:
                boot_sharpe = np.mean(boot_excess) / np.std(boot_excess,
                                                            ddof=1) * np.sqrt(self.trading_days)
                sharpe_boot.append(boot_sharpe)

        if not sharpe_boot:
            return 0.0

        # Calculate probability that true Sharpe > 0
        prob_sharpe = np.mean(np.array(sharpe_boot) > 0)

        return prob_sharpe

    def _calculate_stability_score(self, returns: np.ndarray, window: int = 50) -> float:
        """
        Calculate stability score based on rolling Sharpe variance

        Higher score = more stable performance
        """
        if len(returns) < window * 2:
            return 0.0

        # Calculate rolling Sharpe ratios
        rolling_sharpes = []
        for i in range(window, len(returns), window // 2):
            window_returns = returns[i - window:i]
            if len(window_returns) > 1:
                excess = window_returns - self.risk_free_rate / self.trading_days
                if np.std(excess, ddof=1) > 0:
                    sharpe = np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(self.trading_days)
                    rolling_sharpes.append(sharpe)

        if len(rolling_sharpes) < 3:
            return 0.0

        # Stability = 1 / (1 + variance of rolling Sharpes)
        sharpe_variance = np.var(rolling_sharpes, ddof=1)
        stability_score = 1.0 / (1.0 + sharpe_variance)

        return stability_score

    def detect_snooping(self, optimization_results: Dict[str, Any],
                        n_null_strategies: int = 500) -> Dict[str, Any]:
        """
        Detect data snooping and p-hacking in optimization results

        Args:
            optimization_results: Results from optimization process
            n_null_strategies: Number of null strategies for White's test

        Returns:
            Dict with snooping detection results
        """
        results = {
            'snooping_detected': False,
            'aic_score': None,
            'bic_score': None,
            'whites_p_value': None,
            'bootstrap_ci': None,
            'bonferroni_corrected_p': None,
            'recommendations': []
        }

        # Extract optimization data
        if 'best_returns' in optimization_results:
            returns = np.array(optimization_results['best_returns'])
        elif 'trades_df' in optimization_results:
            returns = optimization_results['trades_df']['PnL %'].values / 100.0
        else:
            return results

        if len(returns) == 0:
            return results

        # AIC/BIC calculation
        log_likelihood = self._calculate_log_likelihood(returns)
        n_params = optimization_results.get('n_parameters', 5)  # Default assumption
        n_obs = len(returns)

        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_obs)

        results['aic_score'] = aic
        results['bic_score'] = bic

        # Compare with baseline (buy-hold)
        baseline_aic = self._calculate_baseline_aic(returns)
        if aic > baseline_aic + 10:  # Significant overfitting
            results['snooping_detected'] = True
            results['recommendations'].append("High AIC indicates overfitting - reduce parameters")

        # White's Reality Check
        whites_p = self._whites_reality_check(returns, n_null_strategies)
        results['whites_p_value'] = whites_p

        if whites_p > 0.05:  # No significant edge
            results['snooping_detected'] = True
            results['recommendations'].append(
                "White's test suggests no significant edge - possible snooping")

        # Bootstrap Confidence Intervals
        ci = self._bootstrap_confidence_interval(returns)
        results['bootstrap_ci'] = ci

        if ci[0] <= 0 <= ci[1]:  # CI includes zero
            results['snooping_detected'] = True
            results['recommendations'].append("Bootstrap CI includes zero - unstable performance")

        # Bonferroni correction for multiple testing
        n_tests = optimization_results.get('n_optimization_runs', 100)
        original_p = optimization_results.get('best_p_value', 0.05)
        bonferroni_p = min(original_p * n_tests, 1.0)
        results['bonferroni_corrected_p'] = bonferroni_p

        if bonferroni_p > 0.05:
            results['snooping_detected'] = True
            results['recommendations'].append("Multiple testing correction removes significance")

        return results

    def _calculate_log_likelihood(self, returns: np.ndarray) -> float:
        """Calculate log-likelihood for AIC/BIC"""
        if len(returns) == 0:
            return 0.0

        # Simple normal distribution assumption
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        if sigma == 0:
            return 0.0

        # Log-likelihood for normal distribution
        log_lik = -0.5 * len(returns) * np.log(2 * np.pi * sigma**2) - \
            0.5 * np.sum((returns - mu)**2) / sigma**2

        return log_lik

    def _calculate_baseline_aic(self, returns: np.ndarray) -> float:
        """Calculate AIC for baseline strategy (random)"""
        # Simulate random strategy returns
        random_returns = np.random.normal(0, np.std(returns, ddof=1), len(returns))
        return -2 * self._calculate_log_likelihood(random_returns) + 2 * 1  # 1 param for random

    def _whites_reality_check(self, returns: np.ndarray, n_null: int) -> float:
        """
        White's Reality Check: Test against null strategies

        Returns: Adjusted p-value
        """
        if len(returns) < 10:
            return 1.0

        # Calculate observed Sharpe
        excess_returns = returns - self.risk_free_rate / self.trading_days
        observed_sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)

        # Generate null strategies (random signals)
        null_sharpes = []
        for _ in range(n_null):
            null_returns = np.random.choice([0.01, -0.01], len(returns), p=[0.5, 0.5])
            null_excess = null_returns - self.risk_free_rate / self.trading_days
            if np.std(null_excess, ddof=1) > 0:
                null_sharpe = np.mean(null_excess) / np.std(null_excess, ddof=1)
                null_sharpes.append(null_sharpe)

        if not null_sharpes:
            return 1.0

        # Calculate p-value: fraction of null strategies better than observed
        null_sharpes = np.array(null_sharpes)
        p_value = np.mean(null_sharpes >= observed_sharpe)

        return p_value

    def _bootstrap_confidence_interval(self, returns: np.ndarray,
                                       n_boot: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for Sharpe ratio"""
        if len(returns) < 10:
            return (0.0, 0.0)

        def sharpe_stat(data):
            excess = data - self.risk_free_rate / self.trading_days
            if np.std(excess, ddof=1) > 0:
                return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(self.trading_days)
            return 0.0

        # Bootstrap resampling
        try:
            boot_result = bootstrap((returns,), sharpe_stat,
                                    confidence_level=self.confidence_level,
                                    n_resamples=n_boot,
                                    method='percentile')
            ci = boot_result.confidence_interval
            return (ci.low, ci.high)
        except Exception:
            return (0.0, 0.0)

    def stress_test_robustness(self, returns: np.ndarray,
                               stress_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stress test strategy under different market conditions

        Args:
            returns: Original strategy returns
            stress_scenarios: Dict of stress test scenarios

        Returns:
            Dict with stress test results
        """
        results = {}

        for scenario_name, scenario_params in stress_scenarios.items():
            scenario_returns = returns.copy()

            # Apply stress modifications
            if 'volatility_multiplier' in scenario_params:
                # Increase volatility
                vol_mult = scenario_params['volatility_multiplier']
                noise = np.random.normal(0, np.std(returns, ddof=1) * (vol_mult - 1), len(returns))
                scenario_returns += noise

            if 'bear_market' in scenario_params and scenario_params['bear_market']:
                # Simulate bear market (negative drift)
                bear_drift = np.linspace(0, -0.5, len(returns))  # Gradual decline
                scenario_returns += bear_drift

            if 'chop_filter' in scenario_params and scenario_params['chop_filter']:
                # Remove profitable trades (simulate chop)
                profitable_mask = scenario_returns > 0
                scenario_returns[profitable_mask] = 0

            # Calculate metrics for stressed scenario
            stressed_metrics = self.calculate_robustness_metrics(scenario_returns)

            # Calculate degradation
            original_sharpe = np.mean(returns) / np.std(returns, ddof=1) * \
                np.sqrt(self.trading_days)
            stressed_sharpe = stressed_metrics['probabilistic_sharpe']  # Approximation

            degradation = (original_sharpe - stressed_sharpe) / \
                abs(original_sharpe) if original_sharpe != 0 else 0

            results[scenario_name] = {
                'metrics': stressed_metrics,
                'degradation': degradation,
                'survives': degradation < 0.5  # Survives if <50% degradation
            }

        return results

    def walk_forward_reoptimization(self, full_data: pd.DataFrame,
                                    strategy_func: callable,
                                    window_size: int = 8,
                                    reopt_frequency: int = 3) -> Dict[str, Any]:
        """
        Walk-forward re-optimization to prevent overfitting

        Args:
            full_data: Complete historical data
            strategy_func: Function that optimizes and returns strategy
            window_size: Size of each window in periods
            reopt_frequency: How often to re-optimize

        Returns:
            Dict with walk-forward results
        """
        results = {
            'periods': [],
            'in_sample_sharpe': [],
            'out_of_sample_sharpe': [],
            'parameter_stability': [],
            'overfitting_detected': False
        }

        # Split data into periods
        total_periods = len(full_data) // (window_size * 12 * 24)  # Approximate

        for i in range(max(1, total_periods - window_size)):
            start_idx = i * (window_size * 12 * 24)
            end_idx = (i + window_size) * (12 * 24)

            if end_idx >= len(full_data):
                break

            # Define train/test split
            train_end = int(start_idx + (end_idx - start_idx) * 0.7)
            train_data = full_data.iloc[start_idx:train_end]
            test_data = full_data.iloc[train_end:end_idx]

            try:
                # Optimize on training data
                opt_result = strategy_func(train_data, optimize=True)
                train_returns = opt_result.get('returns', [])

                # Test on out-of-sample data
                test_result = strategy_func(test_data, parameters=opt_result.get('best_params'))
                test_returns = test_result.get('returns', [])

                # Calculate metrics
                if train_returns and test_returns:
                    train_sharpe = np.mean(train_returns) / np.std(train_returns,
                                                                   ddof=1) * np.sqrt(self.trading_days)
                    test_sharpe = np.mean(test_returns) / np.std(test_returns,
                                                                 ddof=1) * np.sqrt(self.trading_days)

                    results['periods'].append(f'Period_{i+1}')
                    results['in_sample_sharpe'].append(train_sharpe)
                    results['out_of_sample_sharpe'].append(test_sharpe)

                    # Check for overfitting (IS >> OOS)
                    if train_sharpe > test_sharpe * 1.5:
                        results['overfitting_detected'] = True

            except Exception as e:
                print(f"Warning: Walk-forward period {i+1} failed: {e}")
                continue

        return results

    def generate_robustness_report(self, robustness_metrics: Dict[str, Any],
                                   snooping_results: Dict[str, Any],
                                   output_path: str = "results/robustness_report.csv") -> None:
        """
        Generate comprehensive robustness report

        Args:
            robustness_metrics: Robustness metrics results
            snooping_results: Snooping detection results
            output_path: Path to save report
        """
        # Combine results
        report_data = {
            'metric_type': [],
            'metric_name': [],
            'value': [],
            'interpretation': []
        }

        # Add robustness metrics
        for metric_name, value in robustness_metrics.items():
            report_data['metric_type'].append('robustness')
            report_data['metric_name'].append(metric_name)
            report_data['value'].append(value)

            # Add interpretation
            if metric_name == 'information_ratio':
                interp = 'Good (>0.5)' if value > 0.5 else 'Poor (<0.5)'
            elif metric_name == 'sortino_ratio':
                interp = 'Excellent (>1.5)' if value > 1.5 else 'Poor (<1.5)'
            elif metric_name == 'var_95':
                interp = 'Acceptable (>-3%)' if value > -0.03 else 'Risky (<-3%)'
            elif metric_name == 'calmar_ratio':
                interp = 'Good (>2.0)' if value > 2.0 else 'Poor (<2.0)'
            elif metric_name == 'ulcer_index':
                interp = 'Low stress (<10%)' if value < 0.1 else 'High stress (>10%)'
            elif metric_name == 'probabilistic_sharpe':
                interp = 'Robust (>80%)' if value > 0.8 else 'Unreliable (<80%)'
            elif metric_name == 'stability_score':
                interp = 'Stable (>0.7)' if value > 0.7 else 'Unstable (<0.7)'
            else:
                interp = 'Unknown'

            report_data['interpretation'].append(interp)

        # Add snooping results
        for key, value in snooping_results.items():
            if key == 'recommendations':
                continue
            report_data['metric_type'].append('snooping')
            report_data['metric_name'].append(key)
            report_data['value'].append(str(value))
            report_data['interpretation'].append('Detection metric')

        # Create DataFrame and save
        df = pd.DataFrame(report_data)
        Path(output_path).parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✅ Robustness report saved to: {output_path}")

        # Print summary
        print("\n📊 Robustness Analysis Summary:")
        print("=" * 50)

        snooping_detected = snooping_results.get('snooping_detected', False)
        print(f"Snooping Detected: {'❌ YES' if snooping_detected else '✅ NO'}")

        if snooping_detected:
            print("Recommendations:")
            for rec in snooping_results.get('recommendations', []):
                print(f"  - {rec}")

        # Check key robustness metrics
        ir = robustness_metrics.get('information_ratio', 0)
        ps = robustness_metrics.get('probabilistic_sharpe', 0)
        ui = robustness_metrics.get('ulcer_index', 1)

        print("\nKey Metrics:")
        print(f"  Information Ratio: {ir:.3f}")
        print(f"  Probabilistic Sharpe: {ps:.1%}")
        print(f"  Ulcer Index: {ui:.1%}")


# Convenience functions
def calculate_robustness_metrics(
        returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Convenience function for robustness metrics"""
    analyzer = RobustnessAnalyzer()
    return analyzer.calculate_robustness_metrics(returns, benchmark_returns)


def detect_snooping(optimization_results: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for snooping detection"""
    analyzer = RobustnessAnalyzer()
    return analyzer.detect_snooping(optimization_results)


if __name__ == '__main__':
    # Example usage with mock data
    np.random.seed(42)

    # Generate mock BTC strategy returns
    n_trades = 500
    returns = np.random.normal(0.015, 0.08, n_trades)  # Mean 1.5%, std 8%

    # Add some outliers and drawdowns
    returns[returns < -0.05] = returns[returns < -0.05] * 2
    # Simulate a drawdown period
    drawdown_start = 200
    drawdown_end = 250
    returns[drawdown_start:drawdown_end] -= 0.02

    print("🔍 Running Robustness Analysis...")

    # Calculate robustness metrics
    analyzer = RobustnessAnalyzer()
    robustness = analyzer.calculate_robustness_metrics(returns)

    print("\n📈 Robustness Metrics:")
    print("=" * 50)
    for key, value in robustness.items():
        print(f"{key:20}: {value:8.3f}")

    # Mock optimization results for snooping detection
    opt_results = {
        'best_returns': returns,
        'n_parameters': 5,
        'n_optimization_runs': 100,
        'best_p_value': 0.03
    }

    # Detect snooping
    snooping = analyzer.detect_snooping(opt_results)

    print("\n🕵️  Snooping Detection:")
    print("=" * 50)
    print(f"Snooping detected: {snooping['snooping_detected']}")
    print(f"AIC Score: {snooping['aic_score']:.2f}")
    print(f"White's p-value: {snooping['whites_p_value']:.4f}")

    if snooping['recommendations']:
        print("Recommendations:")
        for rec in snooping['recommendations']:
            print(f"  - {rec}")

    # Stress testing
    stress_scenarios = {
        'high_volatility': {'volatility_multiplier': 2.0},
        'bear_market': {'bear_market': True},
        'choppy_market': {'chop_filter': True}
    }

    stress_results = analyzer.stress_test_robustness(returns, stress_scenarios)

    print("\n⚡ Stress Test Results:")
    print("=" * 50)
    for scenario, result in stress_results.items():
        survives = "✅ Survives" if result['survives'] else "❌ Fails"
        print(f"{scenario:15}: Degradation {result['degradation']:.1%} - {survives}")

    print("\n✅ Robustness analysis completed!")
