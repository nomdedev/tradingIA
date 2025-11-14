#!/usr/bin/env python3
"""
Advanced A/B Testing Protocol for BTC Trading Signals
======================================================

Extends base A/B protocol with robustness metrics and anti-snooping detection.
Implements comprehensive statistical validation and bias correction.

Features:
- Robustness metrics: Sortino, Ulcer Index, Probabilistic Sharpe
- Anti-snooping: AIC/BIC, White's Reality Check, Bonferroni correction
- Bootstrap confidence intervals
- Multi-armed bandit variant for dynamic testing
- Comprehensive reporting with CI bounds

Usage:
    from src.ab_advanced import advanced_ab_test
    results = advanced_ab_test(variant_a, variant_b, df_5m)
"""

from src.robustness_snooping import RobustnessAnalyzer
from src.ab_base_protocol import ABTestingBase
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, bootstrap

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dependencies

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedABTesting(ABTestingBase):
    """Advanced A/B testing with robustness and anti-snooping"""

    def __init__(self, capital: float = 10000, slippage: float = 0.001, commission: float = 0.0005):
        super().__init__(capital, slippage, commission)
        self.robustness_analyzer = RobustnessAnalyzer()

    def calculate_robustness_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive robustness metrics

        Args:
            returns: Series of returns

        Returns:
            Dictionary of robustness metrics
        """
        if len(returns) < 2:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'ulcer_index': 0,
                'probabilistic_sharpe': 0,
                'var_95': 0,
                'information_ratio': 0
            }

        # Sharpe Ratio
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino Ratio (downside deviation)
        risk_free_rate = 0.04 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate

        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0

        sortino = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Ulcer Index
        cumulative_returns = (1 + returns).cumprod()
        max_returns = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - max_returns) / max_returns
        ulcer = np.sqrt((drawdowns ** 2).mean()) if len(drawdowns) > 0 else 0

        # Probabilistic Sharpe Ratio (bootstrap)
        def sharpe_statistic(data):
            if len(data) < 2:
                return 0
            return data.mean() / data.std() * np.sqrt(252) if data.std() > 0 else 0

        try:
            boot_result = bootstrap((returns,), sharpe_statistic,
                                    n_resamples=1000, confidence_level=0.95)
            prob_sharpe = np.mean(boot_result.bootstrap_distribution > 0)
        except Exception:
            prob_sharpe = 0.5  # Default neutral

        # VaR 95%
        var_95 = np.percentile(returns, 5)

        # Information Ratio (vs buy-hold BTC, approximated as 0 return)
        benchmark_returns = pd.Series([0] * len(returns))  # Simplified buy-hold
        tracking_error = (returns - benchmark_returns).std()
        information_ratio = returns.mean() / tracking_error if tracking_error > 0 else 0

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'ulcer_index': ulcer,
            'probabilistic_sharpe': prob_sharpe,
            'var_95': var_95,
            'information_ratio': information_ratio
        }

    def anti_snooping_analysis(self, results_a: Dict, results_b: Dict,
                               n_null_tests: int = 500) -> Dict[str, Any]:
        """
        Comprehensive anti-snooping analysis

        Args:
            results_a: Results from variant A
            results_b: Results from variant B
            n_null_tests: Number of null hypothesis tests

        Returns:
            Anti-snooping analysis results
        """
        returns_a = results_a['returns']
        returns_b = results_b['returns']

        # AIC calculation (simplified for trading returns)
        def calculate_aic(returns):
            if len(returns) < 2:
                return float('inf')
            # Simplified AIC for returns series
            n = len(returns)
            k = 3  # Assume 3 parameters (simplified)
            log_likelihood = -n / 2 * np.log(returns.var()) if returns.var() > 0 else 0
            aic = 2 * k - 2 * log_likelihood
            return aic

        aic_a = calculate_aic(returns_a)
        aic_b = calculate_aic(returns_b)
        baseline_aic = calculate_aic(pd.Series(np.random.normal(0, 0.01, len(returns_a))))

        aic_snooping = min(aic_a, aic_b) > baseline_aic + 10

        # White's Reality Check
        null_p_values = []
        for _ in range(n_null_tests):
            # Generate null strategy (random returns)
            null_returns = np.random.normal(0, returns_a.std(), len(returns_a))
            try:
                _, p_null = ttest_rel(returns_a, null_returns)
                null_p_values.append(p_null)
            except Exception:
                null_p_values.append(1.0)

        if null_p_values:
            whites_adj_p = np.min(null_p_values) * n_null_tests
            whites_snooping = whites_adj_p >= 0.05
        else:
            whites_adj_p = 1.0
            whites_snooping = False

        # Bonferroni correction
        n_tests = 5  # Assume 5 metrics tested
        original_p = self.statistical_analysis(results_a, results_b)['p_value']
        bonferroni_p = min(original_p * n_tests, 1.0)
        bonferroni_significant = bonferroni_p < 0.05

        # Bootstrap CI for win rate
        def win_rate_stat(data):
            return (data > 0).mean()

        ci_a = None
        ci_b = None
        try:
            ci_a = bootstrap((returns_a,), win_rate_stat, n_resamples=1000, confidence_level=0.95)
            ci_b = bootstrap((returns_b,), win_rate_stat, n_resamples=1000, confidence_level=0.95)

            ci_overlap = not (ci_a.confidence_interval.high < ci_b.confidence_interval.low or
                              ci_b.confidence_interval.high < ci_a.confidence_interval.low)
        except Exception:
            ci_overlap = True

        snooping_detected = aic_snooping or whites_snooping or ci_overlap

        return {
            'aic_snooping': aic_snooping,
            'aic_values': {'a': aic_a, 'b': aic_b, 'baseline': baseline_aic},
            'whites_reality_check': {
                'adj_p_value': whites_adj_p,
                'snooping_detected': whites_snooping
            },
            'bonferroni_correction': {
                'original_p': original_p,
                'corrected_p': bonferroni_p,
                'significant': bonferroni_significant
            },
            'bootstrap_ci': {
                'ci_a': ci_a.confidence_interval if ci_a is not None else (0, 0),
                'ci_b': ci_b.confidence_interval if ci_b is not None else (0, 0),
                'overlap': ci_overlap
            },
            'overall_snooping_detected': snooping_detected
        }

    def multi_armed_bandit_test(self,
                                variant_a,
                                variant_b,
                                df: pd.DataFrame,
                                n_rounds: int = 10,
                                exploration_rate: float = 0.1) -> Dict[str,
                                                                       Any]:
        """
        Multi-armed bandit approach for dynamic A/B testing

        Args:
            variant_a: Strategy function for variant A
            variant_b: Strategy function for variant B
            df: DataFrame with data
            n_rounds: Number of testing rounds
            exploration_rate: Probability of exploring (trying worse variant)

        Returns:
            MAB test results
        """
        logger.info(f"Running multi-armed bandit test with {n_rounds} rounds")

        # Initialize bandit arms
        arms = {
            'A': {
                'strategy': variant_a,
                'name': 'Variant A',
                'wins': 0,
                'trials': 0,
                'returns': []},
            'B': {
                'strategy': variant_b,
                'name': 'Variant B',
                'wins': 0,
                'trials': 0,
                'returns': []}}

        results_history = []

        for round_num in range(n_rounds):
            # Epsilon-greedy selection
            if np.random.random() < exploration_rate:
                # Explore: choose random arm
                chosen_arm = np.random.choice(['A', 'B'])
            else:
                # Exploit: choose best performing arm
                win_rates = {arm: arms[arm]['wins'] / max(arms[arm]['trials'], 1)
                             for arm in arms}
                chosen_arm = max(win_rates, key=lambda x: win_rates[x])

            # Run test for chosen arm
            arm_data = arms[chosen_arm]

            # Split data for this round
            df_test, _ = self.split_data_random(df, split_ratio=0.3)  # Small test set

            # Generate signals and run backtest
            try:
                signals = arm_data['strategy'](df_test)  # Assume strategy returns signals
                results = self.run_backtest_vectorbt(
                    df_test, signals, f"MAB Round {round_num+1} {chosen_arm}")

                # Update arm statistics
                arm_data['trials'] += 1
                if results['sharpe_ratio'] > 0:  # Simple win condition
                    arm_data['wins'] += 1
                arm_data['returns'].append(results['total_return'])

                results_history.append({
                    'round': round_num + 1,
                    'chosen_arm': chosen_arm,
                    'sharpe': results['sharpe_ratio'],
                    'return': results['total_return']
                })

            except Exception as e:
                logger.warning(f"MAB round {round_num+1} failed: {e}")
                continue

        # Final analysis
        final_win_rates = {arm: arms[arm]['wins'] / max(arms[arm]['trials'], 1)
                           for arm in arms}
        best_arm = max(final_win_rates, key=lambda x: final_win_rates[x])

        return {
            'final_arm_stats': arms,
            'win_rates': final_win_rates,
            'best_arm': best_arm,
            'results_history': results_history,
            'total_rounds': n_rounds
        }

    def comprehensive_ab_analysis(self, results_a: Dict, results_b: Dict) -> Dict[str, Any]:
        """
        Run comprehensive A/B analysis with all advanced metrics

        Args:
            results_a: Results from variant A
            results_b: Results from variant B

        Returns:
            Comprehensive analysis results
        """
        # Base statistical analysis
        base_stats = self.statistical_analysis(results_a, results_b)

        # Robustness metrics
        robustness_a = self.calculate_robustness_metrics(results_a['returns'])
        robustness_b = self.calculate_robustness_metrics(results_b['returns'])

        # Anti-snooping analysis
        snooping_analysis = self.anti_snooping_analysis(results_a, results_b)

        # Bootstrap confidence intervals for key metrics
        def metric_bootstrap(data, metric_func):
            try:
                result = bootstrap((data,), metric_bootstrap,
                                   n_resamples=1000, confidence_level=0.95)
                return result.confidence_interval
            except Exception:
                return (0, 0)

        sharpe_ci_a = self._bootstrap_metric(
            results_a['returns'],
            lambda x: x.mean() /
            x.std() *
            np.sqrt(252) if x.std() > 0 else 0)
        sharpe_ci_b = self._bootstrap_metric(
            results_b['returns'],
            lambda x: x.mean() /
            x.std() *
            np.sqrt(252) if x.std() > 0 else 0)

        # Comprehensive decision making
        decision = self._advanced_decision_making(
            base_stats, robustness_a, robustness_b, snooping_analysis)

        return {
            'base_statistics': base_stats,
            'robustness_metrics': {
                'variant_a': robustness_a,
                'variant_b': robustness_b
            },
            'anti_snooping': snooping_analysis,
            'bootstrap_ci': {
                'sharpe_a': sharpe_ci_a,
                'sharpe_b': sharpe_ci_b
            },
            'decision': decision
        }

    def _bootstrap_metric(self, data: pd.Series, metric_func,
                          n_boot: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a metric"""
        try:
            boot_stats = []
            for _ in range(n_boot):
                sample = data.sample(len(data), replace=True)
                stat = metric_func(sample)
                boot_stats.append(stat)

            return tuple(np.percentile(boot_stats, [2.5, 97.5]))
        except Exception:
            return (0, 0)

    def _advanced_decision_making(self, base_stats: Dict, robust_a: Dict,
                                  robust_b: Dict, snooping: Dict) -> Dict[str, Any]:
        """
        Advanced decision making considering all factors
        """
        # Base criteria
        p_value = base_stats['p_value']
        superiority = base_stats['superiority_percentage']

        # Robustness criteria
        sortino_diff = robust_b['sortino_ratio'] - robust_a['sortino_ratio']
        ulcer_diff = robust_a['ulcer_index'] - robust_b['ulcer_index']  # Lower is better
        prob_sharpe_diff = robust_b['probabilistic_sharpe'] - robust_a['probabilistic_sharpe']

        # Snooping check
        snooping_detected = snooping['overall_snooping_detected']

        # Decision logic
        if snooping_detected:
            recommendation = 'SNOOPING_DETECTED'
            reason = 'Anti-snooping analysis detected potential bias. Results unreliable.'
            confidence = 0.1
        elif p_value < 0.05 and superiority > 0.6 and sortino_diff > 0.2 and prob_sharpe_diff > 0.1:
            recommendation = 'ADOPT_B_STRONG'
            reason = f"B shows strong superiority (p={p_value:.3f}, superiority={superiority:.1%}, robust metrics positive)"
            confidence = 0.9
        elif p_value < 0.1 and (sortino_diff > 0.1 or prob_sharpe_diff > 0.05):
            recommendation = 'ADOPT_B_WEAK'
            reason = "B shows moderate superiority with robustness confirmation"
            confidence = 0.7
        elif ulcer_diff > 0.05:  # B has significantly lower ulcer
            recommendation = 'ADOPT_B_LOW_RISK'
            reason = "B shows better risk-adjusted performance (lower ulcer index)"
            confidence = 0.6
        else:
            recommendation = 'KEEP_A'
            reason = "No significant advantage for B after comprehensive analysis"
            confidence = 0.8

        return {
            'recommendation': recommendation,
            'reason': reason,
            'confidence': confidence,
            'key_metrics': {
                'p_value': p_value,
                'superiority': superiority,
                'sortino_diff': sortino_diff,
                'ulcer_diff': ulcer_diff,
                'prob_sharpe_diff': prob_sharpe_diff,
                'snooping_detected': snooping_detected
            }
        }


def advanced_ab_test(variant_a_signals: pd.Series, variant_b_signals: pd.Series,
                     df_5m: pd.DataFrame, variant_a_name: str = "Variant A",
                     variant_b_name: str = "Variant B") -> Dict[str, Any]:
    """
    Main advanced A/B testing function

    Args:
        variant_a_signals: Signals for variant A
        variant_b_signals: Signals for variant B
        df_5m: 5-minute OHLCV DataFrame
        variant_a_name: Name for variant A
        variant_b_name: Name for variant B

    Returns:
        Comprehensive A/B test results
    """
    logger.info(f"Starting advanced A/B test: {variant_a_name} vs {variant_b_name}")

    # Initialize advanced protocol
    protocol = AdvancedABTesting()

    # Step 1: Data split
    df_a, df_b = protocol.split_data_random(df_5m)

    # Step 2: Parallel backtesting
    logger.info("Running parallel backtests...")
    results_a = protocol.run_backtest_vectorbt(
        df_a, variant_a_signals.loc[df_a.index], variant_a_name)
    results_b = protocol.run_backtest_vectorbt(
        df_b, variant_b_signals.loc[df_b.index], variant_b_name)

    # Step 3: Comprehensive analysis
    analysis = protocol.comprehensive_ab_analysis(results_a, results_b)

    # Step 4: Generate report
    report = _generate_advanced_report(
        results_a,
        results_b,
        analysis,
        variant_a_name,
        variant_b_name)

    # Compile results
    results = {
        'timestamp': pd.Timestamp.now(),
        'variants': {
            'a': {'name': variant_a_name, 'results': results_a},
            'b': {'name': variant_b_name, 'results': results_b}
        },
        'analysis': analysis,
        'report': report,
        'data_summary': {
            'total_bars': len(df_5m),
            'bars_a': len(df_a),
            'bars_b': len(df_b)
        }
    }

    logger.info(f"Advanced A/B test completed: {analysis['decision']['recommendation']}")

    return results


def _generate_advanced_report(results_a: Dict, results_b: Dict,
                              analysis: Dict, name_a: str, name_b: str) -> str:
    """Generate comprehensive analysis report"""
    report = f"""
ADVANCED A/B TESTING REPORT
{'='*50}

VARIANTS:
- A: {name_a}
- B: {name_b}

BASE STATISTICS:
- P-value: {analysis['base_statistics']['p_value']:.4f}
- Significant: {analysis['base_statistics']['significant']}
- Superiority: {analysis['base_statistics']['superiority_percentage']:.1%}
- Effect Size: {analysis['base_statistics']['cohens_d']:.3f} ({analysis['base_statistics']['effect_size_interpretation']})

ROBUSTNESS METRICS:
{name_a}:
  - Sortino: {analysis['robustness_metrics']['variant_a']['sortino_ratio']:.3f}
  - Ulcer Index: {analysis['robustness_metrics']['variant_a']['ulcer_index']:.3f}
  - Prob Sharpe: {analysis['robustness_metrics']['variant_a']['probabilistic_sharpe']:.1%}
  - VaR 95%: {analysis['robustness_metrics']['variant_a']['var_95']:.2%}

{name_b}:
  - Sortino: {analysis['robustness_metrics']['variant_b']['sortino_ratio']:.3f}
  - Ulcer Index: {analysis['robustness_metrics']['variant_b']['ulcer_index']:.3f}
  - Prob Sharpe: {analysis['robustness_metrics']['variant_b']['probabilistic_sharpe']:.1%}
  - VaR 95%: {analysis['robustness_metrics']['variant_b']['var_95']:.2%}

ANTI-SNOOPING ANALYSIS:
- AIC Snooping: {analysis['anti_snooping']['aic_snooping']}
- White's Reality Check: Adj p = {analysis['anti_snooping']['whites_reality_check']['adj_p_value']:.4f}
- Bonferroni Correction: p = {analysis['anti_snooping']['bonferroni_correction']['corrected_p']:.4f}
- Overall Snooping Detected: {analysis['anti_snooping']['overall_snooping_detected']}

DECISION: {analysis['decision']['recommendation']}
Confidence: {analysis['decision']['confidence']:.1%}
Reason: {analysis['decision']['reason']}
"""

    return report


if __name__ == "__main__":
    print("Advanced A/B Testing Protocol")
    print("=" * 40)
    print("Ready for comprehensive A/B testing with robustness and anti-snooping analysis.")
