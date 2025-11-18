"""
Multi-Timeframe Impact Analysis for Squeeze ADX TTM Strategy
Evaluates how higher timeframe trends affect 5-minute signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from scripts.parameter_importance_analyzer import ParameterImportanceAnalyzer
import json
from datetime import datetime
import matplotlib.pyplot as plt


class MultiTimeframeImpactAnalyzer:
    """Analyzes the impact of multi-timeframe confirmation on trading signals"""

    def __init__(self, strategy: SqueezeMomentumADXTTMStrategy):
        self.strategy = strategy
        self.analyzer = ParameterImportanceAnalyzer(strategy)

    def _resample_to_higher_tf(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample 5-minute data to higher timeframe"""
        # Set datetime index if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='5Min')

        # Resample based on target timeframe
        if target_tf == '15Min':
            resampled = df.resample('15Min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_tf == '1H':
            resampled = df.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_tf == '4H':
            resampled = df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        else:
            raise ValueError(f"Unsupported timeframe: {target_tf}")

        return resampled.dropna()

    def _calculate_trend_alignment(self, df_5min: pd.DataFrame,
                                 df_higher: pd.DataFrame) -> pd.Series:
        """Calculate trend alignment between timeframes"""
        # Calculate EMAs for trend direction
        df_higher = df_higher.copy()
        df_higher['ema_fast'] = df_higher['close'].ewm(span=20).mean()
        df_higher['ema_slow'] = df_higher['close'].ewm(span=50).mean()
        df_higher['trend'] = np.where(df_higher['ema_fast'] > df_higher['ema_slow'], 1,
                                    np.where(df_higher['ema_fast'] < df_higher['ema_slow'], -1, 0))

        # Forward fill trend to 5-minute timeframe
        trend_5min = df_higher['trend'].reindex(df_5min.index, method='ffill').fillna(0)

        return trend_5min

    def analyze_timeframe_impact(self, df_5min: pd.DataFrame,
                               higher_timeframes: List[str] = ['15Min', '1H']) -> Dict:
        """
        Analyze how higher timeframes affect 5-minute trading signals

        Args:
            df_5min: 5-minute OHLCV data
            higher_timeframes: List of higher timeframes to analyze

        Returns:
            Dict with analysis results
        """
        print("Analyzing multi-timeframe impact on Squeeze ADX TTM strategy...")

        results = {}

        # Create multi-timeframe data dict
        df_multi_tf = {'5Min': df_5min}

        # Add higher timeframes
        for tf in higher_timeframes:
            df_higher = self._resample_to_higher_tf(df_5min, tf)
            df_multi_tf[tf] = df_higher

            # Calculate trend alignment
            trend_alignment = self._calculate_trend_alignment(df_5min, df_higher)
            df_multi_tf[f'{tf}_trend'] = trend_alignment

        # Test different multi-timeframe weight combinations
        weight_combinations = [
            {'higher_tf_weight': 0.0, 'lower_tf_weight': 0.0},  # No multi-TF
            {'higher_tf_weight': 0.2, 'lower_tf_weight': 0.1},  # Light confirmation
            {'higher_tf_weight': 0.4, 'lower_tf_weight': 0.2},  # Moderate confirmation
            {'higher_tf_weight': 0.6, 'lower_tf_weight': 0.3},  # Strong confirmation
            {'higher_tf_weight': 0.8, 'lower_tf_weight': 0.4},  # Very strong confirmation
        ]

        base_params = self.strategy.get_parameters()

        for i, weights in enumerate(weight_combinations):
            print(f"\nTesting weight combination {i+1}: {weights}")

            # Update strategy parameters
            test_params = base_params.copy()
            test_params.update(weights)
            self.strategy.set_parameters(test_params)

            # Generate signals
            signals_dict = self.strategy.generate_signals(df_multi_tf)
            signals = signals_dict['signals']

            # Calculate performance
            if not signals.empty:
                metrics = self.analyzer._calculate_performance_metrics(signals, df_5min['close'])

                results[f'weights_{i+1}'] = {
                    'weights': weights,
                    'metrics': metrics,
                    'signal_count': len(signals[signals != 0])
                }

                print(f"  Signals: {results[f'weights_{i+1}']['signal_count']}")
                print(".4f")
                print(".2%")

        return results

    def analyze_regime_impact(self, df_5min: pd.DataFrame) -> Dict:
        """
        Analyze strategy performance across different market regimes
        """
        print("\nAnalyzing performance across market regimes...")

        # Identify market regimes using ADX and trend
        df_analysis = df_5min.copy()

        # Calculate ADX for regime detection
        self.strategy._calculate_adx(df_analysis)

        # Define regimes
        df_analysis['regime'] = 'sideways'
        df_analysis.loc[df_analysis['adx'] > 25, 'regime'] = 'trending'
        df_analysis.loc[df_analysis['adx'] > 35, 'regime'] = 'strong_trend'

        # Test strategy in different regimes
        regimes = ['sideways', 'trending', 'strong_trend']
        regime_results = {}

        base_params = self.strategy.get_parameters()

        for regime in regimes:
            print(f"\nAnalyzing {regime} regime...")

            # Filter data for this regime
            regime_data = df_analysis[df_analysis['regime'] == regime]

            if len(regime_data) < 100:  # Skip if too little data
                print(f"  Insufficient data for {regime} regime")
                continue

            # Create multi-TF data for this regime
            df_multi_tf = {'5Min': regime_data}

            # Generate signals
            signals_dict = self.strategy.generate_signals(df_multi_tf)
            signals = signals_dict['signals']

            if not signals.empty:
                metrics = self.analyzer._calculate_performance_metrics(signals, regime_data['close'])

                regime_results[regime] = {
                    'metrics': metrics,
                    'data_points': len(regime_data),
                    'signal_count': len(signals[signals != 0])
                }

                print(f"  Data points: {regime_results[regime]['data_points']}")
                print(f"  Signals: {regime_results[regime]['signal_count']}")
                print(".4f")
                print(".2%")

        return regime_results

    def generate_timeframe_report(self, weight_results: Dict,
                                regime_results: Dict) -> str:
        """Generate comprehensive timeframe impact report"""
        report = []
        report.append("# Multi-Timeframe Impact Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Weight analysis
        report.append("## Multi-Timeframe Weight Impact")
        report.append("")
        report.append("| Weight Config | Total Return | Win Rate | Signals | Sharpe |")
        report.append("|---------------|--------------|----------|---------|--------|")

        for key, result in weight_results.items():
            weights = result['weights']
            metrics = result['metrics']
            report.append(f"| {weights['higher_tf_weight']}/{weights['lower_tf_weight']} | "
                         f"{metrics['total_return']:.4f} | "
                         f"{metrics['win_rate']:.2%} | "
                         f"{result['signal_count']} | "
                         f"{metrics['sharpe_ratio']:.2f} |")

        report.append("")

        # Find optimal weights
        best_result = max(weight_results.values(),
                         key=lambda x: x['metrics']['total_return'])

        report.append("**Optimal Configuration:** "
                     f"Higher TF Weight: {best_result['weights']['higher_tf_weight']}, "
                     f"Lower TF Weight: {best_result['weights']['lower_tf_weight']}")
        report.append(f"**Best Return:** {best_result['metrics']['total_return']:.4f}")
        report.append("")

        # Regime analysis
        report.append("## Performance by Market Regime")
        report.append("")
        report.append("| Regime | Total Return | Win Rate | Signals | Data Points |")
        report.append("|--------|--------------|----------|---------|-------------|")

        for regime, result in regime_results.items():
            metrics = result['metrics']
            report.append(f"| {regime.title()} | "
                         f"{metrics['total_return']:.4f} | "
                         f"{metrics['win_rate']:.2%} | "
                         f"{result['signal_count']} | "
                         f"{result['data_points']} |")

        report.append("")

        # Best performing regime
        if regime_results:
            best_regime = max(regime_results.keys(),
                            key=lambda x: regime_results[x]['metrics']['total_return'])
            best_metrics = regime_results[best_regime]['metrics']
            report.append(f"**Best Regime:** {best_regime.title()} "
                         f"(Return: {best_metrics['total_return']:.4f}, "
                         f"Win Rate: {best_metrics['win_rate']:.2%})")

        # Recommendations
        report.append("")
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Optimal Multi-TF Weights:** Use the configuration that maximizes return while maintaining acceptable win rate")
        report.append("2. **Regime Adaptation:** Consider adjusting strategy parameters based on detected market regime")
        report.append("3. **Signal Filtering:** Higher timeframe confirmation reduces false signals but may miss opportunities")
        report.append("4. **Further Analysis:** Test with different assets and market conditions")

        return "\n".join(report)

    def create_visualizations(self, weight_results: Dict, regime_results: Dict,
                            save_path: str = None) -> None:
        """Create visualizations for the analysis results"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"multitimeframe_analysis_{timestamp}"

        # Multi-TF weight impact chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        weights = [f"{r['weights']['higher_tf_weight']}/{r['weights']['lower_tf_weight']}"
                  for r in weight_results.values()]
        returns = [r['metrics']['total_return'] for r in weight_results.values()]
        win_rates = [r['metrics']['win_rate'] for r in weight_results.values()]
        signal_counts = [r['signal_count'] for r in weight_results.values()]
        sharpes = [r['metrics']['sharpe_ratio'] for r in weight_results.values()]

        ax1.bar(weights, returns)
        ax1.set_title('Total Return by Multi-TF Weight Configuration')
        ax1.set_ylabel('Total Return')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(weights, win_rates)
        ax2.set_title('Win Rate by Multi-TF Weight Configuration')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=45)

        ax3.bar(weights, signal_counts)
        ax3.set_title('Signal Count by Multi-TF Weight Configuration')
        ax3.set_ylabel('Number of Signals')
        ax3.tick_params(axis='x', rotation=45)

        ax4.bar(weights, sharpes)
        ax4.set_title('Sharpe Ratio by Multi-TF Weight Configuration')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{save_path}_weights.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Regime performance chart
        if regime_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            regimes = list(regime_results.keys())
            regime_returns = [r['metrics']['total_return'] for r in regime_results.values()]
            regime_win_rates = [r['metrics']['win_rate'] for r in regime_results.values()]

            ax1.bar(regimes, regime_returns)
            ax1.set_title('Performance by Market Regime')
            ax1.set_ylabel('Total Return')

            ax2.bar(regimes, regime_win_rates)
            ax2.set_title('Win Rate by Market Regime')
            ax2.set_ylabel('Win Rate')

            plt.tight_layout()
            plt.savefig(f"{save_path}_regimes.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizations saved to {save_path}_*.png")


def run_multitimeframe_analysis(strategy: SqueezeMomentumADXTTMStrategy,
                              df_5min: pd.DataFrame) -> None:
    """
    Run complete multi-timeframe impact analysis

    Args:
        strategy: The strategy to analyze
        df_5min: 5-minute OHLCV market data
    """
    analyzer = MultiTimeframeImpactAnalyzer(strategy)

    # Analyze multi-timeframe weight impact
    weight_results = analyzer.analyze_timeframe_impact(df_5min)

    # Analyze regime impact
    regime_results = analyzer.analyze_regime_impact(df_5min)

    # Generate report
    report = analyzer.generate_timeframe_report(weight_results, regime_results)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'weight_analysis': weight_results,
        'regime_analysis': regime_results
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"multitimeframe_analysis_{timestamp}.json"

    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save report
    report_filename = f"multitimeframe_report_{timestamp}.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)

    # Create visualizations
    analyzer.create_visualizations(weight_results, regime_results)

    # Print report
    print("\n" + "="*80)
    print(report)
    print("="*80)

    print(f"\nResults saved to {results_filename}")
    print(f"Report saved to {report_filename}")


if __name__ == "__main__":
    # Example usage
    strategy = SqueezeMomentumADXTTMStrategy()

    # This would be called with actual market data
    # run_multitimeframe_analysis(strategy, df_5min)
    print("Multi-Timeframe Impact Analyzer ready.")
    print("Call run_multitimeframe_analysis() with strategy and 5-minute market data.")