"""
Parameter Importance Analysis for Squeeze ADX TTM Strategy
Evaluates the impact of each parameter on strategy performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import itertools
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime


class ParameterImportanceAnalyzer:
    """Analyzes parameter importance through systematic backtesting"""

    def __init__(self, base_strategy: SqueezeMomentumADXTTMStrategy):
        self.base_strategy = base_strategy
        self.results = {}

    def _calculate_performance_metrics(self, signals: pd.Series, prices: pd.Series) -> Dict:
        """Calculate performance metrics for a signal series"""
        # Simple performance calculation
        returns = []
        position = 0
        entry_price = 0

        for i, signal in enumerate(signals):
            if signal == 1 and position == 0:  # Buy
                position = 1
                entry_price = prices.iloc[i]
            elif signal == -1 and position == 1:  # Sell
                if entry_price > 0:
                    ret = (prices.iloc[i] - entry_price) / entry_price
                    returns.append(ret)
                position = 0
                entry_price = 0

        if not returns:
            return {
                'total_return': 0,
                'win_rate': 0,
                'total_trades': 0,
                'avg_trade': 0,
                'sharpe_ratio': 0
            }

        returns = np.array(returns)
        wins = returns > 0

        return {
            'total_return': np.sum(returns),
            'win_rate': np.mean(wins) if len(wins) > 0 else 0,
            'total_trades': len(returns),
            'avg_trade': np.mean(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }

    def analyze_parameter_importance(self, df_multi_tf: Dict[str, pd.DataFrame],
                                   parameter_ranges: Dict) -> Dict:
        """
        Analyze importance of each parameter by testing different values

        Args:
            df_multi_tf: Multi-timeframe data
            parameter_ranges: Dict with parameter names and list of values to test

        Returns:
            Dict with parameter importance analysis
        """
        base_params = self.base_strategy.get_parameters()
        results = {}

        print("Starting parameter importance analysis...")

        for param_name, param_values in parameter_ranges.items():
            print(f"\nAnalyzing parameter: {param_name}")
            param_results = []

            for value in param_values:
                # Create parameter set with this value
                test_params = base_params.copy()
                test_params[param_name] = value

                # Update strategy parameters
                self.base_strategy.set_parameters(test_params)

                # Generate signals
                signals_dict = self.base_strategy.generate_signals(df_multi_tf)
                signals = signals_dict['signals']

                # Calculate performance
                if not signals.empty and '5Min' in df_multi_tf:
                    prices = df_multi_tf['5Min']['close']
                    metrics = self._calculate_performance_metrics(signals, prices)

                    param_results.append({
                        'value': value,
                        'metrics': metrics
                    })

                    print(f"  Value {value}: Return={metrics['total_return']:.4f}, "
                          f"Win Rate={metrics['win_rate']:.2%}, Trades={metrics['total_trades']}")

            results[param_name] = param_results

        return results

    def analyze_multi_tf_impact(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze the impact of multi-timeframe confirmation
        Compares performance with and without higher timeframe confirmation
        """
        print("\nAnalyzing multi-timeframe impact...")

        base_params = self.base_strategy.get_parameters()

        # Test without multi-timeframe
        self.base_strategy.set_parameters({**base_params, 'higher_tf_weight': 0, 'lower_tf_weight': 0})
        signals_no_mtf = self.base_strategy.generate_signals(df_multi_tf)['signals']

        # Test with multi-timeframe
        self.base_strategy.set_parameters(base_params)
        signals_with_mtf = self.base_strategy.generate_signals(df_multi_tf)['signals']

        results = {}

        if '5Min' in df_multi_tf:
            prices = df_multi_tf['5Min']['close']

            # Performance without multi-timeframe
            metrics_no_mtf = self._calculate_performance_metrics(signals_no_mtf, prices)

            # Performance with multi-timeframe
            metrics_with_mtf = self._calculate_performance_metrics(signals_with_mtf, prices)

            results = {
                'without_multi_tf': metrics_no_mtf,
                'with_multi_tf': metrics_with_mtf,
                'improvement': {
                    'total_return': metrics_with_mtf['total_return'] - metrics_no_mtf['total_return'],
                    'win_rate': metrics_with_mtf['win_rate'] - metrics_no_mtf['win_rate'],
                    'total_trades': metrics_with_mtf['total_trades'] - metrics_no_mtf['total_trades']
                }
            }

            print("Without multi-TF: "
                  f"Return={metrics_no_mtf['total_return']:.4f}, "
                  f"Win Rate={metrics_no_mtf['win_rate']:.2%}")
            print("With multi-TF: "
                  f"Return={metrics_with_mtf['total_return']:.4f}, "
                  f"Win Rate={metrics_with_mtf['win_rate']:.2%}")
            print("Improvement: "
                  f"Return={results['improvement']['total_return']:.4f}, "
                  f"Win Rate={results['improvement']['win_rate']:.2%}")

        return results

    def generate_parameter_report(self, importance_results: Dict,
                                multi_tf_results: Dict) -> str:
        """Generate a comprehensive parameter importance report"""
        report = []
        report.append("# Parameter Importance Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Multi-timeframe impact
        report.append("## Multi-Timeframe Confirmation Impact")
        report.append("")
        if multi_tf_results:
            improvement = multi_tf_results.get('improvement', {})
            report.append(".4f")
            report.append(".2%")
            report.append(f"- **Trade Count Change**: {improvement.get('total_trades', 0)}")
        report.append("")

        # Parameter importance
        report.append("## Parameter Importance Results")
        report.append("")

        for param_name, param_results in importance_results.items():
            report.append(f"### {param_name}")
            report.append("")

            if param_results:
                # Find best and worst performance
                best_result = max(param_results,
                                key=lambda x: x['metrics']['total_return'])
                worst_result = min(param_results,
                                 key=lambda x: x['metrics']['total_return'])

                report.append(f"**Best Value**: {best_result['value']} "
                             f"(Return: {best_result['metrics']['total_return']:.4f}, "
                             f"Win Rate: {best_result['metrics']['win_rate']:.2%})")
                report.append(f"**Worst Value**: {worst_result['value']} "
                             f"(Return: {worst_result['metrics']['total_return']:.4f}, "
                             f"Win Rate: {worst_result['metrics']['win_rate']:.2%})")
                report.append("")

                # Detailed results table
                report.append("| Value | Total Return | Win Rate | Total Trades | Sharpe |")
                report.append("|-------|--------------|----------|--------------|--------|")

                for result in sorted(param_results, key=lambda x: x['value']):
                    metrics = result['metrics']
                    report.append(f"| {result['value']} | "
                                 f"{metrics['total_return']:.4f} | "
                                 f"{metrics['win_rate']:.2%} | "
                                 f"{metrics['total_trades']} | "
                                 f"{metrics['sharpe_ratio']:.2f} |")

            report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Focus on high-impact parameters** that show significant performance variation")
        report.append("2. **Multi-timeframe confirmation** appears to " +
                     ("improve" if multi_tf_results.get('improvement', {}).get('total_return', 0) > 0
                      else "not improve") + " performance")
        report.append("3. **Parameter optimization** should prioritize parameters with >5% return variation")
        report.append("4. **Further testing** needed with out-of-sample data")

        return "\n".join(report)

    def save_results(self, importance_results: Dict, multi_tf_results: Dict,
                    filename: str = None) -> None:
        """Save analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parameter_importance_analysis_{timestamp}.json"

        results = {
            'timestamp': datetime.now().isoformat(),
            'parameter_importance': importance_results,
            'multi_tf_impact': multi_tf_results
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {filename}")


def run_parameter_importance_analysis(strategy: SqueezeMomentumADXTTMStrategy,
                                     df_multi_tf: Dict[str, pd.DataFrame]) -> None:
    """
    Run complete parameter importance analysis

    Args:
        strategy: The strategy to analyze
        df_multi_tf: Multi-timeframe market data
    """
    analyzer = ParameterImportanceAnalyzer(strategy)

    # Define parameter ranges to test
    parameter_ranges = {
        'bb_length': [10, 15, 20, 25, 30],
        'bb_mult': [1.5, 1.8, 2.0, 2.2, 2.5],
        'kc_length': [10, 15, 20, 25, 30],
        'kc_mult': [1.0, 1.2, 1.5, 1.8, 2.0],
        'adx_length': [7, 14, 21, 28],
        'key_level': [15, 20, 23, 25, 30],
        'squeeze_threshold': [0.1, 0.3, 0.5, 0.7, 0.9],
        'adx_threshold': [15, 20, 25, 30],
        'momentum_threshold': [0.05, 0.1, 0.15, 0.2, 0.25]
    }

    # Analyze parameter importance
    importance_results = analyzer.analyze_parameter_importance(df_multi_tf, parameter_ranges)

    # Analyze multi-timeframe impact
    multi_tf_results = analyzer.analyze_multi_tf_impact(df_multi_tf)

    # Generate and save report
    report = analyzer.generate_parameter_report(importance_results, multi_tf_results)

    # Save results
    analyzer.save_results(importance_results, multi_tf_results)

    # Print report
    print("\n" + "="*80)
    print(report)
    print("="*80)

    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"parameter_importance_report_{timestamp}.md"

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to {report_filename}")


if __name__ == "__main__":
    # Example usage
    strategy = SqueezeMomentumADXTTMStrategy()

    # This would be called with actual market data
    # run_parameter_importance_analysis(strategy, df_multi_tf)
    print("Parameter Importance Analyzer ready.")
    print("Call run_parameter_importance_analysis() with strategy and market data.")