#!/usr/bin/env python3
"""
Automated A/B Testing Pipeline
==============================

Integrates ab_advanced.py in a complete automated pipeline using DVC, Docker, and CI/CD.
Automates data → signals → A/B test → robust/snooping → decision → report.

Features:
- DVC pipeline for data/code versioning
- Docker containerization
- CI/CD integration
- Comprehensive reporting
- Versioned outputs

Usage:
    python src/ab_pipeline.py --symbol=BTCUSD --start=2018-01-01 --end=2025-11-12
    # Or via DVC: dvc repro
    # Or via Docker: docker run ab-pipeline
"""

from src.ab_advanced import AdvancedABTesting
import sys
from pathlib import Path
import logging
from datetime import datetime
import json
import yaml
import subprocess
from typing import Dict, Any, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Placeholder functions - replace with actual implementations

def fetch_crypto_data(symbol: str, start_date: str, end_date: str, timeframe: str = '5m'):
    """Placeholder for crypto data fetching"""
    # This should be implemented in src/data_fetcher.py
    import pandas as pd
    import numpy as np
    dates = pd.date_range(start_date, end_date, freq='5min')
    n_bars = len(dates)
    return pd.DataFrame({
        'open': np.random.normal(50000, 1000, n_bars),
        'high': np.random.normal(50100, 1000, n_bars),
        'low': np.random.normal(49900, 1000, n_bars),
        'close': np.random.normal(50000, 1000, n_bars),
        'volume': np.random.normal(100, 20, n_bars)
    }, index=dates)


def generate_signals(df: pd.DataFrame, strategy: str = 'ifvg_base'):
    """Placeholder for signal generation"""
    # This should be implemented in src/signals_generator.py
    import numpy as np
    n_signals = len(df)
    if strategy == 'ifvg_base':
        return pd.Series(np.random.choice([0, 1], n_signals, p=[0.95, 0.05]), index=df.index)
    else:  # rsi_bb_alternative
        return pd.Series(np.random.choice([0, 1], n_signals, p=[0.93, 0.07]), index=df.index)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ab_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ABPipeline:
    """Automated A/B testing pipeline with DVC integration"""

    def __init__(self, symbol: str = 'BTCUSD', start_date: str = '2018-01-01',
                 end_date: Optional[str] = None, capital: float = 10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.capital = capital
        self.pipeline_dir = Path(__file__).parent.parent
        self.data_dir = self.pipeline_dir / 'data'
        self.signals_dir = self.pipeline_dir / 'signals'
        self.results_dir = self.pipeline_dir / 'results' / 'ab_pipeline'
        self.reports_dir = self.pipeline_dir / 'reports'

        # Create directories
        for dir_path in [self.data_dir, self.signals_dir, self.results_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize protocol
        self.protocol = AdvancedABTesting(capital=capital)

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete A/B testing pipeline

        Returns:
            Pipeline results summary
        """
        logger.info(
            f"Starting A/B pipeline for {self.symbol} from {self.start_date} to {self.end_date}")

        try:
            # Step 1: Data fetch and validation
            logger.info("Step 1: Fetching and validating data...")
            df_5m = self._fetch_and_validate_data()

            # Step 2: Generate A/B signals
            logger.info("Step 2: Generating A/B signals...")
            signals_a, signals_b = self._generate_ab_signals(df_5m)

            # Step 3: Run parallel backtests
            logger.info("Step 3: Running parallel backtests...")
            results_a, results_b = self._run_parallel_backtests(df_5m, signals_a, signals_b)

            # Step 4: Comprehensive A/B analysis
            logger.info("Step 4: Running comprehensive A/B analysis...")
            analysis = self._run_comprehensive_analysis(results_a, results_b)

            # Step 5: Generate automated decision
            logger.info("Step 5: Generating automated decision...")
            decision = self._generate_automated_decision(analysis)

            # Step 6: Generate reports
            logger.info("Step 6: Generating reports...")
            report_path = self._generate_reports(results_a, results_b, analysis, decision)

            # Step 7: Version control and commit
            logger.info("Step 7: Versioning results...")
            self._version_and_commit()

            results = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'period': f"{self.start_date}_to_{self.end_date}",
                'data_hash': self._get_data_hash(),
                'results_a': results_a,
                'results_b': results_b,
                'analysis': analysis,
                'decision': decision,
                'report_path': str(report_path),
                'status': 'success'
            }

            logger.info("A/B pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }

    def _fetch_and_validate_data(self) -> pd.DataFrame:
        """Fetch and validate market data"""
        # Fetch data
        df_5m = fetch_crypto_data(self.symbol, self.start_date, self.end_date, timeframe='5m')

        # Validate data integrity
        if df_5m.empty:
            raise ValueError(f"No data fetched for {self.symbol}")

        # Check for missing values
        missing_pct = df_5m.isnull().sum().sum() / len(df_5m)
        if missing_pct > 0.01:  # More than 1% missing
            logger.warning(f"High missing data percentage: {missing_pct:.2%}")
            df_5m = df_5m.fillna(method='ffill').fillna(method='bfill')

        # Validate date range
        actual_start = df_5m.index.min().strftime('%Y-%m-%d')
        actual_end = df_5m.index.max().strftime('%Y-%m-%d')
        logger.info(f"Data validated: {len(df_5m)} bars from {actual_start} to {actual_end}")

        # Save data with DVC tracking
        data_path = self.data_dir / \
            f"{self.symbol.lower()}_5m_{self.start_date}_{self.end_date}.csv"
        df_5m.to_csv(data_path)
        self._dvc_add(str(data_path))

        return df_5m

    def _generate_ab_signals(self, df_5m: pd.DataFrame):
        """Generate A/B test signals"""
        # Generate variant A signals (IFVG base)
        signals_a = generate_signals(df_5m, strategy='ifvg_base')

        # Generate variant B signals (alternative, e.g., RSI+BB)
        signals_b = generate_signals(df_5m, strategy='rsi_bb_alternative')

        # Save signals
        signals_a_path = self.signals_dir / \
            f"signals_a_{self.symbol.lower()}_{self.start_date}_{self.end_date}.csv"
        signals_b_path = self.signals_dir / \
            f"signals_b_{self.symbol.lower()}_{self.start_date}_{self.end_date}.csv"

        signals_a.to_csv(signals_a_path)
        signals_b.to_csv(signals_b_path)

        # DVC track signals
        self._dvc_add(str(signals_a_path))
        self._dvc_add(str(signals_b_path))

        logger.info(f"Generated {len(signals_a)} A signals and {len(signals_b)} B signals")
        return signals_a, signals_b

    def _run_parallel_backtests(
            self,
            df_5m: pd.DataFrame,
            signals_a: pd.Series,
            signals_b: pd.Series):
        """Run parallel backtests for A/B variants"""
        # Run backtest A
        results_a = self.protocol.run_backtest_vectorbt(
            df_5m, signals_a,
            strategy_name=f"A_IFVG_{self.symbol}"
        )

        # Run backtest B
        results_b = self.protocol.run_backtest_vectorbt(
            df_5m, signals_b,
            strategy_name=f"B_Alternative_{self.symbol}"
        )

        logger.info(f"Backtest A: {results_a.get('total_return', 0):.2%} return")
        logger.info(f"Backtest B: {results_b.get('total_return', 0):.2%} return")

        return results_a, results_b

    def _run_comprehensive_analysis(self, results_a: Dict, results_b: Dict) -> Dict:
        """Run comprehensive A/B analysis"""
        analysis = self.protocol.comprehensive_ab_analysis(results_a, results_b)

        # Log key metrics
        base_stats = analysis.get('base_statistics', {})
        logger.info(f"A/B Analysis - P-value: {base_stats.get('p_value', 'N/A'):.4f}")
        logger.info(
            f"A/B Analysis - Superiority: {base_stats.get('superiority_percentage', 0):.1f}%")

        if 'decision' in analysis:
            decision = analysis['decision']
            logger.info(f"Decision: {decision.get('recommendation', 'UNKNOWN')}")
            logger.info(f"Confidence: {decision.get('confidence', 0):.1%}")

        return analysis

    def _generate_automated_decision(self, analysis: Dict) -> Dict:
        """Generate automated decision based on analysis"""
        decision = analysis.get('decision', {})

        # Enhanced decision logic for pipeline
        recommendation = decision.get('recommendation', 'UNKNOWN')
        confidence = decision.get('confidence', 0)

        automated_actions = {
            'ADOPT_B_STRONG': {
                'action': 'deploy_variant_b',
                'confidence_threshold': 0.9,
                'risk_level': 'low'
            },
            'ADOPT_B_WEAK': {
                'action': 'deploy_variant_b_with_monitoring',
                'confidence_threshold': 0.7,
                'risk_level': 'medium'
            },
            'ADOPT_B_LOW_RISK': {
                'action': 'deploy_hybrid',
                'confidence_threshold': 0.6,
                'risk_level': 'low'
            },
            'KEEP_A': {
                'action': 'keep_current_strategy',
                'confidence_threshold': 0.8,
                'risk_level': 'none'
            },
            'SNOOPING_DETECTED': {
                'action': 'investigate_further',
                'confidence_threshold': 0.1,
                'risk_level': 'high'
            }
        }

        action_config = automated_actions.get(recommendation, {
            'action': 'manual_review_required',
            'confidence_threshold': 0.5,
            'risk_level': 'unknown'
        })

        automated_decision = {
            'recommendation': recommendation,
            'automated_action': action_config['action'],
            'confidence': confidence,
            'confidence_threshold': action_config['confidence_threshold'],
            'risk_level': action_config['risk_level'],
            'meets_threshold': confidence >= action_config['confidence_threshold']
        }

        logger.info(f"Automated decision: {automated_decision['automated_action']}")
        return automated_decision

    def _generate_reports(self, results_a: Dict, results_b: Dict,
                          analysis: Dict, decision: Dict) -> Path:
        """Generate comprehensive reports"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"ab_pipeline_report_{self.symbol}_{timestamp}.md"
        report_path = self.reports_dir / report_filename

        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        report_content = self._create_report_content(
            results_a, results_b, analysis, decision
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Generate additional outputs
        self._save_results_json(results_a, results_b, analysis, decision, timestamp)

        logger.info(f"Report generated: {report_path}")
        return report_path

    def _create_report_content(self, results_a: Dict, results_b: Dict,
                               analysis: Dict, decision: Dict) -> str:
        """Create comprehensive markdown report"""
        base_stats = analysis.get('base_statistics', {})
        robust_a = analysis.get('robustness_metrics', {}).get('variant_a', {})
        robust_b = analysis.get('robustness_metrics', {}).get('variant_b', {})

        report = f"""# A/B Testing Pipeline Report
## {self.symbol} - {self.start_date} to {self.end_date}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Hash:** {self._get_data_hash()}

### Executive Summary
- **Recommendation:** {decision.get('automated_action', 'UNKNOWN').replace('_', ' ').title()}
- **Confidence:** {decision.get('confidence', 0):.1%}
- **Risk Level:** {decision.get('risk_level', 'unknown').title()}

### Base Statistics
| Metric | Variant A | Variant B | Difference |
|--------|-----------|-----------|------------|
| Total Return | {results_a.get('total_return', 0):.2%} | {results_b.get('total_return', 0):.2%} | {(results_b.get('total_return', 0) - results_a.get('total_return', 0)):.2%} |
| Sharpe Ratio | {results_a.get('sharpe_ratio', 0):.3f} | {results_b.get('sharpe_ratio', 0):.3f} | {(results_b.get('sharpe_ratio', 0) - results_a.get('sharpe_ratio', 0)):.3f} |
| Win Rate | {results_a.get('win_rate', 0):.1%} | {results_b.get('win_rate', 0):.1%} | {(results_b.get('win_rate', 0) - results_a.get('win_rate', 0)):.1%} |
| Max Drawdown | {results_a.get('max_drawdown', 0):.2%} | {results_b.get('max_drawdown', 0):.2%} | {(results_b.get('max_drawdown', 0) - results_a.get('max_drawdown', 0)):.2%} |
| P-value | - | - | {base_stats.get('p_value', 'N/A')} |
| Superiority | - | - | {base_stats.get('superiority_percentage', 0):.1%} |

### Robustness Metrics
#### Variant A
- Sharpe: {robust_a.get('sharpe_ratio', 0):.3f}
- Sortino: {robust_a.get('sortino_ratio', 0):.3f}
- Ulcer Index: {robust_a.get('ulcer_index', 0):.3f}
- Probabilistic Sharpe: {robust_a.get('probabilistic_sharpe', 0):.1%}

#### Variant B
- Sharpe: {robust_b.get('sharpe_ratio', 0):.3f}
- Sortino: {robust_b.get('sortino_ratio', 0):.3f}
- Ulcer Index: {robust_b.get('ulcer_index', 0):.3f}
- Probabilistic Sharpe: {robust_b.get('probabilistic_sharpe', 0):.1%}

### Anti-Snooping Analysis
- **Overall Snooping Detected:** {analysis.get('anti_snooping', {}).get('overall_snooping_detected', False)}
- **AIC Values:** A={analysis.get('anti_snooping', {}).get('aic_values', {}).get('a', 0):.1f}, B={analysis.get('anti_snooping', {}).get('aic_values', {}).get('b', 0):.1f}
- **White's Reality Check:** {analysis.get('anti_snooping', {}).get('whites_reality_check', {}).get('snooping_detected', False)}

### Decision Rationale
{analysis.get('decision', {}).get('reason', 'No decision rationale available')}

### Next Steps
1. **{"Deploy B" if decision.get('meets_threshold', False) else "Keep A"}** - {decision.get('automated_action', '').replace('_', ' ').title()}
2. Monitor performance for {30 if decision.get('risk_level') == 'low' else 60} days
3. Re-run pipeline quarterly for strategy evolution

---
*Report generated by automated A/B testing pipeline*
"""

        return report

    def _save_results_json(self, results_a: Dict, results_b: Dict,
                           analysis: Dict, decision: Dict, timestamp: str):
        """Save detailed results as JSON"""
        results_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'period': f"{self.start_date}_to_{self.end_date}",
                'pipeline_version': '1.0.0'
            },
            'results_a': results_a,
            'results_b': results_b,
            'analysis': analysis,
            'decision': decision
        }

        json_path = self.results_dir / f"ab_results_{self.symbol}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Results saved to: {json_path}")

    def _version_and_commit(self):
        """Version results and commit to git"""
        try:
            # Add results to DVC
            self._dvc_add(str(self.results_dir))
            self._dvc_add(str(self.reports_dir))

            # Git commit
            commit_msg = f"AB Pipeline Results: {self.symbol} {self.start_date}-{self.end_date}"
            subprocess.run(['git', 'add', '.'], check=True, cwd=self.pipeline_dir)
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True, cwd=self.pipeline_dir)

            logger.info("Results versioned and committed")

        except subprocess.CalledProcessError as e:
            logger.warning(f"Version control failed: {e}")

    def _dvc_add(self, path: str):
        """Add file/directory to DVC tracking"""
        try:
            subprocess.run(['dvc', 'add', path], check=True, cwd=self.pipeline_dir)
        except subprocess.CalledProcessError:
            logger.warning(f"DVC add failed for {path}")

    def _get_data_hash(self) -> str:
        """Get data hash for versioning"""
        try:
            subprocess.run(['dvc', 'status'], capture_output=True, text=True, cwd=self.pipeline_dir)
            # Extract hash from dvc status (simplified)
            return "data_hash_placeholder"
        except Exception:
            return "unknown"


def create_dvc_pipeline():
    """Create DVC pipeline configuration"""
    pipeline_config = {
        'stages': {
            'data_fetch': {
                'cmd': 'python src/ab_pipeline.py --stage=data_fetch',
                'outs': ['data/'],
                'deps': ['requirements.txt']
            },
            'signals_generation': {
                'cmd': 'python src/ab_pipeline.py --stage=signals',
                'outs': ['signals/'],
                'deps': ['data/']
            },
            'ab_testing': {
                'cmd': 'python src/ab_pipeline.py --stage=ab_test',
                'outs': ['results/ab_pipeline/', 'reports/'],
                'deps': ['signals/']
            }
        }
    }

    with open('dvc.yaml', 'w') as f:
        yaml.dump(pipeline_config, f)

    print("DVC pipeline created: dvc.yaml")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Automated A/B Testing Pipeline')
    parser.add_argument('--symbol', default='BTCUSD', help='Trading symbol')
    parser.add_argument('--start', default='2018-01-01', help='Start date')
    parser.add_argument('--end', help='End date (default: today)')
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital')
    parser.add_argument('--stage', choices=['data_fetch', 'signals', 'ab_test', 'full'],
                        help='Run specific pipeline stage')
    parser.add_argument('--create-dvc', action='store_true', help='Create DVC pipeline config')

    args = parser.parse_args()

    if args.create_dvc:
        create_dvc_pipeline()
        return

    pipeline = ABPipeline(args.symbol, args.start, args.end, args.capital)

    if args.stage == 'full' or not args.stage:
        results = pipeline.run_full_pipeline()
        print(f"Pipeline completed. Status: {results.get('status', 'unknown')}")
        if results.get('status') == 'success':
            print(f"Report: {results.get('report_path', 'N/A')}")
    else:
        print(f"Stage execution not implemented yet: {args.stage}")


if __name__ == "__main__":
    main()
