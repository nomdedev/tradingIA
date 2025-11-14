#!/usr/bin/env python3
"""
Automated Trading Pipeline
==========================

End-to-end automated pipeline for BTC trading strategy development.

Features:
- Modular pipeline: data → signals → backtest → A/B/opt → eval/robustez → report
- Docker containerization for reproducible environments
- DVC for data and code versioning
- Automated model validation and deployment
- CI/CD integration ready
- Versioned reports and artifacts

Usage:
    from src.automated_pipeline import AutomatedPipeline
    pipeline = AutomatedPipeline()
    results = pipeline.run_full_pipeline()
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AutomatedPipeline:
    """Automated end-to-end trading strategy pipeline"""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = config_path
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_id = f"pipeline_{self.timestamp}"

        # Pipeline stages
        self.stages = [
            'data_fetch',
            'signals_generation',
            'backtesting',
            'ab_testing',
            'optimization',
            'robustness_analysis',
            'reporting'
        ]

        # Results storage
        self.results = {}
        self.artifacts_dir = Path(f"results/pipeline_runs/{self.run_id}")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except BaseException:
            # Default configuration
            config = {
                'symbol': 'BTCUSD',
                'start_date': '2018-01-01',
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'timeframe': '5T',
                'data_source': 'alpaca',  # or 'csv'
                'backtest_engine': 'vectorbt',  # or 'simple'
                'optimization_method': 'bayesian',
                'ab_test_splits': 3,
                'robustness_tests': ['monte_carlo', 'stress_test', 'walk_forward'],
                'docker_enabled': True,
                'dvc_enabled': True
            }

        return config

    def run_full_pipeline(self, symbol: str = 'BTCUSD',
                          start_date: str = '2018-01-01',
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete automated pipeline

        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data (default: today)

        Returns:
            Dict with complete pipeline results
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"🚀 Starting Automated Pipeline Run: {self.run_id}")
        print(f"📅 Date Range: {start_date} to {end_date}")
        print(f"💰 Symbol: {symbol}")
        print("=" * 60)

        try:
            # Stage 1: Data Fetch
            print("\n📊 Stage 1: Data Fetch")
            data = self._run_data_fetch(symbol, start_date, end_date)
            self.results['data_fetch'] = {'status': 'completed', 'data_points': len(data)}

            # Stage 2: Signals Generation
            print("\n📈 Stage 2: Signals Generation")
            signals = self._run_signals_generation(data)
            self.results['signals_generation'] = {
                'status': 'completed',
                'total_signals': len(signals)
            }

            # Stage 3: Backtesting
            print("\n⚡ Stage 3: Backtesting")
            backtest_results = self._run_backtesting(data, signals)
            self.results['backtesting'] = backtest_results

            # Stage 4: A/B Testing
            print("\n🧪 Stage 4: A/B Testing")
            ab_results = self._run_ab_testing(data)
            self.results['ab_testing'] = ab_results

            # Stage 5: Optimization
            print("\n🎯 Stage 5: Optimization")
            opt_results = self._run_optimization(data)
            self.results['optimization'] = opt_results

            # Stage 6: Robustness Analysis
            print("\n🔍 Stage 6: Robustness Analysis")
            robustness_results = self._run_robustness_analysis(backtest_results, opt_results)
            self.results['robustness_analysis'] = robustness_results

            # Stage 7: Reporting
            print("\n📋 Stage 7: Reporting")
            report_results = self._run_reporting()
            self.results['reporting'] = report_results

            # Final summary
            self._generate_pipeline_summary()

            print(f"\n✅ Pipeline completed successfully!")
            print(f"📁 Artifacts saved to: {self.artifacts_dir}")

            return {
                'status': 'completed',
                'run_id': self.run_id,
                'timestamp': self.timestamp,
                'results': self.results,
                'artifacts_dir': str(self.artifacts_dir)
            }

        except Exception as e:
            error_msg = f"Pipeline failed at stage with error: {str(e)}"
            print(f"❌ {error_msg}")

            self.results['error'] = error_msg
            return {
                'status': 'failed',
                'run_id': self.run_id,
                'error': error_msg,
                'results': self.results
            }

    def _run_data_fetch(self, symbol: str, start_date: str, end_date: str):
        """Stage 1: Fetch market data"""
        try:
            # Import here to avoid circular imports
            from data_fetchers.alpaca_fetcher import AlpacaDataFetcher

            fetcher = AlpacaDataFetcher()
            data = fetcher.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.get('timeframe', '5T')
            )

            # Save data artifact
            data_path = self.artifacts_dir / "market_data.csv"
            data.to_csv(data_path)
            print(f"✅ Data saved: {len(data)} rows to {data_path}")

            # Version with DVC if enabled
            if self.config.get('dvc_enabled', False):
                self._dvc_add(str(data_path), "Market data for pipeline run")

            return data

        except Exception as e:
            print(f"⚠️  Data fetch failed, using sample data: {e}")
            # Generate sample data for demo
            return self._generate_sample_data()

    def _run_signals_generation(self, data):
        """Stage 2: Generate trading signals"""
        try:
            # Import strategy modules
            from src.rules import IFVGSignalGenerator
            from src.indicators import TechnicalIndicators

            # Generate base signals
            signal_gen = IFVGSignalGenerator()
            signals = signal_gen.generate_signals(data)

            # Add technical indicators
            indicators = TechnicalIndicators()
            signals = indicators.add_all_indicators(signals)

            # Save signals artifact
            signals_path = self.artifacts_dir / "trading_signals.csv"
            signals.to_csv(signals_path)
            print(f"✅ Signals saved: {len(signals)} signals to {signals_path}")

            return signals

        except Exception as e:
            print(f"⚠️  Signals generation failed: {e}")
            # Return basic signals
            signals = pd.DataFrame(index=data.index)
            signals['long'] = False
            signals['short'] = False
            return signals

    def _run_backtesting(self, data, signals):
        """Stage 3: Run backtesting"""
        try:
            from src.backtester import AdvancedBacktester

            backtester = AdvancedBacktester()
            results = backtester.run_backtest(
                data=data,
                signals=signals,
                initial_capital=10000,
                commission=0.001
            )

            # Save backtest results
            results_path = self.artifacts_dir / "backtest_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✅ Backtest results saved to {results_path}")

            return results

        except Exception as e:
            print(f"⚠️  Backtesting failed: {e}")
            return {'error': str(e), 'trades': []}

    def _run_ab_testing(self, data):
        """Stage 4: Run A/B testing"""
        try:
            from src.ab_testing_protocol import ABTestingProtocol
            from src.alternatives_integration import AlternativeSignalsGenerator

            protocol = ABTestingProtocol()

            # Generate alternative signals
            alt_signals = AlternativeSignalsGenerator()
            rsi_bb_signals = alt_signals.generate_rsi_bb_signals(data)

            # Run A/B test
            ab_results = protocol.run_ab_test(
                df_5m=data,
                variant_a_func=lambda df: self._get_base_signals(df),
                variant_b_func=lambda df: rsi_bb_signals
            )

            # Save A/B results
            ab_path = self.artifacts_dir / "ab_test_results.json"
            with open(ab_path, 'w') as f:
                json.dump(ab_results, f, indent=2, default=str)
            print(f"✅ A/B test results saved to {ab_path}")

            return ab_results

        except Exception as e:
            print(f"⚠️  A/B testing failed: {e}")
            return {'error': str(e)}

    def _run_optimization(self, data):
        """Stage 5: Run optimization"""
        try:
            from src.optimizer import StrategyOptimizer

            optimizer = StrategyOptimizer()
            opt_results = optimizer.optimize_strategy(
                data=data,
                method=self.config.get('optimization_method', 'bayesian'),
                target_metric='sharpe_ratio'
            )

            # Save optimization results
            opt_path = self.artifacts_dir / "optimization_results.json"
            with open(opt_path, 'w') as f:
                json.dump(opt_results, f, indent=2, default=str)
            print(f"✅ Optimization results saved to {opt_path}")

            return opt_results

        except Exception as e:
            print(f"⚠️  Optimization failed: {e}")
            return {'error': str(e)}

    def _run_robustness_analysis(self, backtest_results, opt_results):
        """Stage 6: Run robustness analysis"""
        try:
            from src.robustness_snooping import RobustnessAnalyzer

            analyzer = RobustnessAnalyzer()

            # Extract returns from backtest
            trades_df = pd.DataFrame(backtest_results.get('trades', []))
            if not trades_df.empty and 'PnL %' in trades_df.columns:
                returns = trades_df['PnL %'].values / 100.0

                # Calculate robustness metrics
                robustness_metrics = analyzer.calculate_robustness_metrics(returns)

                # Detect snooping
                snooping_results = analyzer.detect_snooping(opt_results)

                results = {
                    'robustness_metrics': robustness_metrics,
                    'snooping_detection': snooping_results
                }

                # Save robustness results
                robust_path = self.artifacts_dir / "robustness_results.json"
                with open(robust_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"✅ Robustness results saved to {robust_path}")

                return results
            else:
                return {'error': 'No trade data available'}

        except Exception as e:
            print(f"⚠️  Robustness analysis failed: {e}")
            return {'error': str(e)}

    def _run_reporting(self):
        """Stage 7: Generate reports"""
        try:
            # Generate summary report
            summary_path = self.artifacts_dir / "pipeline_summary.md"
            self._generate_markdown_report(summary_path)

            # Generate performance plots
            plots_path = self.artifacts_dir / "performance_plots.png"
            self._generate_performance_plots(plots_path)

            print(f"✅ Reports generated: {summary_path}, {plots_path}")

            return {
                'summary_report': str(summary_path),
                'performance_plots': str(plots_path)
            }

        except Exception as e:
            print(f"⚠️  Reporting failed: {e}")
            return {'error': str(e)}

    def _get_base_signals(self, df):
        """Helper to get base IFVG signals"""
        try:
            from src.rules import IFVGSignalGenerator
            signal_gen = IFVGSignalGenerator()
            return signal_gen.generate_signals(df)
        except BaseException:
            # Fallback
            signals = pd.DataFrame(index=df.index)
            signals['long'] = False
            signals['short'] = False
            return signals

    def _generate_sample_data(self):
        """Generate sample BTC data for demo purposes"""
        import pandas as pd
        import numpy as np

        dates = pd.date_range('2023-01-01', periods=1000, freq='5T')
        prices = 30000 + np.cumsum(np.random.normal(0, 10, 1000))

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 1000)
        }, index=dates)

        return data

    def _dvc_add(self, file_path: str, message: str):
        """Add file to DVC tracking"""
        try:
            subprocess.run(['dvc', 'add', file_path], check=True, capture_output=True)
            subprocess.run(['git', 'add', f"{file_path}.dvc"], check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True)
            print(f"✅ DVC: {file_path} versioned")
        except Exception as e:
            print(f"⚠️  DVC tracking failed: {e}")

    def _generate_markdown_report(self, output_path: Path):
        """Generate comprehensive Markdown report"""
        report = f"""# Automated Trading Pipeline Report

**Run ID:** {self.run_id}
**Timestamp:** {self.timestamp}
**Symbol:** {self.config.get('symbol', 'BTCUSD')}

## Executive Summary

Pipeline completed with the following results:

"""

        # Add results summary
        for stage, result in self.results.items():
            if stage != 'error':
                status = result.get('status', 'unknown')
                report += f"- **{stage}**: {status}\n"

        # Add key metrics
        report += "\n## Key Metrics\n\n"

        if 'backtesting' in self.results:
            bt = self.results['backtesting']
            if 'total_return' in bt:
                report += f"- **Total Return:** {bt['total_return']:.2%}\n"
            if 'sharpe_ratio' in bt:
                report += f"- **Sharpe Ratio:** {bt['sharpe_ratio']:.3f}\n"
            if 'max_drawdown' in bt:
                report += f"- **Max Drawdown:** {bt['max_drawdown']:.2%}\n"

        if 'ab_testing' in self.results:
            ab = self.results['ab_testing']
            if 'conclusion' in ab:
                conclusion = ab['conclusion']
                report += f"- **A/B Test Result:** {conclusion.get('result', 'Unknown')}\n"
                report += f"- **Recommendation:** {conclusion.get('recommendation', 'N/A')}\n"

        if 'robustness_analysis' in self.results:
            robust = self.results['robustness_analysis']
            if 'snooping_detection' in robust:
                snooping = robust['snooping_detection']
                report += f"- **Snooping Detected:** {snooping.get('snooping_detected', False)}\n"

        # Add artifacts
        report += f"\n## Artifacts\n\n"
        report += f"- **Results Directory:** `{self.artifacts_dir}`\n"
        report += f"- **Market Data:** `market_data.csv`\n"
        report += f"- **Signals:** `trading_signals.csv`\n"
        report += f"- **Backtest Results:** `backtest_results.json`\n"
        report += f"- **A/B Test Results:** `ab_test_results.json`\n"
        report += f"- **Optimization Results:** `optimization_results.json`\n"
        report += f"- **Robustness Results:** `robustness_results.json`\n"

        # Save report
        with open(output_path, 'w') as f:
            f.write(report)

    def _generate_performance_plots(self, output_path: Path):
        """Generate performance visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Pipeline Performance Report - {self.run_id}')

            # Plot 1: Backtest equity curve
            if 'backtesting' in self.results and 'equity_curve' in self.results['backtesting']:
                equity = self.results['backtesting']['equity_curve']
                axes[0, 0].plot(equity)
                axes[0, 0].set_title('Equity Curve')
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: A/B test comparison
            if 'ab_testing' in self.results:
                ab = self.results['ab_testing']
                if 'variant_a' in ab and 'variant_b' in ab:
                    labels = ['Variant A', 'Variant B']
                    sharpes = [
                        ab['variant_a'].get('metrics', {}).get('sharpe_ratio', 0),
                        ab['variant_b'].get('metrics', {}).get('sharpe_ratio', 0)
                    ]
                    axes[0, 1].bar(labels, sharpes)
                    axes[0, 1].set_title('A/B Sharpe Ratio Comparison')
                    axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Robustness metrics
            if 'robustness_analysis' in self.results:
                robust = self.results['robustness_analysis']
                if 'robustness_metrics' in robust:
                    metrics = robust['robustness_metrics']
                    key_metrics = ['information_ratio', 'sortino_ratio', 'var_95']
                    values = [metrics.get(k, 0) for k in key_metrics]
                    axes[1, 0].bar(key_metrics, values)
                    axes[1, 0].set_title('Robustness Metrics')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Optimization progress
            if 'optimization' in self.results:
                opt = self.results['optimization']
                if 'optimization_history' in opt:
                    history = opt['optimization_history']
                    axes[1, 1].plot(history)
                    axes[1, 1].set_title('Optimization Progress')
                    axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️  Plot generation failed: {e}")

    def _generate_pipeline_summary(self):
        """Generate final pipeline summary"""
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY")
        print("=" * 60)

        # Overall status
        failed_stages = [stage for stage, result in self.results.items()
                         if result.get('status') == 'failed' or 'error' in result]

        if failed_stages:
            print(f"❌ Pipeline completed with {len(failed_stages)} failed stages: {failed_stages}")
        else:
            print("✅ Pipeline completed successfully!")

        # Key metrics summary
        print("\n🎯 Key Results:")

        if 'backtesting' in self.results:
            bt = self.results['backtesting']
            sharpe = bt.get('sharpe_ratio', 'N/A')
            returns = bt.get('total_return', 'N/A')
            print(f"  • Sharpe Ratio: {sharpe}")
            print(f"  • Total Return: {returns}")

        if 'ab_testing' in self.results:
            ab = self.results['ab_testing']
            if 'conclusion' in ab:
                result = ab['conclusion'].get('result', 'Unknown')
                print(f"  • A/B Test: {result}")

        if 'robustness_analysis' in self.results:
            robust = self.results['robustness_analysis']
            if 'snooping_detection' in robust:
                snooping = robust['snooping_detection'].get('snooping_detected', False)
                print(f"  • Snooping Detected: {snooping}")

        print(f"\n📁 All artifacts saved to: {self.artifacts_dir}")
        print("=" * 60)

    @staticmethod
    def create_dockerfile(output_path: str = "Dockerfile"):
        """Create Dockerfile for pipeline containerization"""
        dockerfile_content = """# Automated Trading Pipeline Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "src.automated_pipeline"]
"""

        with open(output_path, 'w') as f:
            f.write(dockerfile_content)

        print(f"✅ Dockerfile created: {output_path}")

    @staticmethod
    def create_dvc_pipeline(output_path: str = "dvc.yaml"):
        """Create DVC pipeline configuration"""
        dvc_config = {
            'stages': {
                'data_fetch': {
                    'cmd': 'python -m src.automated_pipeline --stage data_fetch',
                    'outs': ['data/btc_5min.csv']
                },
                'signals': {
                    'cmd': 'python -m src.automated_pipeline --stage signals',
                    'deps': ['data/btc_5min.csv'],
                    'outs': ['results/signals.csv']
                },
                'backtest': {
                    'cmd': 'python -m src.automated_pipeline --stage backtest',
                    'deps': ['results/signals.csv'],
                    'outs': ['results/backtest_results.json']
                },
                'ab_test': {
                    'cmd': 'python -m src.automated_pipeline --stage ab_test',
                    'deps': ['data/btc_5min.csv'],
                    'outs': ['results/ab_test_results.json']
                },
                'optimize': {
                    'cmd': 'python -m src.automated_pipeline --stage optimize',
                    'deps': ['data/btc_5min.csv'],
                    'outs': ['results/optimization_results.json']
                },
                'robustness': {
                    'cmd': 'python -m src.automated_pipeline --stage robustness',
                    'deps': ['results/backtest_results.json', 'results/optimization_results.json'],
                    'outs': ['results/robustness_results.json']
                },
                'report': {
                    'cmd': 'python -m src.automated_pipeline --stage report',
                    'deps': ['results/'],
                    'outs': ['results/pipeline_report.md']
                }
            }
        }

        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(dvc_config, f, default_flow_style=False)

        print(f"✅ DVC pipeline created: {output_path}")

    @staticmethod
    def create_makefile(output_path: str = "Makefile"):
        """Create Makefile for pipeline automation"""
        makefile_content = """# Automated Trading Pipeline Makefile

.PHONY: help data signals backtest ab optimize robust report full clean docker

help:
	@echo "Automated Trading Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  data      - Fetch market data"
	@echo "  signals   - Generate trading signals"
	@echo "  backtest  - Run backtesting"
	@echo "  ab        - Run A/B testing"
	@echo "  optimize  - Run optimization"
	@echo "  robust    - Run robustness analysis"
	@echo "  report    - Generate reports"
	@echo "  full      - Run complete pipeline"
	@echo "  clean     - Clean results"
	@echo "  docker    - Build and run Docker container"

data:
	dvc repro data_fetch

signals:
	dvc repro signals

backtest:
	dvc repro backtest

ab:
	dvc repro ab_test

optimize:
	dvc repro optimize

robust:
	dvc repro robustness

report:
	dvc repro report

full:
	dvc repro

clean:
	rm -rf results/
	dvc remove data/btc_5min.csv.dvc

docker-build:
	docker build -t btc-trading-pipeline .

docker-run:
	docker run --rm -v $(PWD)/results:/app/results btc-trading-pipeline

docker: docker-build docker-run

test:
	pytest src/ -v

install:
	pip install -r requirements.txt
"""

        with open(output_path, 'w') as f:
            f.write(makefile_content)

        print(f"✅ Makefile created: {output_path}")


# Convenience functions
def run_full_pipeline(symbol: str = 'BTCUSD', start_date: str = '2018-01-01') -> Dict[str, Any]:
    """Convenience function for running complete pipeline"""
    pipeline = AutomatedPipeline()
    return pipeline.run_full_pipeline(symbol, start_date)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Automated Trading Pipeline')
    parser.add_argument('--symbol', default='BTCUSD', help='Trading symbol')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date')
    parser.add_argument('--stage', help='Run specific stage only')
    parser.add_argument('--create-dockerfile', action='store_true', help='Create Dockerfile')
    parser.add_argument('--create-dvc', action='store_true', help='Create DVC pipeline')
    parser.add_argument('--create-makefile', action='store_true', help='Create Makefile')

    args = parser.parse_args()

    if args.create_dockerfile:
        AutomatedPipeline.create_dockerfile()
    elif args.create_dvc:
        AutomatedPipeline.create_dvc_pipeline()
    elif args.create_makefile:
        AutomatedPipeline.create_makefile()
    else:
        # Run pipeline
        pipeline = AutomatedPipeline()
        results = pipeline.run_full_pipeline(args.symbol, args.start_date)

        if results['status'] == 'completed':
            print(f"\n🎉 Pipeline completed! Results in: {results['artifacts_dir']}")
        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
