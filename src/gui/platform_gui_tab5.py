"""
BTC Trading Strategy Platform - Tab 5: A/B Testing
GUI component for statistical comparison of trading strategies.

Author: TradingIA Team
Version: 1.0.0
"""

import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QGroupBox, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QFont
from scipy import stats
import traceback

class ABTestWorker(QObject):
    """Worker thread for A/B testing operations"""
    progress_updated = Signal(int, str)
    test_completed = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, backtester_core, strategy_a, strategy_b, data_dict):
        super().__init__()
        self.backtester = backtester_core
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.data_dict = data_dict

    def run_ab_test(self):
        """Run A/B test between two strategies"""
        try:
            self.progress_updated.emit(10, "Loading strategy A...")

            # Load strategy A
            strategy_engine_a = self.backtester.strategy_engine
            params_a = strategy_engine_a.get_strategy_params(self.strategy_a)
            if 'error' in params_a:
                raise ValueError(f"Strategy A params error: {params_a['error']}")

            self.progress_updated.emit(20, "Loading strategy B...")

            # Load strategy B
            strategy_engine_b = self.backtester.strategy_engine
            params_b = strategy_engine_b.get_strategy_params(self.strategy_b)
            if 'error' in params_b:
                raise ValueError(f"Strategy B params error: {params_b['error']}")

            self.progress_updated.emit(30, "Running backtest A...")

            # Run backtest A
            results_a = self.backtester.run_simple_backtest(
                self.data_dict, strategy_engine_a, params_a
            )

            self.progress_updated.emit(60, "Running backtest B...")

            # Run backtest B
            results_b = self.backtester.run_simple_backtest(
                self.data_dict, strategy_engine_b, params_b
            )

            self.progress_updated.emit(80, "Calculating statistics...")

            # Calculate comparative statistics
            comparison = self.calculate_comparison(results_a, results_b)

            self.progress_updated.emit(90, "Generating recommendations...")

            # Generate recommendation
            recommendation = self.generate_recommendation(comparison)

            results = {
                'strategy_a': {
                    'name': self.strategy_a,
                    'results': results_a,
                    'params': params_a
                },
                'strategy_b': {
                    'name': self.strategy_b,
                    'results': results_b,
                    'params': params_b
                },
                'comparison': comparison,
                'recommendation': recommendation
            }

            self.progress_updated.emit(100, "A/B test completed")
            self.test_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"A/B test failed: {str(e)}\n{traceback.format_exc()}")

    def calculate_comparison(self, results_a, results_b):
        """Calculate statistical comparison between strategies"""
        metrics_a = results_a.get('metrics', {})
        metrics_b = results_b.get('metrics', {})

        comparison = {}

        # Basic metrics comparison
        for metric in ['sharpe', 'calmar', 'win_rate', 'max_dd', 'profit_factor', 'num_trades']:
            val_a = metrics_a.get(metric, 0)
            val_b = metrics_b.get(metric, 0)
            delta = val_b - val_a
            comparison[metric] = {
                'a': val_a,
                'b': val_b,
                'delta': delta,
                'percent_change': (delta / val_a * 100) if val_a != 0 else 0
            }

        # Statistical significance tests
        returns_a = [trade.get('pnl_pct', 0) for trade in results_a.get('trades', [])]
        returns_b = [trade.get('pnl_pct', 0) for trade in results_b.get('trades', [])]

        if returns_a and returns_b:
            # t-test for returns
            try:
                t_stat, p_value = stats.ttest_ind(returns_a, returns_b, equal_var=False)
                comparison['returns_ttest'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception:
                comparison['returns_ttest'] = {'error': 'Could not calculate t-test'}

            # Sharpe ratio comparison (approximate)
            sharpe_a = metrics_a.get('sharpe', 0)
            sharpe_b = metrics_b.get('sharpe', 0)
            if sharpe_a != sharpe_b:
                sharpe_diff = sharpe_b - sharpe_a
                # Rough significance test
                comparison['sharpe_comparison'] = {
                    'difference': sharpe_diff,
                    'significant': abs(sharpe_diff) > 0.2  # Rule of thumb
                }

        return comparison

    def generate_recommendation(self, comparison):
        """Generate recommendation based on comparison results"""
        sharpe_comp = comparison.get('sharpe_comparison', {})
        returns_test = comparison.get('returns_ttest', {})

        recommendation = {
            'winner': None,
            'confidence': 'low',
            'reasoning': [],
            'action': 'further_testing'
        }

        # Determine winner
        sharpe_diff = sharpe_comp.get('difference', 0)
        if abs(sharpe_diff) > 0.2:
            recommendation['winner'] = 'B' if sharpe_diff > 0 else 'A'
            recommendation['confidence'] = 'high' if abs(sharpe_diff) > 0.5 else 'medium'

        # Check statistical significance
        if returns_test.get('significant', False):
            p_val = returns_test.get('p_value', 1)
            if p_val < 0.01:
                recommendation['confidence'] = 'very_high'
            elif p_val < 0.05:
                recommendation['confidence'] = 'high'

        # Generate reasoning
        reasoning = []
        if recommendation['winner']:
            reasoning.append(f"Strategy {recommendation['winner']} shows superior Sharpe ratio")

        if returns_test.get('significant'):
            reasoning.append("Statistical significance detected in returns (p < 0.05)")
        else:
            reasoning.append("No statistical significance in returns")

        win_rate_a = comparison.get('win_rate', {}).get('a', 0)
        win_rate_b = comparison.get('win_rate', {}).get('b', 0)
        if win_rate_b > win_rate_a + 5:
            reasoning.append("Strategy B has significantly higher win rate")

        recommendation['reasoning'] = reasoning

        # Determine action
        if recommendation['confidence'] in ['high', 'very_high']:
            recommendation['action'] = f"adopt_strategy_{recommendation['winner'].lower()}"
        elif recommendation['confidence'] == 'medium':
            recommendation['action'] = 'create_hybrid'
        else:
            recommendation['action'] = 'further_testing'

        return recommendation

class Tab5ABTesting(QWidget):
    """A/B Testing tab for strategy comparison"""

    ab_test_completed = Signal(dict)

    def __init__(self, parent_platform, backtester_core):
        super().__init__()
        self.parent = parent_platform
        self.backtester = backtester_core
        self.current_results = None

        self.init_ui()
        self.load_available_strategies()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("A/B Testing - Strategy Comparison")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # Strategy selection
        strategy_group = QGroupBox("Strategy Selection")
        strategy_layout = QHBoxLayout()

        # Strategy A
        strategy_layout.addWidget(QLabel("Strategy A:"))
        self.strategy_a_combo = QComboBox()
        self.strategy_a_combo.setMinimumWidth(200)
        strategy_layout.addWidget(self.strategy_a_combo)

        # Strategy B
        strategy_layout.addWidget(QLabel("Strategy B:"))
        self.strategy_b_combo = QComboBox()
        self.strategy_b_combo.setMinimumWidth(200)
        strategy_layout.addWidget(self.strategy_b_combo)

        # Run test button
        self.run_test_btn = QPushButton("Run A/B Test")
        self.run_test_btn.setStyleSheet("QPushButton { background-color: #FF6B35; color: white; padding: 10px; font-weight: bold; }")
        self.run_test_btn.clicked.connect(self.on_run_ab_test)
        strategy_layout.addWidget(self.run_test_btn)

        strategy_layout.addStretch()
        strategy_group.setLayout(strategy_layout)
        layout.addWidget(strategy_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to run A/B test")
        layout.addWidget(self.status_label)

        # Splitter for results
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Metrics comparison table
        metrics_group = QGroupBox("Metrics Comparison")
        metrics_layout = QVBoxLayout()

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Strategy A", "Strategy B", "Δ", "Significance"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_table)

        metrics_group.setLayout(metrics_layout)
        splitter.addWidget(metrics_group)

        # Recommendation panel
        recommendation_group = QGroupBox("Analysis & Recommendation")
        recommendation_layout = QVBoxLayout()

        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setMaximumHeight(150)
        recommendation_layout.addWidget(self.recommendation_text)

        # Action buttons
        buttons_layout = QHBoxLayout()

        self.create_hybrid_btn = QPushButton("Create Hybrid Strategy")
        self.create_hybrid_btn.clicked.connect(self.on_create_hybrid)
        self.create_hybrid_btn.setEnabled(False)
        buttons_layout.addWidget(self.create_hybrid_btn)

        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.on_export_results)
        self.export_results_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_results_btn)

        buttons_layout.addStretch()
        recommendation_layout.addLayout(buttons_layout)

        recommendation_group.setLayout(recommendation_layout)
        splitter.addWidget(recommendation_group)

        # Set splitter proportions
        splitter.setSizes([400, 200])
        layout.addWidget(splitter)

        # Set dark theme
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTableWidget {
                gridline-color: #444;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
            }
        """)

    def load_available_strategies(self):
        """Load available strategies into combo boxes"""
        try:
            strategies = self.backtester.strategy_engine.list_available_strategies()
            self.strategy_a_combo.clear()
            self.strategy_b_combo.clear()

            for strategy in strategies:
                self.strategy_a_combo.addItem(strategy)
                self.strategy_b_combo.addItem(strategy)

            # Set defaults
            if len(strategies) >= 2:
                self.strategy_a_combo.setCurrentIndex(0)
                self.strategy_b_combo.setCurrentIndex(1)

        except Exception as e:
            self.status_label.setText(f"Error loading strategies: {str(e)}")

    def on_run_ab_test(self):
        """Handle A/B test execution"""
        strategy_a = self.strategy_a_combo.currentText()
        strategy_b = self.strategy_b_combo.currentText()

        if not strategy_a or not strategy_b:
            self.status_label.setText("Please select both strategies")
            return

        if strategy_a == strategy_b:
            self.status_label.setText("Please select different strategies for comparison")
            return

        # Check if data is available
        if not hasattr(self.parent, 'data_dict') or not self.parent.data_dict:
            self.status_label.setText("No data available. Please load data in Tab 1 first.")
            return

        # Disable button and show progress
        self.run_test_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting A/B test...")

        # Create worker thread
        self.worker = ABTestWorker(
            self.backtester,
            strategy_a,
            strategy_b,
            self.parent.data_dict
        )

        self.worker.progress_updated.connect(self.update_progress)
        self.worker.test_completed.connect(self.on_test_completed)
        self.worker.error_occurred.connect(self.on_test_error)

        # Start test
        self.worker.run_ab_test()

    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_test_completed(self, results):
        """Handle test completion"""
        self.current_results = results

        # Re-enable UI
        self.run_test_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update metrics table
        self.update_metrics_table(results['comparison'])

        # Update recommendation
        self.update_recommendation(results['recommendation'])

        # Enable action buttons
        self.create_hybrid_btn.setEnabled(True)
        self.export_results_btn.setEnabled(True)

        self.status_label.setText("A/B test completed successfully")

        # Emit signal
        self.ab_test_completed.emit(results)

    def on_test_error(self, error_msg):
        """Handle test error"""
        self.run_test_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Test failed: {error_msg}")

    def update_metrics_table(self, comparison):
        """Update the metrics comparison table"""
        self.metrics_table.setRowCount(0)

        metrics_order = ['sharpe', 'calmar', 'win_rate', 'max_dd', 'profit_factor', 'num_trades']

        for i, metric in enumerate(metrics_order):
            self.metrics_table.insertRow(i)

            # Metric name
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric.replace('_', ' ').title()))

            # Values
            comp = comparison.get(metric, {})
            val_a = comp.get('a', 0)
            val_b = comp.get('b', 0)
            delta = comp.get('delta', 0)

            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{val_a:.4f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{val_b:.4f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{delta:+.4f}"))

            # Significance
            significance = ""
            if metric == 'sharpe':
                sharpe_comp = comparison.get('sharpe_comparison', {})
                if sharpe_comp.get('significant', False):
                    significance = "***"
                elif abs(delta) > 0.1:
                    significance = "**"
                elif abs(delta) > 0.05:
                    significance = "*"

            self.metrics_table.setItem(i, 4, QTableWidgetItem(significance))

        # Resize columns
        self.metrics_table.resizeColumnsToContents()

    def update_recommendation(self, recommendation):
        """Update the recommendation text"""
        winner = recommendation.get('winner')
        confidence = recommendation.get('confidence', 'low')
        reasoning = recommendation.get('reasoning', [])
        action = recommendation.get('action', 'further_testing')

        text = f"<b>Winner:</b> Strategy {winner or 'Tie'}<br>"
        text += f"<b>Confidence:</b> {confidence.title()}<br><br>"
        text += "<b>Reasoning:</b><br>"
        for reason in reasoning:
            text += f"• {reason}<br>"
        text += "<br>"
        text += f"<b>Recommended Action:</b> {action.replace('_', ' ').title()}"

        self.recommendation_text.setHtml(text)

    def on_create_hybrid(self):
        """Create hybrid strategy from A/B test results"""
        if not self.current_results:
            return

        # This would integrate with strategy creation functionality
        # For now, just show a message
        self.status_label.setText("Hybrid strategy creation - Feature coming soon")

    def on_export_results(self):
        """Export A/B test results"""
        if not self.current_results:
            return

        try:
            # Export to JSON
            import json
            filename = f"ab_test_results_{self.current_results['strategy_a']['name']}_vs_{self.current_results['strategy_b']['name']}.json"
            with open(filename, 'w') as f:
                json.dump(self.current_results, f, indent=2, default=str)

            self.status_label.setText(f"Results exported to {filename}")

        except Exception as e:
            self.status_label.setText(f"Export failed: {str(e)}")

    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.load_available_strategies()