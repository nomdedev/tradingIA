from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QGroupBox, QTextEdit, QSplitter, QFrame, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QFont, QColor
from scipy import stats
import logging
import traceback
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ABTestThread(QThread):
    """Background thread for A/B testing"""
    progress_updated = Signal(int, str)
    test_completed = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, backtester_core, strategy_a, strategy_b, data_dict, params_a, params_b):
        super().__init__()
        self.backtester = backtester_core
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.data_dict = data_dict
        self.params_a = params_a
        self.params_b = params_b

    def run(self):
        try:
            self.progress_updated.emit(10, "Running Strategy A backtest...")
            
            # Run backtest A
            results_a = self.backtester.run_simple_backtest(
                self.data_dict, self.strategy_a, self.params_a
            )
            
            self.progress_updated.emit(50, "Running Strategy B backtest...")
            
            # Run backtest B
            results_b = self.backtester.run_simple_backtest(
                self.data_dict, self.strategy_b, self.params_b
            )
            
            self.progress_updated.emit(80, "Calculating statistical comparison...")
            
            # Calculate comparison
            comparison = self.calculate_comparison(results_a, results_b)
            
            self.progress_updated.emit(95, "Generating recommendations...")
            
            # Generate recommendation
            recommendation = self.generate_recommendation(comparison, results_a, results_b)
            
            results = {
                'strategy_a': {
                    'name': str(self.strategy_a),
                    'results': results_a
                },
                'strategy_b': {
                    'name': str(self.strategy_b),
                    'results': results_b
                },
                'comparison': comparison,
                'recommendation': recommendation
            }
            
            self.progress_updated.emit(100, "A/B test completed!")
            self.test_completed.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"A/B test failed: {str(e)}\n{traceback.format_exc()}")
            
    def calculate_comparison(self, results_a, results_b):
        """Calculate statistical comparison"""
        metrics_a = results_a.get('metrics', {})
        metrics_b = results_b.get('metrics', {})
        
        comparison = {}
        
        # Key metrics to compare
        key_metrics = ['sharpe', 'total_return', 'win_rate', 'max_dd', 'num_trades']
        
        for metric in key_metrics:
            val_a = metrics_a.get(metric, 0)
            val_b = metrics_b.get(metric, 0)
            
            # Calculate difference
            if val_a != 0:
                percent_diff = ((val_b - val_a) / abs(val_a)) * 100
            else:
                percent_diff = 0 if val_b == 0 else 100
                
            comparison[metric] = {
                'a': val_a,
                'b': val_b,
                'diff': val_b - val_a,
                'percent_diff': percent_diff,
                'winner': 'B' if val_b > val_a else 'A' if val_b < val_a else 'Tie'
            }
            
        # Statistical significance test on returns
        trades_a = results_a.get('trades', [])
        trades_b = results_b.get('trades', [])
        
        if trades_a and trades_b:
            returns_a = [t.get('pnl_pct', 0) for t in trades_a]
            returns_b = [t.get('pnl_pct', 0) for t in trades_b]
            
            try:
                t_stat, p_value = stats.ttest_ind(returns_a, returns_b, equal_var=False)
                comparison['statistical_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'confidence': 95 if p_value < 0.05 else 90 if p_value < 0.1 else 0
                }
            except Exception as e:
                comparison['statistical_test'] = {'significant': False, 'p_value': 1.0}
        else:
            comparison['statistical_test'] = {'significant': False, 'p_value': 1.0}
            
        return comparison
        
    def generate_recommendation(self, comparison, results_a, results_b):
        """Generate intelligent recommendation"""
        sharpe_comp = comparison.get('sharpe', {})
        return_comp = comparison.get('total_return', {})
        winrate_comp = comparison.get('win_rate', {})
        dd_comp = comparison.get('max_dd', {})
        stat_test = comparison.get('statistical_test', {})
        
        # Score each strategy
        score_a = 0
        score_b = 0
        
        # Sharpe ratio (most important)
        if sharpe_comp['winner'] == 'A':
            score_a += 3
        elif sharpe_comp['winner'] == 'B':
            score_b += 3
            
        # Total return
        if return_comp['winner'] == 'A':
            score_a += 2
        elif return_comp['winner'] == 'B':
            score_b += 2
            
        # Win rate
        if winrate_comp['winner'] == 'A':
            score_a += 1
        elif winrate_comp['winner'] == 'B':
            score_b += 1
            
        # Lower drawdown is better
        if dd_comp['winner'] == 'B':  # Remember: lower is better for DD
            score_a += 2
        elif dd_comp['winner'] == 'A':
            score_b += 2
            
        # Determine winner
        if score_a > score_b:
            winner = 'A'
            confidence = 'High' if score_a - score_b >= 3 else 'Medium'
        elif score_b > score_a:
            winner = 'B'
            confidence = 'High' if score_b - score_a >= 3 else 'Medium'
        else:
            winner = 'Tie'
            confidence = 'Low'
            
        # Adjust confidence based on statistical test
        if stat_test.get('significant'):
            if confidence == 'Medium':
                confidence = 'High'
            elif confidence == 'Low':
                confidence = 'Medium'
        else:
            if confidence == 'High':
                confidence = 'Medium'
                
        # Generate reasoning
        reasons = []
        
        if sharpe_comp['percent_diff'] != 0:
            reasons.append(
                f"Strategy {sharpe_comp['winner']} has "
                f"{abs(sharpe_comp['percent_diff']):.1f}% better Sharpe ratio"
            )
            
        if return_comp['percent_diff'] != 0:
            reasons.append(
                f"Strategy {return_comp['winner']} has "
                f"{abs(return_comp['percent_diff']):.1f}% higher returns"
            )
            
        if stat_test.get('significant'):
            reasons.append("Difference is statistically significant (p < 0.05)")
        else:
            reasons.append("Difference is NOT statistically significant")
            
        # Determine action
        if winner == 'Tie' or confidence == 'Low':
            action = "further_testing"
            action_text = "Run more tests with different market conditions"
        elif confidence == 'High':
            action = f"adopt_{winner.lower()}"
            action_text = f"Adopt Strategy {winner} for live trading"
        else:
            action = "create_ensemble"
            action_text = "Consider creating an ensemble of both strategies"
            
        return {
            'winner': winner,
            'confidence': confidence,
            'score_a': score_a,
            'score_b': score_b,
            'reasons': reasons,
            'action': action,
            'action_text': action_text
        }


class Tab5ABTesting(QWidget):
    """
    Enhanced A/B Testing Tab with:
    - Head-to-head visual comparison
    - Simplified statistics
    - Automatic recommendations
    - Battle-style interface
    """
    ab_test_completed = Signal(dict)
    status_update = Signal(str, str)

    def __init__(self, parent_platform, backtester_core):
        super().__init__()
        self.parent_platform = parent_platform
        self.backtester = backtester_core
        self.logger = logging.getLogger(__name__)
        self.current_results = None
        self.test_thread = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        
        # Header
        header = QLabel("⚖️ A/B Testing - Strategy Battle")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        main_layout.addWidget(header)
        
        # Strategy selection section
        selection_section = self.create_selection_section()
        main_layout.addWidget(selection_section)
        
        # Progress section (hidden initially)
        self.progress_section = self.create_progress_section()
        self.progress_section.setVisible(False)
        main_layout.addWidget(self.progress_section)
        
        # Results section (hidden initially)
        self.results_section = self.create_results_section()
        self.results_section.setVisible(False)
        main_layout.addWidget(self.results_section, 1)
        
        self.setLayout(main_layout)
        
        # Load strategies
        self.load_available_strategies()
        
    def create_selection_section(self):
        """Create strategy selection section"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Select Strategies to Compare")
        title.setStyleSheet("font-size: 13px; font-weight: 500; color: #cccccc;")
        layout.addWidget(title)
        
        # Selection grid
        grid = QHBoxLayout()
        
        # Strategy A (Blue corner)
        a_layout = QVBoxLayout()
        a_header = QLabel("🔵 Strategy A")
        a_header.setStyleSheet("font-weight: bold; color: #569cd6; font-size: 12px;")
        a_layout.addWidget(a_header)
        
        self.strategy_a_combo = QComboBox()
        self.strategy_a_combo.setMinimumHeight(32)
        self.strategy_a_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e3a52;
                border: 2px solid #569cd6;
                border-radius: 4px;
                padding: 4px;
                color: white;
            }
        """)
        a_layout.addWidget(self.strategy_a_combo)
        grid.addLayout(a_layout)
        
        # VS label
        vs_label = QLabel("VS")
        vs_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #dcdcaa;
            padding: 0 20px;
        """)
        vs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(vs_label)
        
        # Strategy B (Red corner)
        b_layout = QVBoxLayout()
        b_header = QLabel("🔴 Strategy B")
        b_header.setStyleSheet("font-weight: bold; color: #f48771; font-size: 12px;")
        b_layout.addWidget(b_header)
        
        self.strategy_b_combo = QComboBox()
        self.strategy_b_combo.setMinimumHeight(32)
        self.strategy_b_combo.setStyleSheet("""
            QComboBox {
                background-color: #521e1e;
                border: 2px solid #f48771;
                border-radius: 4px;
                padding: 4px;
                color: white;
            }
        """)
        b_layout.addWidget(self.strategy_b_combo)
        grid.addLayout(b_layout)
        
        layout.addLayout(grid)
        
        # Run button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.run_btn = QPushButton("⚔️ Start Battle")
        self.run_btn.clicked.connect(self.on_run_ab_test)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setMinimumWidth(200)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ec9b0;
                color: #1e1e1e;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px 24px;
            }
            QPushButton:hover {
                background-color: #6fdfcf;
            }
            QPushButton:disabled {
                background-color: #3e3e3e;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.run_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        frame.setLayout(layout)
        return frame
        
    def create_progress_section(self):
        """Create progress section"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        
        layout = QVBoxLayout()
        
        title = QLabel("🔄 Testing in Progress...")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4ec9b0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.progress_status = QLabel("Initializing...")
        self.progress_status.setStyleSheet("color: #cccccc; margin-top: 8px;")
        layout.addWidget(self.progress_status)
        
        frame.setLayout(layout)
        return frame
        
    def create_results_section(self):
        """Create results section"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Winner announcement
        self.winner_banner = QFrame()
        self.winner_banner.setMinimumHeight(80)
        self.winner_banner.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 3px solid #4ec9b0;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        winner_layout = QVBoxLayout()
        
        self.winner_text = QLabel("🏆 Winner: Strategy A")
        self.winner_text.setStyleSheet("font-size: 20px; font-weight: bold; color: #4ec9b0;")
        self.winner_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        winner_layout.addWidget(self.winner_text)
        
        self.confidence_text = QLabel("Confidence: High")
        self.confidence_text.setStyleSheet("font-size: 14px; color: #cccccc;")
        self.confidence_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        winner_layout.addWidget(self.confidence_text)
        
        self.winner_banner.setLayout(winner_layout)
        layout.addWidget(self.winner_banner)
        
        # Head-to-head comparison chart
        chart_group = QGroupBox("📊 Head-to-Head Metrics")
        chart_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        chart_layout = QVBoxLayout()
        
        self.comparison_chart = QWebEngineView()
        self.comparison_chart.setMinimumHeight(300)
        chart_layout.addWidget(self.comparison_chart)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        # Bottom section with stats and recommendations
        bottom_layout = QHBoxLayout()
        
        # Detailed metrics table
        metrics_group = QGroupBox("📈 Detailed Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        metrics_layout = QVBoxLayout()
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Strategy A", "Strategy B", "Winner"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setAlternatingRowColors(True)
        metrics_layout.addWidget(self.metrics_table)
        
        metrics_group.setLayout(metrics_layout)
        bottom_layout.addWidget(metrics_group)
        
        # Recommendations
        rec_group = QGroupBox("💡 Recommendations")
        rec_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        rec_layout = QVBoxLayout()
        
        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 8px;
                color: #cccccc;
            }
        """)
        rec_layout.addWidget(self.recommendation_text)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("📄 Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        action_layout.addWidget(self.export_btn)
        
        self.rerun_btn = QPushButton("🔄 Run Again")
        self.rerun_btn.clicked.connect(self.on_run_ab_test)
        self.rerun_btn.setStyleSheet(self.export_btn.styleSheet())
        action_layout.addWidget(self.rerun_btn)
        
        rec_layout.addLayout(action_layout)
        
        rec_group.setLayout(rec_layout)
        bottom_layout.addWidget(rec_group)
        
        layout.addLayout(bottom_layout)
        
        widget.setLayout(layout)
        return widget
        
    def load_available_strategies(self):
        """Load available strategies"""
        try:
            strategies = self.backtester.list_available_strategies()
            
            self.strategy_a_combo.clear()
            self.strategy_b_combo.clear()
            
            for strategy in strategies:
                self.strategy_a_combo.addItem(strategy)
                self.strategy_b_combo.addItem(strategy)
                
            # Set different defaults
            if len(strategies) >= 2:
                self.strategy_a_combo.setCurrentIndex(0)
                self.strategy_b_combo.setCurrentIndex(1)
                
        except Exception as e:
            self.logger.error(f"Error loading strategies: {e}")
            
    def on_run_ab_test(self):
        """Start A/B test"""
        strategy_a = self.strategy_a_combo.currentText()
        strategy_b = self.strategy_b_combo.currentText()
        
        if not strategy_a or not strategy_b:
            QMessageBox.warning(self, "Error", "Please select both strategies")
            return
            
        if strategy_a == strategy_b:
            QMessageBox.warning(self, "Error", "Please select different strategies for comparison")
            return
            
        if not self.parent_platform.data_dict:
            QMessageBox.warning(self, "Error", "No data loaded. Please load data in Tab 1 first.")
            return
            
        # Get strategy parameters (use defaults for now)
        params_a = self.backtester.get_strategy_params(strategy_a)
        params_b = self.backtester.get_strategy_params(strategy_b)
        
        # Update UI
        self.run_btn.setEnabled(False)
        self.progress_section.setVisible(True)
        self.results_section.setVisible(False)
        self.progress_bar.setValue(0)
        
        # Start test thread
        self.test_thread = ABTestThread(
            self.backtester,
            strategy_a, strategy_b,
            self.parent_platform.data_dict,
            params_a, params_b
        )
        
        self.test_thread.progress_updated.connect(self.update_progress)
        self.test_thread.test_completed.connect(self.on_test_completed)
        self.test_thread.error_occurred.connect(self.on_test_error)
        self.test_thread.start()
        
        self.status_update.emit(f"Started A/B test: {strategy_a} vs {strategy_b}", "info")
        
    def update_progress(self, value, message):
        """Update progress"""
        self.progress_bar.setValue(value)
        self.progress_status.setText(message)
        
    def on_test_completed(self, results):
        """Handle test completion"""
        self.current_results = results
        
        # Update UI
        self.run_btn.setEnabled(True)
        self.progress_section.setVisible(False)
        self.results_section.setVisible(True)
        
        # Display results
        self.display_results(results)
        
        # Emit signal
        self.ab_test_completed.emit(results)
        
        winner = results['recommendation']['winner']
        self.status_update.emit(f"A/B test completed - Winner: Strategy {winner}", "success")
        
    def on_test_error(self, error_msg):
        """Handle test error"""
        self.run_btn.setEnabled(True)
        self.progress_section.setVisible(False)
        
        QMessageBox.critical(self, "Test Error", error_msg)
        self.status_update.emit("A/B test failed", "error")
        
    def display_results(self, results):
        """Display test results"""
        recommendation = results['recommendation']
        comparison = results['comparison']
        
        # Update winner banner
        winner = recommendation['winner']
        confidence = recommendation['confidence']
        
        if winner == 'Tie':
            self.winner_text.setText("🤝 It's a Tie!")
            self.winner_banner.setStyleSheet("""
                QFrame {
                    background-color: #2d2d2d;
                    border: 3px solid #dcdcaa;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            self.winner_text.setStyleSheet("font-size: 20px; font-weight: bold; color: #dcdcaa;")
        else:
            color = "#569cd6" if winner == "A" else "#f48771"
            self.winner_text.setText(f"🏆 Winner: Strategy {winner}")
            self.winner_text.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")
            self.winner_banner.setStyleSheet(f"""
                QFrame {{
                    background-color: #2d2d2d;
                    border: 3px solid {color};
                    border-radius: 8px;
                    padding: 16px;
                }}
            """)
            
        self.confidence_text.setText(f"Confidence: {confidence} | Score: A={recommendation['score_a']} vs B={recommendation['score_b']}")
        
        # Create comparison chart
        self.create_comparison_chart(comparison)
        
        # Update metrics table
        self.update_metrics_table(comparison)
        
        # Update recommendations
        self.update_recommendations(recommendation, comparison)
        
    def create_comparison_chart(self, comparison):
        """Create head-to-head comparison chart"""
        try:
            metrics = ['sharpe', 'total_return', 'win_rate', 'max_dd', 'num_trades']
            labels = ['Sharpe Ratio', 'Total Return', 'Win Rate', 'Max Drawdown', 'Total Trades']
            
            values_a = []
            values_b = []
            
            for metric in metrics:
                comp = comparison.get(metric, {})
                val_a = comp.get('a', 0)
                val_b = comp.get('b', 0)
                
                # Normalize for better visualization
                if metric == 'max_dd':
                    # Invert DD (lower is better)
                    val_a = -val_a
                    val_b = -val_b
                elif metric == 'win_rate':
                    # Convert to percentage
                    val_a *= 100
                    val_b *= 100
                elif metric == 'total_return':
                    # Convert to percentage
                    val_a *= 100
                    val_b *= 100
                    
                values_a.append(val_a)
                values_b.append(val_b)
                
            # Create grouped bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Strategy A',
                x=labels,
                y=values_a,
                marker_color='#569cd6',
                text=[f"{v:.2f}" for v in values_a],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Strategy B',
                x=labels,
                y=values_b,
                marker_color='#f48771',
                text=[f"{v:.2f}" for v in values_b],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Strategy Comparison",
                barmode='group',
                template='plotly_dark',
                height=300,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#cccccc'),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.comparison_chart.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error creating comparison chart: {e}")
            
    def update_metrics_table(self, comparison):
        """Update metrics table"""
        metrics = [
            ('sharpe', 'Sharpe Ratio'),
            ('total_return', 'Total Return %'),
            ('win_rate', 'Win Rate %'),
            ('max_dd', 'Max Drawdown %'),
            ('num_trades', 'Total Trades')
        ]
        
        self.metrics_table.setRowCount(len(metrics))
        
        for i, (key, label) in enumerate(metrics):
            comp = comparison.get(key, {})
            
            # Metric name
            self.metrics_table.setItem(i, 0, QTableWidgetItem(label))
            
            # Strategy A
            val_a = comp.get('a', 0)
            if key in ['total_return', 'win_rate']:
                val_a_str = f"{val_a * 100:.2f}%"
            elif key == 'max_dd':
                val_a_str = f"{abs(val_a) * 100:.2f}%"
            elif key == 'num_trades':
                val_a_str = str(int(val_a))
            else:
                val_a_str = f"{val_a:.4f}"
            self.metrics_table.setItem(i, 1, QTableWidgetItem(val_a_str))
            
            # Strategy B
            val_b = comp.get('b', 0)
            if key in ['total_return', 'win_rate']:
                val_b_str = f"{val_b * 100:.2f}%"
            elif key == 'max_dd':
                val_b_str = f"{abs(val_b) * 100:.2f}%"
            elif key == 'num_trades':
                val_b_str = str(int(val_b))
            else:
                val_b_str = f"{val_b:.4f}"
            self.metrics_table.setItem(i, 2, QTableWidgetItem(val_b_str))
            
            # Winner
            winner = comp.get('winner', 'Tie')
            winner_item = QTableWidgetItem(f"🏆 {winner}" if winner != 'Tie' else "Tie")
            
            if winner == 'A':
                winner_item.setForeground(QColor("#569cd6"))
            elif winner == 'B':
                winner_item.setForeground(QColor("#f48771"))
            else:
                winner_item.setForeground(QColor("#dcdcaa"))
                
            self.metrics_table.setItem(i, 3, winner_item)
            
    def update_recommendations(self, recommendation, comparison):
        """Update recommendations text"""
        reasons = recommendation.get('reasons', [])
        action_text = recommendation.get('action_text', '')
        stat_test = comparison.get('statistical_test', {})
        
        html = "<div style='color:#cccccc;'>"
        
        # Key insights
        html += "<b style='color:#4ec9b0;'>📊 Key Insights:</b><br>"
        for reason in reasons:
            html += f"• {reason}<br>"
            
        html += "<br>"
        
        # Statistical significance
        html += "<b style='color:#569cd6;'>📈 Statistical Analysis:</b><br>"
        if stat_test.get('significant'):
            html += f"✓ Difference is statistically significant (p = {stat_test.get('p_value', 0):.4f})<br>"
            html += "The performance difference is unlikely due to random chance.<br>"
        else:
            html += f"⚠️ Difference is NOT statistically significant (p = {stat_test.get('p_value', 1):.4f})<br>"
            html += "Results may vary with different data. Consider more testing.<br>"
            
        html += "<br>"
        
        # Recommended action
        html += "<b style='color:#dcdcaa;'>🎯 Recommended Action:</b><br>"
        html += f"<span style='font-size:13px;'>{action_text}</span><br>"
        
        html += "</div>"
        
        self.recommendation_text.setHtml(html)
        
    def export_results(self):
        """Export A/B test results"""
        if not self.current_results:
            QMessageBox.warning(self, "Error", "No results to export")
            return
            
        try:
            from PySide6.QtWidgets import QFileDialog
            from datetime import datetime
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export A/B Test Results",
                f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON files (*.json)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
                    
                QMessageBox.information(self, "Success", f"Results exported to:\n{filename}")
                self.status_update.emit("A/B test results exported", "success")
                
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
            
    def on_tab_activated(self):
        """Called when tab becomes active"""
        self.load_available_strategies()
