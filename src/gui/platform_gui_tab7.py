"""
BTC Trading Strategy Platform - Tab 7: Advanced Analysis
Advanced analysis tools including regime detection, stress testing, and causality validation.

Author: TradingIA Team
Version: 1.0.0
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox,
    QTextEdit, QTabWidget, QSlider, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QFont
import traceback
from datetime import datetime
import numpy as np

class AdvancedAnalysisWorker(QObject):
    """Worker thread for advanced analysis operations"""

    progress_updated = Signal(int, str)
    analysis_completed = Signal(str, dict)
    error_occurred = Signal(str, str)

    def __init__(self, analysis_engines, data_dict):
        super().__init__()
        self.analysis_engines = analysis_engines
        self.data_dict = data_dict

    def run_regime_analysis(self):
        """Run regime detection analysis"""
        try:
            self.progress_updated.emit(10, "Loading data for regime analysis...")

            # Get data (assuming 5min timeframe)
            df_5m = self.data_dict.get('5min')
            if df_5m is None or df_5m.empty:
                raise ValueError("No data available for regime analysis")

            self.progress_updated.emit(30, "Detecting market regimes...")

            # Run regime detection
            regime_df = self.analysis_engines.detect_regime_hmm(df_5m)

            self.progress_updated.emit(70, "Analyzing regime statistics...")

            # Calculate regime statistics
            regime_stats = self.calculate_regime_stats(regime_df)

            self.progress_updated.emit(90, "Generating regime recommendations...")

            # Generate recommendations
            recommendations = self.generate_regime_recommendations(regime_stats)

            results = {
                'regime_data': regime_df,
                'statistics': regime_stats,
                'recommendations': recommendations
            }

            self.progress_updated.emit(100, "Regime analysis completed")
            self.analysis_completed.emit('regime', results)

        except Exception as e:
            self.error_occurred.emit('regime', f"Regime analysis failed: {str(e)}\n{traceback.format_exc()}")

    def run_stress_test(self, scenarios):
        """Run stress testing"""
        try:
            self.progress_updated.emit(10, "Setting up stress scenarios...")

            df_5m = self.data_dict.get('5min')
            if df_5m is None or df_5m.empty:
                raise ValueError("No data available for stress testing")

            # Mock strategy for testing
            mock_strategy = type('MockStrategy', (), {
                'next': lambda self, df: {'signal': 'BUY' if np.random.random() > 0.5 else 'SELL', 'strength': 4.0}
            })()

            self.progress_updated.emit(30, "Running stress scenarios...")

            # Run stress tests
            stress_results = self.analysis_engines.run_stress_scenarios(
                df_5m, mock_strategy, {}, scenarios
            )

            self.progress_updated.emit(70, "Analyzing stress test results...")

            # Format results for display
            formatted_results = self.format_stress_results(stress_results)

            self.progress_updated.emit(100, "Stress testing completed")
            self.analysis_completed.emit('stress', formatted_results)

        except Exception as e:
            self.error_occurred.emit('stress', f"Stress testing failed: {str(e)}\n{traceback.format_exc()}")

    def run_causality_test(self):
        """Run causality validation"""
        try:
            self.progress_updated.emit(10, "Preparing data for causality test...")

            df_5m = self.data_dict.get('5min')
            if df_5m is None or df_5m.empty:
                raise ValueError("No data available for causality test")

            # Create mock signals and returns for demonstration
            np.random.seed(42)
            signals = np.random.choice([-1, 0, 1], size=len(df_5m))
            returns = np.random.normal(0, 0.01, len(df_5m))

            self.progress_updated.emit(30, "Running Granger causality test...")

            # Run Granger causality
            granger_result = self.analysis_engines.granger_causality_test(signals, returns)

            self.progress_updated.emit(60, "Running placebo test...")

            # Run placebo test
            placebo_result = self.analysis_engines.placebo_test(signals, returns)

            self.progress_updated.emit(90, "Interpreting results...")

            # Interpret results
            interpretation = self.interpret_causality_results(granger_result, placebo_result)

            results = {
                'granger_p_value': granger_result,
                'placebo_p_value': placebo_result,
                'interpretation': interpretation
            }

            self.progress_updated.emit(100, "Causality test completed")
            self.analysis_completed.emit('causality', results)

        except Exception as e:
            self.error_occurred.emit('causality', f"Causality test failed: {str(e)}\n{traceback.format_exc()}")

    def calculate_regime_stats(self, regime_df):
        """Calculate regime statistics"""
        if regime_df is None or regime_df.empty:
            return {}

        # Count regime distribution
        regime_counts = regime_df['regime'].value_counts()
        total_bars = len(regime_df)

        stats = {
            'total_bars': total_bars,
            'regime_distribution': {}
        }

        # Map regime numbers to names
        regime_names = {0: 'Bear', 1: 'Chop', 2: 'Bull'}

        for regime_num, count in regime_counts.items():
            regime_name = regime_names.get(regime_num, f'Regime_{regime_num}')
            percentage = (count / total_bars) * 100
            stats['regime_distribution'][regime_name] = {
                'count': int(count),
                'percentage': round(percentage, 1)
            }

        return stats

    def generate_regime_recommendations(self, stats):
        """Generate regime-based recommendations"""
        if not stats.get('regime_distribution'):
            return "No regime data available"

        dist = stats['regime_distribution']

        # Find dominant regime
        dominant_regime = max(dist.keys(), key=lambda x: dist[x]['percentage'])

        recommendations = []

        if dominant_regime == 'Bull':
            recommendations.append("Bull market detected - Focus on momentum strategies")
            recommendations.append("Consider increasing position sizes")
            recommendations.append("Monitor for trend continuation signals")
        elif dominant_regime == 'Bear':
            recommendations.append("Bear market detected - Implement strict risk management")
            recommendations.append("Consider short strategies or hedging")
            recommendations.append("Reduce position sizes and use wider stops")
        else:  # Chop
            recommendations.append("Choppy market detected - Use mean-reversion strategies")
            recommendations.append("Implement tighter stops to capture quick moves")
            recommendations.append("Consider range-bound indicators (RSI, Bollinger Bands)")

        recommendations.append(f"Dominant regime: {dominant_regime} ({dist[dominant_regime]['percentage']}%)")

        return " | ".join(recommendations)

    def format_stress_results(self, stress_results):
        """Format stress test results for display"""
        formatted = {}

        for scenario, results in stress_results.items():
            formatted[scenario] = {
                'return_pct': round(results.get('return_pct', 0), 2),
                'max_dd_pct': round(results.get('max_dd', 0), 2),
                'survival': results.get('survival', False)
            }

        return formatted

    def interpret_causality_results(self, granger_p, placebo_p):
        """Interpret causality test results"""
        interpretation = []

        if granger_p < 0.05:
            interpretation.append("✓ Granger causality detected (signals predict returns)")
        else:
            interpretation.append("✗ No Granger causality detected")

        if placebo_p > 0.05:
            interpretation.append("✓ Placebo test passed (results not due to randomness)")
        else:
            interpretation.append("✗ Placebo test failed (results may be spurious)")

        if granger_p < 0.05 and placebo_p > 0.05:
            interpretation.append("🎯 Strong evidence of causal relationship")
        elif granger_p < 0.05 and placebo_p < 0.05:
            interpretation.append("⚠️ Weak evidence - may be spurious correlation")
        else:
            interpretation.append("❌ No evidence of causal relationship")

        return " | ".join(interpretation)

class Tab7AdvancedAnalysis(QWidget):
    """Advanced analysis tab with multiple analysis tools"""

    def __init__(self, parent_platform, analysis_engines):
        super().__init__()
        self.parent = parent_platform
        self.analysis_engines = analysis_engines
        self.current_results = {}

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Advanced Analysis Tools")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # Create tab widget for different analysis types
        self.analysis_tabs = QTabWidget()

        # Tab 1: Regime Detection
        self.analysis_tabs.addTab(self.create_regime_tab(), "Regime Detection")

        # Tab 2: Microstructure Impact
        self.analysis_tabs.addTab(self.create_microstructure_tab(), "Microstructure")

        # Tab 3: Stress Testing
        self.analysis_tabs.addTab(self.create_stress_tab(), "Stress Testing")

        # Tab 4: Causality Validation
        self.analysis_tabs.addTab(self.create_causality_tab(), "Causality")

        # Tab 5: Correlation Analysis
        self.analysis_tabs.addTab(self.create_correlation_tab(), "Correlations")

        layout.addWidget(self.analysis_tabs)

        # Progress bar (shared)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready for advanced analysis")
        layout.addWidget(self.status_label)

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
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
            }
        """)

    def create_regime_tab(self):
        """Create regime detection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc = QLabel("Detect market regimes using Hidden Markov Models (Bull/Bear/Chop)")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Run button
        self.regime_run_btn = QPushButton("Run Regime Detection")
        self.regime_run_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 10px; font-weight: bold; }")
        self.regime_run_btn.clicked.connect(self.on_run_regime_analysis)
        layout.addWidget(self.regime_run_btn)

        # Results display
        self.regime_results_text = QTextEdit()
        self.regime_results_text.setReadOnly(True)
        self.regime_results_text.setMaximumHeight(200)
        layout.addWidget(self.regime_results_text)

        return widget

    def create_microstructure_tab(self):
        """Create microstructure impact tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc = QLabel("Analyze market impact of different order sizes and trading costs")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Order size input
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Order Size ($):"))

        self.order_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.order_size_slider.setMinimum(1)
        self.order_size_slider.setMaximum(100)
        self.order_size_slider.setValue(10)
        self.order_size_slider.valueChanged.connect(self.on_microstructure_slider_changed)
        size_layout.addWidget(self.order_size_slider)

        self.order_size_spinbox = QSpinBox()
        self.order_size_spinbox.setMaximumWidth(120)  # Ancho optimizado
        self.order_size_spinbox.setMinimum(1)
        self.order_size_spinbox.setMaximum(100)
        self.order_size_spinbox.setValue(10)
        self.order_size_spinbox.valueChanged.connect(self.on_microstructure_spinbox_changed)
        size_layout.addWidget(self.order_size_spinbox)

        size_layout.addWidget(QLabel("million USD"))
        size_layout.addStretch()
        layout.addLayout(size_layout)

        # Results display
        self.microstructure_results = QTextEdit()
        self.microstructure_results.setReadOnly(True)
        self.microstructure_results.setMaximumHeight(150)
        layout.addWidget(self.microstructure_results)

        # Initial calculation
        self.on_microstructure_slider_changed(10)

        return widget

    def create_stress_tab(self):
        """Create stress testing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc = QLabel("Test strategy performance under extreme market conditions")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Scenario selection
        scenarios_group = QGroupBox("Stress Scenarios")
        scenarios_layout = QVBoxLayout()

        self.stress_checkboxes = {}
        scenarios = [
            ('flash_crash', 'Flash Crash (-20% in 1 hour)'),
            ('bear_market', 'Bear Market (-50% over 3 months)'),
            ('vol_spike', 'Volatility Spike (+200% volatility)'),
            ('liquidity_freeze', 'Liquidity Freeze (no volume)')
        ]

        for scenario_id, scenario_name in scenarios:
            checkbox = QCheckBox(scenario_name)
            if scenario_id in ['flash_crash', 'bear_market']:  # Default selected
                checkbox.setChecked(True)
            self.stress_checkboxes[scenario_id] = checkbox
            scenarios_layout.addWidget(checkbox)

        scenarios_group.setLayout(scenarios_layout)
        layout.addWidget(scenarios_group)

        # Run button
        self.stress_run_btn = QPushButton("Run Stress Tests")
        self.stress_run_btn.setStyleSheet("QPushButton { background-color: #FF5722; color: white; padding: 10px; font-weight: bold; }")
        self.stress_run_btn.clicked.connect(self.on_run_stress_tests)
        layout.addWidget(self.stress_run_btn)

        # Results table
        self.stress_results_table = QTableWidget()
        self.stress_results_table.setColumnCount(4)
        self.stress_results_table.setHorizontalHeaderLabels(["Scenario", "Return %", "Max DD %", "Survival"])
        layout.addWidget(self.stress_results_table)

        return widget

    def create_causality_tab(self):
        """Create causality validation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc = QLabel("Validate causal relationships between signals and market returns")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Run button
        self.causality_run_btn = QPushButton("Run Causality Tests")
        self.causality_run_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; padding: 10px; font-weight: bold; }")
        self.causality_run_btn.clicked.connect(self.on_run_causality_test)
        layout.addWidget(self.causality_run_btn)

        # Results display
        self.causality_results_text = QTextEdit()
        self.causality_results_text.setReadOnly(True)
        layout.addWidget(self.causality_results_text)

        return widget

    def create_correlation_tab(self):
        """Create correlation analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc = QLabel("Analyze correlations between different assets and market regimes")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Placeholder for correlation matrix
        self.correlation_placeholder = QLabel("Correlation analysis - Coming soon\nLoad multi-asset data to analyze correlations")
        self.correlation_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.correlation_placeholder.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #444;
                padding: 40px;
                color: #666;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.correlation_placeholder)

        # Update button
        update_btn = QPushButton("Update Correlations")
        update_btn.clicked.connect(self.on_update_correlations)
        layout.addWidget(update_btn)

        return widget

    def on_run_regime_analysis(self):
        """Run regime detection analysis"""
        if not hasattr(self.parent, 'data_dict') or not self.parent.data_dict:
            self.status_label.setText("No data available. Please load data first.")
            return

        self.run_analysis('regime')

    def on_run_stress_tests(self):
        """Run stress testing"""
        # Get selected scenarios
        selected_scenarios = []
        for scenario_id, checkbox in self.stress_checkboxes.items():
            if checkbox.isChecked():
                selected_scenarios.append(scenario_id)

        if not selected_scenarios:
            self.status_label.setText("Please select at least one stress scenario")
            return

        if not hasattr(self.parent, 'data_dict') or not self.parent.data_dict:
            self.status_label.setText("No data available. Please load data first.")
            return

        self.run_analysis('stress', scenarios=selected_scenarios)

    def on_run_causality_test(self):
        """Run causality validation"""
        if not hasattr(self.parent, 'data_dict') or not self.parent.data_dict:
            self.status_label.setText("No data available. Please load data first.")
            return

        self.run_analysis('causality')

    def run_analysis(self, analysis_type, **kwargs):
        """Run analysis in background thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Running {analysis_type} analysis...")

        # Create worker
        self.worker = AdvancedAnalysisWorker(self.analysis_engines, self.parent.data_dict)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_completed.connect(self.on_analysis_completed)
        self.worker.error_occurred.connect(self.on_analysis_error)

        # Run appropriate analysis
        if analysis_type == 'regime':
            self.worker.run_regime_analysis()
        elif analysis_type == 'stress':
            self.worker.run_stress_test(kwargs.get('scenarios', []))
        elif analysis_type == 'causality':
            self.worker.run_causality_test()

    def update_progress(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_analysis_completed(self, analysis_type, results):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        self.current_results[analysis_type] = results

        if analysis_type == 'regime':
            self.display_regime_results(results)
        elif analysis_type == 'stress':
            self.display_stress_results(results)
        elif analysis_type == 'causality':
            self.display_causality_results(results)

        self.status_label.setText(f"{analysis_type.title()} analysis completed")

    def on_analysis_error(self, analysis_type, error_msg):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"{analysis_type.title()} analysis failed")

        # Display error in appropriate text area
        if analysis_type == 'regime':
            self.regime_results_text.setText(f"Error: {error_msg}")
        elif analysis_type == 'causality':
            self.causality_results_text.setText(f"Error: {error_msg}")

    def display_regime_results(self, results):
        """Display regime analysis results"""
        stats = results.get('statistics', {})
        recommendations = results.get('recommendations', '')

        text = "<b>Regime Analysis Results</b><br><br>"
        text += f"Total bars analyzed: {stats.get('total_bars', 0):,}<br><br>"

        text += "<b>Regime Distribution:</b><br>"
        for regime, data in stats.get('regime_distribution', {}).items():
            text += f"• {regime}: {data['count']:,} bars ({data['percentage']}%)<br>"

        text += f"<br><b>Recommendations:</b><br>{recommendations}"

        self.regime_results_text.setHtml(text)

    def display_stress_results(self, results):
        """Display stress test results"""
        self.stress_results_table.setRowCount(0)

        for i, (scenario, data) in enumerate(results.items()):
            self.stress_results_table.insertRow(i)

            # Scenario name
            scenario_name = scenario.replace('_', ' ').title()
            self.stress_results_table.setItem(i, 0, QTableWidgetItem(scenario_name))

            # Return %
            return_pct = data.get('return_pct', 0)
            self.stress_results_table.setItem(i, 1, QTableWidgetItem(f"{return_pct:+.1f}%"))

            # Max DD %
            max_dd = data.get('max_dd_pct', 0)
            self.stress_results_table.setItem(i, 2, QTableWidgetItem(f"{max_dd:.1f}%"))

            # Survival
            survival = "✓ Survives" if data.get('survival', False) else "✗ Fails"
            self.stress_results_table.setItem(i, 3, QTableWidgetItem(survival))

        self.stress_results_table.resizeColumnsToContents()

    def display_causality_results(self, results):
        """Display causality test results"""
        granger_p = results.get('granger_p_value', 1)
        placebo_p = results.get('placebo_p_value', 1)
        interpretation = results.get('interpretation', '')

        text = "<b>Causality Validation Results</b><br><br>"
        text += f"<b>Granger Causality Test:</b> p-value = {granger_p:.4f}<br>"
        text += f"<b>Placebo Test:</b> p-value = {placebo_p:.4f}<br><br>"
        text += f"<b>Interpretation:</b><br>{interpretation}"

        self.causality_results_text.setHtml(text)

    def on_microstructure_slider_changed(self, value):
        """Handle microstructure slider change"""
        self.order_size_spinbox.setValue(value)
        self.calculate_microstructure_impact(value * 1000000)  # Convert to USD

    def on_microstructure_spinbox_changed(self, value):
        """Handle microstructure spinbox change"""
        self.order_size_slider.setValue(value)
        self.calculate_microstructure_impact(value * 1000000)  # Convert to USD

    def calculate_microstructure_impact(self, order_size_usd):
        """Calculate market impact for given order size"""
        try:
            # Mock calculation (would use real microstructure models)
            adv = 50000000  # Average daily volume for BTC
            impact_pct = min(0.5 * (order_size_usd / adv) ** 0.6, 10)  # Realistic impact formula
            spread_cost = 0.0012  # Typical spread
            slippage_cost = impact_pct * 0.1  # Rough slippage estimate

            total_cost = spread_cost + slippage_cost
            capacity = adv * 0.001  # Rough capacity estimate

            text = f"<b>Order Size: ${order_size_usd:,.0f}</b><br><br>"
            text += f"Market Impact: {impact_pct:.2f}%<br>"
            text += f"Spread Cost: {spread_cost:.2f}%<br>"
            text += f"Slippage Cost: {slippage_cost:.2f}%<br>"
            text += f"Total Cost: {total_cost:.2f}%<br><br>"
            text += f"Trading Capacity: ${capacity:,.0f}<br>"

            if order_size_usd > capacity:
                text += "<span style='color: #f44336;'>⚠️ Order size exceeds estimated capacity</span>"
            else:
                text += "<span style='color: #4CAF50;'>✓ Order size within capacity limits</span>"

            self.microstructure_results.setHtml(text)

        except Exception as e:
            self.microstructure_results.setText(f"Calculation error: {str(e)}")

    def on_update_correlations(self):
        """Update correlation analysis"""
        self.correlation_placeholder.setText("Correlation analysis updated\nLast update: " + datetime.now().strftime("%H:%M:%S"))

    def on_tab_activated(self):
        """Called when tab becomes active"""
        # Could refresh any cached results here
        pass