import sys
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
                               QProgressBar, QComboBox, QPushButton, QTextEdit,
                               QSplitter, QFrame, QScrollArea, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

class Tab11RiskMetrics(QWidget):
    """Advanced Risk Metrics Dashboard - Tab 11"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.backtester = parent.backtester if hasattr(parent, 'backtester') else None
        self.analysis_engines = parent.analysis_engines if hasattr(parent, 'analysis_engines') else None

        # Initialize data storage
        self.current_results = None
        self.mae_mfe_data = None
        self.stress_test_results = None

        self.init_ui()
        self.setup_connections()
        self.load_initial_data()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("📊 Risk Metrics Dashboard")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(self.refresh_btn)

        layout.addLayout(header_layout)

        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)

        # Top section - Key Risk Metrics
        self.create_risk_metrics_section()
        splitter.addWidget(self.risk_metrics_group)

        # Middle section - Charts and Visualizations
        self.create_visualization_section()
        splitter.addWidget(self.visualization_group)

        # Bottom section - Detailed Analysis
        self.create_detailed_analysis_section()
        splitter.addWidget(self.detailed_analysis_group)

        # Set splitter proportions: compact metrics, LARGE charts, medium analysis
        splitter.setStretchFactor(0, 15)  # Risk metrics: 15%
        splitter.setStretchFactor(1, 60)  # Charts: 60%
        splitter.setStretchFactor(2, 25)  # Detailed analysis: 25%
        splitter.setSizes([150, 500, 250])

        layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("Ready - Load backtest results to view risk metrics")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

    def create_risk_metrics_section(self):
        """Create the key risk metrics display section"""
        self.risk_metrics_group = QGroupBox("🎯 Key Risk Metrics")
        layout = QGridLayout()

        # Row 1: Core Risk Metrics
        metrics = [
            ("Maximum Drawdown", "max_dd", "%"),
            ("Value at Risk (95%)", "var_95", "%"),
            ("Expected Shortfall (95%)", "es_95", "%"),
            ("Sharpe Ratio", "sharpe", ""),
            ("Sortino Ratio", "sortino", ""),
            ("Calmar Ratio", "calmar", "")
        ]

        self.metric_labels = {}
        for i, (label, key, suffix) in enumerate(metrics):
            row = i // 3
            col = (i % 3) * 3

            # Label
            lbl = QLabel(f"{label}:")
            lbl.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(lbl, row, col)

            # Value
            val_lbl = QLabel("--")
            val_lbl.setFont(QFont("Arial", 10))
            val_lbl.setStyleSheet("color: #2E86C1; font-weight: bold;")
            layout.addWidget(val_lbl, row, col + 1)
            self.metric_labels[key] = val_lbl

            # Suffix
            if suffix:
                suffix_lbl = QLabel(suffix)
                layout.addWidget(suffix_lbl, row, col + 2)

        # Row 2: MAE/MFE Metrics
        mae_mfe_metrics = [
            ("Average MAE", "avg_mae", "%"),
            ("Average MFE", "avg_mfe", "%"),
            ("MAE/MFE Ratio", "mae_mfe_ratio", ""),
            ("Max MAE", "max_mae", "%"),
            ("Max MFE", "max_mfe", "%"),
            ("Recovery Factor", "recovery_factor", "")
        ]

        for i, (label, key, suffix) in enumerate(mae_mfe_metrics):
            row = 2 + (i // 3)
            col = (i % 3) * 3

            lbl = QLabel(f"{label}:")
            lbl.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(lbl, row, col)

            val_lbl = QLabel("--")
            val_lbl.setFont(QFont("Arial", 10))
            val_lbl.setStyleSheet("color: #E74C3C; font-weight: bold;")
            layout.addWidget(val_lbl, row, col + 1)
            self.metric_labels[key] = val_lbl

            if suffix:
                suffix_lbl = QLabel(suffix)
                layout.addWidget(suffix_lbl, row, col + 2)

        self.risk_metrics_group.setLayout(layout)

    def create_visualization_section(self):
        """Create the visualization section with charts"""
        self.visualization_group = QGroupBox("📈 Risk Visualizations")
        layout = QVBoxLayout()

        # Chart selector
        chart_layout = QHBoxLayout()
        chart_layout.addWidget(QLabel("Chart Type:"))

        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "MAE/MFE Distribution",
            "Drawdown Analysis",
            "Volatility Clustering",
            "Stress Test Scenarios",
            "Risk-Return Scatter",
            "Tail Risk Analysis"
        ])
        self.chart_combo.currentTextChanged.connect(self.update_chart)
        chart_layout.addWidget(self.chart_combo)

        chart_layout.addStretch()
        layout.addLayout(chart_layout)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Chart controls
        controls_layout = QHBoxLayout()

        self.log_scale_cb = QPushButton("Log Scale")
        self.log_scale_cb.setCheckable(True)
        self.log_scale_cb.clicked.connect(self.update_chart)
        controls_layout.addWidget(self.log_scale_cb)

        self.show_grid_cb = QPushButton("Show Grid")
        self.show_grid_cb.setCheckable(True)
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.clicked.connect(self.update_chart)
        controls_layout.addWidget(self.show_grid_cb)

        controls_layout.addStretch()

        self.export_chart_btn = QPushButton("💾 Export Chart")
        self.export_chart_btn.clicked.connect(self.export_chart)
        controls_layout.addWidget(self.export_chart_btn)

        layout.addLayout(controls_layout)

        self.visualization_group.setLayout(layout)

    def create_detailed_analysis_section(self):
        """Create the detailed analysis section"""
        self.detailed_analysis_group = QGroupBox("🔍 Detailed Risk Analysis")
        layout = QHBoxLayout()

        # Left side - Risk Table
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        left_layout.addWidget(QLabel("📋 Risk Events Log"))

        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(5)
        self.risk_table.setHorizontalHeaderLabels([
            "Date", "Type", "Severity", "Impact", "Description"
        ])
        self.risk_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.risk_table)

        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget)

        # Right side - Analysis Text
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        right_layout.addWidget(QLabel("📊 Risk Analysis Report"))

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlainText("Load backtest results to generate risk analysis report...")
        right_layout.addWidget(self.analysis_text)

        # Stress test button
        stress_layout = QHBoxLayout()
        self.stress_test_btn = QPushButton("⚡ Run Stress Test")
        self.stress_test_btn.clicked.connect(self.run_stress_test)
        stress_layout.addWidget(self.stress_test_btn)

        self.stress_progress = QProgressBar()
        self.stress_progress.setVisible(False)
        stress_layout.addWidget(self.stress_progress)

        stress_layout.addStretch()
        right_layout.addLayout(stress_layout)

        right_widget.setLayout(right_layout)
        layout.addWidget(right_widget)

        self.detailed_analysis_group.setLayout(layout)

    def setup_connections(self):
        """Setup signal connections"""
        # Auto-refresh timer (every 30 seconds)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)
        self.refresh_timer.start(30000)  # 30 seconds

    def load_initial_data(self):
        """Load initial data if available"""
        try:
            if self.backtester and hasattr(self.backtester, 'last_results'):
                self.update_risk_metrics(self.backtester.last_results)
        except Exception as e:
            self.status_label.setText(f"Error loading initial data: {str(e)}")

    def update_risk_metrics(self, results):
        """Update all risk metrics displays"""
        try:
            self.current_results = results

            if not results or 'metrics' not in results:
                self.status_label.setText("No valid results to display")
                return

            metrics = results.get('metrics', {})

            # Update core metrics
            core_metrics = {
                'max_dd': metrics.get('max_drawdown', 0) * 100,
                'var_95': metrics.get('var_95', 0) * 100,
                'es_95': metrics.get('expected_shortfall', 0) * 100,
                'sharpe': metrics.get('sharpe_ratio', 0),
                'sortino': metrics.get('sortino_ratio', 0),
                'calmar': metrics.get('calmar_ratio', 0)
            }

            for key, value in core_metrics.items():
                if key in self.metric_labels:
                    if key in ['max_dd', 'var_95', 'es_95']:
                        self.metric_labels[key].setText(f"{value:.2f}")
                    else:
                        self.metric_labels[key].setText(f"{value:.3f}")

            # Update MAE/MFE metrics if available
            if 'mae_mfe' in results:
                mae_mfe = results['mae_mfe']
                mae_mfe_metrics = {
                    'avg_mae': mae_mfe.get('avg_mae', 0) * 100,
                    'avg_mfe': mae_mfe.get('avg_mfe', 0) * 100,
                    'mae_mfe_ratio': mae_mfe.get('ratio', 0),
                    'max_mae': mae_mfe.get('max_mae', 0) * 100,
                    'max_mfe': mae_mfe.get('max_mfe', 0) * 100,
                    'recovery_factor': mae_mfe.get('recovery_factor', 0)
                }

                for key, value in mae_mfe_metrics.items():
                    if key in self.metric_labels:
                        if key in ['avg_mae', 'avg_mfe', 'max_mae', 'max_mfe']:
                            self.metric_labels[key].setText(f"{value:.2f}")
                        else:
                            self.metric_labels[key].setText(f"{value:.3f}")

            # Update risk events table
            self.update_risk_events_table(results)

            # Update analysis report
            self.update_analysis_report(results)

            # Update chart
            self.update_chart()

            self.status_label.setText("Risk metrics updated successfully")

        except Exception as e:
            self.status_label.setText(f"Error updating risk metrics: {str(e)}")

    def update_risk_events_table(self, results):
        """Update the risk events table"""
        try:
            self.risk_table.setRowCount(0)

            if 'risk_events' not in results:
                return

            events = results['risk_events']
            for event in events[-20:]:  # Show last 20 events
                row = self.risk_table.rowCount()
                self.risk_table.insertRow(row)

                self.risk_table.setItem(row, 0, QTableWidgetItem(event.get('date', '')))
                self.risk_table.setItem(row, 1, QTableWidgetItem(event.get('type', '')))
                self.risk_table.setItem(row, 2, QTableWidgetItem(event.get('severity', 'Medium')))
                self.risk_table.setItem(row, 3, QTableWidgetItem(f"{event.get('impact', 0):.2f}%"))
                self.risk_table.setItem(row, 4, QTableWidgetItem(event.get('description', '')))

            # Color code severity
            for row in range(self.risk_table.rowCount()):
                severity_item = self.risk_table.item(row, 2)
                if severity_item:
                    severity = severity_item.text()
                    if severity == 'High':
                        severity_item.setBackground(QColor('#E74C3C'))
                    elif severity == 'Medium':
                        severity_item.setBackground(QColor('#F39C12'))
                    else:
                        severity_item.setBackground(QColor('#27AE60'))

        except Exception as e:
            print(f"Error updating risk events table: {e}")

    def update_analysis_report(self, results):
        """Update the detailed analysis report"""
        try:
            report = "📊 RISK ANALYSIS REPORT\n\n"

            if 'metrics' in results:
                metrics = results['metrics']
                report += f"PERFORMANCE METRICS:\n"
                report += f"• Total Return: {metrics.get('total_return', 0):.2f}%\n"
                report += f"• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
                report += f"• Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
                report += f"• Win Rate: {metrics.get('win_rate', 0):.1f}%\n\n"

            if 'mae_mfe' in results:
                mae_mfe = results['mae_mfe']
                report += f"MAE/MFE ANALYSIS:\n"
                report += f"• Average MAE: {mae_mfe.get('avg_mae', 0):.2f}%\n"
                report += f"• Average MFE: {mae_mfe.get('avg_mfe', 0):.2f}%\n"
                report += f"• MAE/MFE Ratio: {mae_mfe.get('ratio', 0):.2f}\n"
                report += f"• Risk Assessment: {self.assess_risk_level(mae_mfe)}\n\n"

            if 'stress_test' in results:
                stress = results['stress_test']
                report += f"STRESS TEST RESULTS:\n"
                report += f"• Worst Case: {stress.get('worst_case', 0):.2f}%\n"
                report += f"• 99% VaR: {stress.get('var_99', 0):.2f}%\n"
                report += f"• Stress Test Passed: {stress.get('passed', False)}\n\n"

            report += f"RISK RECOMMENDATIONS:\n"
            report += self.generate_risk_recommendations(results)

            self.analysis_text.setPlainText(report)

        except Exception as e:
            self.analysis_text.setPlainText(f"Error generating analysis report: {str(e)}")

    def assess_risk_level(self, mae_mfe):
        """Assess overall risk level based on MAE/MFE"""
        avg_mae = mae_mfe.get('avg_mae', 0)
        avg_mfe = mae_mfe.get('avg_mfe', 0)
        ratio = mae_mfe.get('ratio', 0)

        if avg_mae < 0.02 and ratio > 2.0:
            return "🟢 LOW RISK - Excellent risk control"
        elif avg_mae < 0.05 and ratio > 1.5:
            return "🟡 MEDIUM RISK - Acceptable risk/reward"
        else:
            return "🔴 HIGH RISK - Needs risk management improvement"

    def generate_risk_recommendations(self, results):
        """Generate risk management recommendations"""
        recommendations = []

        if 'metrics' in results:
            metrics = results['metrics']
            max_dd = metrics.get('max_drawdown', 0)

            if max_dd > 0.20:
                recommendations.append("• Consider reducing position sizes - Max DD > 20%")
            if metrics.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("• Sharpe ratio below 1.0 - Consider strategy optimization")

        if 'mae_mfe' in results:
            mae_mfe = results['mae_mfe']
            if mae_mfe.get('avg_mae', 0) > 0.05:
                recommendations.append("• High average MAE - Consider tighter stop losses")
            if mae_mfe.get('ratio', 0) < 1.5:
                recommendations.append("• Poor MAE/MFE ratio - Review reward/risk profile")

        if not recommendations:
            recommendations.append("• Risk profile appears acceptable - Continue monitoring")

        return "\n".join(recommendations)

    def update_chart(self):
        """Update the current chart based on selection"""
        try:
            chart_type = self.chart_combo.currentText()
            self.figure.clear()

            if not self.current_results:
                self.figure.text(0.5, 0.5, 'No data available\nRun a backtest first',
                               ha='center', va='center', fontsize=12)
                self.canvas.draw()
                return

            if chart_type == "MAE/MFE Distribution":
                self.plot_mae_mfe_distribution()
            elif chart_type == "Drawdown Analysis":
                self.plot_drawdown_analysis()
            elif chart_type == "Volatility Clustering":
                self.plot_volatility_clustering()
            elif chart_type == "Stress Test Scenarios":
                self.plot_stress_test_scenarios()
            elif chart_type == "Risk-Return Scatter":
                self.plot_risk_return_scatter()
            elif chart_type == "Tail Risk Analysis":
                self.plot_tail_risk_analysis()

            self.canvas.draw()

        except Exception as e:
            self.figure.clear()
            self.figure.text(0.5, 0.5, f'Error creating chart:\n{str(e)}',
                           ha='center', va='center', fontsize=10)
            self.canvas.draw()

    def plot_mae_mfe_distribution(self):
        """Plot MAE/MFE distribution histogram"""
        if 'mae_mfe' not in self.current_results:
            return

        mae_mfe = self.current_results['mae_mfe']
        mae_values = mae_mfe.get('mae_values', [])
        mfe_values = mae_mfe.get('mfe_values', [])

        ax = self.figure.add_subplot(111)
        ax.hist(mae_values, alpha=0.7, label='MAE (Adverse)', bins=20, color='red')
        ax.hist(mfe_values, alpha=0.7, label='MFE (Favorable)', bins=20, color='green')

        ax.set_xlabel('Excursion (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('MAE/MFE Distribution')
        ax.legend()
        ax.grid(self.show_grid_cb.isChecked())

        if self.log_scale_cb.isChecked():
            ax.set_yscale('log')

    def plot_drawdown_analysis(self):
        """Plot drawdown analysis over time"""
        if 'returns' not in self.current_results:
            return

        returns = self.current_results['returns']
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        ax = self.figure.add_subplot(111)
        ax.fill_between(range(len(drawdown)), 0, drawdown.values * 100,
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.values * 100, color='red', linewidth=1)

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Analysis Over Time')
        ax.legend()
        ax.grid(self.show_grid_cb.isChecked())

    def plot_volatility_clustering(self):
        """Plot volatility clustering analysis"""
        if 'returns' not in self.current_results:
            return

        returns = pd.Series(self.current_results['returns'])
        volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized

        ax = self.figure.add_subplot(111)
        ax.plot(volatility.values * 100, color='blue', linewidth=1, label='20-day Volatility')

        # Add volatility clusters
        high_vol_threshold = volatility.quantile(0.8)
        ax.axhline(y=high_vol_threshold * 100, color='red', linestyle='--',
                  alpha=0.7, label='High Volatility Threshold')

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Annualized Volatility (%)')
        ax.set_title('Volatility Clustering Analysis')
        ax.legend()
        ax.grid(self.show_grid_cb.isChecked())

    def plot_stress_test_scenarios(self):
        """Plot stress test scenario analysis"""
        if not self.stress_test_results:
            self.run_stress_test()
            return

        scenarios = self.stress_test_results.get('scenarios', {})

        ax = self.figure.add_subplot(111)

        scenario_names = list(scenarios.keys())
        losses = [scenarios[s]['loss'] * 100 for s in scenario_names]
        probabilities = [scenarios[s]['probability'] for s in scenario_names]

        bars = ax.bar(scenario_names, losses, color='orange', alpha=0.7)

        # Add probability labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{prob:.1%}', ha='center', va='bottom')

        ax.set_xlabel('Stress Scenario')
        ax.set_ylabel('Portfolio Loss (%)')
        ax.set_title('Stress Test Scenarios')
        ax.grid(self.show_grid_cb.isChecked())

        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')

    def plot_risk_return_scatter(self):
        """Plot risk-return scatter plot"""
        if 'trades' not in self.current_results:
            return

        trades = self.current_results['trades']
        returns = [t.get('pnl', 0) for t in trades]
        risks = [t.get('mae', 0) for t in trades]

        ax = self.figure.add_subplot(111)
        scatter = ax.scatter(risks, returns, alpha=0.6, c=returns,
                           cmap='RdYlGn', edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Risk (MAE %)')
        ax.set_ylabel('Return (PnL %)')
        ax.set_title('Risk-Return Scatter Plot')
        ax.grid(self.show_grid_cb.isChecked())

        # Add colorbar
        cbar = self.figure.colorbar(scatter)
        cbar.set_label('Return (%)')

    def plot_tail_risk_analysis(self):
        """Plot tail risk analysis (VaR, CVaR/ES)"""
        if 'returns' not in self.current_results:
            return

        returns = np.array(self.current_results['returns'])

        # Calculate VaR and CVaR at different confidence levels
        confidence_levels = [0.95, 0.99, 0.999]
        var_levels = []
        cvar_levels = []

        for conf in confidence_levels:
            # VaR: percentile de pérdidas (negativo)
            alpha = 1 - conf
            var = np.percentile(returns, alpha * 100)
            
            # CVaR (Expected Shortfall): promedio de returns <= VaR
            # Es decir, promedio de las peores pérdidas que exceden VaR
            tail_losses = returns[returns <= var]
            if len(tail_losses) > 0:
                cvar = tail_losses.mean()
            else:
                cvar = var  # Si no hay datos, usar VaR
            
            var_levels.append(var * 100)
            cvar_levels.append(cvar * 100)

        ax = self.figure.add_subplot(111)

        x = np.arange(len(confidence_levels))
        width = 0.35

        ax.bar(x - width/2, var_levels, width, label='VaR', color='red', alpha=0.7)
        ax.bar(x + width/2, cvar_levels, width, label='CVaR', color='darkred', alpha=0.7)

        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Loss (%)')
        ax.set_title('Tail Risk Analysis (VaR vs CVaR)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(c*100)}%' for c in confidence_levels])
        ax.legend()
        ax.grid(self.show_grid_cb.isChecked())

    def run_stress_test(self):
        """Run comprehensive stress test analysis"""
        try:
            self.stress_progress.setVisible(True)
            self.stress_progress.setRange(0, 0)  # Indeterminate progress

            # Simulate stress test scenarios
            scenarios = {
                'Market Crash (-20%)': {'shock': -0.20, 'probability': 0.05},
                'Flash Crash (-10%)': {'shock': -0.10, 'probability': 0.10},
                'High Volatility (+50%)': {'shock': 0.0, 'vol_shock': 0.50, 'probability': 0.15},
                'Liquidity Crisis': {'shock': -0.05, 'spread_shock': 2.0, 'probability': 0.08},
                'Interest Rate Hike': {'shock': -0.03, 'probability': 0.12}
            }

            # Calculate portfolio impact for each scenario
            portfolio_value = 10000  # Assume $10k portfolio
            for scenario, params in scenarios.items():
                if 'vol_shock' in params:
                    # Volatility shock affects option-like strategies
                    impact = params['shock'] + params['vol_shock'] * 0.1
                elif 'spread_shock' in params:
                    # Spread shock affects transaction costs
                    impact = params['shock'] * (1 + params['spread_shock'])
                else:
                    impact = params['shock']

                scenarios[scenario]['loss'] = abs(impact)
                scenarios[scenario]['portfolio_impact'] = portfolio_value * impact

            self.stress_test_results = {'scenarios': scenarios}

            # Update results if available
            if self.current_results:
                self.current_results['stress_test'] = self.stress_test_results

            self.stress_progress.setVisible(False)
            self.update_chart()  # Refresh current chart
            self.update_analysis_report(self.current_results)

            self.status_label.setText("Stress test completed successfully")

        except Exception as e:
            self.stress_progress.setVisible(False)
            self.status_label.setText(f"Error running stress test: {str(e)}")

    def export_chart(self):
        """Export current chart to file"""
        try:
            from PySide6.QtWidgets import QFileDialog
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"risk_metrics_{self.chart_combo.currentText().replace(' ', '_').lower()}_{timestamp}.png"

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Chart", default_name,
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )

            if file_path:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_label.setText(f"Chart exported to: {file_path}")

        except Exception as e:
            self.status_label.setText(f"Error exporting chart: {str(e)}")

    def refresh_data(self):
        """Manually refresh risk metrics data"""
        try:
            if self.backtester and hasattr(self.backtester, 'last_results'):
                self.update_risk_metrics(self.backtester.last_results)
            else:
                self.status_label.setText("No backtest results available - run a backtest first")

        except Exception as e:
            self.status_label.setText(f"Error refreshing data: {str(e)}")

    def auto_refresh(self):
        """Auto-refresh data periodically"""
        # Only refresh if tab is visible and we have data
        if self.isVisible() and self.current_results:
            self.refresh_data()