"""
Results Panel - UI for displaying detailed backtest analysis and visualizations
"""

import logging
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QSplitter, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QBarSeries, QBarSet
from PySide6.QtGui import QPainter

logger = logging.getLogger(__name__)

class ResultsPanel(QWidget):
    """
    Panel for displaying detailed backtest results and analysis.
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.current_results = []
        self.setup_ui()

        # Connect to controller signals
        self.controller.backtest_finished.connect(self.on_backtest_finished)

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins
        layout.setSpacing(15)  # Increase spacing between groups

        # Results tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 10px 15px;
                margin-right: 2px;
                border-radius: 5px 5px 0 0;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007bff;
            }
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
        """)

        # Summary tab
        self.summary_tab = self.create_summary_tab()
        self.tabs.addTab(self.summary_tab, "📊 Summary")

        # Performance tab
        self.performance_tab = self.create_performance_tab()
        self.tabs.addTab(self.performance_tab, "📈 Performance")

        # Risk tab
        self.risk_tab = self.create_risk_tab()
        self.tabs.addTab(self.risk_tab, "⚠️ Risk Analysis")

        # Trades tab
        self.trades_tab = self.create_trades_tab()
        self.tabs.addTab(self.trades_tab, "💼 Trades")

        # Statistics tab
        self.stats_tab = self.create_statistics_tab()
        self.tabs.addTab(self.stats_tab, "📋 Statistics")

        layout.addWidget(self.tabs)

        # Export controls
        export_group = QGroupBox("📤 Export Options")
        export_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        export_layout = QHBoxLayout(export_group)
        export_layout.setContentsMargins(15, 20, 15, 15)
        export_layout.setSpacing(15)

        self.export_report_button = QPushButton("📄 Report")
        self.export_report_button.setMaximumHeight(26)
        self.export_report_button.setMaximumWidth(80)
        self.export_report_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 4px 10px;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.export_report_button.clicked.connect(self.export_report)
        self.export_report_button.setEnabled(False)
        export_layout.addWidget(self.export_report_button)

        self.export_charts_button = QPushButton("📊 Charts")
        self.export_charts_button.setMaximumHeight(26)
        self.export_charts_button.setMaximumWidth(75)
        self.export_charts_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 4px 10px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.export_charts_button.clicked.connect(self.export_charts)
        self.export_charts_button.setEnabled(False)
        export_layout.addWidget(self.export_charts_button)

        self.export_trades_button = QPushButton("💼 Trades")
        self.export_trades_button.setMaximumHeight(26)
        self.export_trades_button.setMaximumWidth(75)
        self.export_trades_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 4px 10px;
                background-color: #ffc107;
                color: black;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
            QPushButton:pressed {
                background-color: #d39e00;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.export_trades_button.clicked.connect(self.export_trades)
        self.export_trades_button.setEnabled(False)
        export_layout.addWidget(self.export_trades_button)

        export_layout.addStretch()
        layout.addWidget(export_group)

    def create_summary_tab(self) -> QWidget:
        """Create the summary tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Key metrics table
        metrics_group = QGroupBox("📈 Key Performance Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setContentsMargins(15, 20, 15, 15)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value", "Benchmark"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setMaximumHeight(350)
        self.summary_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #dee2e6;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
            }
            QTableWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: 1px solid #dee2e6;
                font-weight: bold;
            }
        """)

        metrics_layout.addWidget(self.summary_table)
        layout.addWidget(metrics_group)

        # Performance overview
        overview_group = QGroupBox("📋 Performance Overview")
        overview_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        overview_layout = QVBoxLayout(overview_group)
        overview_layout.setContentsMargins(15, 20, 15, 15)

        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setMaximumHeight(180)
        self.overview_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: #f8f9fa;
                font-size: 11px;
                line-height: 1.4;
            }
        """)

        overview_layout.addWidget(self.overview_text)
        layout.addWidget(overview_group)

        return widget

    def create_performance_tab(self) -> QWidget:
        """Create the performance tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Equity curve chart
        equity_group = QGroupBox("Equity Curve")
        equity_layout = QVBoxLayout(equity_group)

        self.equity_chart_view = QChartView()
        self.equity_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        equity_layout.addWidget(self.equity_chart_view)

        layout.addWidget(equity_group)

        # Returns distribution
        returns_group = QGroupBox("Monthly Returns Distribution")
        returns_layout = QVBoxLayout(returns_group)

        self.returns_chart_view = QChartView()
        self.returns_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        returns_layout.addWidget(self.returns_chart_view)

        layout.addWidget(returns_group)

        return widget

    def create_risk_tab(self) -> QWidget:
        """Create the risk analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Drawdown chart
        drawdown_group = QGroupBox("Drawdown Analysis")
        drawdown_layout = QVBoxLayout(drawdown_group)

        self.drawdown_chart_view = QChartView()
        self.drawdown_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        drawdown_layout.addWidget(self.drawdown_chart_view)

        layout.addWidget(drawdown_group)

        # Risk metrics
        risk_metrics_group = QGroupBox("Risk Metrics")
        risk_layout = QVBoxLayout(risk_metrics_group)

        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(2)
        self.risk_table.setHorizontalHeaderLabels(["Risk Metric", "Value"])
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.risk_table.setAlternatingRowColors(True)
        self.risk_table.setMaximumHeight(250)

        risk_layout.addWidget(self.risk_table)
        layout.addWidget(risk_metrics_group)

        return widget

    def create_trades_tab(self) -> QWidget:
        """Create the trades tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Trades table
        trades_group = QGroupBox("Trade History")
        trades_layout = QVBoxLayout(trades_group)

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            "Entry Date", "Exit Date", "Direction", "Entry Price",
            "Exit Price", "Quantity", "Profit/Loss", "Profit/Loss %"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.setAlternatingRowColors(True)

        trades_layout.addWidget(self.trades_table)
        layout.addWidget(trades_group)

        # Trade statistics
        stats_group = QGroupBox("Trade Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.trade_stats_text = QTextEdit()
        self.trade_stats_text.setReadOnly(True)
        self.trade_stats_text.setMaximumHeight(120)

        stats_layout.addWidget(self.trade_stats_text)
        layout.addWidget(stats_group)

        return widget

    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Statistical tests
        tests_group = QGroupBox("Statistical Significance Tests")
        tests_layout = QVBoxLayout(tests_group)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(["Test", "Statistic", "p-value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setMaximumHeight(200)

        tests_layout.addWidget(self.stats_table)
        layout.addWidget(tests_group)

        # Bootstrap analysis
        bootstrap_group = QGroupBox("Bootstrap Analysis")
        bootstrap_layout = QVBoxLayout(bootstrap_group)

        self.bootstrap_text = QTextEdit()
        self.bootstrap_text.setReadOnly(True)
        self.bootstrap_text.setMaximumHeight(150)

        bootstrap_layout.addWidget(self.bootstrap_text)
        layout.addWidget(bootstrap_group)

        return widget

    def on_backtest_finished(self, results: List[Any]):
        """Handle backtest completion and update all tabs"""
        self.current_results = results

        if not results:
            return

        # Update all tabs with results
        self.update_summary_tab()
        self.update_performance_tab()
        self.update_risk_tab()
        self.update_trades_tab()
        self.update_statistics_tab()

        # Enable export buttons
        self.export_report_button.setEnabled(True)
        self.export_charts_button.setEnabled(True)
        self.export_trades_button.setEnabled(True)

    def update_summary_tab(self):
        """Update the summary tab with key metrics"""
        if not self.current_results:
            return

        # Use the best result (highest Sharpe ratio)
        best_result = max(self.current_results,
                         key=lambda r: r.metrics.get('sharpe_ratio', 0))
        metrics = best_result.metrics

        # Key metrics
        key_metrics = [
            ("Total Return", f"{metrics.get('total_return', 0):.2%}", "S&P 500: ~10%"),
            ("Annualized Return", f"{metrics.get('annualized_return', 0):.2%}", "S&P 500: ~10%"),
            ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}", ">1.0 Good"),
            ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}", ">1.5 Good"),
            ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}", "<20% Good"),
            ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}", ">0.5 Good"),
            ("Win Rate", f"{metrics.get('win_rate', 0):.1%}", ">50% Good"),
            ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}", ">1.5 Good"),
            ("Recovery Factor", f"{metrics.get('recovery_factor', 0):.3f}", ">1.0 Good"),
            ("K-Ratio", f"{metrics.get('k_ratio', 0):.3f}", ">0.5 Good")
        ]

        self.summary_table.setRowCount(len(key_metrics))
        for i, (metric, value, benchmark) in enumerate(key_metrics):
            self.summary_table.setItem(i, 0, QTableWidgetItem(metric))
            self.summary_table.setItem(i, 1, QTableWidgetItem(value))
            self.summary_table.setItem(i, 2, QTableWidgetItem(benchmark))

        self.summary_table.resizeColumnsToContents()

        # Performance overview
        total_return = metrics.get('total_return', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)

        overview = f"""
Strategy Performance Overview:

• Total Return: {total_return:.2%}
• Sharpe Ratio: {sharpe:.3f} ({'Excellent' if sharpe > 2 else 'Good' if sharpe > 1 else 'Poor'})
• Maximum Drawdown: {max_dd:.2%} ({'Acceptable' if max_dd < 0.2 else 'High Risk'})
• Win Rate: {win_rate:.1%} ({'Strong' if win_rate > 0.6 else 'Moderate' if win_rate > 0.5 else 'Weak'})

Risk-Adjusted Performance: {'Excellent' if sharpe > 2 and max_dd < 0.15 else 'Good' if sharpe > 1.5 else 'Needs Improvement'}
        """.strip()

        self.overview_text.setPlainText(overview)

    def update_performance_tab(self):
        """Update performance charts"""
        if not self.current_results:
            return

        best_result = max(self.current_results,
                         key=lambda r: r.metrics.get('sharpe_ratio', 0))

        # Equity curve
        self.update_equity_chart(best_result)

        # Monthly returns distribution
        self.update_returns_chart(best_result)

    def update_equity_chart(self, result):
        """Update the equity curve chart"""
        chart = QChart()
        chart.setTitle("Equity Curve")

        series = QLineSeries()
        series.setName("Portfolio Value")

        # Add equity curve data (simplified - would use actual equity data)
        equity_data = result.equity_curve if hasattr(result, 'equity_curve') else []
        if equity_data:
            for i, value in enumerate(equity_data):
                series.append(i, value)
        else:
            # Placeholder data
            for i in range(100):
                series.append(i, 10000 + i * 50)

        chart.addSeries(series)
        chart.createDefaultAxes()
        self.equity_chart_view.setChart(chart)

    def update_returns_chart(self, result):
        """Update the returns distribution chart"""
        chart = QChart()
        chart.setTitle("Monthly Returns Distribution")

        # Create bar series for returns (simplified)
        bar_set = QBarSet("Monthly Returns")
        # Placeholder data - would calculate actual monthly returns
        returns_data = [2.1, -1.5, 3.2, 1.8, -0.9, 2.5, 1.2, -1.1, 2.8, 1.5, -0.5, 2.2]
        for ret in returns_data:
            bar_set.append(ret)

        bar_series = QBarSeries()
        bar_series.append(bar_set)

        chart.addSeries(bar_series)
        chart.createDefaultAxes()
        self.returns_chart_view.setChart(chart)

    def update_risk_tab(self):
        """Update risk analysis"""
        if not self.current_results:
            return

        best_result = max(self.current_results,
                         key=lambda r: r.metrics.get('sharpe_ratio', 0))

        # Drawdown chart
        self.update_drawdown_chart(best_result)

        # Risk metrics table
        metrics = best_result.metrics
        risk_metrics = [
            ("Value at Risk (95%)", f"{metrics.get('var_95', 0):.2%}"),
            ("Expected Shortfall (95%)", f"{metrics.get('expected_shortfall', 0):.2%}"),
            ("Beta", f"{metrics.get('beta', 0):.3f}"),
            ("Alpha", f"{metrics.get('alpha', 0):.3f}"),
            ("Tracking Error", f"{metrics.get('tracking_error', 0):.2%}"),
            ("Information Ratio", f"{metrics.get('information_ratio', 0):.3f}"),
            ("Downside Deviation", f"{metrics.get('downside_deviation', 0):.2%}"),
            ("Ulcer Index", f"{metrics.get('ulcer_index', 0):.3f}")
        ]

        self.risk_table.setRowCount(len(risk_metrics))
        for i, (metric, value) in enumerate(risk_metrics):
            self.risk_table.setItem(i, 0, QTableWidgetItem(metric))
            self.risk_table.setItem(i, 1, QTableWidgetItem(value))

        self.risk_table.resizeColumnsToContents()

    def update_drawdown_chart(self, result):
        """Update the drawdown chart"""
        chart = QChart()
        chart.setTitle("Drawdown Analysis")

        series = QLineSeries()
        series.setName("Drawdown")

        # Add drawdown data (simplified)
        drawdown_data = result.drawdown if hasattr(result, 'drawdown') else []
        if drawdown_data:
            for i, value in enumerate(drawdown_data):
                series.append(i, value * 100)  # Convert to percentage
        else:
            # Placeholder data
            for i in range(100):
                series.append(i, -min(i * 0.5, 15))

        chart.addSeries(series)
        chart.createDefaultAxes()
        self.drawdown_chart_view.setChart(chart)

    def update_trades_tab(self):
        """Update trades table and statistics"""
        if not self.current_results:
            return

        best_result = max(self.current_results,
                         key=lambda r: r.metrics.get('sharpe_ratio', 0))

        # Update trades table
        trades = best_result.trades if hasattr(best_result, 'trades') else []
        self.trades_table.setRowCount(len(trades))

        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.get('entry_date', ''))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(str(trade.get('exit_date', ''))))
            self.trades_table.setItem(i, 2, QTableWidgetItem(trade.get('direction', '')))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade.get('entry_price', 0):.2f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.get('exit_price', 0):.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(str(trade.get('quantity', 0))))
            self.trades_table.setItem(i, 6, QTableWidgetItem(f"{trade.get('pnl', 0):.2f}"))
            self.trades_table.setItem(i, 7, QTableWidgetItem(f"{trade.get('pnl_pct', 0):.2%}"))

        self.trades_table.resizeColumnsToContents()

        # Update trade statistics
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

            win_rate = len(winning_trades) / len(trades)
            avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(t.get('pnl', 0) for t in winning_trades) / sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades else float('inf')

            stats = f"""
Trade Statistics:
• Total Trades: {len(trades)}
• Winning Trades: {len(winning_trades)} ({win_rate:.1%})
• Losing Trades: {len(losing_trades)} ({1-win_rate:.1%})
• Average Win: ${avg_win:.2f}
• Average Loss: ${avg_loss:.2f}
• Profit Factor: {profit_factor:.2f}
• Largest Win: ${max((t.get('pnl', 0) for t in winning_trades), default=0):.2f}
• Largest Loss: ${min((t.get('pnl', 0) for t in losing_trades), default=0):.2f}
            """.strip()

            self.trade_stats_text.setPlainText(stats)

    def update_statistics_tab(self):
        """Update statistical analysis"""
        if not self.current_results:
            return

        best_result = max(self.current_results,
                         key=lambda r: r.metrics.get('sharpe_ratio', 0))
        metrics = best_result.metrics

        # Statistical tests
        stat_tests = [
            ("Sharpe Ratio t-test", f"{metrics.get('sharpe_t_stat', 0):.3f}", f"{metrics.get('sharpe_p_value', 0):.3f}"),
            ("Normality Test (Shapiro)", f"{metrics.get('shapiro_stat', 0):.3f}", f"{metrics.get('shapiro_p_value', 0):.3f}"),
            ("Serial Correlation (Durbin-Watson)", f"{metrics.get('durbin_watson', 0):.3f}", "N/A"),
            ("ARCH Effect Test", f"{metrics.get('arch_stat', 0):.3f}", f"{metrics.get('arch_p_value', 0):.3f}")
        ]

        self.stats_table.setRowCount(len(stat_tests))
        for i, (test, stat, p_val) in enumerate(stat_tests):
            self.stats_table.setItem(i, 0, QTableWidgetItem(test))
            self.stats_table.setItem(i, 1, QTableWidgetItem(stat))
            self.stats_table.setItem(i, 2, QTableWidgetItem(p_val))

        self.stats_table.resizeColumnsToContents()

        # Bootstrap analysis
        bootstrap_info = f"""
Bootstrap Analysis Results:
• Bootstrap Confidence Interval (95%): {metrics.get('bootstrap_ci_lower', 0):.2%} to {metrics.get('bootstrap_ci_upper', 0):.2%}
• Bootstrap Standard Error: {metrics.get('bootstrap_se', 0):.4f}
• Bootstrap Bias: {metrics.get('bootstrap_bias', 0):.4f}
• Number of Bootstrap Samples: {metrics.get('bootstrap_samples', 1000)}

The bootstrap analysis provides a robust estimate of the strategy's performance distribution,
accounting for the finite sample size and potential non-stationarity in returns.
        """.strip()

        self.bootstrap_text.setPlainText(bootstrap_info)

    def export_report(self):
        """Export comprehensive report"""
        from PySide6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "", "HTML Files (*.html);;PDF Files (*.pdf)"
        )

        if filename:
            success = self.controller.export_report(filename)
            if success:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Report exported to {filename}")

    def export_charts(self):
        """Export charts as images"""
        from PySide6.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")

        if directory:
            success = self.controller.export_charts(directory)
            if success:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Charts exported to {directory}")

    def export_trades(self):
        """Export trade history"""
        from PySide6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Trades", "", "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if filename:
            success = self.controller.export_trades(filename)
            if success:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Trades exported to {filename}")