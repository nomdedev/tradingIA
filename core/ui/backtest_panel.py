"""
Backtest Panel - UI for running and monitoring backtests
"""

import logging
from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)

class BacktestPanel(QWidget):
    """
    Panel for running and monitoring backtest operations.
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.current_results = []
        self.setup_ui()

        # Connect to controller signals
        self.controller.backtest_started.connect(self.on_backtest_started)
        self.controller.backtest_finished.connect(self.on_backtest_finished)
        self.controller.backtest_progress.connect(self.on_backtest_progress)
        self.controller.backtest_error.connect(self.on_backtest_error)

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins
        layout.setSpacing(15)  # Increase spacing between groups

        # Control buttons
        controls_group = QGroupBox("🎯 Backtest Controls")
        controls_group.setStyleSheet("""
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
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(15, 20, 15, 15)
        controls_layout.setSpacing(15)

        self.run_button = QPushButton("▶️ Run")
        self.run_button.setMaximumHeight(26)
        self.run_button.setMaximumWidth(80)
        self.run_button.setStyleSheet("""
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
        self.run_button.clicked.connect(self.run_backtests)
        self.run_button.setEnabled(False)
        controls_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.setMaximumHeight(26)
        self.stop_button.setMaximumWidth(70)
        self.stop_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 4px 10px;
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.stop_button.clicked.connect(self.stop_backtests)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        self.export_button = QPushButton("📊 Export")
        self.export_button.setMaximumHeight(26)
        self.export_button.setMaximumWidth(85)
        self.export_button.setStyleSheet("""
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
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        controls_layout.addWidget(self.export_button)

        controls_layout.addStretch()
        layout.addWidget(controls_group)

        # Progress and status
        status_group = QGroupBox("📈 Status & Progress")
        status_group.setStyleSheet("""
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
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(15, 20, 15, 15)
        status_layout.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: #f8f9fa;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
        """)
        status_layout.addWidget(self.status_text)

        layout.addWidget(status_group)

        # Results table
        results_group = QGroupBox("📊 Results Summary")
        results_group.setStyleSheet("""
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
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(15, 20, 15, 15)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Strategy", "Total Return", "Sharpe Ratio", "Max Drawdown",
            "Win Rate", "Profit Factor", "Execution Time", "Parameters"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setMinimumHeight(200)
        self.results_table.setStyleSheet("""
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

        results_layout.addWidget(self.results_table)
        layout.addWidget(results_group)

        # Detailed results
        details_group = QGroupBox("🔍 Detailed Metrics")
        details_group.setStyleSheet("""
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
        details_layout = QVBoxLayout(details_group)
        details_layout.setContentsMargins(15, 20, 15, 15)

        self.details_table = QTableWidget()
        self.details_table.setColumnCount(2)
        self.details_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.details_table.setAlternatingRowColors(True)
        self.details_table.setMaximumHeight(250)
        self.details_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #dee2e6;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #e9ecef;
            }
            QTableWidget::item:selected {
                background-color: #28a745;
                color: white;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 6px;
                border: 1px solid #dee2e6;
                font-weight: bold;
            }
        """)

        # Connect table selection
        self.results_table.itemSelectionChanged.connect(self.show_detailed_metrics)

        details_layout.addWidget(self.details_table)
        layout.addWidget(details_group)

        # Initialize status
        self.update_status("Ready to run backtests")

    def run_backtests(self):
        """Run the configured backtests"""
        # Get configurations from strategy panel (assuming it's accessible)
        # This would typically be connected through signals/slots
        configs = self.get_strategy_configs()

        if not configs:
            QMessageBox.warning(self, "Warning", "No strategy configurations found")
            return

        if not self.controller.current_data is not None:
            QMessageBox.warning(self, "Warning", "No market data loaded")
            return

        self.controller.run_backtests(configs)

    def stop_backtests(self):
        """Stop running backtests"""
        # In a real implementation, this would signal the worker to stop
        self.update_status("Stopping backtests...")
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def export_results(self):
        """Export backtest results"""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        from PySide6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )

        if filename:
            success = self.controller.export_results(filename)
            if success:
                QMessageBox.information(self, "Success", f"Results exported to {filename}")
            else:
                QMessageBox.warning(self, "Error", "Failed to export results")

    def get_strategy_configs(self) -> List[Dict[str, Any]]:
        """Get strategy configurations (would be connected to strategy panel)"""
        # This is a placeholder - in real implementation, this would get configs
        # from the strategy panel through signals or direct reference
        return []

    def set_strategy_configs(self, configs: List[Dict[str, Any]]):
        """Set strategy configurations from strategy panel"""
        self.strategy_configs = configs
        self.run_button.setEnabled(len(configs) > 0)
        self.update_status(f"Ready to run {len(configs)} backtest configurations")

    def on_backtest_started(self):
        """Handle backtest start"""
        self.progress_bar.setVisible(True)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.update_status("Backtests started...")

    def on_backtest_finished(self, results: List[Any]):
        """Handle backtest completion"""
        self.current_results = results
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)

        self.update_results_table()
        self.update_status(f"Backtests completed: {len(results)} results")

    def on_backtest_progress(self, message: str):
        """Handle backtest progress updates"""
        self.update_status(message)

    def on_backtest_error(self, error: str):
        """Handle backtest errors"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status(f"Error: {error}")
        QMessageBox.critical(self, "Backtest Error", error)

    def update_status(self, message: str):
        """Update status display"""
        self.status_text.setPlainText(message)
        logger.info(f"Backtest status: {message}")

    def update_results_table(self):
        """Update the results table with backtest results"""
        self.results_table.setRowCount(len(self.current_results))

        for row, result in enumerate(self.current_results):
            metrics = result.metrics

            # Strategy name (from parameter set if available)
            strategy_name = "Unknown"
            if result.parameter_set:
                # Try to get strategy name from config
                strategy_name = result.parameter_set.get('name', 'Unknown')
            else:
                strategy_name = f"Strategy {row + 1}"

            # Format metrics
            total_return = metrics.get('total_return', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            execution_time = result.execution_time

            # Parameters summary
            params_str = str(result.parameter_set) if result.parameter_set else ""

            # Set table items
            self.results_table.setItem(row, 0, QTableWidgetItem(strategy_name))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{total_return:.2%}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{sharpe_ratio:.3f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{max_drawdown:.2%}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{win_rate:.1%}"))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{profit_factor:.2f}"))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{execution_time:.3f}s"))
            self.results_table.setItem(row, 7, QTableWidgetItem(params_str[:50]))  # Truncate long params

        # Resize columns to content
        self.results_table.resizeColumnsToContents()

    def show_detailed_metrics(self):
        """Show detailed metrics for selected result"""
        selected_rows = set()
        for item in self.results_table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            self.details_table.setRowCount(0)
            return

        # Show details for first selected row
        row = list(selected_rows)[0]
        if row < len(self.current_results):
            result = self.current_results[row]
            metrics = result.metrics

            # Create detailed metrics list
            detailed_metrics = [
                ("Total Return", f"{metrics.get('total_return', 0):.2%}"),
                ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}"),
                ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}"),
                ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}"),
                ("Recovery Factor", f"{metrics.get('recovery_factor', 0):.3f}"),
                ("K-Ratio", f"{metrics.get('k_ratio', 0):.3f}"),
                ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}"),
                ("Win Rate", f"{metrics.get('win_rate', 0):.1%}"),
                ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
                ("Hurst Exponent", f"{metrics.get('hurst_exponent', 0):.3f}"),
                ("Bootstrap Confidence", f"{metrics.get('bootstrap_confidence', 0):.3f}"),
                ("Execution Time", f"{result.execution_time:.3f}s"),
                ("Trades Count", str(len(result.trades) if result.trades else 0))
            ]

            self.details_table.setRowCount(len(detailed_metrics))
            for i, (metric_name, value) in enumerate(detailed_metrics):
                self.details_table.setItem(i, 0, QTableWidgetItem(metric_name))
                self.details_table.setItem(i, 1, QTableWidgetItem(value))

            self.details_table.resizeColumnsToContents()