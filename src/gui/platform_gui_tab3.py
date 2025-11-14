from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
import logging
import pandas as pd

class BacktestThread(QThread):
    progress_updated = Signal(int, str)
    backtest_complete = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, backtester_core, mode, data_dict, strategy_class, strategy_params, periods=8, runs=500):
        super().__init__()
        self.backtester_core = backtester_core
        self.mode = mode
        self.data_dict = data_dict
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.periods = periods
        self.runs = runs

    def run(self):
        try:
            self.progress_updated.emit(5, "Initializing backtest...")

            if self.mode == "Simple":
                self.progress_updated.emit(10, "Running simple backtest...")
                result = self.backtester_core.run_simple_backtest(
                    self.data_dict, self.strategy_class, self.strategy_params
                )

            elif self.mode == "Walk-Forward":
                self.progress_updated.emit(10, f"Running walk-forward analysis ({self.periods} periods)...")
                result = self.backtester_core.run_walk_forward(
                    self.data_dict, self.strategy_class, self.strategy_params, self.periods
                )

            elif self.mode == "Monte Carlo":
                self.progress_updated.emit(10, f"Running Monte Carlo simulation ({self.runs} runs)...")
                result = self.backtester_core.run_monte_carlo(
                    self.data_dict, self.strategy_class, self.strategy_params, self.runs
                )

            if isinstance(result, dict) and 'error' in result:
                self.error_occurred.emit(result['error'])
                return

            self.progress_updated.emit(100, "Backtest completed successfully!")
            self.backtest_complete.emit(result)

        except Exception as e:
            self.error_occurred.emit(f"Backtest thread error: {str(e)}")

class Tab3BacktestRunner(QWidget):
    backtest_complete = Signal(dict)

    def __init__(self, parent_platform, backtester_core):
        super().__init__()
        self.parent_platform = parent_platform
        self.backtester_core = backtester_core
        self.backtest_thread = None
        self.logger = logging.getLogger(__name__)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Backtest Mode Selection
        mode_group = QGroupBox("Backtest Configuration")
        mode_layout = QVBoxLayout()

        # Mode selection
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Backtest Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simple", "Walk-Forward", "Monte Carlo"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)

        # Period/Runs configuration
        self.periods_row = QHBoxLayout()
        self.periods_row.addWidget(QLabel("Number of Periods:"))
        self.periods_spin = QSpinBox()
        self.periods_spin.setMaximumWidth(100)  # Ancho optimizado para valores pequeños
        self.periods_spin.setRange(3, 12)
        self.periods_spin.setValue(8)
        self.periods_row.addWidget(self.periods_spin)
        self.periods_row.addStretch()
        mode_layout.addLayout(self.periods_row)

        self.runs_row = QHBoxLayout()
        self.runs_row.addWidget(QLabel("Number of Runs:"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setMaximumWidth(120)  # Ancho optimizado para valores medianos
        self.runs_spin.setRange(100, 2000)
        self.runs_spin.setValue(500)
        self.runs_row.addWidget(self.runs_spin)
        self.runs_row.addStretch()
        mode_layout.addLayout(self.runs_row)

        # Initially hide periods/runs rows
        self.periods_row.itemAt(0).widget().setVisible(False)
        self.periods_row.itemAt(1).widget().setVisible(False)
        self.runs_row.itemAt(0).widget().setVisible(False)
        self.runs_row.itemAt(1).widget().setVisible(False)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Run Button
        run_layout = QHBoxLayout()
        run_layout.addStretch()
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self.on_run_backtest_clicked)
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 15px 30px; font-size: 16px; font-weight: bold; }")
        self.run_btn.setEnabled(False)  # Disabled until data is loaded
        run_layout.addWidget(self.run_btn)
        run_layout.addStretch()
        layout.addLayout(run_layout)

        # Progress Section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Live Log
        log_group = QGroupBox("Live Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Courier New", 9))
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Results Tables
        results_layout = QHBoxLayout()

        # Walk-Forward Results Table
        wf_group = QGroupBox("Walk-Forward Results")
        wf_layout = QVBoxLayout()

        self.wf_table = QTableWidget()
        self.wf_table.setColumnCount(5)
        self.wf_table.setHorizontalHeaderLabels(["Period", "Train Sharpe", "Test Sharpe", "Degradation %", "Winner"])
        self.wf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.wf_table.setVisible(False)
        wf_layout.addWidget(self.wf_table)

        wf_group.setLayout(wf_layout)
        results_layout.addWidget(wf_group)

        # Summary Metrics Table
        summary_group = QGroupBox("Summary Metrics")
        summary_layout = QVBoxLayout()

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.summary_table)

        summary_group.setLayout(summary_layout)
        results_layout.addWidget(summary_group)

        layout.addLayout(results_layout)

        # Export Buttons
        export_layout = QHBoxLayout()

        self.export_csv_btn = QPushButton("Export Results (CSV)")
        self.export_csv_btn.clicked.connect(lambda: self.export_results('csv'))
        self.export_csv_btn.setEnabled(False)

        self.export_json_btn = QPushButton("Export Results (JSON)")
        self.export_json_btn.clicked.connect(lambda: self.export_results('json'))
        self.export_json_btn.setEnabled(False)

        self.export_equity_btn = QPushButton("Export Equity Curve")
        self.export_equity_btn.clicked.connect(self.export_equity_curve)
        self.export_equity_btn.setEnabled(False)

        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_json_btn)
        export_layout.addWidget(self.export_equity_btn)
        export_layout.addStretch()

        layout.addLayout(export_layout)

        layout.addStretch()
        self.setLayout(layout)

    def on_mode_changed(self, mode):
        # Show/hide relevant controls
        if mode == "Walk-Forward":
            self.periods_row.itemAt(0).widget().setVisible(True)
            self.periods_row.itemAt(1).widget().setVisible(True)
            self.runs_row.itemAt(0).widget().setVisible(False)
            self.runs_row.itemAt(1).widget().setVisible(False)
        elif mode == "Monte Carlo":
            self.periods_row.itemAt(0).widget().setVisible(False)
            self.periods_row.itemAt(1).widget().setVisible(False)
            self.runs_row.itemAt(0).widget().setVisible(True)
            self.runs_row.itemAt(1).widget().setVisible(True)
        else:  # Simple
            self.periods_row.itemAt(0).widget().setVisible(False)
            self.periods_row.itemAt(1).widget().setVisible(False)
            self.runs_row.itemAt(0).widget().setVisible(False)
            self.runs_row.itemAt(1).widget().setVisible(False)

    def on_run_backtest_clicked(self):
        # Validate prerequisites
        if not self.parent_platform.data_dict:
            self.show_error("No data loaded. Please load data in Tab 1 first.")
            return

        if not hasattr(self.parent_platform, 'config_dict') or not self.parent_platform.config_dict:
            self.show_error("No strategy configured. Please configure strategy in Tab 2 first.")
            return

        # Get configuration
        mode = self.mode_combo.currentText()
        config = self.parent_platform.config_dict

        # Prepare parameters
        periods = self.periods_spin.value() if mode == "Walk-Forward" else 8
        runs = self.runs_spin.value() if mode == "Monte Carlo" else 500

        # Disable button and show progress
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Running backtest...")
        self.log_text.clear()

        # Start backtest thread
        self.backtest_thread = BacktestThread(
            self.backtester_core, mode, self.parent_platform.data_dict,
            config['strategy_class'], config['params'], periods, runs
        )
        self.backtest_thread.progress_updated.connect(self.update_progress)
        self.backtest_thread.backtest_complete.connect(self.on_backtest_complete)
        self.backtest_thread.error_occurred.connect(self.on_backtest_error)
        self.backtest_thread.start()

    def update_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"Status: {msg}")
        self.log_text.append(f"{pd.Timestamp.now().strftime('%H:%M:%S')} - {msg}")

    def on_backtest_complete(self, result):
        # Update UI
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Backtest")
        self.progress_bar.setVisible(False)
        self.status_label.setText("Status: Backtest completed")

        # Store results in parent
        self.parent_platform.last_backtest_results = result

        # Update results tables
        self.display_results(result)

        # Enable export buttons
        self.export_csv_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
        self.export_equity_btn.setEnabled('equity_curve' in result)

        # Emit signal
        self.backtest_complete.emit(result)

    def on_backtest_error(self, error_msg):
        self.show_error(error_msg)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Backtest")
        self.progress_bar.setVisible(False)
        self.status_label.setText("Status: Error occurred")

    def display_results(self, result):
        mode = self.mode_combo.currentText()

        if mode == "Walk-Forward" and 'periods' in result:
            # Show walk-forward table
            self.wf_table.setVisible(True)
            self.wf_table.setRowCount(len(result['periods']))

            for i, period in enumerate(result['periods']):
                self.wf_table.setItem(i, 0, QTableWidgetItem(str(period['period'])))
                self.wf_table.setItem(i, 1, QTableWidgetItem(f"{period['train_metrics']['sharpe']:.3f}"))
                self.wf_table.setItem(i, 2, QTableWidgetItem(f"{period['test_metrics']['sharpe']:.3f}"))
                self.wf_table.setItem(i, 3, QTableWidgetItem(f"{period['degradation_pct']:.1f}%"))

                winner = "Train" if period['train_metrics']['sharpe'] > period['test_metrics']['sharpe'] else "Test"
                winner_item = QTableWidgetItem(winner)
                winner_item.setBackground(Qt.GlobalColor.green if winner == "Test" else Qt.GlobalColor.yellow)
                self.wf_table.setItem(i, 4, winner_item)

        # Show summary metrics
        metrics = result.get('metrics', {})
        self.summary_table.setRowCount(len(metrics))

        row = 0
        for metric_name, value in metrics.items():
            self.summary_table.setItem(row, 0, QTableWidgetItem(metric_name.replace('_', ' ').title()))
            if isinstance(value, float):
                self.summary_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
            else:
                self.summary_table.setItem(row, 1, QTableWidgetItem(str(value)))
            row += 1

        # Add additional metrics for Monte Carlo
        if mode == "Monte Carlo" and 'sharpe_mean' in result:
            self.summary_table.insertRow(row)
            self.summary_table.setItem(row, 0, QTableWidgetItem("Sharpe Mean"))
            self.summary_table.setItem(row, 1, QTableWidgetItem(f"{result['sharpe_mean']:.3f}"))
            row += 1

            self.summary_table.insertRow(row)
            self.summary_table.setItem(row, 0, QTableWidgetItem("Sharpe Std"))
            self.summary_table.setItem(row, 1, QTableWidgetItem(f"{result['sharpe_std']:.3f}"))
            row += 1

            self.summary_table.insertRow(row)
            self.summary_table.setItem(row, 0, QTableWidgetItem("Robust"))
            robust_text = "Yes" if result.get('robust', False) else "No"
            robust_item = QTableWidgetItem(robust_text)
            robust_item.setBackground(Qt.GlobalColor.green if result.get('robust', False) else Qt.GlobalColor.red)
            self.summary_table.setItem(row, 1, robust_item)

    def export_results(self, format_type):
        try:
            import pandas as pd
            from PySide6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export Results ({format_type.upper()})",
                f"backtest_results.{format_type}",
                f"{format_type.upper()} files (*.{format_type})"
            )

            if not filename:
                return

            result = self.parent_platform.last_backtest_results

            if format_type == 'csv':
                # Export trades if available
                if 'trades' in result and result['trades']:
                    df_trades = pd.DataFrame(result['trades'])
                    df_trades.to_csv(filename, index=False)
                else:
                    # Export metrics
                    metrics = result.get('metrics', {})
                    pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_csv(filename, index=False)

            elif format_type == 'json':
                import json
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

            self.show_message("Success", f"Results exported to {filename}")

        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")

    def export_equity_curve(self):
        try:
            import pandas as pd
            from PySide6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Equity Curve",
                "equity_curve.csv",
                "CSV files (*.csv)"
            )

            if not filename:
                return

            result = self.parent_platform.last_backtest_results
            if 'equity_curve' in result:
                pd.DataFrame({'equity': result['equity_curve']}).to_csv(filename, index=False)
                self.show_message("Success", f"Equity curve exported to {filename}")
            else:
                self.show_error("No equity curve data available")

        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")

    def show_error(self, msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", msg)

    def show_message(self, title, msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, title, msg)