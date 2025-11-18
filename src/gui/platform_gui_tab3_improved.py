from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView, QFrame, QSplitter, QScrollArea, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QFont
import logging
from datetime import datetime
import json
import os


class BacktestThread(QThread):
    """Enhanced backtest thread with real-time metric updates"""
    progress_updated = Signal(int, str)
    metrics_updated = Signal(dict)  # NEW: Real-time metrics
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
                
                # Emit intermediate metrics if available
                if isinstance(result, dict) and 'metrics' in result:
                    self.metrics_updated.emit(result['metrics'])

            elif self.mode == "Walk-Forward":
                self.progress_updated.emit(10, f"Running walk-forward analysis ({self.periods} periods)...")
                
                # Simulate progress updates per period
                for period in range(self.periods):
                    progress = 10 + (period / self.periods * 80)
                    self.progress_updated.emit(int(progress), f"Processing period {period + 1}/{self.periods}...")
                    
                result = self.backtester_core.run_walk_forward(
                    self.data_dict, self.strategy_class, self.strategy_params, self.periods
                )

            elif self.mode == "Monte Carlo":
                self.progress_updated.emit(10, f"Running Monte Carlo simulation ({self.runs} runs)...")
                
                # Simulate progress updates
                for run in range(0, self.runs, max(1, self.runs // 10)):
                    progress = 10 + (run / self.runs * 80)
                    self.progress_updated.emit(int(progress), f"Completed {run}/{self.runs} runs...")
                    
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


class MetricCard(QFrame):
    """Real-time metric display card"""
    def __init__(self, title, value="0.00", color="#569cd6", suffix="", format_func=None):
        super().__init__()
        self.format_func = format_func or (lambda x: f"{x:.2f}{suffix}")
        self.color = color
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border-left: 4px solid {color};
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #cccccc; font-size: 11px; font-weight: 500;")
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")
        
        # Change indicator (optional)
        self.change_label = QLabel("")
        self.change_label.setStyleSheet("color: #888888; font-size: 10px;")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.change_label)
        
        self.setLayout(layout)
        
    def update_value(self, value, change_text=""):
        """Update the metric value"""
        formatted = self.format_func(value)
        self.value_label.setText(formatted)
        if change_text:
            self.change_label.setText(change_text)


class Tab3BacktestRunner(QWidget):
    """
    Enhanced Backtest Runner with:
    - Real-time progress dashboard
    - Live metric updates during execution
    - Visual feedback with charts
    - Better result organization
    """
    backtest_complete = Signal(dict)
    status_update = Signal(str, str)  # message, type

    def __init__(self, parent_platform, backtester_core):
        super().__init__()
        self.parent_platform = parent_platform
        self.backtester_core = backtester_core
        self.backtest_thread = None
        self.logger = logging.getLogger(__name__)
        self.current_results = None
        self.start_time = None
        
        self.init_ui()
        
        # Check if parent has data/config to enable run button
        QTimer.singleShot(500, self.check_prerequisites)

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        
        # Configuration Section
        config_section = self.create_configuration_section()
        main_layout.addWidget(config_section)
        
        # Run Button (prominent)
        run_section = self.create_run_section()
        main_layout.addWidget(run_section)
        
        # Progress Dashboard (hidden initially)
        self.progress_section = self.create_progress_dashboard()
        self.progress_section.setVisible(False)
        main_layout.addWidget(self.progress_section)
        
        # Results Section (hidden initially)
        self.results_section = self.create_results_section()
        self.results_section.setVisible(False)
        main_layout.addWidget(self.results_section, 1)
        
        self.setLayout(main_layout)
        
    def create_configuration_section(self):
        """Create backtest configuration section"""
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
        title = QLabel("⚙️ Backtest Configuration")
        title.setProperty("class", "title")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Configuration grid
        config_grid = QHBoxLayout()
        
        # Mode selection
        mode_box = QVBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("color: #cccccc; font-weight: 500;")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simple", "Walk-Forward", "Monte Carlo"])
        self.mode_combo.setMinimumHeight(32)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_box.addWidget(mode_label)
        mode_box.addWidget(self.mode_combo)
        config_grid.addLayout(mode_box)
        
        # Periods (Walk-Forward)
        self.periods_box = QVBoxLayout()
        periods_label = QLabel("Periods:")
        periods_label.setStyleSheet("color: #cccccc; font-weight: 500;")
        self.periods_spin = QSpinBox()
        self.periods_spin.setMaximumWidth(100)  # Ancho optimizado
        self.periods_spin.setRange(3, 12)
        self.periods_spin.setValue(8)
        self.periods_spin.setMinimumHeight(32)
        self.periods_box.addWidget(periods_label)
        self.periods_box.addWidget(self.periods_spin)
        config_grid.addLayout(self.periods_box)
        
        # Runs (Monte Carlo)
        self.runs_box = QVBoxLayout()
        runs_label = QLabel("Runs:")
        runs_label.setStyleSheet("color: #cccccc; font-weight: 500;")
        self.runs_spin = QSpinBox()
        self.runs_spin.setMaximumWidth(120)  # Ancho optimizado
        self.runs_spin.setRange(100, 2000)
        self.runs_spin.setValue(500)
        self.runs_spin.setSingleStep(100)
        self.runs_spin.setMinimumHeight(32)
        self.runs_box.addWidget(runs_label)
        self.runs_box.addWidget(self.runs_spin)
        config_grid.addLayout(self.runs_box)
        
        config_grid.addStretch()
        
        layout.addLayout(config_grid)
        
        # FASE 1: Realistic Execution Section
        realistic_section = QHBoxLayout()
        realistic_section.setContentsMargins(0, 12, 0, 0)
        
        # Checkbox for enabling realistic execution
        from PySide6.QtWidgets import QCheckBox
        self.realistic_exec_checkbox = QCheckBox("Enable Realistic Execution (FASE 1)")
        self.realistic_exec_checkbox.setStyleSheet("""
            QCheckBox {
                color: #4ec9b0;
                font-weight: bold;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.realistic_exec_checkbox.setChecked(False)
        self.realistic_exec_checkbox.stateChanged.connect(self.on_realistic_exec_toggled)
        realistic_section.addWidget(self.realistic_exec_checkbox)
        
        # Latency profile dropdown (initially hidden)
        self.latency_profile_label = QLabel("Latency Profile:")
        self.latency_profile_label.setStyleSheet("color: #cccccc; font-weight: 500; margin-left: 20px;")
        self.latency_profile_combo = QComboBox()
        self.latency_profile_combo.addItems([
            "co-located (HFT ~3ms)",
            "institutional (~20ms)",
            "retail_fast (~50ms)",
            "retail_average (~80ms) ⭐",
            "retail_slow (~120ms)",
            "mobile (~165ms)"
        ])
        self.latency_profile_combo.setCurrentIndex(3)  # Default: retail_average
        self.latency_profile_combo.setMinimumHeight(32)
        self.latency_profile_combo.setMinimumWidth(200)
        
        # Hide latency controls initially
        self.latency_profile_label.setVisible(False)
        self.latency_profile_combo.setVisible(False)
        
        realistic_section.addWidget(self.latency_profile_label)
        realistic_section.addWidget(self.latency_profile_combo)
        realistic_section.addStretch()
        
        layout.addLayout(realistic_section)
        
        # Info label for realistic execution
        self.realistic_info_label = QLabel()
        self.realistic_info_label.setWordWrap(True)
        self.realistic_info_label.setStyleSheet("""
            color: #888888; 
            font-size: 10px; 
            margin-top: 4px;
            padding: 8px;
            background-color: #1e1e1e;
            border-left: 3px solid #4ec9b0;
            border-radius: 3px;
        """)
        self.realistic_info_label.setVisible(False)
        layout.addWidget(self.realistic_info_label)
        
        # Mode description
        self.mode_description = QLabel()
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("color: #888888; font-size: 11px; margin-top: 8px;")
        layout.addWidget(self.mode_description)
        
        frame.setLayout(layout)
        
        # Initialize visibility
        self.on_mode_changed("Simple")
        
        return frame
        
    def create_run_section(self):
        """Create run button section"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 6px;
                padding: 12px;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Status indicator
        self.status_icon = QLabel("●")
        self.status_icon.setStyleSheet("color: #888888; font-size: 20px;")
        
        self.status_text = QLabel("Ready to run backtest")
        self.status_text.setStyleSheet("color: #cccccc; font-weight: 500;")
        
        layout.addWidget(self.status_icon)
        layout.addWidget(self.status_text)
        layout.addStretch()
        
        # Run button
        self.run_btn = QPushButton("▶️ Run Backtest")
        self.run_btn.clicked.connect(self.on_run_backtest_clicked)
        self.run_btn.setMinimumHeight(32)
        self.run_btn.setMaximumHeight(36)
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
        self.run_btn.setEnabled(False)
        
        layout.addWidget(self.run_btn)
        
        frame.setLayout(layout)
        return frame
        
    def create_progress_dashboard(self):
        """Create live progress dashboard"""
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
        title = QLabel("📊 Live Progress")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Progress bar
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
                background-color: #0e639c;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.progress_status = QLabel("Initializing...")
        self.progress_status.setStyleSheet("color: #cccccc; margin-top: 8px;")
        layout.addWidget(self.progress_status)
        
        # Live metrics grid
        metrics_label = QLabel("Live Metrics:")
        metrics_label.setStyleSheet("color: #cccccc; font-weight: bold; margin-top: 12px;")
        layout.addWidget(metrics_label)
        
        metrics_grid = QHBoxLayout()
        
        self.metric_sharpe = MetricCard("Sharpe Ratio", "0.00", "#4ec9b0")
        self.metric_return = MetricCard("Total Return", "0.00%", "#569cd6", "%", lambda x: f"{x:.2f}%")
        self.metric_trades = MetricCard("Total Trades", "0", "#dcdcaa", "", lambda x: str(int(x)))
        self.metric_winrate = MetricCard("Win Rate", "0.00%", "#c586c0", "%", lambda x: f"{x:.1f}%")
        
        metrics_grid.addWidget(self.metric_sharpe)
        metrics_grid.addWidget(self.metric_return)
        metrics_grid.addWidget(self.metric_trades)
        metrics_grid.addWidget(self.metric_winrate)
        
        layout.addLayout(metrics_grid)
        
        # Elapsed time
        self.elapsed_label = QLabel("Elapsed: 0s")
        self.elapsed_label.setStyleSheet("color: #888888; font-size: 11px; margin-top: 8px;")
        layout.addWidget(self.elapsed_label)
        
        # Stop button
        stop_layout = QHBoxLayout()
        stop_layout.addStretch()
        
        self.stop_btn = QPushButton("⏹️ Stop Backtest")
        self.stop_btn.clicked.connect(self.on_stop_backtest)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f48771;
                color: #1e1e1e;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #ff9f8f;
            }
        """)
        stop_layout.addWidget(self.stop_btn)
        
        layout.addLayout(stop_layout)
        
        frame.setLayout(layout)
        
        # Timer for elapsed time updates
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)
        
        return frame
        
    def create_results_section(self):
        """Create results display section"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Title with actions
        header_layout = QHBoxLayout()
        
        title = QLabel("📈 Backtest Results")
        title.setProperty("class", "title")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Export buttons
        self.export_csv_btn = QPushButton("📄 Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_results('csv'))
        self.export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        
        self.export_json_btn = QPushButton("📋 Export JSON")
        self.export_json_btn.clicked.connect(lambda: self.export_results('json'))
        self.export_json_btn.setStyleSheet(self.export_csv_btn.styleSheet())
        
        header_layout.addWidget(self.export_csv_btn)
        header_layout.addWidget(self.export_json_btn)
        
        layout.addLayout(header_layout)
        
        # Splitter for tables and chart
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Summary metrics table
        summary_group = QGroupBox("Summary Metrics")
        summary_group.setProperty("class", "metric-card")
        summary_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        summary_layout = QVBoxLayout()
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setMaximumHeight(300)
        summary_layout.addWidget(self.summary_table)
        
        summary_group.setLayout(summary_layout)
        splitter.addWidget(summary_group)
        
        # Walk-Forward results table (conditional)
        self.wf_group = QGroupBox("Walk-Forward Results")
        self.wf_group.setStyleSheet(summary_group.styleSheet())
        wf_layout = QVBoxLayout()
        
        self.wf_table = QTableWidget()
        self.wf_table.setColumnCount(5)
        self.wf_table.setHorizontalHeaderLabels(["Period", "Train Sharpe", "Test Sharpe", "Degradation %", "Status"])
        self.wf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        wf_layout.addWidget(self.wf_table)
        
        self.wf_group.setLayout(wf_layout)
        self.wf_group.setVisible(False)
        splitter.addWidget(self.wf_group)
        
        # Equity curve chart
        chart_group = QGroupBox("Equity Curve")
        chart_group.setStyleSheet(summary_group.styleSheet())
        chart_layout = QVBoxLayout()
        
        self.equity_chart = QWebEngineView()
        self.equity_chart.setMinimumHeight(450)  # Increased from 300
        chart_layout.addWidget(self.equity_chart)
        
        chart_group.setLayout(chart_layout)
        splitter.addWidget(chart_group)
        
        # Set splitter sizes: compact summary, small WF, LARGE chart
        splitter.setStretchFactor(0, 15)  # Summary: 15%
        splitter.setStretchFactor(1, 10)  # Walk-Forward: 10%
        splitter.setStretchFactor(2, 75)  # Chart: 75%
        splitter.setSizes([150, 100, 600])
        
        layout.addWidget(splitter, 1)
        
        widget.setLayout(layout)
        return widget
        
    def on_mode_changed(self, mode):
        """Handle backtest mode change"""
        # Update visibility of mode-specific controls
        if mode == "Simple":
            self.hide_widget_tree(self.periods_box)
            self.hide_widget_tree(self.runs_box)
            self.mode_description.setText(
                "Quick single-pass backtest. Fast execution, basic metrics."
            )
            
        elif mode == "Walk-Forward":
            self.show_widget_tree(self.periods_box)
            self.hide_widget_tree(self.runs_box)
            self.mode_description.setText(
                "Validates strategy across multiple time periods. Tests for overfitting. "
                "Train on historical data, test on out-of-sample."
            )
            
        elif mode == "Monte Carlo":
            self.hide_widget_tree(self.periods_box)
            self.show_widget_tree(self.runs_box)
            self.mode_description.setText(
                "Runs multiple randomized simulations to assess strategy robustness. "
                "Tests resilience to different market conditions."
            )
    
    def on_realistic_exec_toggled(self, state):
        """Handle realistic execution toggle"""
        is_enabled = (state == 2)  # Qt.CheckState.Checked
        
        # Show/hide latency profile controls
        self.latency_profile_label.setVisible(is_enabled)
        self.latency_profile_combo.setVisible(is_enabled)
        self.realistic_info_label.setVisible(is_enabled)
        
        if is_enabled:
            self.realistic_info_label.setText(
                "🚀 Realistic execution adds market impact costs and latency delays. "
                "Expect Sharpe to drop 15-30% and returns to drop 20-35%. "
                "This is REALISTIC and prevents overestimating strategy performance."
            )
        
        self.logger.info(f"Realistic execution {'enabled' if is_enabled else 'disabled'}")
            
    def hide_widget_tree(self, layout):
        """Hide all widgets in a layout"""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setVisible(False)
                
    def show_widget_tree(self, layout):
        """Show all widgets in a layout"""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setVisible(True)
                
    def check_prerequisites(self):
        """Check if data and strategy are configured"""
        has_data = bool(self.parent_platform.data_dict)
        has_config = hasattr(self.parent_platform, 'config_dict') and bool(self.parent_platform.config_dict)
        
        if has_data and has_config:
            self.run_btn.setEnabled(True)
            self.status_icon.setStyleSheet("color: #4ec9b0; font-size: 20px;")
            self.status_text.setText("Ready to run backtest")
        else:
            self.run_btn.setEnabled(False)
            self.status_icon.setStyleSheet("color: #f48771; font-size: 20px;")
            
            if not has_data:
                self.status_text.setText("⚠️ No data loaded (Tab 1)")
            elif not has_config:
                self.status_text.setText("⚠️ No strategy configured (Tab 2)")
                
    def on_run_backtest_clicked(self):
        """Start backtest execution"""
        # Validate prerequisites
        if not self.parent_platform.data_dict:
            QMessageBox.warning(self, "Error", "No data loaded. Please load data in Tab 1 first.")
            return

        if not hasattr(self.parent_platform, 'config_dict') or not self.parent_platform.config_dict:
            QMessageBox.warning(self, "Error", "No strategy configured. Please configure strategy in Tab 2 first.")
            return

        # Get configuration
        mode = self.mode_combo.currentText()
        config = self.parent_platform.config_dict

        # Prepare parameters
        periods = self.periods_spin.value() if mode == "Walk-Forward" else 8
        runs = self.runs_spin.value() if mode == "Monte Carlo" else 500
        
        # FASE 1: Apply realistic execution settings to backtester
        is_realistic = self.realistic_exec_checkbox.isChecked()
        
        if is_realistic:
            # Extract latency profile from combo text
            latency_text = self.latency_profile_combo.currentText()
            # Extract profile key (e.g., "retail_average" from "retail_average (~80ms) ⭐")
            latency_profile = latency_text.split('(')[0].strip().replace(' ', '-').lower()
            
            # Map display names to actual profile keys
            profile_map = {
                'co-located': 'co-located',
                'institutional': 'institutional',
                'retail_fast': 'retail_fast',
                'retail_average': 'retail_average',
                'retail_slow': 'retail_slow',
                'mobile': 'mobile'
            }
            latency_profile = profile_map.get(latency_profile, 'retail_average')
            
            # Reinitialize backtester with realistic execution
            self.backtester_core.enable_realistic_execution = True
            self.backtester_core.latency_profile = latency_profile
            
            # Reinitialize realistic execution components
            try:
                from src.execution.market_impact import MarketImpactModel, VolumeProfileAnalyzer
                from src.execution.latency_model import LatencyModel, LatencyProfile
                
                self.backtester_core.market_impact_model = MarketImpactModel()
                self.backtester_core.volume_analyzer = VolumeProfileAnalyzer()
                self.backtester_core.latency_model = LatencyProfile.get_profile(latency_profile)
                
                self.logger.info(f"🚀 Realistic execution enabled with profile: {latency_profile}")
            except ImportError as e:
                self.logger.warning(f"Realistic execution components not available: {e}")
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Realistic execution components not found. Running with legacy execution."
                )
                is_realistic = False
        else:
            # Disable realistic execution
            self.backtester_core.enable_realistic_execution = False
            self.logger.info("Simple execution model (legacy)")

        # Update UI state
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ Running...")
        self.progress_section.setVisible(True)
        self.results_section.setVisible(False)
        
        # Reset progress dashboard
        self.progress_bar.setValue(0)
        self.progress_status.setText("Initializing backtest...")
        self.metric_sharpe.update_value(0.0)
        self.metric_return.update_value(0.0)
        self.metric_trades.update_value(0)
        self.metric_winrate.update_value(0.0)
        
        # Start elapsed timer
        self.start_time = datetime.now()
        self.elapsed_timer.start(1000)  # Update every second

        # Start backtest thread
        self.backtest_thread = BacktestThread(
            self.backtester_core, mode, self.parent_platform.data_dict,
            config['strategy_class'], config['params'], periods, runs
        )
        self.backtest_thread.progress_updated.connect(self.update_progress)
        self.backtest_thread.metrics_updated.connect(self.update_live_metrics)
        self.backtest_thread.backtest_complete.connect(self.on_backtest_complete)
        self.backtest_thread.error_occurred.connect(self.on_backtest_error)
        self.backtest_thread.start()
        
        self.status_update.emit(f"Started {mode} backtest", "info")
        
    def update_progress(self, pct, msg):
        """Update progress bar and status"""
        self.progress_bar.setValue(pct)
        self.progress_status.setText(msg)
        
    def update_live_metrics(self, metrics):
        """Update live metrics during backtest"""
        if 'sharpe' in metrics:
            self.metric_sharpe.update_value(metrics['sharpe'])
            
        if 'total_return' in metrics:
            self.metric_return.update_value(metrics['total_return'] * 100)
            
        if 'num_trades' in metrics:
            self.metric_trades.update_value(metrics['num_trades'])
            
        if 'win_rate' in metrics:
            self.metric_winrate.update_value(metrics['win_rate'] * 100)
            
    def update_elapsed_time(self):
        """Update elapsed time display"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.elapsed_label.setText(f"Elapsed: {elapsed:.0f}s")
            
    def on_stop_backtest(self):
        """Stop running backtest"""
        if self.backtest_thread and self.backtest_thread.isRunning():
            self.backtest_thread.terminate()
            self.backtest_thread.wait()
            
            self.run_btn.setEnabled(True)
            self.run_btn.setText("▶️ Run Backtest")
            self.elapsed_timer.stop()
            
            self.status_update.emit("Backtest stopped by user", "warning")
            
    def on_backtest_complete(self, result):
        """Handle backtest completion"""
        # Stop timers
        self.elapsed_timer.stop()
        
        # Update UI state
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ Run Backtest")
        
        # Store results
        self.current_results = result
        self.parent_platform.last_backtest_results = result
        
        # Display results
        self.display_results(result)
        
        # Show results section
        self.results_section.setVisible(True)
        
        # Emit signal
        self.backtest_complete.emit(result)
        
        self.status_update.emit("Backtest completed successfully", "success")
        
    def on_backtest_error(self, error_msg):
        """Handle backtest error"""
        self.elapsed_timer.stop()
        
        QMessageBox.critical(self, "Backtest Error", error_msg)
        
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ Run Backtest")
        
        self.status_update.emit(f"Backtest error: {error_msg}", "error")
        
    def display_results(self, result):
        """Display backtest results"""
        mode = self.mode_combo.currentText()
        
        # Update final metrics in progress dashboard
        if 'metrics' in result:
            self.update_live_metrics(result['metrics'])
        
        # Populate summary table
        metrics = result.get('metrics', {})
        self.summary_table.setRowCount(0)
        
        row = 0
        
        # Add realistic execution cost breakdown if available
        if self.realistic_exec_checkbox.isChecked() and 'execution_costs' in result:
            costs = result['execution_costs']
            
            # Section header
            self.summary_table.insertRow(row)
            header_item = QTableWidgetItem("📊 REALISTIC EXECUTION COSTS")
            header_item.setForeground(Qt.GlobalColor.cyan)
            from PySide6.QtGui import QFont
            font = QFont()
            font.setBold(True)
            header_item.setFont(font)
            self.summary_table.setItem(row, 0, header_item)
            self.summary_table.setItem(row, 1, QTableWidgetItem(""))
            row += 1
            
            # Market impact
            if 'total_market_impact' in costs:
                self.summary_table.insertRow(row)
                self.summary_table.setItem(row, 0, QTableWidgetItem("  Market Impact Cost"))
                self.summary_table.setItem(row, 1, QTableWidgetItem(f"${costs['total_market_impact']:.2f}"))
                row += 1
            
            # Latency cost
            if 'total_latency_cost' in costs:
                self.summary_table.insertRow(row)
                self.summary_table.setItem(row, 0, QTableWidgetItem("  Latency Cost"))
                self.summary_table.setItem(row, 1, QTableWidgetItem(f"${costs['total_latency_cost']:.2f}"))
                row += 1
            
            # Total execution cost
            if 'total_execution_cost' in costs:
                self.summary_table.insertRow(row)
                cost_item = QTableWidgetItem("  Total Execution Cost")
                cost_item.setForeground(Qt.GlobalColor.yellow)
                cost_value = QTableWidgetItem(f"${costs['total_execution_cost']:.2f}")
                cost_value.setForeground(Qt.GlobalColor.yellow)
                self.summary_table.setItem(row, 0, cost_item)
                self.summary_table.setItem(row, 1, cost_value)
                row += 1
            
            # Percentage of capital
            if 'total_execution_cost' in costs:
                self.summary_table.insertRow(row)
                pct = (costs['total_execution_cost'] / self.backtester_core.initial_capital) * 100
                self.summary_table.setItem(row, 0, QTableWidgetItem("  Cost % of Capital"))
                self.summary_table.setItem(row, 1, QTableWidgetItem(f"{pct:.3f}%"))
                row += 1
            
            # Separator
            self.summary_table.insertRow(row)
            self.summary_table.setItem(row, 0, QTableWidgetItem(""))
            self.summary_table.setItem(row, 1, QTableWidgetItem(""))
            row += 1
        
        # Add standard metrics
        for metric_name, value in metrics.items():
            self.summary_table.insertRow(row)
            
            # Metric name
            name_item = QTableWidgetItem(metric_name.replace('_', ' ').title())
            self.summary_table.setItem(row, 0, name_item)
            
            # Metric value
            if isinstance(value, float):
                value_item = QTableWidgetItem(f"{value:.4f}")
            else:
                value_item = QTableWidgetItem(str(value))
            self.summary_table.setItem(row, 1, value_item)
            
            row += 1
            
        # Handle Walk-Forward results
        if mode == "Walk-Forward" and 'periods' in result:
            self.wf_group.setVisible(True)
            self.wf_table.setRowCount(0)
            
            for i, period in enumerate(result['periods']):
                self.wf_table.insertRow(i)
                
                self.wf_table.setItem(i, 0, QTableWidgetItem(str(period['period'])))
                self.wf_table.setItem(i, 1, QTableWidgetItem(f"{period['train_metrics']['sharpe']:.3f}"))
                self.wf_table.setItem(i, 2, QTableWidgetItem(f"{period['test_metrics']['sharpe']:.3f}"))
                
                degradation = period.get('degradation_pct', 0)
                self.wf_table.setItem(i, 3, QTableWidgetItem(f"{degradation:.1f}%"))
                
                # Status indicator
                status = "✓ Pass" if abs(degradation) < 20 else "⚠️ High Degradation"
                status_item = QTableWidgetItem(status)
                
                if "Pass" in status:
                    status_item.setForeground(Qt.GlobalColor.green)
                else:
                    status_item.setForeground(Qt.GlobalColor.yellow)
                    
                self.wf_table.setItem(i, 4, status_item)
        else:
            self.wf_group.setVisible(False)
            
        # Create equity curve chart
        if 'equity_curve' in result:
            self.create_equity_chart(result['equity_curve'])
        elif 'portfolio_value' in result:
            self.create_equity_chart(result['portfolio_value'])
            
    def create_equity_chart(self, equity_data):
        """Create Plotly equity curve chart"""
        try:
            import plotly.graph_objects as go
            import pandas as pd
            
            fig = go.Figure()
            
            # Equity curve
            fig.add_trace(
                go.Scatter(
                    y=equity_data,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#4ec9b0', width=2)
                )
            )
            
            # Add drawdown area (if available)
            # Calculate drawdown
            cummax = pd.Series(equity_data).cummax()
            drawdown = (pd.Series(equity_data) - cummax) / cummax * 100
            
            fig.add_trace(
                go.Scatter(
                    y=drawdown,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='#f48771', width=1),
                    yaxis='y2'
                )
            )
            
            # Layout
            fig.update_layout(
                template='plotly_dark',
                height=300,
                showlegend=True,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#cccccc'),
                xaxis=dict(title='Time'),
                yaxis=dict(title='Portfolio Value ($)'),
                yaxis2=dict(
                    title='Drawdown (%)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.equity_chart.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error creating equity chart: {e}")
            
    def export_results(self, format_type):
        """Export backtest results"""
        if not self.current_results:
            QMessageBox.warning(self, "Error", "No results to export")
            return
            
        try:
            from PySide6.QtWidgets import QFileDialog
            import pandas as pd
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                f"Export Results ({format_type.upper()})",
                f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}",
                f"{format_type.upper()} files (*.{format_type})"
            )
            
            if not filename:
                return
                
            if format_type == 'csv':
                # Export metrics as CSV
                metrics = self.current_results.get('metrics', {})
                df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                df.to_csv(filename, index=False)
                
            elif format_type == 'json':
                # Export full results as JSON
                with open(filename, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
                    
            QMessageBox.information(self, "Success", f"Results exported to:\n{filename}")
            self.status_update.emit(f"Results exported to {format_type.upper()}", "success")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            QMessageBox.critical(self, "Export Error", str(e))
