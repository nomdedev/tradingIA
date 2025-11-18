"""
Main Window - Main application window integrating all UI components
"""

import logging
import sys
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSplitter,
    QMenuBar, QStatusBar, QLabel, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon

from .dashboard_controller import DashboardController
from .charts_widget import ChartsWidget
from .strategy_panel import StrategyPanel
from .backtest_panel import BacktestPanel
from .results_panel import ResultsPanel
from core.optimization.optimization_panel import OptimizationPanel
from core.alerts import AlertType, AlertSeverity

try:
    from core.alerts.alerts_panel import AlertsPanel
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    Main application window for the Trading IA platform.
    """

    def __init__(self):
        super().__init__()
        self.controller = DashboardController()
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        self.connect_signals()

        # Initialize application
        self.initialize_app()

    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("Trading IA - Strategy Testing Platform")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Strategy configuration
        self.strategy_panel = StrategyPanel(self.controller)
        self.main_splitter.addWidget(self.strategy_panel)

        # Right splitter for charts and results
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top right - Charts
        self.charts_widget = ChartsWidget(self.controller)
        self.right_splitter.addWidget(self.charts_widget)

        # Bottom right - Tab widget with Backtest, Results, and Optimization
        from PySide6.QtWidgets import QTabWidget
        self.tab_widget = QTabWidget()

        # Backtest panel
        self.backtest_panel = BacktestPanel(self.controller)
        self.tab_widget.addTab(self.backtest_panel, "Backtest")

        # Results panel
        self.results_panel = ResultsPanel(self.controller)
        self.tab_widget.addTab(self.results_panel, "Results")

        # Optimization panel
        self.optimization_panel = OptimizationPanel(self.controller)
        self.tab_widget.addTab(self.optimization_panel, "Optimization")

        # Alerts panel
        if ALERTS_AVAILABLE:
            self.alerts_panel = AlertsPanel(self.controller.get_alert_manager())
            self.tab_widget.addTab(self.alerts_panel, "Alertas")
        else:
            logger.warning("Alerts panel not available - PyQt6 may not be installed")

        self.right_splitter.addWidget(self.tab_widget)
        self.main_splitter.addWidget(self.right_splitter)

        # Set splitter proportions
        self.main_splitter.setSizes([350, 1050])  # Strategy panel narrower
        self.right_splitter.setSizes([400, 500])  # Charts and bottom panels

        main_layout.addWidget(self.main_splitter)

    def setup_menu(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Load Data action
        load_data_action = QAction("Load Market Data", self)
        load_data_action.triggered.connect(self.load_market_data)
        file_menu.addAction(load_data_action)

        # Load Strategy action
        load_strategy_action = QAction("Load Strategy", self)
        load_strategy_action.triggered.connect(self.load_strategy)
        file_menu.addAction(load_strategy_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Toggle panels
        toggle_strategy_action = QAction("Strategy Panel", self)
        toggle_strategy_action.setCheckable(True)
        toggle_strategy_action.setChecked(True)
        toggle_strategy_action.triggered.connect(lambda: self.toggle_panel(self.strategy_panel))
        view_menu.addAction(toggle_strategy_action)

        toggle_charts_action = QAction("Charts Panel", self)
        toggle_charts_action.setCheckable(True)
        toggle_charts_action.setChecked(True)
        toggle_charts_action.triggered.connect(lambda: self.toggle_panel(self.charts_widget))
        view_menu.addAction(toggle_charts_action)

        toggle_backtest_action = QAction("Backtest Panel", self)
        toggle_backtest_action.setCheckable(True)
        toggle_backtest_action.setChecked(True)
        toggle_backtest_action.triggered.connect(lambda: self.toggle_panel(self.backtest_panel))
        view_menu.addAction(toggle_backtest_action)

        toggle_results_action = QAction("Results Panel", self)
        toggle_results_action.setCheckable(True)
        toggle_results_action.setChecked(True)
        toggle_results_action.triggered.connect(lambda: self.toggle_panel(self.results_panel))
        view_menu.addAction(toggle_results_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        # Diagnostic action
        diagnostic_action = QAction("Run Diagnostics", self)
        diagnostic_action.triggered.connect(self.run_diagnostics)
        tools_menu.addAction(diagnostic_action)

        # Clear Cache action
        clear_cache_action = QAction("Clear Cache", self)
        clear_cache_action.triggered.connect(self.clear_cache)
        tools_menu.addAction(clear_cache_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Documentation action
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = self.statusBar()

        # Status label
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        self.status_bar.addPermanentWidget(QLabel("v1.0.0"))

        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def connect_signals(self):
        """Connect controller signals to UI updates"""
        self.controller.data_loaded.connect(self.on_data_loaded)
        self.controller.strategy_loaded.connect(self.on_strategy_loaded)
        self.controller.backtest_started.connect(self.on_backtest_started)
        self.controller.backtest_progress.connect(self.on_backtest_progress)
        self.controller.backtest_finished.connect(self.on_backtest_finished)
        self.controller.backtest_error.connect(self.on_backtest_error)

    def initialize_app(self):
        """Initialize the application"""
        try:
            # Load available strategies
            # self.controller.load_available_strategies()

            self.update_status("Application initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            self.update_status(f"Initialization error: {str(e)}")

    def load_market_data(self):
        """Load market data from file"""
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Market Data",
            "", "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                # self.controller.load_data_from_file(filename)
                self.update_status(f"Data loading not implemented yet: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def load_strategy(self):
        """Load strategy from file"""
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Strategy",
            "", "Python Files (*.py);;All Files (*)"
        )

        if filename:
            try:
                # self.controller.load_strategy_from_file(filename)
                self.update_status(f"Strategy loading not implemented yet: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load strategy: {str(e)}")

    def toggle_panel(self, panel):
        """Toggle visibility of a panel"""
        panel.setVisible(not panel.isVisible())

    def run_diagnostics(self):
        """Run system diagnostics"""
        try:
            # results = self.controller.run_diagnostics()
            # self.show_diagnostic_results(results)
            QMessageBox.information(self, "Diagnostics", "Diagnostics not implemented yet")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Diagnostics failed: {str(e)}")

    def clear_cache(self):
        """Clear application cache"""
        try:
            # self.controller.clear_cache()
            QMessageBox.information(self, "Cache", "Cache clearing not implemented yet")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear cache: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>Trading IA - Strategy Testing Platform</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Description:</b> Professional trading strategy testing and benchmarking platform</p>
        <p><b>Features:</b></p>
        <ul>
            <li>Advanced backtesting engine with parallel processing</li>
            <li>Comprehensive performance metrics and risk analysis</li>
            <li>Interactive charts and visualizations</li>
            <li>Strategy optimization and parameter tuning</li>
            <li>Real-time performance monitoring</li>
        </ul>
        <p><b>Built with:</b> Python 3.11, PyQt6, Pandas, NumPy</p>
        """

        QMessageBox.about(self, "About Trading IA", about_text)

    def show_documentation(self):
        """Show documentation"""
        # This would typically open a help file or web page
        QMessageBox.information(self, "Documentation",
                               "Documentation is available in the 'docs/' folder of the project.")

    def show_diagnostic_results(self, results):
        """Show diagnostic results in a dialog"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Diagnostic Results")
        dialog.setText("System Diagnostics Completed")

        # Format results
        details = "Diagnostic Results:\n\n"
        for key, value in results.items():
            details += f"{key}: {value}\n"

        dialog.setDetailedText(details)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()

    def on_data_loaded(self, data_info):
        """Handle data loading completion"""
        self.update_status(f"Data loaded: {data_info}")
        # self.charts_widget.update_data_display()

    def on_strategy_loaded(self, strategy_info):
        """Handle strategy loading completion"""
        self.update_status(f"Strategy loaded: {strategy_info}")
        # self.strategy_panel.update_strategy_display()

    def on_backtest_started(self):
        """Handle backtest start"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.update_status("Backtest running...")

    def on_backtest_progress(self, message):
        """Handle backtest progress updates"""
        self.update_status(message)

    def on_backtest_finished(self, results):
        """Handle backtest completion"""
        self.progress_bar.setVisible(False)
        count = len(results) if results else 0
        self.update_status(f"Backtest completed: {count} results")

        # Update all panels with results
        # self.charts_widget.update_backtest_results(results)
        self.backtest_panel.on_backtest_finished(results)
        self.results_panel.on_backtest_finished(results)

    def on_backtest_error(self, error):
        """Handle backtest errors"""
        self.progress_bar.setVisible(False)
        self.update_status(f"Backtest error: {error}")
        QMessageBox.critical(self, "Backtest Error", error)

        # Trigger alert for backtest error
        self.controller.trigger_alert(
            AlertType.STRATEGY_ERROR,
            AlertSeverity.HIGH,
            "Error en Backtest",
            f"Error durante la ejecución de backtests: {error}",
            "backtester",
            {"error": error}
        )

    def update_status(self, message):
        """Update status bar message"""
        self.status_label.setText(message)
        logger.info(f"Status: {message}")

    def closeEvent(self, event):
        """Handle application close event"""
        # Save any pending work or settings
        try:
            # self.controller.save_settings()
            pass
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

        # Confirm exit if backtest is running
        # if hasattr(self.controller, 'is_backtest_running') and self.controller.is_backtest_running:
        #     reply = QMessageBox.question(
        #         self, "Confirm Exit",
        #         "A backtest is currently running. Are you sure you want to exit?",
        #         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        #         QMessageBox.StandardButton.No
        #     )
        #
        #     if reply == QMessageBox.StandardButton.No:
        #         event.ignore()
        #         return

        event.accept()

def main():
    """Main application entry point"""
    from PySide6.QtWidgets import QApplication

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Trading IA")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Trading IA Team")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()