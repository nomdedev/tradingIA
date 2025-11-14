import sys
import logging
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                               QVBoxLayout, QHBoxLayout, QStatusBar, QLabel, 
                               QPushButton, QFrame, QSplitter)
from PySide6.QtCore import Qt, QDateTime, QTimer
from PySide6.QtGui import QIcon, QFont, QPalette, QColor

# Import platform components
from backend_core import DataManager, StrategyEngine
from backtester_core import BacktesterCore
from analysis_engines import AnalysisEngines
from settings_manager import SettingsManager
from reporters_engine import ReportersEngine
from live_monitor_engine import LiveMonitorEngine

# Import GUI tabs
from gui.platform_gui_tab0 import Tab0Dashboard
from gui.platform_gui_tab1_improved import Tab1DataManagement
from gui.platform_gui_tab2_improved import Tab2StrategyConfig
from gui.platform_gui_tab3_improved import Tab3BacktestRunner
from gui.platform_gui_tab4_improved import Tab4ResultsAnalysis
from gui.platform_gui_tab5_improved import Tab5ABTesting
from gui.platform_gui_tab6_improved import Tab6LiveMonitoring
from gui.platform_gui_tab7_improved import Tab7AdvancedAnalysis
from gui.platform_gui_tab8 import Tab8SystemSettings
from gui.platform_gui_tab9_data_download import Tab9DataDownload
from gui.platform_gui_tab10_help import Tab10Help


class TradingPlatform(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 TradingIA - Advanced Trading Platform v2.0")
        self.setGeometry(50, 50, 1800, 1000)
        
        # Apply modern dark theme
        self.apply_modern_theme()

        # Apply modern dark theme
        self.apply_modern_theme()

        # Initialize backend engines
        self.data_manager = DataManager()
        self.strategy_engine = StrategyEngine()
        self.backtester = BacktesterCore()
        self.analysis_engines = AnalysisEngines()
        self.live_monitor = LiveMonitorEngine("", "")
        self.settings = SettingsManager()
        self.reporters = ReportersEngine()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Shared data
        self.data_dict = {}
        self.config_dict = {}
        self.last_backtest_results = {}

        # Create modern tabs with icons
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(False)
        self.tabs.setDocumentMode(True)
        
        # Add Dashboard as first tab
        self.dashboard_tab = Tab0Dashboard(self)
        self.tabs.addTab(self.dashboard_tab, "🏠 Dashboard")
        
        # Add other tabs with modern styling
        self.tabs.addTab(Tab1DataManagement(self), "📊 Data")
        self.tabs.addTab(Tab2StrategyConfig(self, self.strategy_engine), "⚙️ Strategy")
        self.tabs.addTab(Tab3BacktestRunner(self, self.backtester), "▶️ Backtest")
        self.tabs.addTab(Tab4ResultsAnalysis(self), "📈 Results")
        self.tabs.addTab(Tab5ABTesting(self, self.backtester), "⚖️ A/B Test")
        self.tabs.addTab(Tab6LiveMonitoring(self), "🔴 Live")
        self.tabs.addTab(Tab7AdvancedAnalysis(self, self.analysis_engines), "🔧 Research")
        self.tabs.addTab(Tab8SystemSettings(self), "⚙️ Settings")
        self.tabs.addTab(Tab9DataDownload(self), "📥 Data Download")
        self.tabs.addTab(Tab10Help(self), "❓ Help")

        # Set central widget
        self.setCentralWidget(self.tabs)

        # Create modern status bar
        self.create_modern_statusbar()

        # Load configuration
        self.settings.load_config()

        # Auto-load default BTC/USD data
        QTimer.singleShot(1000, self.auto_load_default_data)  # Delay to allow UI to show first

        # Connect tab change signal
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.show()

    def apply_modern_theme(self):
        """Apply modern dark theme with professional styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #252525;
                border-radius: 8px;
                margin-top: -1px;
            }
            
            QTabWidget::tab-bar {
                alignment: left;
            }
            
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 13px;
                font-weight: 500;
                min-width: 100px;
            }
            
            QTabBar::tab:selected {
                background-color: #0e639c;
                color: white;
                font-weight: 600;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #3d3d3d;
                color: #ffffff;
            }
            
            QStatusBar {
                background-color: #1e1e1e;
                color: #cccccc;
                border-top: 1px solid #3d3d3d;
                font-size: 11px;
                padding: 6px;
            }
            
            QLabel {
                color: #cccccc;
            }
            
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background-color: #1177bb;
            }
            
            QPushButton:pressed {
                background-color: #0d5689;
            }
            
            QPushButton:disabled {
                background-color: #3d3d3d;
                color: #666666;
            }
            
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
                font-size: 12px;
            }
            
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #0e639c;
            }
            
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
                min-width: 120px;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-style: solid;
                border-width: 4px 4px 0px 4px;
                border-color: #cccccc transparent transparent transparent;
            }
            
            QTableWidget, QTableView {
                background-color: #2d2d2d;
                alternate-background-color: #252525;
                color: #cccccc;
                gridline-color: #3d3d3d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            
            QHeaderView::section {
                background-color: #1e1e1e;
                color: #cccccc;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #3d3d3d;
                font-weight: 600;
            }
            
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #4d4d4d;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #5d5d5d;
            }
            
            QScrollBar:horizontal {
                background-color: #2d2d2d;
                height: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:horizontal {
                background-color: #4d4d4d;
                border-radius: 6px;
                min-width: 20px;
            }
            
            QScrollBar::add-line, QScrollBar::sub-line {
                border: none;
                background: none;
            }
            
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
                color: white;
                background-color: #2d2d2d;
            }
            
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 3px;
            }
        """)

    def create_modern_statusbar(self):
        """Create modern status bar with indicators"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status label
        self.status_label = QLabel("🟢 Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4ec9b0;
                font-weight: 600;
                padding: 4px 12px;
            }
        """)
        
        # Timestamp label
        self.time_label = QLabel()
        self.update_time()
        self.time_label.setStyleSheet("""
            QLabel {
                color: #808080;
                padding: 4px 12px;
            }
        """)
        
        # Version label
        version_label = QLabel("v2.0.0")
        version_label.setStyleSheet("""
            QLabel {
                color: #0e639c;
                font-weight: 600;
                padding: 4px 12px;
            }
        """)
        
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.time_label)
        self.status_bar.addPermanentWidget(version_label)
        
        # Update time every second
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)

    def update_time(self):
        """Update timestamp in status bar"""
        timestamp = QDateTime.currentDateTime().toString("MMM dd, yyyy | hh:mm:ss")
        self.time_label.setText(timestamp)

    def on_tab_changed(self, index):
        """Handle tab changes"""
        current_tab = self.tabs.widget(index)

        # Call tab-specific activation methods
        if hasattr(current_tab, 'on_tab_activated'):
            current_tab.on_tab_activated()

    def update_status(self, msg, status_type="info"):
        """Update status bar with color coding"""
        status_icons = {
            "success": "🟢",
            "warning": "🟡",
            "error": "🔴",
            "info": "🔵",
            "processing": "⏳"
        }
        
        status_colors = {
            "success": "#4ec9b0",
            "warning": "#dcdcaa",
            "error": "#f48771",
            "info": "#569cd6",
            "processing": "#c586c0"
        }
        
        icon = status_icons.get(status_type, "🔵")
        color = status_colors.get(status_type, "#569cd6")
        
        self.status_label.setText(f"{icon} {msg}")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: 600;
                padding: 4px 12px;
            }}
        """)

    def _create_placeholder_tab(self, tab_name):
        """Create placeholder tab for unimplemented features"""
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel(f"{tab_name} - Coming Soon!")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("QLabel { font-size: 18px; color: #666; }")

        layout.addWidget(label)
        widget.setLayout(layout)
        return widget

    def auto_load_default_data(self):
        """Automatically load BTC/USD data on startup"""
        try:
            self.logger.info("Auto-loading default BTC/USD data...")
            
            # Import required modules
            from datetime import datetime, timedelta
            
            # Default parameters for BTC/USD
            symbol = "BTC/USD"
            timeframe = "1Hour"  # Default timeframe
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            # Load data using data manager
            df = self.data_manager.load_alpaca_data(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                timeframe=timeframe
            )
            
            if isinstance(df, dict) and 'error' in df:
                self.logger.warning(f"Failed to auto-load BTC/USD data: {df['error']}")
                return
            
            # Store in platform data dict
            data_key = f"{symbol.replace('/', '-')}_{timeframe}"
            self.data_dict[data_key] = df
            
            self.logger.info(f"✅ Auto-loaded {len(df)} records of BTC/USD {timeframe} data")
            
            # Update status bar
            self.statusBar().showMessage(f"✅ Auto-loaded BTC/USD data ({len(df)} records)", 5000)
            
        except Exception as e:
            self.logger.error(f"Error in auto_load_default_data: {e}")


def main():
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("BTC Trading Strategy Platform")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("TradingIA")

    # Create and show main window
    platform = TradingPlatform()

    # Start event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
