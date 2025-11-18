import sys
import logging
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                               QVBoxLayout, QHBoxLayout, QStatusBar, QLabel, 
                               QPushButton, QFrame, QSplitter, QGroupBox)
from PySide6.QtCore import Qt, QDateTime, QTimer
from PySide6.QtGui import QIcon, QFont, QPalette, QColor

# Import platform components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backend_core import DataManager, StrategyEngine
from core.execution.backtester_core import BacktesterCore
from src.analysis_engines import AnalysisEngines
from utils.settings_manager import SettingsManager
from src.reporters_engine import ReportersEngine
from src.live_monitor_engine import LiveMonitorEngine

# Import configuration and logging
from src.config.user_config import UserConfigManager
from src.utils.session_logger import SessionLogger

# Import GUI tabs
from src.gui.platform_gui_tab0_enhanced import EnhancedTab0Dashboard
from src.gui.platform_gui_tab1_improved import Tab1DataManagement
from src.gui.platform_gui_tab2_improved import Tab2StrategyConfig
from src.gui.platform_gui_tab3_improved import Tab3BacktestRunner
from src.gui.platform_gui_tab4_improved import Tab4ResultsAnalysis
from src.gui.platform_gui_tab5_improved import Tab5ABTesting
from src.gui.platform_gui_tab6_user_friendly import Tab6LiveMonitoringUserFriendly
from src.gui.platform_gui_tab7_improved import Tab7AdvancedAnalysis
from src.gui.platform_gui_tab9_data_download import Tab9DataDownload
from src.gui.platform_gui_tab10_help import Tab10Help
from src.gui.platform_gui_tab11_risk_metrics import Tab11RiskMetrics

# Import broker components
try:
    from core.brokers import BrokerManager, BrokersPanel
    BROKERS_AVAILABLE = True
except ImportError:
    BROKERS_AVAILABLE = False
    BrokerManager = None
    BrokersPanel = None

# Import API components
try:
    from core.api import API_AVAILABLE
except ImportError:
    API_AVAILABLE = False

# Import loading screen
from src.loading_screen import LoadingScreen

# Import help system
from src.gui.help_system import init_help_system


class TradingPlatform(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 TradingIA - Advanced Trading Platform v2.0")
        self.setGeometry(50, 50, 1800, 1000)

        # Apply modern dark theme
        self.apply_modern_theme()

        # Initialize basic attributes (heavy components will be loaded later)
        self.config_manager = None
        self.session_logger = None
        self.data_manager = None
        self.strategy_engine = None
        self.backtester = None
        self.analysis_engines = None
        self.live_monitor = None
        self.settings = None
        self.reporters = None
        self.broker_manager = None
        self.api_app = None
        self.api_thread = None
        self.api_running = False
        self.BROKERS_AVAILABLE = BROKERS_AVAILABLE
        self.API_AVAILABLE = API_AVAILABLE

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Shared data
        self.data_dict = {}
        self.config_dict = {}
        self.last_backtest_results = {}

        # Don't initialize tabs yet - will be done after loading screen
        self.tabs = None
        self.dashboard_tab = None

        # Show loading screen instead of initializing everything
        self.show_loading_screen()

    def show_loading_screen(self):
        """Show the loading screen and start component initialization"""
        self.loading_screen = LoadingScreen(self)
        self.loading_screen.show()

    def _create_tabs(self):
        """Create all GUI tabs (called after loading is complete)"""
        # Create modern tabs with icons
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(False)
        self.tabs.setDocumentMode(True)

        # Add Dashboard as first tab
        self.dashboard_tab = EnhancedTab0Dashboard(self)
        self.tabs.addTab(self.dashboard_tab, "🏠 Dashboard")

        # Add other tabs with modern styling
        self.tabs.addTab(Tab1DataManagement(self), "📊 Data")
        self.tabs.addTab(Tab2StrategyConfig(self, self.strategy_engine), "⚙️ Strategy")
        self.tabs.addTab(Tab3BacktestRunner(self, self.backtester), "▶️ Backtest")
        self.tabs.addTab(Tab4ResultsAnalysis(self), "📈 Results")
        self.tabs.addTab(Tab5ABTesting(self, self.backtester), "⚖️ A/B Test")
        self.tabs.addTab(Tab6LiveMonitoringUserFriendly(self), "🔴 Live")

        # Add Brokers tab if available
        if self.BROKERS_AVAILABLE and self.broker_manager and BrokersPanel:
            self.brokers_tab = BrokersPanel(self.broker_manager)
            self.tabs.addTab(self.brokers_tab, "💰 Brokers")

        # Add API tab if available
        if self.API_AVAILABLE:
            self.api_tab = self._create_api_tab()
            self.tabs.addTab(self.api_tab, "🌐 API")

        self.tabs.addTab(Tab7AdvancedAnalysis(self, self.analysis_engines), "🔧 Research")
        self.tabs.addTab(Tab9DataDownload(self), "📥 Data Download")
        self.tabs.addTab(Tab10Help(self), "❓ Help")
        self.tabs.addTab(Tab11RiskMetrics(self), "📊 Risk Metrics")

        # Set central widget
        self.setCentralWidget(self.tabs)

        # Create modern status bar
        self.create_modern_statusbar()

        # Load saved configuration
        self.load_saved_config()

        # Connect tab change signal
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Connect dashboard signals
        self.connect_dashboard_signals()
    
    def _create_api_tab(self):
        """Create the API management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("🌐 REST API Management")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setProperty("class", "title")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Control the REST API server for remote access to the TradingIA platform.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Status section
        status_group = QFrame()
        status_group.setFrameStyle(QFrame.Shape.Box)
        status_group.setProperty("class", "metric-card")
        status_layout = QVBoxLayout(status_group)
        
        status_title = QLabel("📊 Server Status")
        status_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        status_title.setProperty("class", "metric-label")
        status_layout.addWidget(status_title)
        
        self.api_status_label = QLabel("🛑 Server Stopped")
        self.api_status_label.setProperty("class", "metric-value")
        status_layout.addWidget(self.api_status_label)
        
        layout.addWidget(status_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.start_api_btn = QPushButton("▶️ Start Server")
        self.start_api_btn.clicked.connect(self.start_api_server)
        self.start_api_btn.setMinimumHeight(32)
        self.start_api_btn.setMaximumHeight(32)
        buttons_layout.addWidget(self.start_api_btn)
        
        self.stop_api_btn = QPushButton("⏹️ Stop Server")
        self.stop_api_btn.clicked.connect(self.stop_api_server)
        self.stop_api_btn.setMinimumHeight(32)
        self.stop_api_btn.setMaximumHeight(32)
        self.stop_api_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_api_btn)
        
        layout.addLayout(buttons_layout)
        
        # API Information
        info_group = QFrame()
        info_group.setFrameStyle(QFrame.Shape.Box)
        info_group.setProperty("class", "card")
        info_layout = QVBoxLayout(info_group)
        
        info_title = QLabel("ℹ️ API Information")
        info_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        info_title.setProperty("class", "metric-label")
        info_layout.addWidget(info_title)
        
        api_info = QLabel("""
<b>Base URL:</b> http://127.0.0.1:8000<br>
<b>Endpoints:</b><br>
• GET /health - Health check<br>
• GET /status - System status<br>
• GET /data - Market data<br>
• POST /backtest - Run backtest<br>
• GET /backtest/{task_id} - Backtest status<br>
• GET /strategies - Available strategies<br>
• POST /brokers - Add broker<br>
• POST /orders - Place orders<br>
        """)
        api_info.setTextFormat(Qt.TextFormat.RichText)
        api_info.setWordWrap(True)
        info_layout.addWidget(api_info)
        
        layout.addWidget(info_group)
        
        # Update status timer
        self.api_status_timer = QTimer()
        self.api_status_timer.timeout.connect(self.update_api_status)
        self.api_status_timer.start(2000)  # Update every 2 seconds
        
        layout.addStretch()
        return tab
    
    def update_api_status(self):
        """Update API server status display."""
        if self.api_running:
            self.api_status_label.setText("✅ Server Running on http://127.0.0.1:8000")
            self.start_api_btn.setEnabled(False)
            self.stop_api_btn.setEnabled(True)
        else:
            self.api_status_label.setText("🛑 Server Stopped")
            self.start_api_btn.setEnabled(True)
            self.stop_api_btn.setEnabled(False)
    
    def load_saved_config(self):
        """Load saved user configuration from previous session"""
        try:
            live_config = self.config_manager.get_live_trading_config()
            if live_config:
                self.session_logger.log_action('config_loaded', {
                    'ticker': live_config.get('ticker'),
                    'strategy': live_config.get('strategy'),
                    'last_session': self.config_manager.config.get('last_session')
                })
                
                # Apply loaded config (would need to update relevant tabs)
                self.logger.info(f"Loaded configuration from previous session: {live_config}")
        except Exception as e:
            self.session_logger.log_error('config_load_error', str(e))
            self.logger.warning(f"Could not load previous configuration: {e}")

    def start_api_server(self):
        """Start the REST API server in a background thread."""
        if not API_AVAILABLE:
            self.logger.warning("API not available - FastAPI not installed")
            return

        if self.api_running:
            self.logger.info("API server already running")
            return

        def run_api():
            try:
                self.logger.info("Starting REST API server on port 8000...")
                uvicorn.run(
                    api_app,
                    host="127.0.0.1",  # Localhost only for security
                    port=8000,
                    log_level="warning"  # Reduce uvicorn logging
                )
            except Exception as e:
                self.logger.error(f"Failed to start API server: {e}")

        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.api_thread.start()
        self.api_running = True
        self.logger.info("REST API server started in background thread")
        self._update_api_status()

    def stop_api_server(self):
        """Stop the REST API server."""
        if self.api_running:
            self.api_running = False
            self.logger.info("REST API server stopped")
            self._update_api_status()
        else:
            self.logger.info("API server not running")

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
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
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
                padding: 5px 8px;
                font-size: 13px;
                min-height: 24px;
                max-height: 32px;
            }
            
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #0e639c;
            }
            
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px 8px;
                min-width: 120px;
                min-height: 26px;
                max-height: 32px;
                font-size: 13px;
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

            /* ===== MODERN ENHANCEMENTS ===== */
            /* Optimized button styling - compact and modern */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0e639c, stop:1 #0a4f7a);
                border: 1px solid #0a4f7a;
                padding: 6px 10px;
                min-height: 28px;
                max-height: 36px;
                font-size: 13px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1177bb, stop:1 #0d5a8c);
                border: 1px solid #1177bb;
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0a4f7a, stop:1 #083d5f);
                padding-top: 7px;
                padding-bottom: 5px;
            }

            /* Enhanced focus states */
            QLineEdit:focus, QComboBox:focus {
                border-color: #0e639c;
            }

            /* Better table styling */
            QTableWidget::item:hover {
                background-color: #3d3d3d;
            }

            /* Card-like appearance for frames */
            QFrame[class="card"] {
                background-color: #2d2d2d;
                border: 2px solid #333333;
                border-radius: 12px;
                padding: 20px;
            }

            /* Metric cards with accent borders */
            QFrame[class="metric-card"] {
                background-color: #2d2d2d;
                border: 2px solid #0e639c;
                border-radius: 12px;
                padding: 20px;
                margin: 4px;
            }

            /* Status cards */
            QFrame[class="success-card"] {
                background-color: #2d2d2d;
                border: 2px solid #28a745;
                border-radius: 12px;
                padding: 20px;
            }

            QFrame[class="warning-card"] {
                background-color: #2d2d2d;
                border: 2px solid #ffc107;
                border-radius: 12px;
                padding: 20px;
            }

            QFrame[class="error-card"] {
                background-color: #2d2d2d;
                border: 2px solid #dc3545;
                border-radius: 12px;
                padding: 20px;
            }

            /* Enhanced typography */
            QLabel[class="title"] {
                font-size: 20px;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 12px;
                letter-spacing: 0.5px;
            }

            QLabel[class="metric-value"] {
                font-size: 36px;
                font-weight: 700;
                color: #0e639c;
                margin: 4px 0;
            }

            QLabel[class="metric-label"] {
                font-size: 12px;
                color: #888888;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            /* Better group boxes */
            QGroupBox {
                font-size: 15px;
                font-weight: 600;
                color: #ffffff;
                border: 2px solid #333333;
                border-radius: 12px;
                margin-top: 16px;
                padding-top: 16px;
                background-color: #2d2d2d;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 12px 0 12px;
                color: #0e639c;
                font-weight: 600;
                font-size: 14px;
                background-color: #2d2d2d;
                border-radius: 6px;
            }

            /* Enhanced scrollbars */
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 16px;
                border-radius: 8px;
                margin: 2px;
            }

            QScrollBar::handle:vertical {
                background-color: #0e639c;
                border-radius: 8px;
                min-height: 30px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #1177bb;
            }

            /* Better table headers */
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 12px 8px;
                border: 1px solid #333333;
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            QHeaderView::section:hover {
                background-color: #2d2d2d;
            }

            /* Enhanced checkboxes */
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #333333;
                border-radius: 4px;
                background-color: #2d2d2d;
            }

            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border-color: #0e639c;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }

            QCheckBox::indicator:hover {
                border-color: #0e639c;
            }

            /* Tooltips */
            QToolTip {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
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
        tab_name = self.tabs.tabText(index)
        
        # Log tab visit
        self.session_logger.log_tab_visit(tab_name)

        # Call tab-specific activation methods
        if hasattr(current_tab, 'on_tab_activated'):
            current_tab.on_tab_activated()
    
    def closeEvent(self, event):
        """Handle application close - save config and generate session report"""
        try:
            # Stop API server
            self.stop_api_server()
            
            # Save configuration
            self.config_manager.save_config()
            self.session_logger.log_action('config_saved', {'success': True})
            
            # End session and generate report
            self.session_logger.end_session()
            
            # Show message to user
            self.logger.info(f"Session report saved: {self.session_logger.session_file}")
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            event.accept()
    
    def resizeEvent(self, event):
        """Log window resize events"""
        try:
            super().resizeEvent(event)
            
            # Only log significant resizes (> 50px difference)
            old_size = event.oldSize()
            new_size = event.size()
            
            if old_size.width() > 0 and old_size.height() > 0:  # Skip initial resize
                width_diff = abs(new_size.width() - old_size.width())
                height_diff = abs(new_size.height() - old_size.height())
                
                if width_diff > 50 or height_diff > 50:
                    if hasattr(self, 'session_logger') and self.session_logger:
                        self.session_logger.log_window_event('resize', {
                            'old_size': f"{old_size.width()}x{old_size.height()}",
                            'new_size': f"{new_size.width()}x{new_size.height()}",
                            'difference': f"{width_diff}x{height_diff}"
                        })
        except Exception as e:
            self.logger.error(f"Error logging resize event: {e}")
    
    def moveEvent(self, event):
        """Log window move events"""
        try:
            super().moveEvent(event)
            
            # Only log if window moved significantly (> 100px)
            old_pos = event.oldPos()
            new_pos = event.pos()
            
            if old_pos.x() >= 0 and old_pos.y() >= 0:  # Skip initial move
                x_diff = abs(new_pos.x() - old_pos.x())
                y_diff = abs(new_pos.y() - old_pos.y())
                
                if x_diff > 100 or y_diff > 100:
                    if hasattr(self, 'session_logger') and self.session_logger:
                        self.session_logger.log_window_event('move', {
                            'old_pos': f"{old_pos.x()},{old_pos.y()}",
                            'new_pos': f"{new_pos.x()},{new_pos.y()}",
                            'difference': f"{x_diff},{y_diff}"
                        })
        except Exception as e:
            self.logger.error(f"Error logging move event: {e}")

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

    def _update_api_status(self):
        """Update API status display."""
        if not API_AVAILABLE:
            self.api_status_label.setText("❌ API not available (FastAPI not installed)")
            self.api_status_label.setStyleSheet("QLabel { font-size: 14px; color: #dc3545; }")
            self.btn_start_api.setEnabled(False)
            return

        if self.api_running:
            self.api_status_label.setText("🟢 API server running on port 8000")
            self.api_status_label.setStyleSheet("QLabel { font-size: 14px; color: #28a745; }")
            self.btn_start_api.setEnabled(False)
            self.btn_stop_api.setEnabled(True)
        else:
            self.api_status_label.setText("🔴 API server not running")
            self.api_status_label.setStyleSheet("QLabel { font-size: 14px; color: #dc3545; }")
            self.btn_start_api.setEnabled(True)
            self.btn_stop_api.setEnabled(False)

    def _create_placeholder_tab(self, tab_name):
        """Create placeholder tab for unimplemented features"""
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel(f"{tab_name} - Coming Soon!")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setProperty("class", "title")
        label.setStyleSheet("QLabel { color: #666; }")

        layout.addWidget(label)
        widget.setLayout(layout)
        return widget

    def auto_load_default_data(self):
        """Automatically load BTC/USD data on startup"""
        try:
            self.logger.info("Auto-loading default BTC/USD data...")
            
            # Try to load from local files first (more reliable)
            self._load_local_btc_data()
            
        except Exception as e:
            self.logger.error(f"Error in auto_load_default_data: {e}")
            self.statusBar().showMessage("❌ Error loading default data", 5000)
    
    def _load_local_btc_data(self):
        """Load BTC data from local files as fallback"""
        try:
            import os
            import pandas as pd
            data_dir = "data"
            
            # Look for BTC data files, prefer 5Min
            btc_files = [f for f in os.listdir(data_dir) if f.startswith("btc") and f.endswith(".csv")]
            
            if not btc_files:
                self.logger.warning("No local BTC data files found")
                return
            
            # Prefer 5Min data, then 15Min, then 1H
            preferred_files = []
            for pref in ["5min", "15min", "1h"]:
                for file in btc_files:
                    if pref in file.lower():
                        preferred_files.append(file)
                        break
            
            if not preferred_files:
                preferred_files = btc_files
            
            # Load the preferred BTC file
            file_path = os.path.join(data_dir, preferred_files[0])
            
            self.logger.info(f"Loading BTC data from: {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if df.empty:
                self.logger.warning("Loaded BTC data file is empty")
                return
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure we have required OHLC columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns in BTC data: {missing_cols}")
                return
            
            # Determine timeframe from filename
            filename_lower = preferred_files[0].lower()
            if "15min" in filename_lower:
                timeframe = "15Min"
            elif "5min" in filename_lower:
                timeframe = "5Min"
            elif "1h" in filename_lower or "1hour" in filename_lower:
                timeframe = "1Hour"
            else:
                # Check if it's actually 15min data by looking at time differences
                if len(df) > 1:
                    time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
                    if time_diff == 15:
                        timeframe = "15Min"
                    elif time_diff == 60:
                        timeframe = "1Hour"
                    else:
                        timeframe = "5Min"  # default
                else:
                    timeframe = "5Min"  # default
            
            data_key = f"BTC-USD_{timeframe}"
            self.data_dict[data_key] = df
            
            self.logger.info(f"✅ Loaded local BTC data: {len(df)} records, timeframe: {timeframe}")
            self.statusBar().showMessage(f"✅ Loaded BTC/USD {timeframe} data ({len(df)} records)", 5000)
            
            # Log to session logger
            if hasattr(self, 'session_logger') and self.session_logger:
                self.session_logger.log_data_loading_event('auto_load_success', {
                    'symbol': 'BTC-USD',
                    'timeframe': timeframe,
                    'records': len(df),
                    'date_range': f"{df.index[0]} to {df.index[-1]}",
                    'file_path': file_path
                })
            
            # Notify tabs about new data
            self._notify_tabs_data_loaded()
            
        except Exception as e:
            self.logger.error(f"Error loading local BTC data: {e}")
            self.statusBar().showMessage("❌ Error loading BTC data", 5000)
            
            # Log to session logger
            if hasattr(self, 'session_logger') and self.session_logger:
                import traceback
                self.session_logger.log_error(
                    error_type='data_loading_error',
                    error_message=str(e),
                    context={'action': 'auto_load_btc_data'},
                    stack_trace=traceback.format_exc()
                )
    
    def _notify_tabs_data_loaded(self):
        """Notify relevant tabs that new data has been loaded"""
        try:
            # Notify Tab1 (Data Management)
            if hasattr(self, 'tabs'):
                tab1 = self.tabs.widget(1)  # Tab1 is index 1
                if hasattr(tab1, 'on_data_loaded'):
                    tab1.on_data_loaded(self.data_dict)
                    
                    # Log UI update
                    if hasattr(self, 'session_logger') and self.session_logger:
                        self.session_logger.log_ui_event('data_loaded_notification', {
                            'tab': 'Tab1_DataManagement',
                            'data_keys': list(self.data_dict.keys())
                        })
        except Exception as e:
            self.logger.error(f"Error notifying tabs about data load: {e}")
            
            # Log error
            if hasattr(self, 'session_logger') and self.session_logger:
                import traceback
                self.session_logger.log_error(
                    error_type='ui_notification_error',
                    error_message=str(e),
                    context={'action': 'notify_tabs_data_loaded'},
                    stack_trace=traceback.format_exc()
                )

    def start_api_server(self):
        """Start the REST API server in a background thread."""
        if not API_AVAILABLE:
            self.statusBar().showMessage("❌ API not available - FastAPI not installed", 5000)
            return

        if self.api_running:
            self.statusBar().showMessage("ℹ️ API server already running", 3000)
            return

        try:
            def run_server():
                import uvicorn
                uvicorn.run(api_app, host="127.0.0.1", port=8000, log_level="info")

            self.api_thread = threading.Thread(target=run_server, daemon=True)
            self.api_thread.start()
            self.api_running = True
            self.logger.info("API server started on http://127.0.0.1:8000")
            self.statusBar().showMessage("✅ API server started on http://127.0.0.1:8000", 5000)

        except Exception as e:
            self.logger.error(f"Error starting API server: {e}")
            self.statusBar().showMessage(f"❌ Error starting API server: {str(e)}", 5000)

    def stop_api_server(self):
        """Stop the REST API server."""
        if not self.api_running:
            self.statusBar().showMessage("ℹ️ API server not running", 3000)
            return

        try:
            # Note: In a real implementation, you'd need a way to gracefully shutdown uvicorn
            # For now, we just mark it as stopped
            self.api_running = False
            self.api_thread = None
            self.logger.info("API server stopped")
            self.statusBar().showMessage("🛑 API server stopped", 5000)

        except Exception as e:
            self.logger.error(f"Error stopping API server: {e}")
            self.statusBar().showMessage(f"❌ Error stopping API server: {str(e)}", 5000)

    def toggle_api_server(self):
        """Toggle API server on/off."""
        if self.api_running:
            self.stop_api_server()
        else:
            self.start_api_server()

    def connect_dashboard_signals(self):
        """Connect dashboard signals to main platform actions"""
        if hasattr(self.dashboard_tab, 'tutorial_requested'):
            self.dashboard_tab.tutorial_requested.connect(self.show_tutorial)

        if hasattr(self.dashboard_tab, 'load_data_clicked'):
            self.dashboard_tab.load_data_clicked.connect(lambda: self.tabs.setCurrentIndex(1))  # Data tab

        if hasattr(self.dashboard_tab, 'select_strategy_clicked'):
            self.dashboard_tab.select_strategy_clicked.connect(lambda: self.tabs.setCurrentIndex(2))  # Strategy tab

        if hasattr(self.dashboard_tab, 'run_backtest_clicked'):
            self.dashboard_tab.run_backtest_clicked.connect(lambda: self.tabs.setCurrentIndex(3))  # Backtest tab

        if hasattr(self.dashboard_tab, 'setup_risk_clicked'):
            self.dashboard_tab.setup_risk_clicked.connect(lambda: self.tabs.setCurrentIndex(10))  # Risk tab

        if hasattr(self.dashboard_tab, 'start_live_clicked'):
            self.dashboard_tab.start_live_clicked.connect(lambda: self.tabs.setCurrentIndex(5))  # Live tab

        if hasattr(self.dashboard_tab, 'help_requested'):
            self.dashboard_tab.help_requested.connect(self.show_help)

    def show_tutorial(self):
        """Show tutorial dialog"""
        from src.gui.help_system import show_contextual_help
        show_contextual_help(
            "Tutorial Interactivo",
            "<h2>🎓 Tutorial de TradingIA</h2>"
            "<p>¡Bienvenido al tutorial interactivo!</p>"
            "<p>Sigue estos pasos para comenzar:</p>"
            "<ol>"
            "<li><b>Carga datos históricos</b> - Ve a la pestaña 'Data'</li>"
            "<li><b>Selecciona una estrategia</b> - Ve a la pestaña 'Strategy'</li>"
            "<li><b>Ejecuta un backtest</b> - Ve a la pestaña 'Backtest'</li>"
            "<li><b>Revisa los resultados</b> - Ve a la pestaña 'Results'</li>"
            "<li><b>Configura riesgos</b> - Ve a la pestaña 'Risk Metrics'</li>"
            "</ol>"
            "<p>¡Presiona F1 en cualquier momento para obtener ayuda!</p>"
        )

    def show_help(self):
        """Show help center"""
        from src.gui.help_system import show_help_center
        show_help_center()


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
