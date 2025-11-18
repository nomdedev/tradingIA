"""
Loading Screen for TradingIA Platform
Displays a progress bar and status updates during application startup
"""

import time
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                               QProgressBar, QFrame, QApplication)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QPalette, QColor


class LoadingWorker(QThread):
    """Worker thread for checking component availability and updating progress"""
    progress_updated = Signal(str, int)  # component_name, progress_percentage
    component_checked = Signal(str, bool)  # component_name, available
    checking_complete = Signal()

    def __init__(self, platform_instance):
        super().__init__()
        self.platform = platform_instance
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        """Check component availability with progress updates"""
        components = [
            ("Config Manager", self._check_config_manager),
            ("Session Logger", self._check_session_logger),
            ("Data Manager", self._check_data_manager),
            ("Strategy Engine", self._check_strategy_engine),
            ("Backtester Core", self._check_backtester),
            ("Analysis Engines", self._check_analysis_engines),
            ("Live Monitor", self._check_live_monitor),
            ("Settings Manager", self._check_settings_manager),
            ("Reporters Engine", self._check_reporters_engine),
            ("Broker Manager", self._check_broker_manager),
            ("API Server", self._check_api_server),
            ("GUI Components", self._check_gui_components),
        ]

        total_components = len(components)
        completed = 0

        for component_name, check_func in components:
            if self.is_cancelled:
                break

            self.progress_updated.emit(component_name, int((completed / total_components) * 100))

            try:
                available = check_func()
                self.component_checked.emit(component_name, available)
            except Exception as e:
                print(f"Error checking {component_name}: {e}")
                self.component_checked.emit(component_name, False)

            completed += 1
            time.sleep(0.2)  # Small delay for visual feedback

        self.progress_updated.emit("Ready", 100)
        self.checking_complete.emit()

    def _check_config_manager(self):
        """Check if config manager can be imported"""
        try:
            from src.config.user_config import UserConfigManager
            return True
        except Exception as e:
            print(f"Config Manager check error: {e}")
            return False

    def _check_session_logger(self):
        """Check if session logger can be imported"""
        try:
            from src.utils.session_logger import SessionLogger
            return True
        except Exception as e:
            print(f"Session Logger check error: {e}")
            return False

    def _check_data_manager(self):
        """Check if data manager can be imported"""
        try:
            from core.backend_core import DataManager
            return True
        except Exception as e:
            print(f"Data Manager check error: {e}")
            return False

    def _check_strategy_engine(self):
        """Check if strategy engine can be imported"""
        try:
            from core.backend_core import StrategyEngine
            return True
        except Exception as e:
            print(f"Strategy Engine check error: {e}")
            return False

    def _check_backtester(self):
        """Check if backtester can be imported"""
        try:
            from core.execution.backtester_core import BacktesterCore
            return True
        except Exception as e:
            print(f"Backtester check error: {e}")
            return False

    def _check_analysis_engines(self):
        """Check if analysis engines can be imported"""
        try:
            from src.analysis_engines import AnalysisEngines
            return True
        except Exception as e:
            print(f"Analysis Engines check error: {e}")
            return False

    def _check_live_monitor(self):
        """Check if live monitor can be imported"""
        try:
            from src.live_monitor_engine import LiveMonitorEngine
            return True
        except Exception as e:
            print(f"Live Monitor check error: {e}")
            return False

    def _check_settings_manager(self):
        """Check if settings manager can be imported"""
        try:
            from utils.settings_manager import SettingsManager
            return True
        except Exception as e:
            print(f"Settings Manager check error: {e}")
            return False

    def _check_reporters_engine(self):
        """Check if reporters engine can be imported"""
        try:
            from src.reporters_engine import ReportersEngine
            return True
        except Exception as e:
            print(f"Reporters Engine check error: {e}")
            return False

    def _check_broker_manager(self):
        """Check if broker manager can be imported"""
        try:
            from core.brokers import BrokerManager
            return True
        except ImportError:
            return False  # Not an error, just not available
        except Exception as e:
            print(f"Broker Manager check error: {e}")
            return False

    def _check_api_server(self):
        """Check if API server can be imported"""
        try:
            from core.api import API_AVAILABLE
            return API_AVAILABLE
        except ImportError:
            return False  # Not an error, just not available
        except Exception as e:
            print(f"API Server check error: {e}")
            return False

    def _check_gui_components(self):
        """Check if GUI components can be imported"""
        try:
            # Import all GUI tabs to check availability
            from src.gui.platform_gui_tab0 import Tab0Dashboard
            from src.gui.platform_gui_tab1_improved import Tab1DataManagement
            from src.gui.platform_gui_tab2_improved import Tab2StrategyConfig
            from src.gui.platform_gui_tab3_improved import Tab3BacktestRunner
            from src.gui.platform_gui_tab4_improved import Tab4ResultsAnalysis
            from src.gui.platform_gui_tab5_improved import Tab5ABTesting
            from src.gui.platform_gui_tab6_user_friendly import Tab6LiveMonitoringUserFriendly
            from src.gui.platform_gui_tab7_improved import Tab7AdvancedAnalysis
            from src.gui.platform_gui_tab9_data_download import Tab9DataDownload
            from src.gui.platform_gui_tab10_help import Tab10Help
            return True
        except Exception as e:
            print(f"GUI Components check error: {e}")
            return False


class LoadingScreen(QWidget):
    """Loading screen with progress bar and status updates"""

    def __init__(self, platform_instance):
        super().__init__()
        self.platform = platform_instance
        self.worker = None
        self.init_ui()
        self.start_loading()

    def init_ui(self):
        """Initialize the loading screen UI"""
        self.setWindowTitle("🚀 TradingIA - Loading...")
        self.setFixedSize(700, 500)  # Increased size for better readability
        self.center_window()

        # Apply dark theme
        self.apply_dark_theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)  # Increased margins
        layout.setSpacing(25)  # Increased spacing

        # Logo/Title section
        title_layout = QVBoxLayout()
        title_layout.setSpacing(10)

        self.title_label = QLabel("🚀 TradingIA")
        self.title_label.setFont(QFont("Arial", 32, QFont.Weight.Bold))  # Increased font size
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("Advanced Trading Platform v2.0")
        self.subtitle_label.setFont(QFont("Arial", 16))  # Increased font size
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("color: #888888;")
        title_layout.addWidget(self.subtitle_label)

        layout.addLayout(title_layout)

        # Progress section
        progress_group = QFrame()
        progress_group.setFrameStyle(QFrame.Shape.Box)
        progress_group.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 2px solid #444444;
                border-radius: 10px;
                padding: 20px;  /* Increased padding */
            }
        """)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(20)  # Increased spacing

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(30)  # Increased height
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 10px;
                text-align: center;
                background-color: #1e1e1e;
                font-size: 12px;  /* Added font size */
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 8px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        # Current component label
        self.component_label = QLabel("Initializing platform components...")
        self.component_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))  # Increased font size and made bold
        self.component_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.component_label.setStyleSheet("color: #ffffff;")
        self.component_label.setWordWrap(True)  # Allow text wrapping
        self.component_label.setMinimumHeight(40)  # Set minimum height
        progress_layout.addWidget(self.component_label)

        # Status messages
        self.status_label = QLabel("Starting up...")
        self.status_label.setFont(QFont("Arial", 12))  # Increased font size
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #cccccc;")  # Made text lighter
        self.status_label.setWordWrap(True)  # Allow text wrapping
        self.status_label.setMinimumHeight(30)  # Set minimum height
        progress_layout.addWidget(self.status_label)

        layout.addWidget(progress_group)

        # Loading animation dots
        self.dots_label = QLabel("⏳")
        self.dots_label.setFont(QFont("Arial", 28))  # Increased font size
        self.dots_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dots_label.setMinimumHeight(50)  # Set minimum height
        layout.addWidget(self.dots_label)

        # Start dots animation
        self.dots_timer = QTimer()
        self.dots_timer.timeout.connect(self.animate_dots)
        self.dots_timer.start(500)
        self.dot_count = 0

    def apply_dark_theme(self):
        """Apply dark theme to the loading screen"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(33, 33, 33))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)

        self.setStyleSheet("""
            QWidget {
                background-color: #212121;
                color: #ffffff;
            }
        """)

    def center_window(self):
        """Center the window on screen"""
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

    def animate_dots(self):
        """Animate the loading dots"""
        dots = ["⏳", "⏳.", "⏳..", "⏳..."]
        self.dot_count = (self.dot_count + 1) % len(dots)
        self.dots_label.setText(dots[self.dot_count])

    def start_loading(self):
        """Start the loading process"""
        self.worker = LoadingWorker(self.platform)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.component_checked.connect(self.on_component_checked)
        self.worker.checking_complete.connect(self.on_checking_complete)
        self.worker.start()

    def update_progress(self, component_name, progress):
        """Update progress bar and component label"""
        try:
            self.progress_bar.setValue(progress)
            self.component_label.setText(f"Loading: {component_name}...")
            QApplication.processEvents()  # Keep UI responsive
        except Exception as e:
            print(f"ERROR in update_progress: {e}")
            import traceback
            traceback.print_exc()

    def on_component_checked(self, component_name, available):
        """Handle component check completion"""
        status = "✅" if available else "❌"
        self.status_label.setText(f"{status} {component_name} checked")
        QApplication.processEvents()

    def on_checking_complete(self):
        """Handle checking completion and start actual loading"""
        self.dots_timer.stop()
        self.progress_bar.setValue(100)
        self.component_label.setText("Checks complete! Loading components...")
        self.status_label.setText("🚀 Initializing TradingIA...")

        # Now actually load the components in the main thread
        self.load_components()

        # Small delay to show completion
        QTimer.singleShot(500, self.finish_loading)

    def load_components(self):
        """Load all platform components in the main thread"""
        try:
            # Initialize configuration manager
            try:
                from src.config.user_config import UserConfigManager
                self.platform.config_manager = UserConfigManager()
                print("✅ Config Manager loaded")
            except Exception as e:
                print(f"⚠️ Config Manager failed: {e}")
                self.platform.config_manager = None

            # Initialize session logger
            try:
                from src.utils.session_logger import SessionLogger
                self.platform.session_logger = SessionLogger()
                self.platform.session_logger.log_action('platform_start', {'version': '2.0.0'})
                print("✅ Session Logger loaded")
            except Exception as e:
                print(f"⚠️ Session Logger failed: {e}")
                self.platform.session_logger = None

            # Initialize backend engines
            try:
                from core.backend_core import DataManager, StrategyEngine
                self.platform.data_manager = DataManager()
                self.platform.strategy_engine = StrategyEngine()
                print("✅ Backend engines loaded")
            except Exception as e:
                print(f"⚠️ Backend engines failed: {e}")
                self.platform.data_manager = None
                self.platform.strategy_engine = None

            try:
                from core.execution.backtester_core import BacktesterCore
                self.platform.backtester = BacktesterCore()
                print("✅ Backtester loaded")
            except Exception as e:
                print(f"⚠️ Backtester failed: {e}")
                self.platform.backtester = None

            try:
                from src.analysis_engines import AnalysisEngines
                self.platform.analysis_engines = AnalysisEngines()
                print("✅ Analysis engines loaded")
            except Exception as e:
                print(f"⚠️ Analysis engines failed: {e}")
                self.platform.analysis_engines = None

            try:
                from src.live_monitor_engine import LiveMonitorEngine
                self.platform.live_monitor = LiveMonitorEngine("", "")
                print("✅ Live monitor loaded")
            except Exception as e:
                print(f"⚠️ Live monitor failed: {e}")
                self.platform.live_monitor = None

            try:
                from utils.settings_manager import SettingsManager
                self.platform.settings = SettingsManager()
                print("✅ Settings manager loaded")
            except Exception as e:
                print(f"⚠️ Settings manager failed: {e}")
                self.platform.settings = None

            try:
                from src.reporters_engine import ReportersEngine
                self.platform.reporters = ReportersEngine()
                print("✅ Reporters engine loaded")
            except Exception as e:
                print(f"⚠️ Reporters engine failed: {e}")
                self.platform.reporters = None

            # Initialize broker manager if available
            try:
                from core.brokers import BrokerManager
                self.platform.broker_manager = BrokerManager()
                print("✅ Broker manager loaded")
            except ImportError:
                print("ℹ️ Broker manager not available (optional)")
                self.platform.broker_manager = None
            except Exception as e:
                print(f"⚠️ Broker manager failed: {e}")
                self.platform.broker_manager = None

            # Initialize API components
            try:
                from core.api.main import app as api_app
                self.platform.api_app = api_app
                print("✅ API components loaded")
            except ImportError:
                print("ℹ️ API components not available (optional)")
                self.platform.api_app = None
            except Exception as e:
                print(f"⚠️ API components failed: {e}")
                self.platform.api_app = None

            # Create GUI tabs
            try:
                self.platform._create_tabs()
                print("✅ GUI tabs created")
            except Exception as e:
                print(f"❌ CRITICAL: Failed to create GUI tabs: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Create status bar
            try:
                self.platform.create_modern_statusbar()
                print("✅ Status bar created")
            except Exception as e:
                print(f"⚠️ Status bar creation failed: {e}")

            # Initialize help system
            try:
                from src.gui.help_system import init_help_system
                init_help_system(self.platform)
                print("✅ Help system initialized")
            except Exception as e:
                print(f"⚠️ Help system initialization failed: {e}")

            # Load configuration
            try:
                self.platform.load_saved_config()
                print("✅ Configuration loaded")
            except Exception as e:
                print(f"⚠️ Configuration loading failed: {e}")

            # Check if onboarding is needed
            try:
                from src.gui.onboarding_wizard import run_onboarding_wizard
                print("🔍 Checking if onboarding wizard is needed...")

                # Run onboarding wizard if needed
                wizard = run_onboarding_wizard(self.platform)
                if wizard:
                    print("✅ Onboarding wizard completed")
                    # Configuration will be handled by the wizard signal
                else:
                    print("ℹ️ Onboarding not needed or skipped")

            except Exception as e:
                print(f"⚠️ Onboarding wizard failed: {e}")
                # Continue without onboarding

            self.status_label.setText("✅ All components loaded successfully")
            print("=" * 80)
            print("✅ LOADING COMPLETE - All components initialized")
            print("=" * 80)

        except Exception as e:
            self.status_label.setText(f"❌ Error loading components: {e}")
            print(f"❌ CRITICAL ERROR loading components: {e}")
            import traceback
            traceback.print_exc()
            
            # Log to session logger if available
            if hasattr(self.platform, 'session_logger') and self.platform.session_logger:
                self.platform.session_logger.log_component_load_failure(
                    'loading_screen',
                    str(e),
                    traceback.format_exc()
                )

    def finish_loading(self):
        """Finish loading and show main window"""
        try:
            print("=" * 80)
            print("🚀 FINISHING LOADING PROCESS")
            print("=" * 80)
            
            self.hide()

            # Initialize remaining components that need the UI
            try:
                # Auto-load default BTC/USD data after UI is fully initialized
                print("⏱️  Scheduling auto-load of BTC data in 2 seconds...")
                QTimer.singleShot(2000, self.platform.auto_load_default_data)
            except Exception as e:
                print(f"⚠️ Warning: Could not auto-load default data: {e}")
                pass

            # Show the main platform window
            print("📺 Showing main platform window...")
            self.platform.show()
            
            print("✅ Main window displayed successfully")
            print("=" * 80)

            # Close loading screen
            self.close()
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in finish_loading: {e}")
            import traceback
            traceback.print_exc()
            
            # Log error
            if hasattr(self.platform, 'session_logger') and self.platform.session_logger:
                self.platform.session_logger.log_error(
                    error_type='loading_screen_finish_error',
                    error_message=str(e),
                    context={'stage': 'finish_loading'},
                    stack_trace=traceback.format_exc()
                )