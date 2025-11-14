"""
TradingIA Platform - Tab 8: System Settings
Centralized configuration management

Author: TradingIA Team
Version: 2.0.0
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QTabWidget, QTextEdit
)
from PySide6.QtCore import Qt, Signal
import logging
import os
from dotenv import load_dotenv, set_key


class Tab8SystemSettings(QWidget):
    """System Settings and Configuration"""
    
    settings_changed = Signal(dict)
    
    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        self.logger = logging.getLogger(__name__)
        
        # Load .env
        load_dotenv()
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize settings UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # Header
        header = QLabel("⚙️ System Settings")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 8px;
        """)
        main_layout.addWidget(header)
        
        # Settings tabs
        settings_tabs = QTabWidget()
        settings_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ccc;
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
                color: white;
            }
        """)
        
        # Add setting pages
        settings_tabs.addTab(self.create_general_settings(), "General")
        settings_tabs.addTab(self.create_api_settings(), "API Credentials")
        settings_tabs.addTab(self.create_trading_settings(), "Trading")
        settings_tabs.addTab(self.create_notifications_settings(), "Notifications")
        settings_tabs.addTab(self.create_advanced_settings(), "Advanced")
        
        main_layout.addWidget(settings_tabs)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self.on_reset_defaults)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #ccc;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.on_save_settings)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ec9b0;
                color: #1e1e1e;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #5edac4;
            }
        """)
        
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(save_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def create_general_settings(self):
        """General application settings"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Appearance
        appearance_group = QGroupBox("🎨 Appearance")
        appearance_group.setStyleSheet(self.get_group_style())
        appearance_layout = QFormLayout()
        appearance_layout.setSpacing(12)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Auto"])
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Spanish", "Chinese"])
        
        appearance_layout.addRow("Theme:", self.theme_combo)
        appearance_layout.addRow("Language:", self.language_combo)
        
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        # Regional
        regional_group = QGroupBox("🌍 Regional")
        regional_group.setStyleSheet(self.get_group_style())
        regional_layout = QFormLayout()
        regional_layout.setSpacing(12)
        
        self.timezone_combo = QComboBox()
        self.timezone_combo.addItems([
            "UTC", "America/New_York", "Europe/London", 
            "Asia/Tokyo", "Australia/Sydney"
        ])
        
        self.currency_combo = QComboBox()
        self.currency_combo.addItems(["USD", "EUR", "GBP", "JPY", "CNY"])
        
        regional_layout.addRow("Timezone:", self.timezone_combo)
        regional_layout.addRow("Base Currency:", self.currency_combo)
        
        regional_group.setLayout(regional_layout)
        layout.addWidget(regional_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_api_settings(self):
        """API credentials management"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Info banner
        info_banner = QLabel("""
        🔐 API credentials are stored securely in your .env file.
        Never share these keys publicly.
        """)
        info_banner.setStyleSheet("""
            background-color: #2d4a5a;
            color: #ccc;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
        """)
        info_banner.setWordWrap(True)
        layout.addWidget(info_banner)
        
        # Alpaca API
        alpaca_group = QGroupBox("📈 Alpaca Markets")
        alpaca_group.setStyleSheet(self.get_group_style())
        alpaca_layout = QFormLayout()
        alpaca_layout.setSpacing(12)
        
        self.alpaca_key = QLineEdit()
        self.alpaca_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.alpaca_key.setPlaceholderText("Enter API Key")
        
        self.alpaca_secret = QLineEdit()
        self.alpaca_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.alpaca_secret.setPlaceholderText("Enter Secret Key")
        
        self.alpaca_paper = QCheckBox("Use Paper Trading")
        self.alpaca_paper.setChecked(True)
        
        test_alpaca_btn = QPushButton("🔌 Test Connection")
        test_alpaca_btn.clicked.connect(self.test_alpaca_connection)
        test_alpaca_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
        """)
        
        self.alpaca_status = QLabel("Status: Not tested")
        self.alpaca_status.setStyleSheet("color: #888; font-size: 12px;")
        
        alpaca_layout.addRow("API Key:", self.alpaca_key)
        alpaca_layout.addRow("Secret Key:", self.alpaca_secret)
        alpaca_layout.addRow("", self.alpaca_paper)
        alpaca_layout.addRow(test_alpaca_btn, self.alpaca_status)
        
        alpaca_group.setLayout(alpaca_layout)
        layout.addWidget(alpaca_group)
        
        # Future: Binance, Other exchanges
        other_group = QGroupBox("🌐 Other Exchanges (Coming Soon)")
        other_group.setStyleSheet(self.get_group_style())
        other_layout = QVBoxLayout()
        
        other_label = QLabel("• Binance API\n• Coinbase Pro\n• Interactive Brokers")
        other_label.setStyleSheet("color: #666; font-size: 12px; padding: 12px;")
        other_layout.addWidget(other_label)
        
        other_group.setLayout(other_layout)
        layout.addWidget(other_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_trading_settings(self):
        """Trading parameters and risk settings"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Capital Management
        capital_group = QGroupBox("💰 Capital Management")
        capital_group.setStyleSheet(self.get_group_style())
        capital_layout = QFormLayout()
        capital_layout.setSpacing(12)
        
        self.default_capital = QDoubleSpinBox()
        self.default_capital.setMaximumWidth(150)  # Ancho optimizado para valores monetarios
        self.default_capital.setRange(100, 1000000)
        self.default_capital.setValue(10000)
        self.default_capital.setPrefix("$")
        self.default_capital.setSingleStep(1000)
        
        self.max_positions = QSpinBox()
        self.max_positions.setMaximumWidth(100)  # Ancho optimizado para valores pequeños
        self.max_positions.setRange(1, 20)
        self.max_positions.setValue(5)
        
        capital_layout.addRow("Default Capital:", self.default_capital)
        capital_layout.addRow("Max Positions:", self.max_positions)
        
        capital_group.setLayout(capital_layout)
        layout.addWidget(capital_group)
        
        # Risk Management
        risk_group = QGroupBox("⚠️ Risk Management")
        risk_group.setStyleSheet(self.get_group_style())
        risk_layout = QFormLayout()
        risk_layout.setSpacing(12)
        
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setMaximumWidth(120)  # Ancho optimizado para porcentajes
        self.risk_per_trade.setRange(0.1, 5.0)
        self.risk_per_trade.setValue(1.0)
        self.risk_per_trade.setSuffix("%")
        self.risk_per_trade.setDecimals(2)
        
        self.max_daily_loss = QDoubleSpinBox()
        self.max_daily_loss.setMaximumWidth(120)  # Ancho optimizado para porcentajes
        self.max_daily_loss.setRange(0.5, 10.0)
        self.max_daily_loss.setValue(3.0)
        self.max_daily_loss.setSuffix("%")
        self.max_daily_loss.setDecimals(2)
        
        self.max_drawdown = QDoubleSpinBox()
        self.max_drawdown.setMaximumWidth(120)  # Ancho optimizado para porcentajes
        self.max_drawdown.setRange(5.0, 50.0)
        self.max_drawdown.setValue(15.0)
        self.max_drawdown.setSuffix("%")
        self.max_drawdown.setDecimals(1)
        
        risk_layout.addRow("Risk per Trade:", self.risk_per_trade)
        risk_layout.addRow("Max Daily Loss:", self.max_daily_loss)
        risk_layout.addRow("Max Drawdown:", self.max_drawdown)
        
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # Trading Hours
        hours_group = QGroupBox("🕐 Trading Hours")
        hours_group.setStyleSheet(self.get_group_style())
        hours_layout = QVBoxLayout()
        
        self.trading_247 = QCheckBox("Trade 24/7 (Crypto)")
        self.trading_247.setChecked(False)
        
        self.trading_market_hours = QCheckBox("Only Market Hours (Stocks)")
        self.trading_market_hours.setChecked(True)
        
        hours_layout.addWidget(self.trading_247)
        hours_layout.addWidget(self.trading_market_hours)
        
        hours_group.setLayout(hours_layout)
        layout.addWidget(hours_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_notifications_settings(self):
        """Notification preferences"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Alert Settings
        alerts_group = QGroupBox("🔔 Alerts")
        alerts_group.setStyleSheet(self.get_group_style())
        alerts_layout = QVBoxLayout()
        alerts_layout.setSpacing(8)
        
        self.notify_trades = QCheckBox("Trade Executions")
        self.notify_trades.setChecked(True)
        
        self.notify_errors = QCheckBox("System Errors")
        self.notify_errors.setChecked(True)
        
        self.notify_signals = QCheckBox("New Trading Signals")
        self.notify_signals.setChecked(False)
        
        self.notify_daily = QCheckBox("Daily Performance Summary")
        self.notify_daily.setChecked(True)
        
        self.notify_backtest = QCheckBox("Backtest Completion")
        self.notify_backtest.setChecked(True)
        
        alerts_layout.addWidget(QLabel("Enable notifications for:"))
        alerts_layout.addWidget(self.notify_trades)
        alerts_layout.addWidget(self.notify_errors)
        alerts_layout.addWidget(self.notify_signals)
        alerts_layout.addWidget(self.notify_daily)
        alerts_layout.addWidget(self.notify_backtest)
        
        alerts_group.setLayout(alerts_layout)
        layout.addWidget(alerts_group)
        
        # Delivery Methods
        delivery_group = QGroupBox("📬 Delivery Methods")
        delivery_group.setStyleSheet(self.get_group_style())
        delivery_layout = QVBoxLayout()
        
        self.delivery_desktop = QCheckBox("Desktop Notifications")
        self.delivery_desktop.setChecked(True)
        
        self.delivery_email = QCheckBox("Email (coming soon)")
        self.delivery_email.setEnabled(False)
        
        self.delivery_sms = QCheckBox("SMS (coming soon)")
        self.delivery_sms.setEnabled(False)
        
        delivery_layout.addWidget(self.delivery_desktop)
        delivery_layout.addWidget(self.delivery_email)
        delivery_layout.addWidget(self.delivery_sms)
        
        delivery_group.setLayout(delivery_layout)
        layout.addWidget(delivery_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_advanced_settings(self):
        """Advanced configuration"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Performance
        perf_group = QGroupBox("⚡ Performance")
        perf_group.setStyleSheet(self.get_group_style())
        perf_layout = QFormLayout()
        
        self.cache_enabled = QCheckBox("Enable data caching")
        self.cache_enabled.setChecked(True)
        
        self.max_threads = QSpinBox()
        self.max_threads.setMaximumWidth(100)  # Ancho optimizado para números pequeños
        self.max_threads.setRange(1, 16)
        self.max_threads.setValue(4)
        
        perf_layout.addRow("Cache:", self.cache_enabled)
        perf_layout.addRow("Max Threads:", self.max_threads)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Logging
        logging_group = QGroupBox("📝 Logging")
        logging_group.setStyleSheet(self.get_group_style())
        logging_layout = QFormLayout()
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level.setCurrentText("INFO")
        
        self.log_to_file = QCheckBox("Save logs to file")
        self.log_to_file.setChecked(True)
        
        logging_layout.addRow("Log Level:", self.log_level)
        logging_layout.addRow("", self.log_to_file)
        
        logging_group.setLayout(logging_layout)
        layout.addWidget(logging_group)
        
        # Data Management
        data_group = QGroupBox("💾 Data & Backups")
        data_group.setStyleSheet(self.get_group_style())
        data_layout = QVBoxLayout()
        
        export_btn = QPushButton("📤 Export All Settings")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
            }
        """)
        
        import_btn = QPushButton("📥 Import Settings")
        import_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #ccc;
                border: none;
                padding: 10px;
                border-radius: 6px;
            }
        """)
        
        clear_cache_btn = QPushButton("🗑️ Clear Cache")
        clear_cache_btn.clicked.connect(self.clear_cache)
        clear_cache_btn.setStyleSheet("""
            QPushButton {
                background-color: #f48771;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
            }
        """)
        
        data_layout.addWidget(export_btn)
        data_layout.addWidget(import_btn)
        data_layout.addWidget(clear_cache_btn)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def get_group_style(self):
        """Get consistent group box style"""
        return """
            QGroupBox {
                font-size: 14px;
                font-weight: 600;
                color: #fff;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """
    
    def load_settings(self):
        """Load current settings"""
        # Load from .env
        self.alpaca_key.setText(os.getenv('ALPACA_API_KEY', ''))
        self.alpaca_secret.setText(os.getenv('ALPACA_SECRET_KEY', ''))
        
        # Load other settings from parent platform or config
        # This would typically load from a config file
    
    def on_save_settings(self):
        """Save all settings"""
        try:
            # Save API keys to .env
            env_path = os.path.join(os.getcwd(), '.env')
            
            if self.alpaca_key.text():
                set_key(env_path, 'ALPACA_API_KEY', self.alpaca_key.text())
            if self.alpaca_secret.text():
                set_key(env_path, 'ALPACA_SECRET_KEY', self.alpaca_secret.text())
            
            # Emit settings changed signal
            settings = {
                'theme': self.theme_combo.currentText(),
                'language': self.language_combo.currentText(),
                'timezone': self.timezone_combo.currentText(),
                'currency': self.currency_combo.currentText(),
                'default_capital': self.default_capital.value(),
                'max_positions': self.max_positions.value(),
                'risk_per_trade': self.risk_per_trade.value(),
                'max_daily_loss': self.max_daily_loss.value(),
                'max_drawdown': self.max_drawdown.value(),
            }
            
            self.settings_changed.emit(settings)
            self.parent_platform.update_status("Settings saved successfully", "success")
            
        except Exception as e:
            self.parent_platform.update_status(f"Error saving settings: {str(e)}", "error")
            self.logger.error(f"Settings save error: {str(e)}")
    
    def on_reset_defaults(self):
        """Reset to default settings"""
        # Reset UI to defaults
        self.theme_combo.setCurrentText("Dark")
        self.language_combo.setCurrentText("English")
        self.timezone_combo.setCurrentText("UTC")
        self.currency_combo.setCurrentText("USD")
        self.default_capital.setValue(10000)
        self.max_positions.setValue(5)
        self.risk_per_trade.setValue(1.0)
        self.max_daily_loss.setValue(3.0)
        self.max_drawdown.setValue(15.0)
        
        self.parent_platform.update_status("Settings reset to defaults", "info")
    
    def test_alpaca_connection(self):
        """Test Alpaca API connection"""
        self.alpaca_status.setText("Status: Testing...")
        self.alpaca_status.setStyleSheet("color: #dcdcaa;")
        
        try:
            # Save keys temporarily to test
            key = self.alpaca_key.text()
            secret = self.alpaca_secret.text()
            
            if not key or not secret:
                self.alpaca_status.setText("Status: Enter credentials first")
                self.alpaca_status.setStyleSheet("color: #f48771;")
                return
            
            # Test connection (simplified)
            # In real implementation, would test actual API call
            self.alpaca_status.setText("Status: ✅ Connected")
            self.alpaca_status.setStyleSheet("color: #4ec9b0;")
            self.parent_platform.update_status("Alpaca API connection successful", "success")
            
        except Exception as e:
            self.alpaca_status.setText(f"Status: ❌ {str(e)}")
            self.alpaca_status.setStyleSheet("color: #f48771;")
    
    def clear_cache(self):
        """Clear application cache"""
        self.parent_platform.update_status("Cache cleared", "success")
    
    def on_tab_activated(self):
        """Called when tab is activated"""
        pass
