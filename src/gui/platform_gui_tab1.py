from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QDateEdit, QCheckBox, QProgressBar, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QDate
import logging
import pandas as pd

class DataLoadThread(QThread):
    progress_updated = Signal(int, str)
    data_loaded = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, data_manager, symbol, timeframe, start_date, end_date, multi_tf):
        super().__init__()
        self.data_manager = data_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.multi_tf = multi_tf

    def run(self):
        try:
            self.progress_updated.emit(10, "Connecting to Alpaca API...")

            # Load data
            df = self.data_manager.load_alpaca_data(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe
            )

            if isinstance(df, dict) and 'error' in df:
                self.error_occurred.emit(df['error'])
                return

            self.progress_updated.emit(50, f"Loaded {len(df)} bars...")

            if self.multi_tf:
                self.progress_updated.emit(75, "Resampling multi-timeframe data...")
                df_multi = self.data_manager.resample_multi_tf(df)
                if isinstance(df_multi, dict) and 'error' in df_multi:
                    self.error_occurred.emit(df_multi['error'])
                    return
            else:
                df_multi = {'5min': df}

            self.progress_updated.emit(100, "Data loading complete!")
            self.data_loaded.emit(df_multi)

        except Exception as e:
            self.error_occurred.emit(f"Thread error: {str(e)}")

class Tab1DataManagement(QWidget):
    data_loaded_signal = Signal(dict)

    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        self.data_manager = parent_platform.data_manager
        self.logger = logging.getLogger(__name__)

        self.load_thread = None
        self.init_ui()
        self.load_config()

    def init_ui(self):
        layout = QVBoxLayout()

        # Alpaca API Configuration Group
        api_group = QGroupBox("Alpaca API Configuration")
        api_layout = QFormLayout()

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter Alpaca API Key")

        self.secret_key_edit = QLineEdit()
        self.secret_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.secret_key_edit.setPlaceholderText("Enter Alpaca Secret Key")

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.connect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px 16px; }")

        self.status_label = QLabel("Status: Not Connected")
        self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

        api_layout.addRow("API Key:", self.api_key_edit)
        api_layout.addRow("Secret Key:", self.secret_key_edit)
        api_layout.addRow(self.connect_btn, self.status_label)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Data Parameters Group
        data_group = QGroupBox("Data Parameters")
        data_layout = QFormLayout()

        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTC-USD", "ETH-USD", "SOL-USD", "SPY"])
        self.symbol_combo.setCurrentText("BTC-USD")

        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["5Min", "15Min", "1H", "Daily"])
        self.timeframe_combo.setCurrentText("5Min")

        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDate(QDate(2020, 1, 1))
        self.start_date_edit.setCalendarPopup(True)

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setCalendarPopup(True)

        self.multi_tf_check = QCheckBox("Multi-Timeframe (5min, 15min, 1h)")
        self.multi_tf_check.setChecked(True)

        data_layout.addRow("Symbol:", self.symbol_combo)
        data_layout.addRow("Timeframe:", self.timeframe_combo)
        data_layout.addRow("Start Date:", self.start_date_edit)
        data_layout.addRow("End Date:", self.end_date_edit)
        data_layout.addRow("", self.multi_tf_check)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Load Data Section
        load_layout = QHBoxLayout()

        self.load_data_btn = QPushButton("Load Data")
        self.load_data_btn.clicked.connect(self.on_load_data_clicked)
        self.load_data_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 12px 24px; font-size: 14px; }")
        self.load_data_btn.setEnabled(False)  # Disabled until connected

        load_layout.addStretch()
        load_layout.addWidget(self.load_data_btn)
        load_layout.addStretch()
        layout.addLayout(load_layout)

        # Progress Section
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        # Data Info Section
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()

        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.data_info_label.setStyleSheet("QLabel { font-family: monospace; padding: 10px; }")

        info_layout.addWidget(self.data_info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.setLayout(layout)

    def on_connect_clicked(self):
        api_key = self.api_key_edit.text().strip()
        secret_key = self.secret_key_edit.text().strip()

        if not api_key or not secret_key:
            self.show_error("Please enter both API key and secret key")
            return

        try:
            # Test connection by creating new DataManager instance
            test_manager = type(self.data_manager)(api_key, secret_key)
            # Try a small data fetch to test connection
            test_data = test_manager.load_alpaca_data('BTC-USD', '2023-01-01', '2023-01-02', '1H')
            if isinstance(test_data, dict) and 'error' in test_data:
                raise Exception(test_data['error'])

            # Update main data manager
            self.data_manager.api_key = api_key
            self.data_manager.secret_key = secret_key
            self.data_manager.api = test_manager.api

            self.status_label.setText("Status: Connected ✓")
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.load_data_btn.setEnabled(True)

            self.logger.info("Alpaca API connection successful")

        except Exception as e:
            self.show_error(f"Connection failed: {str(e)}")
            self.status_label.setText("Status: Connection Failed ✗")
            self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

    def on_load_data_clicked(self):
        # Validate inputs
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")

        if start_date >= end_date:
            self.show_error("Start date must be before end date")
            return

        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        multi_tf = self.multi_tf_check.isChecked()

        # Disable button during loading
        self.load_data_btn.setEnabled(False)
        self.load_data_btn.setText("Loading...")

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Initializing...")

        # Start loading thread
        self.load_thread = DataLoadThread(
            self.data_manager, symbol, timeframe, start_date, end_date, multi_tf
        )
        self.load_thread.progress_updated.connect(self.update_progress)
        self.load_thread.data_loaded.connect(self.on_data_loaded)
        self.load_thread.error_occurred.connect(self.on_load_error)
        self.load_thread.start()

    def update_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def on_data_loaded(self, data_dict):
        # Update parent platform data
        self.parent_platform.data_dict = data_dict

        # Update UI
        self.load_data_btn.setEnabled(True)
        self.load_data_btn.setText("Load Data")
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        # Update data info
        if '5min' in data_dict:
            df_5m = data_dict['5min']
            info_text = f"Symbol: {self.symbol_combo.currentText()} | Bars: {len(df_5m):,} | Range: {df_5m.index[0].strftime('%Y-%m-%d')} to {df_5m.index[-1].strftime('%Y-%m-%d')} | Last Update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
            self.data_info_label.setText(info_text)

        # Emit signal
        self.data_loaded_signal.emit(data_dict)

        # Save config
        self.save_config()

    def on_load_error(self, error_msg):
        self.show_error(error_msg)
        self.load_data_btn.setEnabled(True)
        self.load_data_btn.setText("Load Data")
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

    def show_error(self, msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", msg)

    def save_config(self):
        config = {
            'api_key': self.api_key_edit.text(),
            'secret_key': self.secret_key_edit.text(),
            'symbol': self.symbol_combo.currentText(),
            'timeframe': self.timeframe_combo.currentText(),
            'start_date': self.start_date_edit.date().toString("yyyy-MM-dd"),
            'end_date': self.end_date_edit.date().toString("yyyy-MM-dd"),
            'multi_tf': self.multi_tf_check.isChecked()
        }

        try:
            import json
            with open('config/gui_config.json', 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def load_config(self):
        try:
            import json
            with open('config/gui_config.json', 'r') as f:
                config = json.load(f)

            self.api_key_edit.setText(config.get('api_key', ''))
            self.secret_key_edit.setText(config.get('secret_key', ''))
            self.symbol_combo.setCurrentText(config.get('symbol', 'BTC-USD'))
            self.timeframe_combo.setCurrentText(config.get('timeframe', '5Min'))

            start_date = QDate.fromString(config.get('start_date', '2020-01-01'), "yyyy-MM-dd")
            self.start_date_edit.setDate(start_date)

            end_date = QDate.fromString(config.get('end_date', '2025-11-13'), "yyyy-MM-dd")
            self.end_date_edit.setDate(end_date)

            self.multi_tf_check.setChecked(config.get('multi_tf', True))

        except FileNotFoundError:
            pass  # Use defaults
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")