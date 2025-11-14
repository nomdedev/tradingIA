"""
TradingIA Platform - Tab 1: Data Manager (Improved)
Modern data management interface with visual preview

Author: TradingIA Team
Version: 2.0.0
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QDateEdit, QCheckBox, QProgressBar, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QDate
from PySide6.QtWebEngineWidgets import QWebEngineView
import logging
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime


class DataLoadThread(QThread):
    """Background thread for data loading"""
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
            self.progress_updated.emit(10, "Connecting to data source...")

            # Load data from Alpaca (uses .env credentials automatically)
            df = self.data_manager.load_alpaca_data(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=self.timeframe
            )

            if isinstance(df, dict) and 'error' in df:
                self.error_occurred.emit(df['error'])
                return

            self.progress_updated.emit(50, f"Loaded {len(df):,} bars...")

            if self.multi_tf:
                self.progress_updated.emit(75, "Resampling multi-timeframe data...")
                df_multi = self.data_manager.resample_multi_tf(df)
                if isinstance(df_multi, dict) and 'error' in df_multi:
                    self.error_occurred.emit(df_multi['error'])
                    return
            else:
                df_multi = {self.timeframe.lower(): df}

            self.progress_updated.emit(100, "Data loading complete!")
            self.data_loaded.emit(df_multi)

        except Exception as e:
            self.error_occurred.emit(f"Error loading data: {str(e)}")


class DataCard(QFrame):
    """Card showing loaded dataset information"""
    
    def __init__(self, timeframe, bars, date_range):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            DataCard {
                background-color: #2d2d2d;
                border: 2px solid #4ec9b0;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(6)
        
        # Timeframe
        tf_label = QLabel(f"⏱️ {timeframe}")
        tf_label.setStyleSheet("color: #4ec9b0; font-size: 14px; font-weight: 600;")
        
        # Bars count
        bars_label = QLabel(f"{bars:,} bars")
        bars_label.setStyleSheet("color: #ccc; font-size: 18px; font-weight: 700;")
        
        # Date range
        range_label = QLabel(date_range)
        range_label.setStyleSheet("color: #888; font-size: 11px;")
        
        layout.addWidget(tf_label)
        layout.addWidget(bars_label)
        layout.addWidget(range_label)
        
        self.setLayout(layout)


class Tab1DataManagement(QWidget):
    """Improved Data Management Tab"""
    data_loaded_signal = Signal(dict)

    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        self.data_manager = parent_platform.data_manager
        self.logger = logging.getLogger(__name__)

        self.load_thread = None
        self.loaded_datasets = []
        
        self.init_ui()

    def init_ui(self):
        """Initialize modern UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QLabel("📊 Data Manager")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 8px;
        """)
        main_layout.addWidget(header)

        # Connection Status Banner
        self.status_banner = self.create_status_banner()
        main_layout.addWidget(self.status_banner)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Configuration
        config_widget = self.create_configuration_panel()
        splitter.addWidget(config_widget)
        
        # Right side: Preview and loaded data
        preview_widget = self.create_preview_panel()
        splitter.addWidget(preview_widget)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)

    def create_status_banner(self):
        """Create connection status banner"""
        banner = QFrame()
        banner.setStyleSheet("""
            QFrame {
                background-color: #2d5a47;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        icon = QLabel("✓")
        icon.setStyleSheet("font-size: 12px; color: #4ec9b0;")

        self.status_text = QLabel("Alpaca Connected")
        self.status_text.setStyleSheet("""
            color: #4ec9b0;
            font-weight: 500;
            font-size: 11px;
        """)

        layout.addWidget(icon)
        layout.addWidget(self.status_text)
        layout.addStretch()

        banner.setLayout(layout)
        return banner

    def create_configuration_panel(self):
        """Create data configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Data Source Selection
        source_group = QGroupBox("📥 Data Source")
        source_group.setStyleSheet("""
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
        """)
        
        source_layout = QVBoxLayout()
        source_layout.setSpacing(8)
        
        # Source buttons
        source_btn_layout = QHBoxLayout()
        
        self.alpaca_btn = QPushButton("Alpaca")
        self.alpaca_btn.setCheckable(True)
        self.alpaca_btn.setChecked(True)
        self.alpaca_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:checked {
                background-color: #4ec9b0;
            }
        """)
        
        binance_btn = QPushButton("Binance")
        binance_btn.setCheckable(True)
        binance_btn.setEnabled(False)
        binance_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #888;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
        """)
        
        csv_btn = QPushButton("CSV Upload")
        csv_btn.setCheckable(True)
        csv_btn.setEnabled(False)
        csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #888;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
        """)
        
        source_btn_layout.addWidget(self.alpaca_btn)
        source_btn_layout.addWidget(binance_btn)
        source_btn_layout.addWidget(csv_btn)
        
        source_layout.addLayout(source_btn_layout)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Data Parameters
        params_group = QGroupBox("⚙️ Parameters")
        params_group.setStyleSheet("""
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
        """)
        
        params_layout = QFormLayout()
        params_layout.setSpacing(12)

        # Symbol
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTC/USD", "ETH/USD", "SOL/USD", "AAPL", "SPY", "QQQ"])
        self.symbol_combo.setCurrentText("BTC/USD")  # Set BTC/USD as default
        self.symbol_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 13px;
            }
        """)

        # Timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["5Min", "15Min", "1Hour", "Daily"])
        self.timeframe_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 13px;
            }
        """)

        # Date range
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDate(QDate(2023, 1, 1))
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setStyleSheet("padding: 8px; font-size: 13px;")

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setStyleSheet("padding: 8px; font-size: 13px;")

        # Multi-timeframe
        self.multi_tf_check = QCheckBox("Enable Multi-Timeframe Analysis")
        self.multi_tf_check.setChecked(True)
        self.multi_tf_check.setStyleSheet("font-size: 12px; color: #ccc;")

        params_layout.addRow("Symbol:", self.symbol_combo)
        params_layout.addRow("Timeframe:", self.timeframe_combo)
        params_layout.addRow("Start Date:", self.start_date_edit)
        params_layout.addRow("End Date:", self.end_date_edit)
        params_layout.addRow("", self.multi_tf_check)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Load button
        self.load_data_btn = QPushButton("📥 Load Data")
        self.load_data_btn.setFixedHeight(50)
        self.load_data_btn.clicked.connect(self.on_load_data_clicked)
        self.load_data_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0e639c, stop:1 #0a4d7a);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #1177bb;
            }
            QPushButton:pressed {
                background: #0d5689;
            }
        """)
        layout.addWidget(self.load_data_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 6px;
                text-align: center;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #4ec9b0;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #4ec9b0; font-size: 12px; font-weight: 600;")
        layout.addWidget(self.progress_label)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_preview_panel(self):
        """Create data preview panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Loaded Datasets Section
        loaded_group = QGroupBox("📚 Loaded Datasets")
        loaded_group.setStyleSheet("""
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
        """)
        
        loaded_layout = QVBoxLayout()
        
        # Cards container
        self.datasets_container = QWidget()
        self.datasets_layout = QHBoxLayout()
        self.datasets_layout.setSpacing(12)
        self.datasets_container.setLayout(self.datasets_layout)
        
        # Placeholder
        self.no_data_label = QLabel("No data loaded yet. Load data to see preview.")
        self.no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_data_label.setStyleSheet("""
            color: #666;
            font-size: 14px;
            padding: 40px;
        """)
        
        loaded_layout.addWidget(self.no_data_label)
        loaded_layout.addWidget(self.datasets_container)
        self.datasets_container.setVisible(False)
        
        loaded_group.setLayout(loaded_layout)
        layout.addWidget(loaded_group)

        # Chart Preview
        preview_group = QGroupBox("📈 Data Preview")
        preview_group.setStyleSheet("""
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
        """)
        
        preview_layout = QVBoxLayout()
        
        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(400)
        preview_layout.addWidget(self.chart_view)
        
        # Stats row
        stats_layout = QHBoxLayout()
        
        self.stats_labels = {}
        stats = ['Mean', 'Std Dev', 'Min', 'Max', 'Volatility']
        for stat in stats:
            label = QLabel(f"{stat}: --")
            label.setStyleSheet("""
                color: #888;
                font-size: 11px;
                padding: 4px 8px;
            """)
            stats_layout.addWidget(label)
            self.stats_labels[stat] = label
        
        preview_layout.addLayout(stats_layout)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        widget.setLayout(layout)
        return widget

    def on_load_data_clicked(self):
        """Handle load data button click"""
        if self.load_thread and self.load_thread.isRunning():
            self.parent_platform.update_status("Data loading already in progress", "warning")
            return

        # Get parameters
        symbol = self.symbol_combo.currentText().replace('/', '-')
        timeframe = self.timeframe_combo.currentText()
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        multi_tf = self.multi_tf_check.isChecked()

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Initializing...")
        self.load_data_btn.setEnabled(False)

        # Start loading thread
        self.load_thread = DataLoadThread(
            self.data_manager, symbol, timeframe,
            start_date, end_date, multi_tf
        )
        self.load_thread.progress_updated.connect(self.on_progress_updated)
        self.load_thread.data_loaded.connect(self.on_data_loaded)
        self.load_thread.error_occurred.connect(self.on_error_occurred)
        self.load_thread.start()

        self.parent_platform.update_status(f"Loading {symbol} data...", "processing")

    def on_progress_updated(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def on_data_loaded(self, data_dict):
        """Handle successful data load"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.load_data_btn.setEnabled(True)

        # Store data in parent platform
        self.parent_platform.data_dict = data_dict
        self.data_loaded_signal.emit(data_dict)

        # Update UI
        self.update_loaded_datasets(data_dict)
        self.update_chart_preview(data_dict)
        
        total_bars = sum(len(df) for df in data_dict.values())
        self.parent_platform.update_status(
            f"Loaded {len(data_dict)} timeframes ({total_bars:,} total bars)",
            "success"
        )

    def update_loaded_datasets(self, data_dict):
        """Update loaded datasets display"""
        # Clear existing cards
        while self.datasets_layout.count():
            item = self.datasets_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Hide placeholder
        self.no_data_label.setVisible(False)
        self.datasets_container.setVisible(True)
        
        # Add cards for each timeframe
        for tf, df in data_dict.items():
            date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            card = DataCard(tf.upper(), len(df), date_range)
            self.datasets_layout.addWidget(card)
        
        self.datasets_layout.addStretch()

    def update_chart_preview(self, data_dict):
        """Update chart preview with candlestick"""
        # Use first timeframe for preview
        tf_key = list(data_dict.keys())[0]
        df = data_dict[tf_key]
        
        # Get last 100 bars for preview
        df_preview = df.tail(100)
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df_preview.index,
            open=df_preview['open'],
            high=df_preview['high'],
            low=df_preview['low'],
            close=df_preview['close'],
            name='OHLC'
        )])
        
        fig.update_layout(
            title=f"Last 100 bars - {tf_key.upper()}",
            template='plotly_dark',
            height=400,
            xaxis_title="Date",
            yaxis_title="Price",
            font=dict(size=10),
            margin=dict(l=50, r=20, t=40, b=40),
            paper_bgcolor='#252525',
            plot_bgcolor='#2d2d2d'
        )
        
        # Display chart
        html = fig.to_html(include_plotlyjs='cdn')
        self.chart_view.setHtml(html)
        
        # Update stats
        self.stats_labels['Mean'].setText(f"Mean: ${df['close'].mean():.2f}")
        self.stats_labels['Std Dev'].setText(f"Std Dev: ${df['close'].std():.2f}")
        self.stats_labels['Min'].setText(f"Min: ${df['close'].min():.2f}")
        self.stats_labels['Max'].setText(f"Max: ${df['close'].max():.2f}")
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100
        self.stats_labels['Volatility'].setText(f"Volatility: {volatility:.1f}%")

    def on_error_occurred(self, error_message):
        """Handle data loading error"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.load_data_btn.setEnabled(True)
        
        self.parent_platform.update_status(f"Error: {error_message}", "error")
        self.logger.error(f"Data loading error: {error_message}")

    def on_tab_activated(self):
        """Called when tab is activated"""
        pass
