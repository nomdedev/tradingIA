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
    """Compact card showing loaded dataset information"""
    
    def __init__(self, timeframe, bars, date_range):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            DataCard {
                background-color: #2d2d2d;
                border: 2px solid #4ec9b0;
                border-radius: 6px;
                padding: 6px 10px;
            }
        """)
        self.setMaximumWidth(140)  # Compact width
        self.setMaximumHeight(65)  # Compact height
        
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Timeframe
        tf_label = QLabel(f"⏱️ {timeframe}")
        tf_label.setStyleSheet("color: #4ec9b0; font-size: 11px; font-weight: 600;")
        
        # Bars count
        bars_label = QLabel(f"{bars:,} bars")
        bars_label.setStyleSheet("color: #ccc; font-size: 13px; font-weight: 700;")
        
        # Date range (abbreviated)
        date_parts = date_range.split(' to ')
        if len(date_parts) == 2:
            short_range = f"{date_parts[0][-5:]} - {date_parts[1][-5:]}"
        else:
            short_range = date_range[:15] + "..."
        range_label = QLabel(short_range)
        range_label.setStyleSheet("color: #888; font-size: 10px;")
        
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

        # Main content splitter - Make preview more prominent
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Configuration
        config_widget = self.create_configuration_panel()
        splitter.addWidget(config_widget)
        
        # Right side: Preview and loaded data (more prominent)
        preview_widget = self.create_preview_panel()
        splitter.addWidget(preview_widget)
        
        # Adjust splitter ratios to make preview DOMINANT
        splitter.setStretchFactor(0, 3)  # Configuration panel: 30%
        splitter.setStretchFactor(1, 7)  # Preview panel: 70%
        splitter.setSizes([300, 900])  # Set initial pixel sizes
        
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
            font-size: 14px;
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
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        # Data Source Selection
        source_group = QGroupBox("📥 Data Source")
        source_group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: 600;
                color: #fff;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        source_layout = QVBoxLayout()
        source_layout.setSpacing(6)
        source_layout.setContentsMargins(8, 6, 8, 8)
        
        # Source buttons
        source_btn_layout = QHBoxLayout()
        
        self.alpaca_btn = QPushButton("Alpaca")
        self.alpaca_btn.setCheckable(True)
        self.alpaca_btn.setChecked(True)
        self.alpaca_btn.setMinimumHeight(28)
        self.alpaca_btn.setMaximumHeight(28)
        self.alpaca_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 5px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:checked {
                background-color: #1177bb;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        
        binance_btn = QPushButton("Binance")
        binance_btn.setCheckable(True)
        binance_btn.setEnabled(False)
        binance_btn.setMinimumHeight(28)
        binance_btn.setMaximumHeight(28)
        binance_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #888;
                border: none;
                padding: 6px 10px;
                border-radius: 5px;
                font-size: 13px;
            }
        """)
        
        csv_btn = QPushButton("CSV Upload")
        csv_btn.setCheckable(True)
        csv_btn.setEnabled(False)
        csv_btn.setMinimumHeight(28)
        csv_btn.setMaximumHeight(28)
        csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #888;
                border: none;
                padding: 6px 10px;
                border-radius: 5px;
                font-size: 13px;
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
                font-size: 13px;
                font-weight: 600;
                color: #fff;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(10, 8, 10, 8)

        # Symbol
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTC/USD", "ETH/USD", "SOL/USD", "AAPL", "SPY", "QQQ"])
        self.symbol_combo.setCurrentText("BTC/USD")  # Set BTC/USD as default

        # Timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["5Min", "15Min", "1Hour", "Daily"])

        # Date range
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDate(QDate(2023, 1, 1))
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setMaximumHeight(30)

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setMaximumHeight(30)

        # Multi-timeframe
        self.multi_tf_check = QCheckBox("Enable Multi-Timeframe Analysis")
        self.multi_tf_check.setChecked(True)
        self.multi_tf_check.setStyleSheet("font-size: 13px; color: #ccc;")

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

        # Status banner below load button
        self.status_banner = self.create_status_banner()
        layout.addWidget(self.status_banner)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                height: 20px;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #4ec9b0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #4ec9b0; font-size: 13px; font-weight: 600;")
        layout.addWidget(self.progress_label)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_preview_panel(self):
        """Create data preview panel - MAXIMIZED for visibility"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Loaded Datasets - COMPACT horizontal bar at top
        self.datasets_container = QWidget()
        self.datasets_container.setMaximumHeight(90)  # Compact height
        datasets_container_layout = QVBoxLayout()
        datasets_container_layout.setContentsMargins(6, 4, 6, 4)
        datasets_container_layout.setSpacing(4)
        
        # Small header
        datasets_header = QLabel("📚 Loaded:")
        datasets_header.setStyleSheet("""
            color: #888;
            font-size: 11px;
            font-weight: 600;
        """)
        datasets_container_layout.addWidget(datasets_header)
        
        # Cards in horizontal layout
        self.datasets_layout = QHBoxLayout()
        self.datasets_layout.setSpacing(6)
        self.datasets_layout.setContentsMargins(0, 0, 0, 0)
        datasets_container_layout.addLayout(self.datasets_layout)
        
        # Placeholder
        self.no_data_label = QLabel("No data loaded yet")
        self.no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_data_label.setStyleSheet("""
            color: #555;
            font-size: 12px;
            padding: 8px;
        """)
        self.datasets_layout.addWidget(self.no_data_label)
        self.datasets_layout.addStretch()
        
        self.datasets_container.setLayout(datasets_container_layout)
        self.datasets_container.setStyleSheet("""
            QWidget {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.datasets_container)

        # Chart Preview - MAXIMIZED
        preview_group = QGroupBox("📈 Data Preview")
        preview_group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: 600;
                color: #fff;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(8, 6, 8, 8)
        preview_layout.setSpacing(8)
        
        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(500)  # Much taller
        # No maximum height - let it expand!
        preview_layout.addWidget(self.chart_view, stretch=1)  # Add stretch to expand
        
        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(6)
        
        self.stats_labels = {}
        stats = ['Mean', 'Std Dev', 'Min', 'Max', 'Volatility']
        for stat in stats:
            label = QLabel(f"{stat}: --")
            label.setStyleSheet("""
                color: #888;
                font-size: 12px;
                padding: 3px 6px;
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
        symbol = self.symbol_combo.currentText()  # Keep original format with '/'
        timeframe = self.timeframe_combo.currentText()
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        multi_tf = self.multi_tf_check.isChecked()
        
        # Log button click
        if hasattr(self.parent_platform, 'session_logger') and self.parent_platform.session_logger:
            self.parent_platform.session_logger.log_ui_event('load_data_button_clicked', {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'multi_timeframe': multi_tf
            })

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
        
        # Log successful data load
        if hasattr(self.parent_platform, 'session_logger') and self.parent_platform.session_logger:
            self.parent_platform.session_logger.log_data_loading_event('manual_load_success', {
                'timeframes': list(data_dict.keys()),
                'total_bars': total_bars,
                'timeframes_count': len(data_dict)
            })
        
        self.parent_platform.update_status(
            f"Loaded {len(data_dict)} timeframes ({total_bars:,} total bars)",
            "success"
        )

    def update_loaded_datasets(self, data_dict):
        """Update loaded datasets display - compact version"""
        # Clear existing cards
        while self.datasets_layout.count():
            item = self.datasets_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Hide placeholder
        self.no_data_label.setVisible(False)
        
        # Add compact cards for each timeframe
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
            height=550,  # Much taller chart
            xaxis_title="Date",
            yaxis_title="Price",
            font=dict(size=11),  # Slightly larger font
            margin=dict(l=60, r=30, t=50, b=50),  # Better margins
            paper_bgcolor='#252525',
            plot_bgcolor='#2d2d2d',
            xaxis=dict(rangeslider=dict(visible=False))  # Remove rangeslider for more space
        )
        
        # Display chart
        html = fig.to_html(include_plotlyjs='cdn')
        self.chart_view.setHtml(html)
        
        # Update stats with comprehensive error handling
        try:
            # Handle different column name cases
            close_col = None
            for col_name in ['close', 'Close', 'CLOSE']:
                if col_name in df.columns:
                    close_col = col_name
                    break
            
            if close_col is None:
                self.logger.warning(f"No close column found. Available columns: {list(df.columns)}")
                self.stats_labels['Mean'].setText("Mean: N/A")
                self.stats_labels['Std Dev'].setText("Std Dev: N/A")
                self.stats_labels['Min'].setText("Min: N/A")
                self.stats_labels['Max'].setText("Max: N/A")
                self.stats_labels['Volatility'].setText("Volatility: N/A")
                return
                
            self.stats_labels['Mean'].setText(f"Mean: ${df[close_col].mean():.2f}")
            self.stats_labels['Std Dev'].setText(f"Std Dev: ${df[close_col].std():.2f}")
            self.stats_labels['Min'].setText(f"Min: ${df[close_col].min():.2f}")
            self.stats_labels['Max'].setText(f"Max: ${df[close_col].max():.2f}")
            
            returns = df[close_col].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100
            self.stats_labels['Volatility'].setText(f"Volatility: {volatility:.1f}%")
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def on_error_occurred(self, error_message):
        """Handle data loading error"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.load_data_btn.setEnabled(True)
        
        # Log error
        if hasattr(self.parent_platform, 'session_logger') and self.parent_platform.session_logger:
            self.parent_platform.session_logger.log_error(
                error_type='data_loading_error',
                error_message=error_message,
                context={'tab': 'Tab1_DataManagement'}
            )
        
        self.parent_platform.update_status(f"Error: {error_message}", "error")
        self.logger.error(f"Data loading error: {error_message}")

    def on_tab_activated(self):
        """Called when tab is activated"""
        pass
