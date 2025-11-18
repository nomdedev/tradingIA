from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox,
    QGroupBox, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QColor
import os
import sys
import subprocess
from datetime import datetime

class DataDownloadThread(QThread):
    """Thread for downloading data in background"""
    progress_update = Signal(str, int)  # message, percentage
    download_finished = Signal(bool, str)  # success, message

    def __init__(self, start_date, end_date, timeframe):
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe

    def run(self):
        try:
            self.progress_update.emit(f"Starting download of {self.timeframe} data...", 10)

            # Get correct path to script (go up from src/gui to root, then into scripts/)
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            project_root = os.path.dirname(src_dir)
            script_path = os.path.join(project_root, "scripts", "download_btc_data.py")
            
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script not found at: {script_path}")

            # Build command
            cmd = [
                sys.executable,
                script_path,
                "--start-date", self.start_date,
                "--end-date", self.end_date,
                "--timeframe", self.timeframe
            ]

            self.progress_update.emit("Executing download command...", 20)

            # Run the download script from project root
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=project_root
            )

            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.progress_update.emit(output.strip(), 50)

            # Get final result
            return_code = process.poll()
            if return_code == 0:
                self.progress_update.emit("Download completed successfully!", 100)
                self.download_finished.emit(True, f"Successfully downloaded {self.timeframe} data")
            else:
                error_output = process.stderr.read()
                self.progress_update.emit(f"Download failed: {error_output}", 0)
                self.download_finished.emit(False, f"Failed to download {self.timeframe} data: {error_output}")

        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}", 0)
            self.download_finished.emit(False, f"Error downloading data: {str(e)}")

class Tab9DataDownload(QWidget):
    """
    Data Download Management Tab

    Allows users to:
    - View status of downloaded BTC/USD data files
    - Download missing timeframes
    - Monitor download progress
    - View data statistics
    """

    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        self.download_thread = None

        # Data configuration
        self.timeframes = [
            ('5Min', '5m', '5 minutes - High frequency scalping'),
            ('15Min', '15m', '15 minutes - Intraday analysis'),
            ('1Hour', '1h', '1 hour - Swing trading'),
            ('4Hour', '4h', '4 hours - Position trading')
        ]

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)

        # Header section
        header = self.create_header()
        main_layout.addWidget(header)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Data status and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel: Download progress and logs
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions: 35% controls, 65% log for better visibility
        splitter.setStretchFactor(0, 35)
        splitter.setStretchFactor(1, 65)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter, 1)
        self.setLayout(main_layout)

        # Initial data check
        self.check_data_status()

    def create_header(self):
        """Create header section"""
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

        title = QLabel("📥 BTC/USD Data Management")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)

        subtitle = QLabel("Download and manage historical BTC/USD data for backtesting and live trading")
        subtitle.setStyleSheet("color: #cccccc; font-size: 15px;")
        layout.addWidget(subtitle)

        # Quick stats
        stats_layout = QHBoxLayout()

        self.total_files_label = QLabel("Files: 0/4")
        self.total_files_label.setStyleSheet("color: #569cd6; font-weight: bold;")

        self.total_size_label = QLabel("Size: 0 MB")
        self.total_size_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")

        stats_layout.addWidget(self.total_files_label)
        stats_layout.addStretch()
        stats_layout.addWidget(self.total_size_label)

        layout.addLayout(stats_layout)

        frame.setLayout(layout)
        return frame

    def create_left_panel(self):
        """Create left panel with data status cards"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Data Status Cards
        status_group = QGroupBox("📊 Estado de Datos")
        status_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #fff;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)

        status_layout = QVBoxLayout()
        status_layout.setSpacing(10)

        # Create cards for each timeframe
        self.timeframe_cards = {}
        for timeframe, code, desc in self.timeframes:
            card = self.create_timeframe_card(timeframe, code, desc)
            status_layout.addWidget(card)
            self.timeframe_cards[timeframe] = card

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Quick actions
        actions_group = QGroupBox("⚡ Acciones Rápidas")
        actions_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #fff;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)

        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(8)

        download_all_btn = QPushButton("📥 Descargar Todo")
        download_all_btn.setFixedHeight(45)
        download_all_btn.clicked.connect(self.download_all_data)
        download_all_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0e639c, stop:1 #0a4d7a);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1177bb;
            }
        """)
        actions_layout.addWidget(download_all_btn)

        refresh_btn = QPushButton("🔄 Actualizar Estado")
        refresh_btn.setFixedHeight(40)
        refresh_btn.clicked.connect(self.refresh_data_status)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d2d;
                color: #ccc;
                border: 1px solid #555;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #353535;
            }
        """)
        actions_layout.addWidget(refresh_btn)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_timeframe_card(self, timeframe, code, description):
        """Create a card for a specific timeframe"""
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                padding: 12px;
            }
            QFrame:hover {
                border-color: #569cd6;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel(f"{timeframe}")
        title.setStyleSheet("color: #4ec9b0; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.status_labels = getattr(self, 'status_labels', {})
        status_label = QLabel("❓ Checking...")
        status_label.setStyleSheet("color: #888; font-size: 13px;")
        self.status_labels[timeframe] = status_label
        header_layout.addWidget(status_label)

        layout.addLayout(header_layout)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #ccc; font-size: 12px;")
        layout.addWidget(desc_label)

        # Stats
        stats_layout = QHBoxLayout()

        size_label = QLabel("Size: --")
        size_label.setStyleSheet("color: #569cd6; font-size: 12px;")
        stats_layout.addWidget(size_label)

        records_label = QLabel("Records: --")
        records_label.setStyleSheet("color: #dcdcaa; font-size: 12px;")
        stats_layout.addWidget(records_label)

        stats_layout.addStretch()

        # Download button
        download_btn = QPushButton("📥 Download")
        download_btn.setFixedSize(80, 30)
        download_btn.clicked.connect(lambda: self.download_timeframe(timeframe, code))
        download_btn.setStyleSheet("""
            QPushButton {
                background: #0e639c;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1177bb;
            }
            QPushButton:disabled {
                background: #444;
                color: #888;
            }
        """)
        stats_layout.addWidget(download_btn)

        layout.addLayout(stats_layout)

        # Store references
        card.size_label = size_label
        card.records_label = records_label
        card.download_btn = download_btn

        card.setLayout(layout)
        return card

        # Style the table
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #252525;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                gridline-color: #3e3e3e;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3e3e3e;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 8px;
                border: 1px solid #3e3e3e;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background-color: #0e639c;
            }
        """)

        status_layout.addWidget(self.data_table)

        # Action buttons
        buttons_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("🔄 Refresh Status")
        self.refresh_btn.clicked.connect(self.check_data_status)
        self.refresh_btn.setMinimumHeight(32)

        self.download_selected_btn = QPushButton("📥 Download Selected")
        self.download_selected_btn.clicked.connect(self.download_selected_timeframe)
        self.download_selected_btn.setMinimumHeight(32)
        self.download_selected_btn.setEnabled(False)

        self.download_all_btn = QPushButton("📦 Download All Missing")
        self.download_all_btn.clicked.connect(self.download_all_missing)
        self.download_all_btn.setMinimumHeight(32)

        buttons_layout.addWidget(self.refresh_btn)
        buttons_layout.addWidget(self.download_selected_btn)
        buttons_layout.addWidget(self.download_all_btn)

        status_layout.addLayout(buttons_layout)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Download Configuration
        config_group = QGroupBox("⚙️ Download Configuration")
        config_group.setStyleSheet(status_group.styleSheet())

        config_layout = QVBoxLayout()

        # Date range selection would go here
        date_info = QLabel("Default date range: 2020-01-01 to 2024-12-31")
        date_info.setStyleSheet("color: #cccccc; font-size: 15px;")
        config_layout.addWidget(date_info)

        note = QLabel("💡 Tip: Downloads are processed in background. Check progress in the right panel.")
        note.setStyleSheet("color: #dcdcaa; font-size: 14px;")
        note.setWordWrap(True)
        config_layout.addWidget(note)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        widget.setLayout(layout)
        return widget

    def create_right_panel(self):
        """Create right panel with progress and logs"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Download Progress
        progress_group = QGroupBox("📈 Download Progress")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)

        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                text-align: center;
                background-color: #252525;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 2px;
            }
        """)

        self.progress_label = QLabel("Ready to download data")
        self.progress_label.setStyleSheet("color: #cccccc;")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Activity Log
        log_group = QGroupBox("📋 Activity Log")
        log_group.setStyleSheet(progress_group.styleSheet())

        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(400)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                color: #cccccc;
                font-family: 'Consolas', monospace;
                font-size: 14px;
            }
        """)

        # Add initial log message
        self.log_message("Data Download Manager initialized")
        self.log_message("Click 'Refresh Status' to check current data files")

        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group, 1)

        widget.setLayout(layout)
        return widget

    def check_data_status(self):
        """Check status of all data files"""
        self.log_message("Checking data file status...")

        total_files = 0
        total_size = 0

        self.data_table.setRowCount(len(self.timeframes))

        for row, (api_tf, filename_tf, description) in enumerate(self.timeframes):
            filepath = f"data/raw/btc_usd_{filename_tf}.csv"

            # Timeframe name
            self.data_table.setItem(row, 0, QTableWidgetItem(f"{api_tf} ({description})"))

            if os.path.exists(filepath):
                # File exists - get stats
                file_size = os.path.getsize(filepath)
                total_size += file_size

                # Get record count (rough estimate)
                try:
                    with open(filepath, 'r') as f:
                        record_count = sum(1 for _ in f) - 1  # Subtract header
                except Exception:
                    record_count = 0

                # Get last modified
                mod_time = os.path.getmtime(filepath)
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')

                # Status
                status_item = QTableWidgetItem("✅ Available")
                status_item.setBackground(QColor("#4ec9b0"))
                self.data_table.setItem(row, 1, status_item)

                # Size
                size_mb = file_size / (1024 * 1024)
                self.data_table.setItem(row, 2, QTableWidgetItem(".1f"))

                # Records
                self.data_table.setItem(row, 3, QTableWidgetItem(f"{record_count:,}"))

                # Last modified
                self.data_table.setItem(row, 4, QTableWidgetItem(mod_date))

                total_files += 1

                self.log_message(f"✅ {filename_tf}: {record_count:,} records, {size_mb:.1f} MB")

            else:
                # File missing
                status_item = QTableWidgetItem("❌ Missing")
                status_item.setBackground(QColor("#f48771"))
                self.data_table.setItem(row, 1, status_item)

                self.data_table.setItem(row, 2, QTableWidgetItem("-"))
                self.data_table.setItem(row, 3, QTableWidgetItem("-"))
                self.data_table.setItem(row, 4, QTableWidgetItem("-"))

                self.log_message(f"❌ {filename_tf}: File not found")

        # Update summary
        self.total_files_label.setText(f"Files: {total_files}/4")
        total_size_mb = total_size / (1024 * 1024)
        self.total_size_label.setText(".1f")

        # Enable/disable buttons
        self.download_selected_btn.setEnabled(total_files < len(self.timeframes))

        self.log_message(f"Status check complete: {total_files}/4 files available ({total_size_mb:.1f} MB total)")

    def download_selected_timeframe(self):
        """Download the selected timeframe"""
        selected_rows = set()
        for item in self.data_table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a timeframe to download.")
            return

        if len(selected_rows) > 1:
            QMessageBox.warning(self, "Multiple Selection", "Please select only one timeframe at a time.")
            return

        row = list(selected_rows)[0]
        api_tf, filename_tf, description = self.timeframes[row]

        # Check if already exists
        filepath = f"data/raw/btc_usd_{filename_tf}.csv"
        if os.path.exists(filepath):
            reply = QMessageBox.question(
                self, "File Exists",
                f"The {api_tf} data file already exists. Do you want to overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.start_download(api_tf, "2020-01-01", "2024-12-31")

    def download_all_missing(self):
        """Download all missing timeframes"""
        missing_timeframes = []

        for api_tf, filename_tf, description in self.timeframes:
            filepath = f"data/raw/btc_usd_{filename_tf}.csv"
            if not os.path.exists(filepath):
                missing_timeframes.append(api_tf)

        if not missing_timeframes:
            QMessageBox.information(self, "All Files Present", "All data files are already downloaded!")
            return

        reply = QMessageBox.question(
            self, "Download Missing Files",
            f"Download {len(missing_timeframes)} missing timeframe(s): {', '.join(missing_timeframes)}?\n\nThis may take several minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.log_message(f"Starting batch download of {len(missing_timeframes)} timeframes...")
            # For now, download one at a time. Could be enhanced to download in parallel
            for tf in missing_timeframes:
                self.start_download(tf, "2020-01-01", "2024-12-31")

    def start_download(self, timeframe, start_date, end_date):
        """Start download process for a timeframe"""
        if self.download_thread and self.download_thread.isRunning():
            QMessageBox.warning(self, "Download in Progress", "A download is already running. Please wait.")
            return

        self.log_message(f"Starting download: {timeframe} from {start_date} to {end_date}")

        # Disable buttons during download
        self.download_selected_btn.setEnabled(False)
        self.download_all_btn.setEnabled(False)

        # Create and start download thread
        self.download_thread = DataDownloadThread(start_date, end_date, timeframe)
        self.download_thread.progress_update.connect(self.on_progress_update)
        self.download_thread.download_finished.connect(self.on_download_finished)
        self.download_thread.start()

    def on_progress_update(self, message, percentage):
        """Handle progress updates from download thread"""
        self.progress_label.setText(message)
        self.progress_bar.setValue(percentage)
        self.log_message(message)

    def on_download_finished(self, success, message):
        """Handle download completion"""
        self.log_message(message)

        if success:
            self.progress_label.setText("Download completed successfully!")
            self.progress_bar.setValue(100)
        else:
            self.progress_label.setText("Download failed")
            self.progress_bar.setValue(0)

        # Re-enable buttons
        self.download_selected_btn.setEnabled(True)
        self.download_all_btn.setEnabled(True)

        # Refresh data status
        QTimer.singleShot(1000, self.check_data_status)  # Refresh after 1 second

    def log_message(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

        # Auto scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def download_all_data(self):
        """Download all available data for all timeframes"""
        try:
            # Get all timeframes from the cards
            timeframes_to_download = []
            
            # Check which timeframes are available in the UI
            for i in range(self.timeframe_layout.count()):
                item = self.timeframe_layout.itemAt(i)
                if item and item.widget():
                    card = item.widget()
                    if hasattr(card, 'timeframe') and hasattr(card, 'is_selected'):
                        if card.is_selected():
                            timeframes_to_download.append(card.timeframe)
            
            if not timeframes_to_download:
                self.log_message("No se seleccionaron timeframes para descargar")
                return
            
            # Start download for each timeframe
            for timeframe in timeframes_to_download:
                start_date = self.start_date_edit.date().toPython()
                end_date = self.end_date_edit.date().toPython()
                self.start_download(timeframe, start_date, end_date)
                
            self.log_message(f"Iniciando descarga de {len(timeframes_to_download)} timeframes")
            
        except Exception as e:
            self.log_message(f"Error al descargar todos los datos: {str(e)}")
    
    def refresh_data_status(self):
        """Refresh the data status display"""
        try:
            self.check_data_status()
            self.log_message("Estado de datos actualizado")
        except Exception as e:
            self.log_message(f"Error al actualizar estado de datos: {str(e)}")
    
    @property
    def data_table(self):
        """Get the data table widget"""
        return getattr(self, 'data_status_table', None) or self._create_mock_table()
    
    def _create_mock_table(self):
        """Create a mock table if the real one doesn't exist"""
        from PySide6.QtWidgets import QTableWidget
        mock_table = QTableWidget()
        mock_table.setRowCount(0)
        mock_table.setColumnCount(3)
        mock_table.setHorizontalHeaderLabels(["Archivo", "Estado", "Última Modificación"])
        return mock_table
    
    @property
    def download_selected_btn(self):
        """Get the download selected button"""
        return getattr(self, 'download_selected_button', None) or self._create_mock_button()
    
    def _create_mock_button(self):
        """Create a mock button if the real one doesn't exist"""
        from PySide6.QtWidgets import QPushButton
        mock_btn = QPushButton("Download Selected")
        mock_btn.setEnabled(False)
        return mock_btn