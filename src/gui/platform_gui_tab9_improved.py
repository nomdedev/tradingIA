from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QProgressBar,
    QMessageBox,
    QGroupBox,
    QTextEdit,
    QSplitter,
    QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QColor, QFont
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
                "--start-date",
                self.start_date,
                "--end-date",
                self.end_date,
                "--timeframe",
                self.timeframe,
            ]

            self.progress_update.emit("Executing download command...", 20)

            # Run the download script from project root
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=project_root)

            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
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
    Data Download Management Tab (Improved)
    """

    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        self.download_thread = None

        # Data configuration
        self.timeframes = [
            ("5Min", "5m", "5 minutes - High frequency scalping"),
            ("15Min", "15m", "15 minutes - Intraday analysis"),
            ("1Hour", "1h", "1 hour - Swing trading"),
            ("4Hour", "4h", "4 hours - Position trading"),
        ]

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #3e3e3e;
            }
        """
        )

        # Left panel: Data status and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel: Download progress and logs
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setStretchFactor(0, 30)
        splitter.setStretchFactor(1, 70)
        splitter.setSizes([350, 850])
        splitter.setCollapsible(0, False)

        main_layout.addWidget(splitter)

        # Initial data check
        self.check_data_status()

    def create_left_panel(self):
        """Create left panel with data status cards"""
        panel = QFrame()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(400)
        panel.setStyleSheet("background-color: #252526; border-right: 1px solid #3e3e3e;")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        title = QLabel("📥 Data Management")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        subtitle = QLabel("Manage historical BTC/USD data")
        subtitle.setStyleSheet("color: #888888; font-size: 14px;")
        layout.addWidget(subtitle)

        # Data Status Cards
        status_group = QGroupBox("Data Status")
        status_group.setStyleSheet(self.get_group_style())
        
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
        actions_group = QGroupBox("Quick Actions")
        actions_group.setStyleSheet(self.get_group_style())
        
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(10)

        download_all_btn = QPushButton("📥 Download All")
        download_all_btn.setFixedHeight(45)
        download_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        download_all_btn.setStyleSheet(self.get_button_style("#4ec9b0"))
        download_all_btn.clicked.connect(self.download_all_data)
        actions_layout.addWidget(download_all_btn)
        
        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.setFixedHeight(45)
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.setStyleSheet(self.get_button_style("#569cd6"))
        refresh_btn.clicked.connect(self.check_data_status)
        actions_layout.addWidget(refresh_btn)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        layout.addStretch()
        
        # Stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: #1e1e1e; border-radius: 6px; padding: 10px;")
        stats_layout = QHBoxLayout(stats_frame)
        
        self.total_files_label = QLabel("Files: 0/4")
        self.total_files_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        stats_layout.addWidget(self.total_files_label)
        
        stats_layout.addStretch()
        
        self.total_size_label = QLabel("Size: 0 MB")
        self.total_size_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        stats_layout.addWidget(self.total_size_label)
        
        layout.addWidget(stats_frame)

        return panel

    def create_right_panel(self):
        """Create right panel with logs"""
        panel = QFrame()
        panel.setStyleSheet("background-color: #1e1e1e;")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Progress Section
        progress_group = QGroupBox("Download Progress")
        progress_group.setStyleSheet(self.get_group_style())
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready to download")
        self.progress_label.setStyleSheet("color: #cccccc; font-size: 14px;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4ec9b0;
                border-radius: 3px;
            }
        """
        )
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Logs Section
        log_group = QGroupBox("Download Logs")
        log_group.setStyleSheet(self.get_group_style())
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', monospace;
            }
        """
        )
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        return panel

    def create_timeframe_card(self, timeframe, code, desc):
        """Create a status card for a timeframe"""
        card = QFrame()
        card.setObjectName(f"card_{timeframe}")
        card.setStyleSheet(
            """
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
            }
        """
        )
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        title = QLabel(f"{timeframe} ({code})")
        title.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px;")
        info_layout.addWidget(title)
        
        desc_lbl = QLabel(desc.split(" - ")[1] if " - " in desc else desc)
        desc_lbl.setStyleSheet("color: #888888; font-size: 11px;")
        info_layout.addWidget(desc_lbl)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Status
        status_lbl = QLabel("Checking...")
        status_lbl.setObjectName(f"status_{timeframe}")
        status_lbl.setStyleSheet("color: #888888; font-weight: bold;")
        layout.addWidget(status_lbl)
        
        # Download button
        btn = QPushButton("⬇")
        btn.setFixedSize(30, 30)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(f"Download {timeframe}")
        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3e3e3e;
                color: #ffffff;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4e4e4e;
            }
        """
        )
        btn.clicked.connect(lambda: self.download_data(timeframe))
        layout.addWidget(btn)
        
        # Store references
        card.status_lbl = status_lbl
        
        return card

    def get_group_style(self):
        return """
            QGroupBox {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                color: #ffffff;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                background-color: #2d2d2d;
            }
        """

    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {color}, stop:1 {self.adjust_color(color, -20)});
                color: #1e1e1e;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                border: none;
            }}
            QPushButton:hover {{
                background: {self.adjust_color(color, 20)};
            }}
            QPushButton:pressed {{
                background: {self.adjust_color(color, -40)};
            }}
        """

    def adjust_color(self, hex_color, factor):
        """Adjust color brightness"""
        color = QColor(hex_color)
        h, s, v, a = color.getHsv()
        v = max(0, min(255, v + factor))
        return QColor.fromHsv(h, s, v, a).name()

    def check_data_status(self):
        """Check status of data files"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        
        files_found = 0
        total_size = 0
        
        for timeframe, _, _ in self.timeframes:
            filename = f"btc_{timeframe}.csv"
            path = os.path.join(data_dir, filename)
            
            card = self.timeframe_cards.get(timeframe)
            if not card:
                continue
                
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                total_size += size_mb
                files_found += 1
                
                card.status_lbl.setText(f"✓ {size_mb:.1f} MB")
                card.status_lbl.setStyleSheet("color: #4ec9b0; font-weight: bold;")
            else:
                card.status_lbl.setText("Missing")
                card.status_lbl.setStyleSheet("color: #f48771; font-weight: bold;")
                
        self.total_files_label.setText(f"Files: {files_found}/{len(self.timeframes)}")
        self.total_size_label.setText(f"Size: {total_size:.1f} MB")

    def download_data(self, timeframe):
        """Start download for a specific timeframe"""
        if self.download_thread and self.download_thread.isRunning():
            QMessageBox.warning(self, "Download in Progress", "Please wait for the current download to finish.")
            return
            
        self.log_text.append(f"Starting download for {timeframe}...")
        
        # Default to last 2 years
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = "2022-01-01"
        
        self.download_thread = DataDownloadThread(start_date, end_date, timeframe)
        self.download_thread.progress_update.connect(self.update_progress)
        self.download_thread.download_finished.connect(self.on_download_finished)
        self.download_thread.start()

    def download_all_data(self):
        """Download all timeframes sequentially"""
        # This is a simplified version, ideally we'd queue them
        self.download_data("5Min") # Start with 5Min as example

    def update_progress(self, message, value):
        """Update progress bar and log"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def on_download_finished(self, success, message):
        """Handle download completion"""
        if success:
            self.log_text.append(f"✅ {message}")
            self.check_data_status()
        else:
            self.log_text.append(f"❌ {message}")
            
        self.progress_bar.setValue(100 if success else 0)
