"""
BTC Trading Strategy Platform - Tab 6: Live Monitoring
Real-time monitoring and paper trading interface.

Author: TradingIA Team
Version: 1.0.0
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox,
    QTextEdit, QSplitter, QListWidget, QFrame, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QThread
from PySide6.QtGui import QFont, QPainter, QColor, QPen
import traceback
import time
from datetime import datetime, timedelta
import random  # For demo purposes
import math

class LiveMonitorEngine(QObject):
    """Live monitoring engine for paper trading"""

    pnl_updated = Signal(float)
    metrics_updated = Signal(dict)
    signal_detected = Signal(dict)
    chart_updated = Signal(list)
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.current_pnl = 0.0
        self.trades_today = []
        self.signals_log = []

    def start_monitoring(self):
        """Start live monitoring"""
        self.is_running = True
        self.status_changed.emit("Monitoring started")

    def stop_monitoring(self):
        """Stop live monitoring"""
        self.is_running = False
        self.status_changed.emit("Monitoring stopped")

    def get_current_metrics(self):
        """Get current live metrics"""
        # Mock data for demonstration
        return {
            'sharpe_live': round(random.uniform(1.2, 2.1), 2),
            'calmar_live': round(random.uniform(1.8, 2.8), 2),
            'win_rate_live': round(random.uniform(65, 75), 1),
            'dd_live': round(random.uniform(2, 8), 1),
            'trades_today': len(self.trades_today)
        }

    def simulate_live_data(self):
        """Simulate live data updates (for demo)"""
        if not self.is_running:
            return

        # Simulate PnL changes
        pnl_change = random.uniform(-50, 50)
        self.current_pnl += pnl_change
        self.pnl_updated.emit(self.current_pnl)

        # Simulate metrics update
        metrics = self.get_current_metrics()
        self.metrics_updated.emit(metrics)

        # Simulate occasional signals
        if random.random() < 0.1:  # 10% chance every update
            signal = {
                'timestamp': datetime.now(),
                'type': random.choice(['BUY', 'SELL']),
                'price': round(random.uniform(45000, 55000), 2),
                'strength': round(random.uniform(3.0, 5.0), 1),
                'reason': random.choice(['IFVG Break', 'RSI Divergence', 'Volume Spike'])
            }
            self.signals_log.append(signal)
            self.signal_detected.emit(signal)

            # Keep only last 20 signals
            if len(self.signals_log) > 20:
                self.signals_log.pop(0)

class PnLGauge(QWidget):
    """Custom circular gauge for PnL display"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setMinimumSize(200, 200)

    def setValue(self, value):
        """Set the gauge value"""
        self.value = value
        self.update()

    def paintEvent(self, event):
        """Paint the gauge"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get dimensions
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 2 - 10

        # Draw background circle
        painter.setPen(QPen(QColor(100, 100, 100), 3))
        painter.setBrush(QColor(43, 43, 43))
        painter.drawEllipse(center, radius, radius)

        # Draw scale
        painter.setPen(QPen(QColor(150, 150, 150), 2))
        for i in range(0, 360, 45):
            angle = i - 90  # Start from top
            x = center.x() + radius * 0.8 * math.cos(math.radians(angle))
            y = center.y() + radius * 0.8 * math.sin(math.radians(angle))
            painter.drawLine(
                int(center.x() + radius * 0.7 * math.cos(math.radians(angle))),
                int(center.y() + radius * 0.7 * math.sin(math.radians(angle))),
                int(x), int(y)
            )

        # Draw needle
        # Map value to angle (-1000 to +1000 -> -135 to +135 degrees)
        angle = max(-135, min(135, self.value * 135 / 1000)) - 90
        painter.setPen(QPen(QColor(255, 107, 53), 4))
        needle_length = radius * 0.7
        x = center.x() + needle_length * math.cos(math.radians(angle))
        y = center.y() + needle_length * math.sin(math.radians(angle))
        painter.drawLine(center.x(), center.y(), int(x), int(y))

        # Draw center dot
        painter.setBrush(QColor(255, 107, 53))
        painter.drawEllipse(center, 8, 8)

        # Draw value text
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        text = f"${self.value:,.0f}"
        painter.drawText(rect, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, text)

class Tab6LiveMonitoring(QWidget):
    """Live monitoring tab for real-time trading"""

    def __init__(self, parent_platform, live_monitor_engine=None):
        super().__init__()
        self.parent = parent_platform
        self.live_monitor = live_monitor_engine or LiveMonitorEngine()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_live_data)

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Live Monitoring - Paper Trading")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Gauges and metrics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # PnL Gauge
        pnl_group = QGroupBox("Current PnL")
        pnl_layout = QVBoxLayout()
        self.pnl_gauge = PnLGauge()
        pnl_layout.addWidget(self.pnl_gauge)
        pnl_group.setLayout(pnl_layout)
        left_layout.addWidget(pnl_group)

        # Key Metrics
        metrics_group = QGroupBox("Live Metrics")
        metrics_layout = QGridLayout()

        self.metrics_labels = {}
        metrics = ['Sharpe (Live)', 'Calmar (Live)', 'Win Rate (Live)', 'DD (Live)', 'Trades Today']

        for i, metric in enumerate(metrics):
            # Label
            label = QLabel(f"{metric}:")
            label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            metrics_layout.addWidget(label, i, 0)

            # Value
            value_label = QLabel("--")
            value_label.setFont(QFont("Arial", 10))
            value_label.setStyleSheet("color: #ffffff;")
            metrics_layout.addWidget(value_label, i, 1)
            self.metrics_labels[metric] = value_label

        metrics_group.setLayout(metrics_layout)
        left_layout.addWidget(metrics_group)

        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        self.start_btn = QPushButton("Start Paper Trading")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-weight: bold; }")
        self.start_btn.clicked.connect(self.on_start_trading)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Paper Trading")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; font-weight: bold; }")
        self.stop_btn.clicked.connect(self.on_stop_trading)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        self.manual_trade_btn = QPushButton("Manual Trade")
        self.manual_trade_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 10px; }")
        self.manual_trade_btn.clicked.connect(self.on_manual_trade)
        controls_layout.addWidget(self.manual_trade_btn)

        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        # Status
        self.status_label = QLabel("Status: Not running")
        self.status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        main_splitter.addWidget(left_panel)

        # Right panel - Signals and charts
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Signals log
        signals_group = QGroupBox("Signal Alerts (Last 20)")
        signals_layout = QVBoxLayout()

        self.signals_list = QListWidget()
        self.signals_list.setMaximumHeight(200)
        signals_layout.addWidget(self.signals_list)

        signals_group.setLayout(signals_layout)
        right_layout.addWidget(signals_group)

        # Live chart placeholder (would integrate with Plotly)
        chart_group = QGroupBox("Live Price Chart (BTC/USD)")
        chart_layout = QVBoxLayout()

        self.chart_placeholder = QLabel("Chart integration - Coming soon\nLast update: --")
        self.chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_placeholder.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #444;
                padding: 20px;
                color: #666;
                font-size: 14px;
            }
        """)
        self.chart_placeholder.setMinimumHeight(300)
        chart_layout.addWidget(self.chart_placeholder)

        chart_group.setLayout(chart_layout)
        right_layout.addWidget(chart_group)

        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([300, 500])
        layout.addWidget(main_splitter)

        # Set dark theme
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

    def connect_signals(self):
        """Connect live monitor signals"""
        self.live_monitor.pnl_updated.connect(self.update_pnl_gauge)
        self.live_monitor.metrics_updated.connect(self.update_metrics)
        self.live_monitor.signal_detected.connect(self.add_signal_alert)
        self.live_monitor.status_changed.connect(self.update_status)

    def on_start_trading(self):
        """Start paper trading"""
        try:
            self.live_monitor.start_monitoring()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.manual_trade_btn.setEnabled(True)

            # Start update timer (every 5 seconds)
            self.update_timer.start(5000)

        except Exception as e:
            self.status_label.setText(f"Failed to start: {str(e)}")

    def on_stop_trading(self):
        """Stop paper trading"""
        try:
            self.live_monitor.stop_monitoring()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.manual_trade_btn.setEnabled(False)

            # Stop update timer
            self.update_timer.stop()

        except Exception as e:
            self.status_label.setText(f"Failed to stop: {str(e)}")

    def on_manual_trade(self):
        """Handle manual trade button"""
        # For now, just show a message
        self.status_label.setText("Manual trading - Feature coming soon")

    def update_live_data(self):
        """Update live data (called by timer)"""
        if self.live_monitor.is_running:
            self.live_monitor.simulate_live_data()

    def update_pnl_gauge(self, pnl):
        """Update PnL gauge"""
        self.pnl_gauge.setValue(pnl)

    def update_metrics(self, metrics):
        """Update metrics display"""
        for metric, label in self.metrics_labels.items():
            key = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
            value = metrics.get(key, '--')
            label.setText(str(value))

    def add_signal_alert(self, signal):
        """Add signal to alerts list"""
        timestamp = signal['timestamp'].strftime('%H:%M:%S')
        signal_type = signal['type']
        price = signal['price']
        strength = signal['strength']
        reason = signal['reason']

        # Color based on signal type
        color = "#4CAF50" if signal_type == "BUY" else "#f44336"

        item_text = f"{timestamp} | {signal_type} | ${price:,.0f} | Strength: {strength} | {reason}"

        # Add to list
        self.signals_list.insertItem(0, item_text)

        # Color the item
        if self.signals_list.count() > 0:
            item = self.signals_list.item(0)
            item.setForeground(QColor(color))

        # Keep only last 20
        while self.signals_list.count() > 20:
            self.signals_list.takeItem(self.signals_list.count() - 1)

    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(f"Status: {status}")

        # Update color based on status
        if "started" in status.lower():
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif "stopped" in status.lower():
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: #ff9800; font-weight: bold;")

    def on_tab_activated(self):
        """Called when tab becomes active"""
        # Refresh metrics if running
        if self.live_monitor.is_running:
            metrics = self.live_monitor.get_current_metrics()
            self.update_metrics(metrics)