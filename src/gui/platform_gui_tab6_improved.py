"""
Tab 6 - Live Trading Monitor (Improved)
Risk Dashboard with real-time alerts and emergency controls
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QTextEdit
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor, QPainter, QFont, QPen, QBrush
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from datetime import datetime
import random


# ============================================================================
# RISK DASHBOARD - MetricCard Component
# ============================================================================
class MetricCard(QFrame):
    """Reusable metric card with icon and value"""
    def __init__(self, title, value="--", unit="", color="#569cd6"):
        super().__init__()
        self.color = color
        self.title_text = title
        self.value_text = value
        self.unit_text = unit
        
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: #2d2d2d;
                border-left: 4px solid {color};
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 11px; font-weight: normal;")
        layout.addWidget(title_label)
        
        # Value container
        value_layout = QHBoxLayout()
        value_layout.setContentsMargins(0, 4, 0, 0)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        value_layout.addWidget(self.value_label)
        
        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet("color: #888; font-size: 14px; margin-left: 4px;")
            value_layout.addWidget(unit_label)
        
        value_layout.addStretch()
        layout.addLayout(value_layout)
    
    def update_value(self, value, color=None):
        """Update card value and optionally color"""
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")


# ============================================================================
# CUSTOM WIDGETS - PnL Gauge
# ============================================================================
class PnLGauge(QWidget):
    """Visual gauge for P&L display"""
    def __init__(self):
        super().__init__()
        self.value = 0.0
        self.setMinimumSize(180, 180)
    
    def setValue(self, value):
        """Set gauge value (-100 to +100)"""
        self.value = max(-100, min(100, value))
        self.update()
    
    def paintEvent(self, event):
        """Custom paint for gauge"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.setBrush(QBrush(QColor("#1e1e1e")))
        painter.setPen(QPen(QColor("#444"), 2))
        painter.drawEllipse(10, 10, 160, 160)
        
        # Arc segments
        center_x, center_y, radius = 90, 90, 70
        
        # Negative zone (red)
        painter.setPen(QPen(QColor("#f48771"), 8))
        painter.drawArc(20, 20, 140, 140, 180*16, -90*16)
        
        # Positive zone (green)
        painter.setPen(QPen(QColor("#4ec9b0"), 8))
        painter.drawArc(20, 20, 140, 140, 90*16, -90*16)
        
        # Needle
        angle = 180 - (self.value + 100) * 0.9  # Map -100..100 to 180..0 degrees
        import math
        rad = math.radians(angle)
        needle_x = center_x + radius * math.cos(rad)
        needle_y = center_y - radius * math.sin(rad)
        
        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.drawLine(center_x, center_y, int(needle_x), int(needle_y))
        
        # Center dot
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)
        
        # Value text
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        painter.drawText(event.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.value:+.1f}%")


# ============================================================================
# BACKGROUND THREAD - Live Data Simulator
# ============================================================================
class LiveMonitorThread(QThread):
    """Background thread for live market monitoring"""
    pnl_update = Signal(float)
    metrics_update = Signal(dict)
    position_update = Signal(list)
    alert_triggered = Signal(str, str)  # (message, level)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.current_pnl = 0.0
        self.positions = []
        
    def run(self):
        """Main monitoring loop"""
        self.running = True
        iteration = 0
        
        while self.running:
            iteration += 1
            
            # Simulate P&L changes
            self.current_pnl += random.uniform(-2, 2.5)
            self.current_pnl = max(-50, min(50, self.current_pnl))
            self.pnl_update.emit(self.current_pnl)
            
            # Simulate metrics every 5 iterations
            if iteration % 5 == 0:
                metrics = {
                    'sharpe': round(random.uniform(0.5, 2.5), 2),
                    'max_dd': round(random.uniform(-15, -2), 2),
                    'win_rate': round(random.uniform(45, 65), 1),
                    'exposure': round(random.uniform(30, 80), 1),
                    'trades_today': random.randint(5, 20),
                    'active_positions': random.randint(0, 4)
                }
                self.metrics_update.emit(metrics)
                
                # Check alert conditions
                if metrics['max_dd'] < -12:
                    self.alert_triggered.emit(f"Drawdown crítico: {metrics['max_dd']:.1f}%", "error")
                elif self.current_pnl < -20:
                    self.alert_triggered.emit(f"Pérdida diaria alta: {self.current_pnl:.1f}%", "warning")
            
            # Simulate position updates
            if iteration % 10 == 0:
                positions = []
                for i in range(random.randint(0, 3)):
                    positions.append({
                        'symbol': random.choice(['BTC/USD', 'ETH/USD', 'AAPL']),
                        'side': random.choice(['LONG', 'SHORT']),
                        'size': round(random.uniform(0.1, 2.0), 2),
                        'entry': round(random.uniform(30000, 50000), 2),
                        'current': round(random.uniform(30000, 50000), 2),
                        'pnl': round(random.uniform(-500, 800), 2)
                    })
                self.position_update.emit(positions)
            
            self.msleep(1000)  # 1 second updates
    
    def stop(self):
        """Stop monitoring"""
        self.running = False


# ============================================================================
# MAIN TAB CLASS
# ============================================================================
class Tab6LiveMonitoring(QWidget):
    """Tab 6: Live Trading Monitor with Risk Dashboard"""
    status_update = Signal(str, str)
    
    def __init__(self, parent_platform=None):
        super().__init__()
        self.parent = parent_platform
        self.monitor_thread = None
        self.alert_config = {
            'max_drawdown': -15.0,
            'daily_loss_limit': -25.0,
            'win_streak_notify': 5,
            'loss_streak_notify': 3
        }
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # === HEADER ===
        header_layout = QHBoxLayout()
        
        title = QLabel("🔴 LIVE TRADING MONITOR")
        title.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Connection status
        self.connection_status = QLabel("⚫ Disconnected")
        self.connection_status.setStyleSheet("color: #888; font-size: 12px; padding: 6px 12px; background: #2d2d2d; border-radius: 4px;")
        header_layout.addWidget(self.connection_status)
        
        layout.addLayout(header_layout)
        
        # === MAIN SPLITTER ===
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT PANEL: Risk Dashboard ---
        left_panel = self.create_risk_dashboard()
        main_splitter.addWidget(left_panel)
        
        # --- RIGHT PANEL: Monitoring & Controls ---
        right_panel = self.create_monitor_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([400, 700])
        layout.addWidget(main_splitter)
        
        # === APPLY THEME ===
        self.apply_modern_theme()
    
    def create_risk_dashboard(self):
        """Create left panel with risk metrics and alerts"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # --- P&L Gauge ---
        gauge_group = QGroupBox("📊 P&L Actual")
        gauge_group.setMaximumHeight(240)
        gauge_layout = QVBoxLayout()
        gauge_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gauge_layout.setContentsMargins(8, 12, 8, 8)
        
        self.pnl_gauge = PnLGauge()
        gauge_layout.addWidget(self.pnl_gauge, alignment=Qt.AlignmentFlag.AlignCenter)
        
        gauge_group.setLayout(gauge_layout)
        layout.addWidget(gauge_group)
        
        # --- Risk Metrics Cards ---
        metrics_group = QGroupBox("⚡ Métricas de Riesgo")
        metrics_group.setMaximumHeight(180)
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(6)
        metrics_layout.setContentsMargins(8, 12, 8, 8)
        
        self.sharpe_card = MetricCard("Sharpe Ratio", "--", "", "#569cd6")
        self.dd_card = MetricCard("Max Drawdown", "--", "%", "#f48771")
        self.win_rate_card = MetricCard("Win Rate", "--", "%", "#4ec9b0")
        self.exposure_card = MetricCard("Exposición", "--", "%", "#dcdcaa")
        
        metrics_layout.addWidget(self.sharpe_card, 0, 0)
        metrics_layout.addWidget(self.dd_card, 0, 1)
        metrics_layout.addWidget(self.win_rate_card, 1, 0)
        metrics_layout.addWidget(self.exposure_card, 1, 1)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # --- Alert Configuration ---
        alert_group = QGroupBox("🔔 Configuración de Alertas")
        alert_group.setMaximumHeight(200)
        alert_layout = QGridLayout()
        alert_layout.setSpacing(6)
        alert_layout.setContentsMargins(8, 12, 8, 8)
        
        # Max Drawdown alert
        dd_label = QLabel("Max Drawdown:")
        dd_label.setStyleSheet("color: #ccc; font-size: 11px;")
        alert_layout.addWidget(dd_label, 0, 0)
        self.dd_alert_spin = QDoubleSpinBox()
        self.dd_alert_spin.setRange(-50, 0)
        self.dd_alert_spin.setValue(-15)
        self.dd_alert_spin.setSuffix(" %")
        self.dd_alert_spin.setStyleSheet("background: #2d2d2d; color: #fff; padding: 6px; border: 1px solid #555; border-radius: 3px; font-size: 11px;")
        self.dd_alert_spin.setMaximumHeight(28)
        alert_layout.addWidget(self.dd_alert_spin, 0, 1)
        
        # Daily loss limit
        loss_label = QLabel("Pérdida Diaria:")
        loss_label.setStyleSheet("color: #ccc; font-size: 11px;")
        alert_layout.addWidget(loss_label, 1, 0)
        self.loss_alert_spin = QDoubleSpinBox()
        self.loss_alert_spin.setRange(-100, 0)
        self.loss_alert_spin.setValue(-25)
        self.loss_alert_spin.setSuffix(" %")
        self.loss_alert_spin.setStyleSheet("background: #2d2d2d; color: #fff; padding: 6px; border: 1px solid #555; border-radius: 3px; font-size: 11px;")
        self.loss_alert_spin.setMaximumHeight(28)
        alert_layout.addWidget(self.loss_alert_spin, 1, 1)
        
        # Win streak notification
        win_label = QLabel("Racha Ganadora:")
        win_label.setStyleSheet("color: #ccc; font-size: 11px;")
        alert_layout.addWidget(win_label, 2, 0)
        self.win_streak_spin = QSpinBox()
        self.win_streak_spin.setRange(3, 20)
        self.win_streak_spin.setValue(5)
        self.win_streak_spin.setSuffix(" trades")
        self.win_streak_spin.setStyleSheet("background: #2d2d2d; color: #fff; padding: 6px; border: 1px solid #555; border-radius: 3px; font-size: 11px;")
        self.win_streak_spin.setMaximumHeight(28)
        alert_layout.addWidget(self.win_streak_spin, 2, 1)
        
        # Loss streak notification
        loss_streak_label = QLabel("Racha Perdedora:")
        loss_streak_label.setStyleSheet("color: #ccc; font-size: 11px;")
        alert_layout.addWidget(loss_streak_label, 3, 0)
        self.loss_streak_spin = QSpinBox()
        self.loss_streak_spin.setRange(2, 10)
        self.loss_streak_spin.setValue(3)
        self.loss_streak_spin.setSuffix(" trades")
        self.loss_streak_spin.setStyleSheet("background: #2d2d2d; color: #fff; padding: 6px; border: 1px solid #555; border-radius: 3px; font-size: 11px;")
        self.loss_streak_spin.setMaximumHeight(28)
        alert_layout.addWidget(self.loss_streak_spin, 3, 1)
        
        # Sound alerts checkbox
        self.sound_alerts_check = QCheckBox("Alertas sonoras")
        self.sound_alerts_check.setChecked(True)
        self.sound_alerts_check.setStyleSheet("color: #ccc; font-size: 11px;")
        alert_layout.addWidget(self.sound_alerts_check, 4, 0, 1, 2)
        
        alert_group.setLayout(alert_layout)
        layout.addWidget(alert_group)
        
        # --- Emergency Controls ---
        emergency_group = QGroupBox("🚨 Controles de Emergencia")
        emergency_group.setMaximumHeight(180)
        emergency_layout = QVBoxLayout()
        emergency_layout.setSpacing(6)
        emergency_layout.setContentsMargins(8, 12, 8, 8)
        
        self.pause_btn = QPushButton("⏸ PAUSAR")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background: #dcdcaa;
                color: #1e1e1e;
                border: none;
                padding: 8px;
                font-weight: bold;
                font-size: 11px;
                border-radius: 4px;
                min-height: 32px;
            }
            QPushButton:hover { background: #e8e4b7; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        self.pause_btn.clicked.connect(self.on_pause_trading)
        self.pause_btn.setEnabled(False)
        emergency_layout.addWidget(self.pause_btn)
        
        self.stop_all_btn = QPushButton("🛑 STOP ALL")
        self.stop_all_btn.setStyleSheet("""
            QPushButton {
                background: #f48771;
                color: #fff;
                border: none;
                padding: 8px;
                font-weight: bold;
                font-size: 11px;
                border-radius: 4px;
                min-height: 32px;
            }
            QPushButton:hover { background: #ff9d8a; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        self.stop_all_btn.clicked.connect(self.on_stop_all)
        self.stop_all_btn.setEnabled(False)
        emergency_layout.addWidget(self.stop_all_btn)
        
        self.kill_switch_btn = QPushButton("☠ KILL SWITCH")
        self.kill_switch_btn.setStyleSheet("""
            QPushButton {
                background: #8B0000;
                color: #fff;
                border: 2px solid #ff0000;
                padding: 8px;
                font-weight: bold;
                font-size: 11px;
                border-radius: 4px;
                min-height: 32px;
            }
            QPushButton:hover { background: #a00000; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        self.kill_switch_btn.clicked.connect(self.on_kill_switch)
        self.kill_switch_btn.setEnabled(False)
        emergency_layout.addWidget(self.kill_switch_btn)
        
        emergency_group.setLayout(emergency_layout)
        layout.addWidget(emergency_group)
        
        layout.addStretch()
        
        return container
    
    def create_monitor_panel(self):
        """Create right panel with monitoring and positions"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # --- Trading Controls ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        
        self.start_btn = QPushButton("▶ START TRADING")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4ec9b0;
                color: #1e1e1e;
                border: none;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background: #6eddc5; }
        """)
        self.start_btn.clicked.connect(self.on_start_trading)
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ STOP TRADING")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #f48771;
                color: #fff;
                border: none;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background: #ff9d8a; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        self.stop_btn.clicked.connect(self.on_stop_trading)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        
        # Mode selector
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Paper Trading", "Live Trading (PRODUCCIÓN)"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d2d;
                color: #fff;
                border: 1px solid #555;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 13px;
            }
            QComboBox:hover { border-color: #0e639c; }
            QComboBox::drop-down { border: none; }
        """)
        controls_layout.addWidget(QLabel("Modo:"))
        controls_layout.addWidget(self.mode_combo)
        
        layout.addLayout(controls_layout)
        
        # --- Status Bar ---
        self.status_bar = QLabel("⚫ Sistema detenido - Presione START para comenzar")
        self.status_bar.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                color: #888;
                padding: 12px;
                border-radius: 4px;
                border-left: 4px solid #888;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.status_bar)
        
        # --- Live Positions Table ---
        positions_group = QGroupBox("📈 Posiciones Activas")
        positions_layout = QVBoxLayout()
        positions_layout.setContentsMargins(8, 12, 8, 8)
        
        self.positions_table = QTableWidget(0, 7)
        self.positions_table.setHorizontalHeaderLabels([
            "Symbol", "Side", "Size", "Entry", "Current", "P&L", "P&L %"
        ])
        self.positions_table.setStyleSheet("""
            QTableWidget {
                background: #1e1e1e;
                color: #fff;
                border: 1px solid #444;
                gridline-color: #333;
                font-size: 11px;
            }
            QHeaderView::section {
                background: #2d2d2d;
                color: #fff;
                padding: 6px;
                border: none;
                font-weight: bold;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 6px;
            }
        """)
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.setAlternatingRowColors(True)
        self.positions_table.setMinimumHeight(140)
        self.positions_table.setMaximumHeight(180)
        positions_layout.addWidget(self.positions_table)
        
        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)
        
        # --- Alert Log ---
        alerts_group = QGroupBox("🔔 Registro de Alertas")
        alerts_layout = QVBoxLayout()
        alerts_layout.setContentsMargins(8, 12, 8, 8)
        
        self.alerts_log = QTextEdit()
        self.alerts_log.setReadOnly(True)
        self.alerts_log.setMinimumHeight(120)
        self.alerts_log.setMaximumHeight(140)
        self.alerts_log.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #fff;
                border: 1px solid #444;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                line-height: 1.4;
            }
        """)
        alerts_layout.addWidget(self.alerts_log)
        
        alerts_group.setLayout(alerts_layout)
        layout.addWidget(alerts_group)
        
        # --- Live Chart ---
        chart_group = QGroupBox("📊 Gráfico en Tiempo Real")
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(8, 12, 8, 8)
        
        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(280)
        chart_layout.addWidget(self.chart_view)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        
        # Initialize empty chart
        self.update_live_chart([])
        
        return container
    
    def apply_modern_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #ffffff;
                font-size: 13px;
            }
            QLabel {
                color: #cccccc;
            }
            QPushButton {
                min-height: 32px;
            }
        """)
    
    # === SLOT HANDLERS ===
    
    def on_start_trading(self):
        """Start live trading monitor"""
        try:
            mode = self.mode_combo.currentText()
            
            if "PRODUCCIÓN" in mode:
                # Production mode - show warning
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.warning(
                    self,
                    "⚠️ MODO PRODUCCIÓN",
                    "Está a punto de activar trading en VIVO con dinero real.\n\n"
                    "¿Está seguro de continuar?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    return
            
            # Start monitor thread
            self.monitor_thread = LiveMonitorThread()
            self.monitor_thread.pnl_update.connect(self.update_pnl)
            self.monitor_thread.metrics_update.connect(self.update_metrics)
            self.monitor_thread.position_update.connect(self.update_positions)
            self.monitor_thread.alert_triggered.connect(self.handle_alert)
            self.monitor_thread.start()
            
            # Update UI state
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.stop_all_btn.setEnabled(True)
            self.kill_switch_btn.setEnabled(True)
            self.mode_combo.setEnabled(False)
            
            self.connection_status.setText("🟢 Connected")
            self.connection_status.setStyleSheet("color: #4ec9b0; font-size: 12px; padding: 6px 12px; background: #2d2d2d; border-radius: 4px;")
            
            self.update_status_bar(f"✅ Trading activo en modo: {mode}", "success")
            self.add_alert(f"Sistema iniciado en {mode}", "info")
            
            self.status_update.emit(f"Trading iniciado: {mode}", "success")
            
        except Exception as e:
            self.update_status_bar(f"❌ Error al iniciar: {str(e)}", "error")
            self.status_update.emit(f"Error: {str(e)}", "error")
    
    def on_stop_trading(self):
        """Stop live trading"""
        try:
            if self.monitor_thread and self.monitor_thread.isRunning():
                self.monitor_thread.stop()
                self.monitor_thread.wait(3000)  # Wait up to 3 seconds
            
            # Update UI state
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_all_btn.setEnabled(False)
            self.kill_switch_btn.setEnabled(False)
            self.mode_combo.setEnabled(True)
            
            self.connection_status.setText("⚫ Disconnected")
            self.connection_status.setStyleSheet("color: #888; font-size: 12px; padding: 6px 12px; background: #2d2d2d; border-radius: 4px;")
            
            self.update_status_bar("⏹ Trading detenido correctamente", "info")
            self.add_alert("Sistema detenido por usuario", "info")
            
            self.status_update.emit("Trading detenido", "info")
            
        except Exception as e:
            self.update_status_bar(f"❌ Error al detener: {str(e)}", "error")
    
    def on_pause_trading(self):
        """Pause trading (keep monitoring but stop new orders)"""
        self.update_status_bar("⏸ Trading pausado - Monitoreando posiciones existentes", "warning")
        self.add_alert("Trading pausado - No se abrirán nuevas posiciones", "warning")
        self.status_update.emit("Trading pausado", "warning")
    
    def on_stop_all(self):
        """Emergency stop - close all positions"""
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.critical(
            self,
            "🛑 STOP ALL POSITIONS",
            "¿Cerrar TODAS las posiciones activas inmediatamente?\n\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.update_status_bar("🛑 Cerrando todas las posiciones...", "error")
            self.add_alert("🛑 STOP ALL ejecutado - Cerrando posiciones", "error")
            self.status_update.emit("STOP ALL ejecutado", "error")
            
            # Clear positions table
            self.positions_table.setRowCount(0)
    
    def on_kill_switch(self):
        """Ultimate emergency - stop everything and disconnect"""
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.critical(
            self,
            "☠ KILL SWITCH",
            "⚠️ ACTIVAR KILL SWITCH ⚠️\n\n"
            "Esto detendrá INMEDIATAMENTE:\n"
            "- Todas las posiciones abiertas\n"
            "- Todas las órdenes pendientes\n"
            "- Conexión al broker\n"
            "- Sistema de trading\n\n"
            "¿CONFIRMAR KILL SWITCH?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.update_status_bar("☠ KILL SWITCH ACTIVADO - Sistema detenido", "error")
            self.add_alert("☠☠☠ KILL SWITCH ACTIVADO ☠☠☠", "error")
            self.status_update.emit("KILL SWITCH activado", "error")
            
            # Stop trading
            self.on_stop_trading()
    
    # === UPDATE METHODS ===
    
    def update_pnl(self, pnl):
        """Update P&L gauge"""
        self.pnl_gauge.setValue(pnl)
    
    def update_metrics(self, metrics):
        """Update risk metrics cards"""
        sharpe = metrics.get('sharpe', 0)
        max_dd = metrics.get('max_dd', 0)
        win_rate = metrics.get('win_rate', 0)
        exposure = metrics.get('exposure', 0)
        
        self.sharpe_card.update_value(f"{sharpe:.2f}")
        self.dd_card.update_value(f"{max_dd:.1f}")
        self.win_rate_card.update_value(f"{win_rate:.1f}")
        self.exposure_card.update_value(f"{exposure:.1f}")
    
    def update_positions(self, positions):
        """Update positions table"""
        self.positions_table.setRowCount(len(positions))
        
        for i, pos in enumerate(positions):
            self.positions_table.setItem(i, 0, QTableWidgetItem(pos['symbol']))
            
            # Side with color
            side_item = QTableWidgetItem(pos['side'])
            side_item.setForeground(QColor("#4ec9b0" if pos['side'] == "LONG" else "#f48771"))
            self.positions_table.setItem(i, 1, side_item)
            
            self.positions_table.setItem(i, 2, QTableWidgetItem(f"{pos['size']:.2f}"))
            self.positions_table.setItem(i, 3, QTableWidgetItem(f"${pos['entry']:,.2f}"))
            self.positions_table.setItem(i, 4, QTableWidgetItem(f"${pos['current']:,.2f}"))
            
            # P&L with color
            pnl_item = QTableWidgetItem(f"${pos['pnl']:,.2f}")
            pnl_item.setForeground(QColor("#4ec9b0" if pos['pnl'] > 0 else "#f48771"))
            self.positions_table.setItem(i, 5, pnl_item)
            
            pnl_pct = (pos['pnl'] / (pos['entry'] * pos['size'])) * 100
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            pnl_pct_item.setForeground(QColor("#4ec9b0" if pnl_pct > 0 else "#f48771"))
            self.positions_table.setItem(i, 6, pnl_pct_item)
    
    def handle_alert(self, message, level):
        """Handle incoming alerts"""
        self.add_alert(message, level)
        
        # Check if alert should trigger emergency action
        if "crítico" in message.lower():
            self.update_status_bar(f"⚠️ {message}", "error")
    
    def add_alert(self, message, level):
        """Add alert to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on level
        color_map = {
            'error': '#f48771',
            'warning': '#dcdcaa',
            'info': '#569cd6',
            'success': '#4ec9b0'
        }
        color = color_map.get(level, '#cccccc')
        
        html = f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        self.alerts_log.append(html)
        
        # Auto-scroll to bottom
        self.alerts_log.verticalScrollBar().setValue(
            self.alerts_log.verticalScrollBar().maximum()
        )
    
    def update_status_bar(self, message, status_type):
        """Update status bar with color coding"""
        color_map = {
            'success': ('#4ec9b0', '#4ec9b0'),
            'error': ('#f48771', '#f48771'),
            'warning': ('#dcdcaa', '#dcdcaa'),
            'info': ('#569cd6', '#569cd6'),
            'processing': ('#c586c0', '#c586c0')
        }
        
        text_color, border_color = color_map.get(status_type, ('#888', '#888'))
        
        self.status_bar.setText(message)
        self.status_bar.setStyleSheet(f"""
            QLabel {{
                background: #2d2d2d;
                color: {text_color};
                padding: 12px;
                border-radius: 4px;
                border-left: 4px solid {border_color};
                font-size: 13px;
            }}
        """)
    
    def update_live_chart(self, price_data):
        """Update live price chart"""
        try:
            # Create sample data if empty
            if not price_data:
                import numpy as np
                x = list(range(50))
                y = 40000 + np.cumsum(np.random.randn(50) * 100)
                price_data = list(zip(x, y))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[p[0] for p in price_data],
                y=[p[1] for p in price_data],
                mode='lines',
                name='Price',
                line=dict(color='#0e639c', width=2)
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#ffffff', size=11),
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis=dict(
                    title='Time',
                    gridcolor='#2d2d2d',
                    showgrid=True
                ),
                yaxis=dict(
                    title='Price (USD)',
                    gridcolor='#2d2d2d',
                    showgrid=True
                ),
                hovermode='x unified',
                height=280
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.chart_view.setHtml(html)
            
        except Exception as e:
            print(f"Error updating chart: {e}")
    
    def on_tab_activated(self):
        """Called when tab becomes active"""
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.add_alert("Tab activada - Sistema operativo", "info")
