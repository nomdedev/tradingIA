
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGridLayout,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QTextEdit,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QScrollArea,
    QProgressBar,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, Slot
from PySide6.QtGui import QColor, QFont, QIcon
import random
from datetime import datetime
import json
import logging
from core.data.realtime_provider import MockRealTimeProvider
from src.gui.styles import DarkTheme

# ============================================================================
# HELP DIALOG
# ============================================================================
class HelpDialog(QDialog):
    """Dialog showing help for the Live tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayuda - Live Trading Monitor")
        self.setMinimumSize(700, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.setStyleSheet(f"background-color: {DarkTheme.BG_PRIMARY}; color: {DarkTheme.TEXT_HIGHLIGHT};")

        title = QLabel("📚 Guía de Uso - Live Trading Monitor")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {DarkTheme.SUCCESS}; margin-bottom: 10px;")
        layout.addWidget(title)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet(
            f"""
            QTextEdit {{
                background: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_PRIMARY};
                border: 1px solid {DarkTheme.BG_HOVER};
                border-radius: 4px;
                padding: 12px;
                font-size: 14px;
            }}
        """
        )

        help_content = (
            "<h2 style='color: #4ec9b0;'>🎯 ¿Qué hace esta pestaña?</h2>"
            "<p>Esta pestaña te permite <b>ejecutar trading automático en MODO SIMULADO (Paper Trading)</b> usando la API de Alpaca.</p>"
            "<p><b style='color: #f48771;'>IMPORTANTE:</b> NO se ejecuta trading real. Todo es simulación para probar estrategias sin riesgo.</p>"
            "<hr style='border: 1px solid #444; margin: 15px 0;'>"
            "<h2 style='color: #569cd6;'>📊 Elementos de la Interfaz</h2>"
            "<h3 style='color: #dcdcaa;'>1. Panel Izquierdo - Configuración</h3>"
            "<ul>"
            "<li><b>Selector de Ticker:</b> Elige qué activo tradear (BTC/USD, ETH/USD, AAPL, etc.)</li>"
            "<li><b>Selector de Estrategia:</b> Escoge la estrategia de trading (RSI, MACD, etc.)</li>"
            "<li><b>Información de Estrategia:</b> Muestra parámetros actuales de la estrategia</li>"
            "</ul>"
            "<h3 style='color: #dcdcaa;'>2. Panel Central - Métricas</h3>"
            "<ul>"
            "<li><b style='color: #4ec9b0;'>P&L:</b> Ganancia/pérdida del día. Verde = ganancia, Rojo = pérdida</li>"
            "<li><b style='color: #569cd6;'>Sharpe Ratio:</b> Relación riesgo/retorno. Mayor que 1.5 es bueno</li>"
            "<li><b style='color: #f48771;'>Max Drawdown:</b> Peor caída desde el pico. Entre -5% y -10% es aceptable</li>"
            "<li><b style='color: #4ec9b0;'>Win Rate:</b> Porcentaje de trades ganadores. Mayor a 55% es bueno</li>"
            "</ul>"
            "<h3 style='color: #dcdcaa;'>3. Panel Derecho - Decisiones</h3>"
            "<ul>"
            "<li><b>Registro de Decisiones:</b> Log en tiempo real que explica cada decisión del bot</li>"
            "<li>Muestra: timestamp, acción (BUY/SELL/HOLD), razón, e indicadores usados</li>"
            "</ul>"
            "<hr style='border: 1px solid #444; margin: 15px 0;'>"
            "<h2 style='color: #4ec9b0;'>🚀 Cómo Usar</h2>"
            "<h3 style='color: #dcdcaa;'>Paso 1: Configurar</h3>"
            "<ol>"
            "<li>Selecciona el <b>ticker</b> que quieres tradear (ej: BTC/USD)</li>"
            "<li>Elige una <b>estrategia</b> del dropdown</li>"
            "<li>Revisa los <b>parámetros</b> mostrados</li>"
            "<li>Haz clic en <b>Cargar Estrategia</b> para aplicar cambios</li>"
            "</ol>"
            "<h3 style='color: #dcdcaa;'>Paso 2: Iniciar Trading</h3>"
            "<ol>"
            "<li>Verifica que el modo sea <b>Paper Trading</b> (simulación)</li>"
            "<li>Haz clic en <b>▶ START TRADING</b></li>"
            "<li>Observa cómo el bot toma decisiones en tiempo real</li>"
            "</ol>"
            "<h3 style='color: #dcdcaa;'>Paso 3: Detener</h3>"
            "<p>Haz clic en <b>⏹ STOP TRADING</b> para detener el bot y cerrar posiciones abiertas (opcional).</p>"
        )

        help_text.setHtml(help_content)
        layout.addWidget(help_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        buttons.setStyleSheet(
            """
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """
        )
        layout.addWidget(buttons)


class Tab6LiveMonitor(QWidget):
    """
    Improved Live Trading Monitor Tab
    """
    
    # Signal to update UI from background threads
    price_update_signal = Signal(float)

    def __init__(self, parent_platform, backtester_core):
        print("DEBUG: Tab6 __init__ called")
        super().__init__()
        self.parent_platform = parent_platform
        self.backtester = backtester_core
        self.logger = logging.getLogger(__name__)
        
        # State
        self.is_running = False
        self.selected_ticker = "BTC/USD"
        self.selected_strategy = "RSI_Bollinger"
        self.paper_trading_mode = True
        
        # Real-time provider
        self.data_provider = MockRealTimeProvider(interval_sec=1.0)
        self.data_provider.subscribe(self.on_market_data)
        self.price_update_signal.connect(self.on_price_update)
        
        # Mock data for simulation
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)

        self.init_ui()
        
    def on_market_data(self, data):
        """Callback for real-time data (runs in background thread)"""
        if data['type'] == 'ticker' and data['symbol'] == self.selected_ticker:
            # Emit signal to update UI in main thread
            self.price_update_signal.emit(data['price'])
            
    @Slot(float)
    def on_price_update(self, price):
        """Handle price update in main thread"""
        if hasattr(self, 'metric_pnl'):
             self.metric_pnl.subtext_lbl.setText(f"Current Price: ${price:,.2f}")
             
             # Simple logic to update PnL based on price movement (Simulated)
             # In a real app, this would calculate based on open positions
             if self.is_running:
                 change = (random.random() - 0.5) * 10
                 current_pnl_text = self.metric_pnl.value_lbl.text().replace('$', '').replace(',', '')
                 try:
                     current_pnl = float(current_pnl_text)
                 except (ValueError, AttributeError):
                     current_pnl = 0.0
                 
                 new_pnl = current_pnl + change
                 self.metric_pnl.value_lbl.setText(f"${new_pnl:,.2f}")
                 self.metric_pnl.value_lbl.setStyleSheet(
                    f"color: {'#4ec9b0' if new_pnl >= 0 else '#f48771'}; font-size: 24px; font-weight: bold;"
                )
        
    def init_ui(self):
        """Initialize the UI"""
        print(f"DEBUG: init_ui called. Current layout: {self.layout()}")
        # Check if layout already exists to avoid re-initialization
        if self.layout():
            print("DEBUG: Layout exists, returning")
            return

        print("DEBUG: Creating layout")
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter for resizable layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"""
            QSplitter::handle {{
                background-color: {DarkTheme.BG_HOVER};
            }}
        """
        )

        # 1. Left Panel: Configuration
        config_panel = self.create_config_panel()
        splitter.addWidget(config_panel)

        # 2. Right Panel: Monitor & Logs
        monitor_panel = self.create_monitor_panel()
        splitter.addWidget(monitor_panel)

        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([350, 850])
        splitter.setCollapsible(0, False)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_config_panel(self):
        """Create the left configuration panel"""
        panel = QFrame()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(400)
        panel.setStyleSheet(f"background-color: {DarkTheme.BG_SECONDARY}; border-right: 1px solid {DarkTheme.BG_HOVER};")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header_layout = QHBoxLayout()
        
        help_btn = QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        help_btn.setToolTip("Show Help Guide")
        help_btn.clicked.connect(self.show_help)
        help_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {DarkTheme.BG_HOVER};
                color: {DarkTheme.TEXT_HIGHLIGHT};
                border-radius: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #4e4e4e;
            }}
        """
        )
        header_layout.addStretch()
        header_layout.addWidget(help_btn)
        layout.addLayout(header_layout)

        # Ticker Selection
        ticker_group = QGroupBox("Asset Selection")
        ticker_group.setStyleSheet(self.get_group_style())
        ticker_layout = QVBoxLayout()
        
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(["BTC/USD", "ETH/USD", "SOL/USD", "AAPL", "TSLA", "SPY"])
        self.ticker_combo.currentTextChanged.connect(self.on_ticker_changed)
        self.ticker_combo.setStyleSheet(self.get_combo_style())
        ticker_layout.addWidget(self.ticker_combo)
        
        ticker_group.setLayout(ticker_layout)
        layout.addWidget(ticker_group)

        # Strategy Selection
        strategy_group = QGroupBox("Strategy Selection")
        strategy_group.setStyleSheet(self.get_group_style())
        strategy_layout = QVBoxLayout()
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["RSI_Bollinger", "MACD_Crossover", "MovingAverage_Cross", "SuperTrend"])
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        self.strategy_combo.setStyleSheet(self.get_combo_style())
        strategy_layout.addWidget(self.strategy_combo)
        
        # Strategy Description
        self.strategy_desc = QLabel("RSI + Bollinger Bands strategy for mean reversion.")
        self.strategy_desc.setWordWrap(True)
        self.strategy_desc.setStyleSheet("color: #aaaaaa; font-style: italic; margin-top: 5px;")
        strategy_layout.addWidget(self.strategy_desc)
        
        strategy_group.setLayout(strategy_layout)
        layout.addWidget(strategy_group)

        # Trading Mode
        mode_group = QGroupBox("Trading Mode")
        mode_group.setStyleSheet(self.get_group_style())
        mode_layout = QVBoxLayout()
        
        self.mode_label = QLabel("🟢 PAPER TRADING (Simulation)")
        self.mode_label.setStyleSheet("color: #4ec9b0; font-weight: bold; font-size: 14px;")
        mode_layout.addWidget(self.mode_label)
        
        mode_desc = QLabel("Executes trades in a simulated environment. No real funds are used.")
        mode_desc.setWordWrap(True)
        mode_desc.setStyleSheet("color: #888888; font-size: 12px;")
        mode_layout.addWidget(mode_desc)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        layout.addStretch()

        # Control Buttons
        self.start_btn = QPushButton("▶ START TRADING")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setFixedHeight(50)
        self.start_btn.clicked.connect(self.toggle_trading)
        self.start_btn.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4ec9b0, stop:1 #3aa890);
                color: #1e1e1e;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: #5fdcd0;
            }
            QPushButton:pressed {
                background: #2d9680;
            }
        """
        )
        layout.addWidget(self.start_btn)

        return panel

    def create_monitor_panel(self):
        """Create the right monitor panel"""
        panel = QFrame()
        panel.setStyleSheet("background-color: #1e1e1e;")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Status Bar
        status_layout = QHBoxLayout()
        
        self.status_indicator = QLabel("● STOPPED")
        self.status_indicator.setStyleSheet("color: #f48771; font-weight: bold; font-size: 16px;")
        status_layout.addWidget(self.status_indicator)
        
        status_layout.addStretch()
        
        # Latency Indicator
        self.latency_label = QLabel("Latency: -- ms")
        self.latency_label.setStyleSheet("color: #888888; font-family: monospace;")
        status_layout.addWidget(self.latency_label)
        
        status_layout.addSpacing(20)

        self.last_update_label = QLabel("Last Update: --:--:--")
        self.last_update_label.setStyleSheet("color: #888888;")
        status_layout.addWidget(self.last_update_label)
        
        layout.addLayout(status_layout)

        # Metrics Grid
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(15)
        
        self.metric_pnl = self.create_metric_card("Daily P&L", "$0.00", "0.00%")
        self.metric_sharpe = self.create_metric_card("Sharpe Ratio", "0.00", "")
        self.metric_dd = self.create_metric_card("Max Drawdown", "0.00%", "")
        self.metric_winrate = self.create_metric_card("Win Rate", "0.00%", "0/0 Trades")
        
        metrics_layout.addWidget(self.metric_pnl, 0, 0)
        metrics_layout.addWidget(self.metric_sharpe, 0, 1)
        metrics_layout.addWidget(self.metric_dd, 0, 2)
        metrics_layout.addWidget(self.metric_winrate, 0, 3)
        
        layout.addLayout(metrics_layout)

        # Splitter for Charts and Logs
        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.setHandleWidth(1)
        content_splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #3e3e3e;
            }
        """
        )

        # Active Positions
        positions_group = QGroupBox("Active Positions")
        positions_group.setStyleSheet(self.get_group_style())
        positions_layout = QVBoxLayout()
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels(["Symbol", "Side", "Entry Price", "Current Price", "P&L"])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.positions_table.verticalHeader().setVisible(False)
        self.positions_table.setStyleSheet(self.get_table_style())
        positions_layout.addWidget(self.positions_table)
        
        positions_group.setLayout(positions_layout)
        content_splitter.addWidget(positions_group)

        # Decision Log
        log_group = QGroupBox("Decision Log")
        log_group.setStyleSheet(self.get_group_style())
        log_layout = QVBoxLayout()
        
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(5) # Added Indicators column
        self.log_table.setHorizontalHeaderLabels(["Time", "Action", "Price", "Reason", "Indicators"])
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setStyleSheet(self.get_table_style())
        log_layout.addWidget(self.log_table)
        
        log_group.setLayout(log_layout)
        content_splitter.addWidget(log_group)
        
        layout.addWidget(content_splitter)

        return panel

    def create_metric_card(self, title, value, subtext):
        """Create a styled metric card"""
        card = QFrame()
        card.setStyleSheet(
            """
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
            }
        """
        )
        layout = QVBoxLayout(card)
        
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(title_lbl)
        
        value_lbl = QLabel(value)
        value_lbl.setStyleSheet("color: #ffffff; font-size: 24px; font-weight: bold;")
        layout.addWidget(value_lbl)
        
        subtext_lbl = QLabel(subtext)
        subtext_lbl.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        layout.addWidget(subtext_lbl)
        
        # Store references for updates
        card.value_lbl = value_lbl
        card.subtext_lbl = subtext_lbl
        
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

    def get_combo_style(self):
        return """
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 5px;
            }
        """

    def get_table_style(self):
        return """
            QTableWidget {
                background-color: #1e1e1e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                color: #cccccc;
                gridline-color: #3e3e3e;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 5px;
                border: none;
                border-bottom: 1px solid #3e3e3e;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """

    def show_help(self):
        """Show help dialog"""
        dialog = HelpDialog(self)
        dialog.exec()

    def on_ticker_changed(self, ticker):
        """Handle ticker change"""
        self.selected_ticker = ticker
        self.logger.info(f"Selected ticker: {ticker}")

    def on_strategy_changed(self, strategy):
        """Handle strategy change"""
        self.selected_strategy = strategy
        self.logger.info(f"Selected strategy: {strategy}")
        
        # Update description based on strategy
        descriptions = {
            "RSI_Bollinger": "RSI + Bollinger Bands strategy for mean reversion.",
            "MACD_Crossover": "Classic trend following strategy using MACD line crossovers.",
            "MovingAverage_Cross": "Golden Cross / Death Cross strategy using SMA 50/200.",
            "SuperTrend": "Trend following strategy using ATR-based SuperTrend indicator."
        }
        self.strategy_desc.setText(descriptions.get(strategy, "Custom strategy configuration."))

    def toggle_trading(self):
        """Start or stop trading"""
        if not self.is_running:
            # Start
            self.is_running = True
            self.start_btn.setText("⏹ STOP TRADING")
            self.start_btn.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f48771, stop:1 #d65d45);
                    color: #1e1e1e;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                    border: none;
                }
                QPushButton:hover {
                    background: #ff9d8a;
                }
                QPushButton:pressed {
                    background: #c44b35;
                }
            """
            )
            self.status_indicator.setText("● RUNNING")
            self.status_indicator.setStyleSheet("color: #4ec9b0; font-weight: bold; font-size: 16px;")
            
            # Start data stream
            self.data_provider.start([self.selected_ticker])
            
            # Start simulation timer (for trades/logs)
            self.simulation_timer.start(2000)  # Update every 2 seconds
            self.add_log_entry("SYSTEM", "Trading started", f"Strategy: {self.selected_strategy} on {self.selected_ticker}", "-")
            
        else:
            # Stop
            self.is_running = False
            self.start_btn.setText("▶ START TRADING")
            self.start_btn.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4ec9b0, stop:1 #3aa890);
                    color: #1e1e1e;
                    font-size: 16px;
                    font-weight: bold;
                    border-radius: 8px;
                    border: none;
                }
                QPushButton:hover {
                    background: #5fdcd0;
                }
                QPushButton:pressed {
                    background: #2d9680;
                }
            """
            )
            self.status_indicator.setText("● STOPPED")
            self.status_indicator.setStyleSheet("color: #f48771; font-weight: bold; font-size: 16px;")
            
            # Stop data stream
            self.data_provider.stop()
            
            # Stop simulation timer
            self.simulation_timer.stop()
            self.add_log_entry("SYSTEM", "Trading stopped", "User requested stop", "-")

    def update_simulation(self):
        """Update simulated data (Trades and Logs)"""
        # Update timestamp
        now = datetime.now().strftime("%H:%M:%S")
        self.last_update_label.setText(f"Last Update: {now}")
        
        # Update Latency (Simulated)
        latency = random.randint(15, 120)
        self.latency_label.setText(f"Latency: {latency} ms")
        if latency < 50:
            self.latency_label.setStyleSheet("color: #4ec9b0; font-family: monospace;")
        elif latency < 100:
            self.latency_label.setStyleSheet("color: #dcdcaa; font-family: monospace;")
        else:
            self.latency_label.setStyleSheet("color: #f48771; font-family: monospace;")
        
        # Randomly generate a trade or update
        if random.random() < 0.1:  # 10% chance of action
            action = random.choice(["BUY", "SELL"])
            # Use real price if available, else random
            price = getattr(self, 'latest_price', random.uniform(30000, 60000))
            if hasattr(self, 'metric_pnl'):
                 # Parse current price from label if latest_price not set
                 pass
                 
            reason = f"Signal from {self.selected_strategy}"
            
            # Generate fake indicators
            rsi = random.randint(20, 80)
            bb_pos = random.choice(["Upper", "Lower", "Middle"])
            indicators = f"RSI: {rsi} | BB: {bb_pos}"
            
            self.add_log_entry(action, f"${price:,.2f}", reason, indicators)

    def add_log_entry(self, action, price, reason, indicators):
        """Add entry to log table"""
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        
        time_item = QTableWidgetItem(datetime.now().strftime("%H:%M:%S"))
        action_item = QTableWidgetItem(action)
        price_item = QTableWidgetItem(price)
        reason_item = QTableWidgetItem(reason)
        indicators_item = QTableWidgetItem(indicators)
        
        # Color code action
        if action == "BUY":
            action_item.setForeground(QColor("#4ec9b0"))
        elif action == "SELL":
            action_item.setForeground(QColor("#f48771"))
        elif action == "SYSTEM":
            action_item.setForeground(QColor("#569cd6"))
            
        self.log_table.setItem(row, 0, time_item)
        self.log_table.setItem(row, 1, action_item)
        self.log_table.setItem(row, 2, price_item)
        self.log_table.setItem(row, 3, reason_item)
        self.log_table.setItem(row, 4, indicators_item)
        
        self.log_table.scrollToBottom()
