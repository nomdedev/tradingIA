"""
Tab 6 - Live Trading Monitor (User-Friendly Version)
With help panels, ticker selection, and clear explanations
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QTextEdit, QComboBox, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor, QFont
import random
from datetime import datetime
import json


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
        
        title = QLabel("📚 Guía de Uso - Live Trading Monitor")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #4ec9b0; margin-bottom: 10px;")
        layout.addWidget(title)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #ccc;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 12px;
                font-size: 14px;
            }
        """)
        
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
            "<li>El indicador cambiará a 🟢 <b>EN VIVO</b></li>"
            "</ol>"
            "<h3 style='color: #dcdcaa;'>Paso 3: Monitorear</h3>"
            "<ol>"
            "<li>Observa el <b>P&L</b> en el centro</li>"
            "<li>Revisa las <b>métricas</b> para evaluar rendimiento</li>"
            "<li>Lee el <b>log de decisiones</b> para entender qué hace el bot</li>"
            "</ol>"
            "<hr style='border: 1px solid #444; margin: 15px 0;'>"
            "<h2 style='color: #f48771;'>⚠️ Advertencias</h2>"
            "<ul>"
            "<li><b>SIMULACIÓN SOLAMENTE:</b> No se pierde dinero real</li>"
            "<li><b>Datos de Alpaca:</b> Precios reales pero operaciones simuladas</li>"
            "<li><b>No es consejo financiero:</b> Solo educativo</li>"
            "</ul>"
            "<h2 style='color: #4ec9b0;'>❓ Preguntas Frecuentes</h2>"
            "<p><b>P: ¿Por qué aparecen tickers que no seleccioné?</b></p>"
            "<p>R: Asegúrate de seleccionar el ticker y hacer clic en Cargar Estrategia antes de iniciar.</p>"
            "<p><b>P: ¿Los datos son en tiempo real?</b></p>"
            "<p>R: Sí, cuando estás conectado (🟢 EN VIVO), los precios vienen de Alpaca.</p>"
            "<p><b>P: ¿Puedo perder dinero real?</b></p>"
            "<p>R: NO. Todo es simulación.</p>"
        )
        
        help_text.setHtml(help_content)
        layout.addWidget(help_text)
        
        # Close button
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)


# ============================================================================
# METRIC CARD COMPONENT
# ============================================================================
class MetricCard(QFrame):
    """Enhanced metric card with tooltip"""
    def __init__(self, title, value="--", unit="", color="#569cd6", tooltip=""):
        super().__init__()
        self.color = color
        
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: #2d2d2d;
                border-left: 4px solid {color};
                border-radius: 6px;
                padding: 12px;
                min-height: 80px;
            }}
        """)
        
        if tooltip:
            self.setToolTip(tooltip)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 14px; font-weight: normal;")
        layout.addWidget(title_label)
        
        # Value container
        value_layout = QHBoxLayout()
        value_layout.setContentsMargins(0, 4, 0, 0)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")
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
            self.value_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")


# ============================================================================
# TICKER SELECTOR
# ============================================================================
class TickerSelector(QGroupBox):
    """Ticker selection panel"""
    ticker_changed = Signal(str)
    
    def __init__(self):
        super().__init__("🎯 Selección de Activo")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 15, 12, 12)
        layout.setSpacing(10)
        
        # Info label
        info = QLabel("Selecciona qué activo quieres tradear:")
        info.setStyleSheet("color: #888; font-size: 13px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Ticker dropdown
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems([
            "BTC/USD - Bitcoin",
            "ETH/USD - Ethereum",
            "AAPL - Apple",
            "TSLA - Tesla",
            "SPY - S&P 500 ETF",
            "QQQ - Nasdaq 100 ETF"
        ])
        self.ticker_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d2d;
                color: #fff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QComboBox:hover { border: 1px solid #777; }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #888;
                margin-right: 8px;
            }
        """)
        self.ticker_combo.currentTextChanged.connect(self._on_ticker_changed)
        layout.addWidget(self.ticker_combo)
        
        # Current ticker display
        self.current_ticker = QLabel("Activo actual: BTC/USD")
        self.current_ticker.setStyleSheet("color: #4ec9b0; font-size: 14px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(self.current_ticker)
    
    def _on_ticker_changed(self, text):
        ticker = text.split(" - ")[0]
        self.current_ticker.setText(f"Activo actual: {ticker}")
        self.ticker_changed.emit(ticker)
    
    def get_current_ticker(self):
        """Get currently selected ticker"""
        return self.ticker_combo.currentText().split(" - ")[0]


# ============================================================================
# STRATEGY INFO PANEL
# ============================================================================
class StrategyInfoPanel(QGroupBox):
    """Panel showing current strategy information"""
    def __init__(self):
        super().__init__("🎯 Estrategia Activa")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 15, 12, 12)
        layout.setSpacing(8)
        
        # Strategy name
        name_layout = QHBoxLayout()
        name_label = QLabel("Estrategia:")
        name_label.setStyleSheet("color: #888; font-size: 14px; min-width: 80px;")
        self.strategy_name = QLabel("No activa")
        self.strategy_name.setStyleSheet("color: #4ec9b0; font-size: 15px; font-weight: bold;")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.strategy_name)
        name_layout.addStretch()
        layout.addLayout(name_layout)
        
        # Description
        desc_label = QLabel("Descripción:")
        desc_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(desc_label)
        
        self.strategy_desc = QLabel("No hay estrategia cargada")
        self.strategy_desc.setStyleSheet("color: #ccc; font-size: 11px; padding: 8px; background: #1e1e1e; border-radius: 4px;")
        self.strategy_desc.setWordWrap(True)
        layout.addWidget(self.strategy_desc)
        
        # Parameters
        params_label = QLabel("Parámetros:")
        params_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 8px;")
        layout.addWidget(params_label)
        
        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setMaximumHeight(120)
        self.params_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #dcdcaa;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        self.params_text.setPlainText("No hay parámetros disponibles")
        layout.addWidget(self.params_text)
    
    def update_strategy(self, name, description, parameters):
        """Update strategy information"""
        self.strategy_name.setText(name)
        self.strategy_desc.setText(description)
        
        # Format parameters as JSON
        params_formatted = json.dumps(parameters, indent=2, ensure_ascii=False)
        self.params_text.setPlainText(params_formatted)


# ============================================================================
# DATA SOURCE INDICATOR
# ============================================================================
class DataSourceIndicator(QFrame):
    """Clear indicator of data source"""
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # Title
        title = QLabel("📡 Fuente de Datos")
        title.setStyleSheet("color: #888; font-size: 11px; font-weight: normal;")
        layout.addWidget(title)
        
        # Status
        status_layout = QHBoxLayout()
        self.status_indicator = QLabel("⚫")
        self.status_indicator.setStyleSheet("font-size: 16px;")
        status_layout.addWidget(self.status_indicator)
        
        self.status_text = QLabel("Desconectado")
        self.status_text.setStyleSheet("color: #888; font-size: 13px; font-weight: bold;")
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        # Details
        self.details_label = QLabel("Esperando conexión...")
        self.details_label.setStyleSheet("color: #888; font-size: 10px;")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
    
    def set_live_mode(self, is_live=True, provider="Alpaca Paper Trading"):
        """Update to show live data mode"""
        if is_live:
            self.status_indicator.setText("🟢")
            self.status_text.setText("EN VIVO")
            self.status_text.setStyleSheet("color: #4ec9b0; font-size: 13px; font-weight: bold;")
            self.details_label.setText(f"Conectado a {provider}\nPrecios en tiempo real actualizándose cada 5 segundos")
        else:
            self.status_indicator.setText("🔴")
            self.status_text.setText("DESCONECTADO")
            self.status_text.setStyleSheet("color: #f48771; font-size: 13px; font-weight: bold;")
            self.details_label.setText("No hay conexión activa con el mercado")


# ============================================================================
# DECISION LOG PANEL
# ============================================================================
class DecisionLogPanel(QGroupBox):
    """Panel showing bot's decision-making process"""
    def __init__(self):
        super().__init__("🤖 Registro de Decisiones")
        self.setToolTip("Este log muestra en tiempo real cada decisión que toma el bot,\nla razón detrás de ella, y los valores de indicadores usados.")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 15, 12, 12)
        
        # Info label
        info = QLabel("📝 Aquí verás cada decisión del bot con su razón y los indicadores que la motivaron:")
        info.setStyleSheet("color: #888; font-size: 10px; margin-bottom: 8px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #ccc;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.log_text)
        
        # Clear button
        clear_btn = QPushButton("🗑 Limpiar Log")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #444;
                color: #ccc;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 10px;
            }
            QPushButton:hover { background: #555; }
        """)
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)
    
    def add_decision(self, timestamp, action, reason, indicators):
        """Add a decision entry to the log"""
        time_str = timestamp.strftime("%H:%M:%S")
        entry = f"[{time_str}] {action}\n"
        entry += f"  Razón: {reason}\n"
        entry += f"  Indicadores: {indicators}\n"
        entry += "-" * 50 + "\n"
        
        self.log_text.append(entry)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# ============================================================================
# STRATEGY SELECTOR
# ============================================================================
class StrategySelector(QGroupBox):
    """Strategy selection"""
    strategy_changed = Signal(str)
    
    def __init__(self):
        super().__init__("🔧 Selector de Estrategia")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 15, 12, 12)
        layout.setSpacing(10)
        
        # Strategy dropdown
        strategy_label = QLabel("Seleccionar estrategia:")
        strategy_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "RSI Mean Reversion",
            "MACD Momentum",
            "Bollinger Bands Breakout",
            "MA Crossover",
            "Volume Breakout"
        ])
        self.strategy_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d2d;
                color: #fff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
            }
            QComboBox:hover { border: 1px solid #777; }
        """)
        self.strategy_combo.currentTextChanged.connect(self.strategy_changed.emit)
        layout.addWidget(self.strategy_combo)
        
        # Load strategy button
        load_btn = QPushButton("📥 Cargar Estrategia")
        load_btn.setStyleSheet("""
            QPushButton {
                background: #569cd6;
                color: #fff;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: #6aaae1; }
        """)
        load_btn.clicked.connect(self.load_selected_strategy)
        layout.addWidget(load_btn)


# ============================================================================
# LIVE MONITOR THREAD
# ============================================================================
class EnhancedLiveMonitorThread(QThread):
    """Background thread for live monitoring"""
    pnl_update = Signal(float)
    metrics_update = Signal(dict)
    position_update = Signal(list)
    decision_made = Signal(dict)
    connection_status = Signal(bool)
    
    def __init__(self, strategy_name="RSI Mean Reversion", ticker="BTC/USD"):
        super().__init__()
        self.running = False
        self.current_pnl = 0.0
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.iteration = 0
        
    def run(self):
        """Main monitoring loop"""
        self.running = True
        self.connection_status.emit(True)
        
        while self.running:
            self.iteration += 1
            
            # Simulate P&L changes
            self.current_pnl += random.uniform(-1.5, 2.0)
            self.current_pnl = max(-50, min(50, self.current_pnl))
            self.pnl_update.emit(self.current_pnl)
            
            # Update metrics every 3 seconds
            if self.iteration % 3 == 0:
                metrics = {
                    'sharpe': round(random.uniform(0.8, 2.2), 2),
                    'max_dd': round(random.uniform(-12, -3), 2),
                    'win_rate': round(random.uniform(48, 68), 1),
                    'exposure': round(random.uniform(35, 75), 1),
                    'trades_today': random.randint(8, 25),
                    'active_positions': random.randint(0, 3)
                }
                self.metrics_update.emit(metrics)
            
            # Simulate trading decisions every 10 seconds
            if self.iteration % 10 == 0:
                decision = {
                    'timestamp': datetime.now(),
                    'action': random.choice(['BUY', 'SELL', 'HOLD']),
                    'reason': self._generate_decision_reason(),
                    'indicators': self._generate_indicator_values()
                }
                self.decision_made.emit(decision)
            
            # Simulate position updates (ONLY for selected ticker)
            if self.iteration % 8 == 0:
                positions = []
                for i in range(random.randint(0, 2)):
                    positions.append({
                        'symbol': self.ticker,  # Use selected ticker
                        'side': random.choice(['LONG', 'SHORT']),
                        'size': round(random.uniform(0.05, 1.5), 3),
                        'entry': round(random.uniform(42000, 45000), 2),
                        'current': round(random.uniform(42000, 45000), 2),
                        'pnl': round(random.uniform(-300, 500), 2),
                        'pnl_pct': round(random.uniform(-2, 3), 2)
                    })
                self.position_update.emit(positions)
            
            self.msleep(1000)
    
    def _generate_decision_reason(self):
        """Generate decision reason"""
        reasons = [
            "RSI sobrevendido (< 30) + MACD cruce alcista",
            "RSI sobrecomprado (> 70) + Divergencia bajista",
            "Precio cerca de banda inferior de Bollinger",
            "Confirmación de tendencia alcista",
            "Stop loss preventivo por volatilidad",
            "Take profit alcanzado (objetivo: 2%)",
            "Volumen aumentando + Rompimiento de resistencia",
            "Señal débil, mantener posición actual"
        ]
        return random.choice(reasons)
    
    def _generate_indicator_values(self):
        """Generate indicator values"""
        return {
            'RSI': round(random.uniform(25, 75), 1),
            'MACD': round(random.uniform(-50, 50), 2),
            'BB_position': round(random.uniform(0, 1), 2),
            'Volume_ratio': round(random.uniform(0.8, 2.5), 2)
        }
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.connection_status.emit(False)


# ============================================================================
# MAIN TAB CLASS
# ============================================================================
class Tab6LiveMonitoringUserFriendly(QWidget):
    """User-friendly Live Trading Monitor"""
    status_update = Signal(str, str)
    
    def __init__(self, parent_platform=None):
        super().__init__()
        self.parent = parent_platform
        self.monitor_thread = None
        self.current_ticker = "BTC/USD"
        self.current_strategy = {
            'name': 'RSI Mean Reversion',
            'description': 'Estrategia de reversión a la media basada en RSI. Compra cuando RSI < 30 y vende cuando RSI > 70.',
            'parameters': {
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'take_profit': 2.0,
                'stop_loss': 1.5
            }
        }
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # === HEADER - Simplified ===
        header_layout = QHBoxLayout()
        
        title = QLabel("🔴 Live Trading Monitor")
        title.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Status indicator
        self.status_indicator = QLabel("🔴 DESCONECTADO")
        self.status_indicator.setStyleSheet("color: #f48771; font-size: 14px; font-weight: bold; padding: 8px 16px; background-color: #2d2d2d; border-radius: 4px;")
        header_layout.addWidget(self.status_indicator)
        
        # Help button
        help_btn = QPushButton("❓ Ayuda")
        help_btn.setStyleSheet("""
            QPushButton {
                background: #569cd6;
                color: #fff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background: #6aaae1; }
        """)
        help_btn.clicked.connect(self.show_help)
        header_layout.addWidget(help_btn)
        
        layout.addLayout(header_layout)
        
        # === CONFIGURATION SECTION ===
        config_group = QGroupBox("⚙️ Configuración")
        config_group.setStyleSheet("""
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
        
        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(12, 12, 12, 12)
        config_layout.setSpacing(20)
        
        # Left: Ticker and Strategy
        left_config = QVBoxLayout()
        left_config.setSpacing(12)
        
        # Ticker selector
        ticker_group = QGroupBox("🎯 Activo")
        ticker_group.setStyleSheet("QGroupBox { border: none; font-weight: bold; color: #4ec9b0; }")
        ticker_layout = QVBoxLayout()
        
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems([
            "BTC/USD - Bitcoin",
            "ETH/USD - Ethereum", 
            "AAPL - Apple",
            "SPY - S&P 500",
            "QQQ - Nasdaq 100"
        ])
        self.ticker_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 13px;
                background: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        self.ticker_combo.currentTextChanged.connect(self.on_ticker_changed)
        ticker_layout.addWidget(self.ticker_combo)
        ticker_group.setLayout(ticker_layout)
        left_config.addWidget(ticker_group)
        
        # Strategy selector
        strategy_group = QGroupBox("🤖 Estrategia")
        strategy_group.setStyleSheet("QGroupBox { border: none; font-weight: bold; color: #c586c0; }")
        strategy_layout = QVBoxLayout()
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "RSI Mean Reversion",
            "MACD Momentum", 
            "Bollinger Bands Breakout",
            "MA Crossover",
            "Volume Breakout"
        ])
        self.strategy_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 13px;
                background: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        strategy_layout.addWidget(self.strategy_combo)
        
        load_btn = QPushButton("📥 Cargar Estrategia")
        load_btn.setStyleSheet("""
            QPushButton {
                background: #569cd6;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background: #6aaae1; }
        """)
        load_btn.clicked.connect(self.load_selected_strategy)
        strategy_layout.addWidget(load_btn)
        
        strategy_group.setLayout(strategy_layout)
        left_config.addWidget(strategy_group)
        
        config_layout.addLayout(left_config)
        
        # Center: Mode and Strategy Info
        center_config = QVBoxLayout()
        
        # Trading Mode
        mode_group = QGroupBox("🎭 Modo de Trading")
        mode_group.setStyleSheet("QGroupBox { border: none; font-weight: bold; color: #f48771; }")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Paper Trading (Simulación)", "Live Trading (REAL - No usar)"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 13px;
                background: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
                color: #f48771;
                font-weight: bold;
            }
        """)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        center_config.addWidget(mode_group)
        
        # Strategy info
        strategy_info_group = QGroupBox("📋 Información")
        strategy_info_group.setStyleSheet("QGroupBox { border: none; font-weight: bold; color: #dcdcaa; }")
        
        info_layout = QVBoxLayout()
        self.strategy_info_label = QLabel("Selecciona una estrategia para ver detalles...")
        self.strategy_info_label.setWordWrap(True)
        self.strategy_info_label.setStyleSheet("color: #ccc; font-size: 12px; padding: 8px; background: #1e1e1e; border-radius: 4px;")
        info_layout.addWidget(self.strategy_info_label)
        
        strategy_info_group.setLayout(info_layout)
        center_config.addWidget(strategy_info_group)
        
        config_layout.addLayout(center_config)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === MONITORING SECTION ===
        monitoring_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Performance Metrics
        metrics_group = QGroupBox("📊 Rendimiento")
        metrics_group.setStyleSheet("""
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
        
        metrics_layout = QGridLayout()
        metrics_layout.setContentsMargins(12, 12, 12, 12)
        metrics_layout.setSpacing(12)
        
        # Performance metrics
        self.pnl_card = MetricCard("P&L", "$0.00", color="#4ec9b0", tooltip="Ganancia/Pérdida total de la sesión")
        self.sharpe_card = MetricCard("Sharpe Ratio", "--", color="#569cd6", tooltip="Relación riesgo/retorno (>1.5 es bueno)")
        self.drawdown_card = MetricCard("Max Drawdown", "--", "%", color="#f48771", tooltip="Peor caída desde el pico")
        self.winrate_card = MetricCard("Win Rate", "--", "%", color="#dcdcaa", tooltip="Porcentaje de trades ganadores")
        
        metrics_layout.addWidget(self.pnl_card, 0, 0)
        metrics_layout.addWidget(self.sharpe_card, 0, 1)
        metrics_layout.addWidget(self.drawdown_card, 1, 0)
        metrics_layout.addWidget(self.winrate_card, 1, 1)
        
        metrics_group.setLayout(metrics_layout)
        monitoring_splitter.addWidget(metrics_group)
        
        # Right: Decision Log
        decisions_group = QGroupBox("🤖 Decisiones del Bot")
        decisions_group.setStyleSheet("""
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
        
        decisions_layout = QVBoxLayout()
        decisions_layout.setContentsMargins(12, 12, 12, 12)
        
        self.decision_log = QTextEdit()
        self.decision_log.setReadOnly(True)
        self.decision_log.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #ccc;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.decision_log.setPlainText("Esperando decisiones del bot...\nSelecciona una estrategia y haz clic en START TRADING.")
        decisions_layout.addWidget(self.decision_log)
        
        decisions_group.setLayout(decisions_layout)
        monitoring_splitter.addWidget(decisions_group)
        
        # Optimize proportions: 35% metrics, 65% decision log for better visibility
        monitoring_splitter.setStretchFactor(0, 35)
        monitoring_splitter.setStretchFactor(1, 65)
        monitoring_splitter.setSizes([350, 650])
        layout.addWidget(monitoring_splitter)
        
        # === CONTROL BUTTONS ===
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.start_btn = QPushButton("▶ START TRADING")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4ec9b0;
                color: #fff;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover { background: #5fd9c0; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        self.start_btn.clicked.connect(self.start_trading)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ STOP TRADING")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #f48771;
                color: #fff;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover { background: #ff9d8a; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        self.stop_btn.clicked.connect(self.stop_trading)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Apply theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-size: 12px;
                font-weight: bold;
                color: #888;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                color: #ccc;
            }
        """)
    
    def create_left_panel(self):
        """Create left panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Ticker selector
        self.ticker_selector = TickerSelector()
        self.ticker_selector.ticker_changed.connect(self.on_ticker_changed)
        layout.addWidget(self.ticker_selector)
        
        # Strategy selector
        self.strategy_selector = StrategySelector()
        self.strategy_selector.strategy_changed.connect(self.on_strategy_changed)
        layout.addWidget(self.strategy_selector)
        
        # Strategy info
        self.strategy_info = StrategyInfoPanel()
        self.strategy_info.update_strategy(
            self.current_strategy['name'],
            self.current_strategy['description'],
            self.current_strategy['parameters']
        )
        layout.addWidget(self.strategy_info)
        
        # Data source indicator
        self.data_source = DataSourceIndicator()
        layout.addWidget(self.data_source)
        
        layout.addStretch()
        return container
    
    def create_center_panel(self):
        """Create center panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # P&L Display
        pnl_group = QGroupBox("💰 P&L Actual (Ganancia/Pérdida del Día)")
        pnl_group.setToolTip("Muestra tu ganancia o pérdida porcentual del día.\nVerde = ganando, Rojo = perdiendo")
        pnl_layout = QVBoxLayout()
        pnl_layout.setContentsMargins(12, 15, 12, 12)
        
        self.pnl_label = QLabel("+0.00%")
        self.pnl_label.setStyleSheet("color: #4ec9b0; font-size: 48px; font-weight: bold;")
        self.pnl_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pnl_layout.addWidget(self.pnl_label)
        
        self.pnl_usd = QLabel("$0.00 USD")
        self.pnl_usd.setStyleSheet("color: #888; font-size: 14px;")
        self.pnl_usd.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pnl_layout.addWidget(self.pnl_usd)
        
        pnl_group.setLayout(pnl_layout)
        pnl_group.setMaximumHeight(160)
        layout.addWidget(pnl_group)
        
        # Metrics Grid
        metrics_group = QGroupBox("📊 Métricas en Tiempo Real")
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(8)
        metrics_layout.setContentsMargins(12, 15, 12, 12)
        
        self.sharpe_card = MetricCard("Sharpe Ratio", "--", "", "#569cd6", 
                                     tooltip="Relación riesgo/retorno\n>1.5 = Bueno\n>2.0 = Excelente")
        self.dd_card = MetricCard("Max Drawdown", "--", "%", "#f48771",
                                 tooltip="Peor caída desde el pico\nEntre -5% y -10% es aceptable")
        self.win_rate_card = MetricCard("Win Rate", "--", "%", "#4ec9b0",
                                       tooltip="% de trades ganadores\n>55% es bueno")
        self.exposure_card = MetricCard("Exposición", "--", "%", "#dcdcaa",
                                       tooltip="% de capital en uso")
        
        metrics_layout.addWidget(self.sharpe_card, 0, 0)
        metrics_layout.addWidget(self.dd_card, 0, 1)
        metrics_layout.addWidget(self.win_rate_card, 1, 0)
        metrics_layout.addWidget(self.exposure_card, 1, 1)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Positions table
        positions_group = QGroupBox("📈 Posiciones Activas (Solo tu activo seleccionado)")
        positions_layout = QVBoxLayout()
        positions_layout.setContentsMargins(12, 15, 12, 12)
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels(['Symbol', 'Side', 'Size', 'Entry', 'Current', 'P&L'])
        self.positions_table.setStyleSheet("""
            QTableWidget {
                background: #1e1e1e;
                color: #ccc;
                border: 1px solid #444;
                border-radius: 4px;
                gridline-color: #333;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QHeaderView::section {
                background: #2d2d2d;
                color: #888;
                border: none;
                padding: 8px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self.positions_table.setMaximumHeight(180)
        positions_layout.addWidget(self.positions_table)
        
        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)
        
        return container
    
    def create_right_panel(self):
        """Create right panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Decision log
        self.decision_log = DecisionLogPanel()
        layout.addWidget(self.decision_log)
        
        return container
    
    def show_help(self):
        """Show help dialog"""
        dialog = HelpDialog(self)
        dialog.exec()
    
    def start_trading(self):
        """Start trading"""
        if self.monitor_thread and self.monitor_thread.isRunning():
            return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.data_source.set_live_mode(True, "Alpaca Paper Trading API")
        
        # Use selected ticker
        self.monitor_thread = EnhancedLiveMonitorThread(
            self.current_strategy['name'], 
            self.current_ticker
        )
        self.monitor_thread.pnl_update.connect(self.update_pnl)
        self.monitor_thread.metrics_update.connect(self.update_metrics)
        self.monitor_thread.position_update.connect(self.update_positions)
        self.monitor_thread.decision_made.connect(self.log_decision)
        self.monitor_thread.connection_status.connect(self.update_connection_status)
        self.monitor_thread.start()
        
        self.decision_log.add_decision(
            datetime.now(),
            "STARTED",
            f"Iniciando trading en {self.current_ticker} con estrategia: {self.current_strategy['name']}",
            self.current_strategy['parameters']
        )
    
    def stop_trading(self):
        """Stop trading"""
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.data_source.set_live_mode(False)
        
        self.decision_log.add_decision(
            datetime.now(),
            "STOPPED",
            "Trading detenido por el usuario",
            {}
        )
    
    def update_pnl(self, pnl):
        """Update P&L"""
        color = "#4ec9b0" if pnl >= 0 else "#f48771"
        sign = "+" if pnl >= 0 else ""
        self.pnl_label.setText(f"{sign}{pnl:.2f}%")
        self.pnl_label.setStyleSheet(f"color: {color}; font-size: 48px; font-weight: bold;")
        
        usd_value = pnl * 100
        self.pnl_usd.setText(f"${sign}{usd_value:.2f} USD")
    
    def update_metrics(self, metrics):
        """Update metrics"""
        self.sharpe_card.update_value(f"{metrics['sharpe']:.2f}")
        
        dd_color = "#f48771" if metrics['max_dd'] < -10 else "#dcdcaa"
        self.dd_card.update_value(f"{metrics['max_dd']:.1f}", dd_color)
        
        wr_color = "#4ec9b0" if metrics['win_rate'] > 55 else "#f48771" if metrics['win_rate'] < 45 else "#dcdcaa"
        self.win_rate_card.update_value(f"{metrics['win_rate']:.1f}", wr_color)
        
        self.exposure_card.update_value(f"{metrics['exposure']:.1f}")
    
    def update_positions(self, positions):
        """Update positions"""
        self.positions_table.setRowCount(len(positions))
        
        for i, pos in enumerate(positions):
            self.positions_table.setItem(i, 0, QTableWidgetItem(pos['symbol']))
            
            side_item = QTableWidgetItem(pos['side'])
            side_color = "#4ec9b0" if pos['side'] == 'LONG' else "#f48771"
            side_item.setForeground(QColor(side_color))
            self.positions_table.setItem(i, 1, side_item)
            
            self.positions_table.setItem(i, 2, QTableWidgetItem(f"{pos['size']:.3f}"))
            self.positions_table.setItem(i, 3, QTableWidgetItem(f"${pos['entry']:.2f}"))
            self.positions_table.setItem(i, 4, QTableWidgetItem(f"${pos['current']:.2f}"))
            
            pnl_item = QTableWidgetItem(f"${pos['pnl']:.2f} ({pos['pnl_pct']:+.2f}%)")
            pnl_color = "#4ec9b0" if pos['pnl'] >= 0 else "#f48771"
            pnl_item.setForeground(QColor(pnl_color))
            self.positions_table.setItem(i, 5, pnl_item)
    
    def log_decision(self, decision):
        """Log decision"""
        self.decision_log.add_decision(
            decision['timestamp'],
            decision['action'],
            decision['reason'],
            decision['indicators']
        )
    
    def update_connection_status(self, connected):
        """Update connection status"""
        if connected:
            self.data_source.set_live_mode(True, "Alpaca Paper Trading API")
        else:
            self.data_source.set_live_mode(False)
    
    def on_ticker_changed(self, ticker):
        """Handle ticker change"""
        self.current_ticker = ticker
        print(f"Ticker changed to: {ticker}")
    
    def on_strategy_changed(self, strategy_name):
        """Handle strategy change"""
        strategies_info = {
            'RSI Mean Reversion': {
                'description': 'Compra cuando RSI < 30 (sobrevendido), vende cuando RSI > 70 (sobrecomprado). Busca reversiones a la media.',
                'parameters': {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30, 'take_profit': 2.0, 'stop_loss': 1.5}
            },
            'MACD Momentum': {
                'description': 'Compra en cruce alcista de MACD, vende en cruce bajista. Sigue tendencias de momento.',
                'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'take_profit': 2.5, 'stop_loss': 1.5}
            },
            'Bollinger Bands Breakout': {
                'description': 'Compra cuando precio rompe banda superior, vende en banda inferior. Captura volatilidad.',
                'parameters': {'period': 20, 'std_dev': 2.0, 'take_profit': 3.0, 'stop_loss': 1.5}
            },
            'MA Crossover': {
                'description': 'Compra cuando MA rápida cruza arriba de MA lenta, vende en cruce inverso. Estrategia clásica de tendencias.',
                'parameters': {'fast_ma': 50, 'slow_ma': 200, 'take_profit': 2.0, 'stop_loss': 1.5}
            },
            'Volume Breakout': {
                'description': 'Compra cuando volumen supera promedio + precio rompe resistencia. Captura momentum fuerte.',
                'parameters': {'volume_threshold': 1.5, 'breakout_period': 20, 'take_profit': 3.0, 'stop_loss': 2.0}
            }
        }
        
        if strategy_name in strategies_info:
            self.current_strategy['name'] = strategy_name
            self.current_strategy['description'] = strategies_info[strategy_name]['description']
            self.current_strategy['parameters'] = strategies_info[strategy_name]['parameters']
            # Update info label instead of calling non-existent method
            params_text = '\n'.join([f'{k}: {v}' for k, v in strategies_info[strategy_name]['parameters'].items()])
            self.strategy_info_label.setText(f"""{strategy_name}

{strategies_info[strategy_name]['description']}

Parámetros:
{params_text}""")
    
    def load_selected_strategy(self):
        """Load the selected strategy"""
        strategy_name = self.strategy_combo.currentText()
        print(f"Loading strategy: {strategy_name}")
        
        strategies_info = {
            'RSI Mean Reversion': {
                'description': 'Compra cuando RSI < 30 (sobrevendido), vende cuando RSI > 70 (sobrecomprado). Busca reversiones a la media.',
                'parameters': {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30, 'take_profit': 2.0, 'stop_loss': 1.5}
            },
            'MACD Momentum': {
                'description': 'Compra en cruce alcista de MACD, vende en cruce bajista. Sigue tendencias de momento.',
                'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'take_profit': 2.5, 'stop_loss': 1.5}
            },
            'Bollinger Bands Breakout': {
                'description': 'Compra cuando precio rompe banda superior, vende en banda inferior. Captura volatilidad.',
                'parameters': {'period': 20, 'std_dev': 2.0, 'take_profit': 3.0, 'stop_loss': 1.5}
            },
            'MA Crossover': {
                'description': 'Compra cuando MA rápida cruza arriba de MA lenta, vende en cruce inverso. Estrategia clásica de tendencias.',
                'parameters': {'fast_ma': 50, 'slow_ma': 200, 'take_profit': 2.0, 'stop_loss': 1.5}
            },
            'Volume Breakout': {
                'description': 'Compra cuando volumen supera promedio + precio rompe resistencia. Captura momentum fuerte.',
                'parameters': {'volume_threshold': 1.5, 'breakout_period': 20, 'take_profit': 3.0, 'stop_loss': 2.0}
            }
        }
        
        if strategy_name in strategies_info:
            self.current_strategy['name'] = strategy_name
            self.current_strategy['description'] = strategies_info[strategy_name]['description']
            self.current_strategy['parameters'] = strategies_info[strategy_name]['parameters']
            # Update info label instead of calling non-existent method
            params_text = '\n'.join([f'{k}: {v}' for k, v in strategies_info[strategy_name]['parameters'].items()])
            self.strategy_info_label.setText(f"""{strategy_name}

{strategies_info[strategy_name]['description']}

Parámetros:
{params_text}""")
