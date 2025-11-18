"""
Help System for TradingIA Platform
Provides contextual help, tooltips, and documentation access
"""

import sys
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTextBrowser, QDialog, QDialogButtonBox,
                               QSplitter, QTreeWidget, QTreeWidgetItem, QLineEdit)
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut, QKeySequence

# Import strategy documentation
from strategies.strategy_docs import get_strategy_info


class ContextualHelpDialog(QDialog):
    """Dialog for showing contextual help"""

    def __init__(self, title="Ayuda Contextual", content="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"❓ {title}")
        self.setModal(False)
        self.setMinimumSize(600, 400)
        self.setMaximumSize(800, 600)

        layout = QVBoxLayout()

        # Content area
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        self.content_browser.setHtml(content)
        layout.addWidget(self.content_browser)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

        # Apply styling
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QTextBrowser {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)


class HelpCenterDialog(QDialog):
    """Main help center dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📚 Centro de Ayuda - TradingIA")
        self.setModal(False)
        self.setMinimumSize(900, 700)

        self.setup_ui()
        self.load_help_content()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QHBoxLayout()

        # Navigation sidebar
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderLabel("Contenido")
        self.nav_tree.setMaximumWidth(250)
        self.nav_tree.itemClicked.connect(self.on_nav_item_clicked)

        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout()

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("🔍 Buscar:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar en la ayuda...")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        search_layout.addWidget(self.search_input)
        content_layout.addLayout(search_layout)

        # Content browser
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        content_layout.addWidget(self.content_browser)

        content_widget.setLayout(content_layout)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.nav_tree)
        splitter.addWidget(content_widget)
        splitter.setSizes([250, 650])

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Apply styling
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QTreeWidget {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QTextBrowser {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 10px;
            }
        """)

    def load_help_content(self):
        """Load help content into navigation tree"""
        # Getting Started
        getting_started = QTreeWidgetItem(["🚀 Primeros Pasos"])
        getting_started.addChild(QTreeWidgetItem(["¿Qué es TradingIA?"]))
        getting_started.addChild(QTreeWidgetItem(["Instalación y Configuración"]))
        getting_started.addChild(QTreeWidgetItem(["Interfaz de Usuario"]))
        self.nav_tree.addTopLevelItem(getting_started)

        # Strategies
        strategies = QTreeWidgetItem(["🎯 Estrategias"])
        strategies.addChild(QTreeWidgetItem(["Tipos de Estrategias"]))
        strategies.addChild(QTreeWidgetItem(["Crear Nueva Estrategia"]))
        strategies.addChild(QTreeWidgetItem(["Backtesting"]))
        strategies.addChild(QTreeWidgetItem(["Optimización"]))
        self.nav_tree.addTopLevelItem(strategies)

        # Trading
        trading = QTreeWidgetItem(["📈 Trading"])
        trading.addChild(QTreeWidgetItem(["Trading en Vivo"]))
        trading.addChild(QTreeWidgetItem(["Gestión de Riesgos"]))
        trading.addChild(QTreeWidgetItem(["Monitoreo"]))
        trading.addChild(QTreeWidgetItem(["Brokers Integrados"]))
        self.nav_tree.addTopLevelItem(trading)

        # Analysis
        analysis = QTreeWidgetItem(["📊 Análisis"])
        analysis.addChild(QTreeWidgetItem(["Análisis Técnico"]))
        analysis.addChild(QTreeWidgetItem(["Métricas de Rendimiento"]))
        analysis.addChild(QTreeWidgetItem(["Reportes"]))
        analysis.addChild(QTreeWidgetItem(["Comparaciones"]))
        self.nav_tree.addTopLevelItem(analysis)

        # Advanced
        advanced = QTreeWidgetItem(["🔧 Avanzado"])
        advanced.addChild(QTreeWidgetItem(["API REST"]))
        advanced.addChild(QTreeWidgetItem(["Configuración Personalizada"]))
        advanced.addChild(QTreeWidgetItem(["Solución de Problemas"]))
        advanced.addChild(QTreeWidgetItem(["Desarrollo de Estrategias"]))
        self.nav_tree.addTopLevelItem(advanced)

        # Expand first level
        self.nav_tree.expandToDepth(0)

    def on_nav_item_clicked(self, item, column):
        """Handle navigation item clicks"""
        topic = item.text(0)
        content = self.get_help_content(topic)
        self.content_browser.setHtml(content)

    def on_search_text_changed(self, text):
        """Handle search text changes"""
        if len(text) < 3:
            return

        # Simple search implementation
        results = self.search_help_content(text)
        if results:
            self.content_browser.setHtml(results)
        else:
            self.content_browser.setHtml(f"<h3>No se encontraron resultados para: '{text}'</h3>")

    def get_help_content(self, topic):
        """Get help content for a specific topic"""
        content_map = {
            "¿Qué es TradingIA?": """
                <h2>🚀 ¿Qué es TradingIA?</h2>
                <p>TradingIA es una plataforma avanzada para trading algorítmico que te permite:</p>
                <ul>
                    <li><b>Crear estrategias</b> de trading automatizadas</li>
                    <li><b>Probar estrategias</b> con datos históricos (backtesting)</li>
                    <li><b>Ejecutar trading</b> en vivo con brokers reales</li>
                    <li><b>Analizar rendimiento</b> con métricas detalladas</li>
                    <li><b>Gestionar riesgos</b> automáticamente</li>
                </ul>
                <h3>¿Por qué usar trading algorítmico?</h3>
                <ul>
                    <li>Elimina emociones del trading</li>
                    <li>Opera 24/7 sin intervención manual</li>
                    <li>Mayor velocidad y precisión</li>
                    <li>Resultados consistentes y medibles</li>
                </ul>
            """,

            "Interfaz de Usuario": """
                <h2>🖥️ Interfaz de Usuario</h2>
                <p>TradingIA cuenta con 11 pestañas principales organizadas por funcionalidad:</p>

                <h3>🏠 Dashboard</h3>
                <p>Vista general del sistema con métricas clave y estado actual.</p>

                <h3>📊 Data</h3>
                <p>Gestión de datos históricos y configuración de fuentes de datos.</p>

                <h3>⚙️ Strategy</h3>
                <p>Configuración y gestión de estrategias de trading.</p>

                <h3>▶️ Backtest</h3>
                <p>Ejecución de pruebas históricas de estrategias.</p>

                <h3>📈 Results</h3>
                <p>Análisis detallado de resultados de backtesting.</p>

                <h3>⚖️ A/B Test</h3>
                <p>Comparación de estrategias mediante pruebas A/B.</p>

                <h3>🔴 Live</h3>
                <p>Monitoreo y control de trading en vivo.</p>

                <h3>💰 Brokers</h3>
                <p>Configuración de conexiones con brokers.</p>

                <h3>🌐 API</h3>
                <p>Control del servidor API REST.</p>

                <h3>🔧 Research</h3>
                <p>Herramientas avanzadas de investigación.</p>

                <h3>📥 Data Download</h3>
                <p>Descarga de datos históricos.</p>

                <h3>❓ Help</h3>
                <p>Centro de ayuda y documentación.</p>

                <h3>📊 Risk Metrics</h3>
                <p>Análisis detallado de métricas de riesgo.</p>
            """,

            "Tipos de Estrategias": """
                <h2>🎯 Tipos de Estrategias Disponibles</h2>
                <p>TradingIA incluye varias estrategias preconfiguradas:</p>

                <h3>📈 Bollinger Bands</h3>
                <p>Opera basado en bandas de volatilidad. Compra cuando el precio toca la banda inferior y vende cuando toca la superior.</p>

                <h3>📊 RSI Mean Reversion</h3>
                <p>Usa el índice de fuerza relativa (RSI) para identificar condiciones de sobrecompra y sobreventa.</p>

                <h3>💹 MACD Momentum</h3>
                <p>Sigue el momentum del mercado usando la divergencia/convergencia de medias móviles (MACD).</p>

                <h3>📉 MA Crossover</h3>
                <p>Genera señales cuando medias móviles de diferentes períodos se cruzan.</p>

                <h3>📊 Volume Breakout</h3>
                <p>Identifica rupturas de volumen significativas para entrar en posiciones.</p>

                <h3>🔮 Oracle Numeris Safeguard</h3>
                <p>Estrategia avanzada que combina múltiples indicadores técnicos con protección de capital.</p>
            """,

            "Backtesting": """
                <h2>🔬 Backtesting - Pruebas Históricas</h2>
                <p>El backtesting te permite probar estrategias con datos históricos antes de usarlas en vivo.</p>

                <h3>¿Cómo funciona?</h3>
                <ol>
                    <li>Selecciona una estrategia</li>
                    <li>Configura parámetros (capital inicial, período, etc.)</li>
                    <li>Ejecuta la simulación</li>
                    <li>Analiza los resultados</li>
                </ol>

                <h3>Métricas importantes</h3>
                <ul>
                    <li><b>Retorno total</b>: Ganancia/perdida total</li>
                    <li><b>Sharpe ratio</b>: Riesgo ajustado al retorno</li>
                    <li><b>Max drawdown</b>: Máxima pérdida desde el pico</li>
                    <li><b>Win rate</b>: Porcentaje de operaciones ganadoras</li>
                    <li><b>Profit factor</b>: Ganancias totales / pérdidas totales</li>
                </ul>

                <h3>Consideraciones importantes</h3>
                <p>Recuerda que el pasado no garantiza el futuro. Una estrategia que funciona bien en backtesting puede no hacerlo en condiciones de mercado reales.</p>
            """,

            "Trading en Vivo": """
                <h2>🔴 Trading en Vivo</h2>
                <p>Ejecuta tus estrategias con dinero real en los mercados.</p>

                <h3>Antes de comenzar</h3>
                <ul>
                    <li>✅ Configura tu broker</li>
                    <li>✅ Define límites de riesgo</li>
                    <li>✅ Prueba en modo demo primero</li>
                    <li>✅ Monitorea constantemente</li>
                </ul>

                <h3>Configuración necesaria</h3>
                <ul>
                    <li>API keys del broker</li>
                    <li>Límites de posición</li>
                    <li>Stop losses automáticos</li>
                    <li>Notificaciones de alertas</li>
                </ul>

                <h3>Monitoreo continuo</h3>
                <p>El trading algorítmico requiere supervisión constante. Monitorea:</p>
                <ul>
                    <li>Estado de conexión con el broker</li>
                    <li>Rendimiento de la estrategia</li>
                    <li>Niveles de drawdown</li>
                    <li>Eventos del mercado</li>
                </ul>
            """
        }

        return content_map.get(topic, f"<h2>{topic}</h2><p>Contenido de ayuda para '{topic}' próximamente.</p>")

    def search_help_content(self, query):
        """Search help content for query"""
        # Simple search implementation
        query_lower = query.lower()

        if "estrategia" in query_lower or "strategy" in query_lower:
            return self.get_help_content("Tipos de Estrategias")
        elif "backtest" in query_lower:
            return self.get_help_content("Backtesting")
        elif "live" in query_lower or "trading" in query_lower:
            return self.get_help_content("Trading en Vivo")
        elif "interfaz" in query_lower or "ui" in query_lower:
            return self.get_help_content("Interfaz de Usuario")
        else:
            return self.get_help_content("¿Qué es TradingIA?")


class HelpSystem:
    """Main help system manager"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.help_dialogs = {}
        self.setup_help_system()

    def setup_help_system(self):
        """Setup the help system"""
        # Add F1 key shortcut for help
        from PySide6.QtGui import QShortcut
        help_shortcut = QShortcut(QKeySequence("F1"), self.main_window)
        help_shortcut.activated.connect(self.show_help_center)

        # Add help button to status bar if available
        if hasattr(self.main_window, 'statusBar'):
            help_button = QPushButton("❓ Ayuda")
            help_button.clicked.connect(self.show_help_center)
            help_button.setMaximumWidth(80)
            self.main_window.statusBar().addPermanentWidget(help_button)

    def show_contextual_help(self, topic, content=""):
        """Show contextual help dialog"""
        if topic in self.help_dialogs:
            self.help_dialogs[topic].raise_()
            self.help_dialogs[topic].activateWindow()
            return

        dialog = ContextualHelpDialog(topic, content, self.main_window)
        self.help_dialogs[topic] = dialog
        dialog.finished.connect(lambda: self.help_dialogs.pop(topic, None))
        dialog.show()

    def show_help_center(self):
        """Show main help center"""
        if "center" in self.help_dialogs:
            self.help_dialogs["center"].raise_()
            self.help_dialogs["center"].activateWindow()
            return

        dialog = HelpCenterDialog(self.main_window)
        self.help_dialogs["center"] = dialog
        dialog.finished.connect(lambda: self.help_dialogs.pop("center", None))
        dialog.show()

    def show_strategy_help(self, strategy_name):
        """Show help for a specific strategy"""
        try:
            strategy_info = get_strategy_info(strategy_name)
            if strategy_info:
                content = f"""
                    <h2>🎯 {strategy_info.get('name', strategy_name)}</h2>
                    <p><b>Descripción:</b> {strategy_info.get('description', 'No disponible')}</p>
                    <p><b>Señales:</b> {strategy_info.get('signals', 'No disponible')}</p>
                    <p><b>Parámetros:</b> {strategy_info.get('parameters', 'No disponible')}</p>
                    <p><b>Recomendaciones:</b> {strategy_info.get('recommendations', 'No disponible')}</p>
                """
                self.show_contextual_help(f"Estrategia: {strategy_name}", content)
            else:
                self.show_contextual_help(
                    f"Estrategia: {strategy_name}",
                    f"<h3>Información de estrategia</h3><p>No se encontró documentación para '{strategy_name}'</p>"
                )
        except Exception as e:
            self.show_contextual_help(
                f"Estrategia: {strategy_name}",
                f"<h3>Error</h3><p>No se pudo cargar la información de la estrategia: {str(e)}</p>"
            )

    def get_tooltip_for_widget(self, widget_name):
        """Get tooltip text for a specific widget"""
        tooltips = {
            "strategy_combo": "Selecciona la estrategia de trading que deseas usar",
            "backtest_button": "Ejecuta una prueba histórica de la estrategia seleccionada",
            "live_start_button": "Inicia el trading automatizado en vivo (¡usa con precaución!)",
            "data_load_button": "Carga datos históricos para análisis y backtesting",
            "results_analyze_button": "Analiza los resultados del último backtest ejecutado",
            "risk_calculate_button": "Calcula métricas de riesgo para la estrategia actual"
        }

        return tooltips.get(widget_name, "")

    def setup_tab_help(self, tab_widget, tab_name):
        """Setup help for a specific tab"""
        help_tips = {
            "Dashboard": "Vista general del sistema. Muestra métricas clave y estado actual.",
            "Data": "Gestiona datos históricos. Carga, visualiza y configura fuentes de datos.",
            "Strategy": "Configura estrategias de trading. Selecciona y personaliza algoritmos.",
            "Backtest": "Ejecuta pruebas históricas. Evalúa estrategias con datos pasados.",
            "Results": "Analiza resultados. Revisa métricas de rendimiento y estadísticas.",
            "A/B Test": "Compara estrategias. Ejecuta pruebas A/B entre diferentes algoritmos.",
            "Live": "Trading en vivo. Monitorea y controla operaciones automatizadas.",
            "Brokers": "Configura brokers. Conecta con plataformas de trading externas.",
            "API": "Servidor REST. Controla el acceso remoto a la plataforma.",
            "Research": "Herramientas avanzadas. Análisis técnico y investigación.",
            "Data Download": "Descarga datos. Obtén datos históricos de mercados.",
            "Help": "Centro de ayuda. Documentación y soporte de la plataforma.",
            "Risk Metrics": "Análisis de riesgo. Métricas detalladas de exposición y volatilidad."
        }

        tip = help_tips.get(tab_name, f"Ayuda para la pestaña {tab_name}")
        tab_widget.setToolTip(tip)


# Global help system instance
_help_system_instance = None

def init_help_system(main_window):
    """Initialize the global help system"""
    global _help_system_instance
    _help_system_instance = HelpSystem(main_window)
    return _help_system_instance

def get_help_system():
    """Get the global help system instance"""
    return _help_system_instance

def show_help_center():
    """Show help center (convenience function)"""
    if _help_system_instance:
        _help_system_instance.show_help_center()

def show_contextual_help(topic, content=""):
    """Show contextual help (convenience function)"""
    if _help_system_instance:
        _help_system_instance.show_contextual_help(topic, content)

def show_strategy_help(strategy_name):
    """Show strategy help (convenience function)"""
    if _help_system_instance:
        _help_system_instance.show_strategy_help(strategy_name)