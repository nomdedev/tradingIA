"""
TradingIA Platform - Tab 10: Help & Documentation
Integrated help system and user manual

Author: TradingIA Team
Version: 2.0.0
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QTextBrowser,
    QSplitter, QTreeWidget, QTreeWidgetItem
)
from PySide6.QtCore import Qt


class Tab10Help(QWidget):
    """
    Help & Documentation Tab

    Provides integrated help system with:
    - User manual for all platform features
    - Quick start guides
    - Troubleshooting
    - FAQ
    - Video tutorials (links)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """Initialize the help interface"""
        layout = QHBoxLayout()

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Navigation
        self.nav_tree = self.create_navigation_tree()
        splitter.addWidget(self.nav_tree)

        # Right panel - Content
        self.content_browser = self.create_content_browser()
        splitter.addWidget(self.content_browser)

        # Set splitter proportions
        splitter.setSizes([300, 1000])

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Load initial content
        self.show_welcome()

    def create_navigation_tree(self):
        """Create navigation tree for help topics"""
        tree = QTreeWidget()
        tree.setHeaderLabel("📚 Manual de Usuario")
        tree.setStyleSheet("""
            QTreeWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #555;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 5px;
                border-bottom: 1px solid #444;
            }
            QTreeWidget::item:selected {
                background-color: #0e639c;
            }
            QTreeWidget::item:hover {
                background-color: #3d3d3d;
            }
        """)

        # Connect signal
        tree.itemClicked.connect(self.on_topic_selected)

        # Create main categories
        self.create_help_structure(tree)

        tree.expandAll()
        return tree

    def create_help_structure(self, tree):
        """Create the hierarchical help structure"""
        categories = {
            "🚀 Inicio Rápido": [
                "Bienvenido a TradingIA",
                "Primeros Pasos",
                "Configuración Inicial",
                "Carga Automática de Datos"
            ],
            "📊 Dashboard": [
                "Vista General",
                "Métricas del Sistema",
                "Acciones Rápidas",
                "Estado del Sistema"
            ],
            "📥 Gestión de Datos": [
                "Descarga de Datos",
                "Formatos Soportados",
                "Almacenamiento",
                "Verificación de Integridad"
            ],
            "⚙️ Estrategias": [
                "Configuración de Estrategias",
                "Parámetros",
                "Optimización",
                "Backtesting"
            ],
            "▶️ Backtesting": [
                "Ejecución de Backtests",
                "Análisis de Resultados",
                "Métricas de Rendimiento",
                "Validación de Estrategias"
            ],
            "📈 Análisis de Resultados": [
                "Gráficos de Rendimiento",
                "Estadísticas Detalladas",
                "Comparación de Estrategias",
                "Exportación de Reportes"
            ],
            "🆚 A/B Testing": [
                "Configuración de Tests",
                "Ejecución Automatizada",
                "Análisis Estadístico",
                "Recomendaciones"
            ],
            "📊 Monitoreo en Vivo": [
                "Paper Trading",
                "Conexión con Alpaca",
                "Monitoreo en Tiempo Real",
                "Alertas y Notificaciones"
            ],
            "🔬 Análisis Avanzado": [
                "Análisis Técnico",
                "Machine Learning",
                "Risk Management",
                "Optimización Avanzada"
            ],
            "📥 Descarga de Datos": [
                "Configuración de APIs",
                "Descargas Automáticas",
                "Gestión de Progreso",
                "Solución de Problemas"
            ],
            "🔧 Configuración": [
                "Ajustes del Sistema",
                "Preferencias de Usuario",
                "Configuración de APIs",
                "Backup y Restauración"
            ],
            "❓ Solución de Problemas": [
                "Problemas Comunes",
                "Mensajes de Error",
                "Performance Issues",
                "Soporte Técnico"
            ]
        }

        for category, topics in categories.items():
            category_item = QTreeWidgetItem([category])
            category_item.setExpanded(True)

            for topic in topics:
                topic_item = QTreeWidgetItem([topic])
                category_item.addChild(topic_item)

            tree.addTopLevelItem(category_item)

    def create_content_browser(self):
        """Create content display browser"""
        browser = QTextBrowser()
        browser.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                font-size: 11px;
                padding: 10px;
            }
        """)
        browser.setOpenExternalLinks(True)
        return browser

    def on_topic_selected(self, item, column):
        """Handle topic selection"""
        topic_text = item.text(column)

        # Map topics to content methods
        content_methods = {
            # Inicio Rápido
            "Bienvenido a TradingIA": self.show_welcome,
            "Primeros Pasos": self.show_getting_started,
            "Configuración Inicial": self.show_initial_setup,
            "Carga Automática de Datos": self.show_auto_load,

            # Dashboard
            "Vista General": self.show_dashboard_overview,
            "Métricas del Sistema": self.show_system_metrics,
            "Acciones Rápidas": self.show_quick_actions,
            "Estado del Sistema": self.show_system_status,

            # Gestión de Datos
            "Descarga de Datos": self.show_data_download,
            "Formatos Soportados": self.show_supported_formats,
            "Almacenamiento": self.show_data_storage,
            "Verificación de Integridad": self.show_data_integrity,

            # Estrategias
            "Configuración de Estrategias": self.show_strategy_config,
            "Parámetros": self.show_strategy_parameters,
            "Optimización": self.show_strategy_optimization,
            "Backtesting": self.show_strategy_backtesting,

            # Backtesting
            "Ejecución de Backtests": self.show_backtest_execution,
            "Análisis de Resultados": self.show_backtest_analysis,
            "Métricas de Rendimiento": self.show_performance_metrics,
            "Validación de Estrategias": self.show_strategy_validation,

            # Análisis de Resultados
            "Gráficos de Rendimiento": self.show_performance_charts,
            "Estadísticas Detalladas": self.show_detailed_stats,
            "Comparación de Estrategias": self.show_strategy_comparison,
            "Exportación de Reportes": self.show_report_export,

            # A/B Testing
            "Configuración de Tests": self.show_ab_test_config,
            "Ejecución Automatizada": self.show_ab_test_execution,
            "Análisis Estadístico": self.show_statistical_analysis,
            "Recomendaciones": self.show_recommendations,

            # Monitoreo en Vivo
            "Paper Trading": self.show_paper_trading,
            "Conexión con Alpaca": self.show_alpaca_connection,
            "Monitoreo en Tiempo Real": self.show_live_monitoring,
            "Alertas y Notificaciones": self.show_alerts_notifications,

            # Análisis Avanzado
            "Análisis Técnico": self.show_technical_analysis,
            "Machine Learning": self.show_machine_learning,
            "Risk Management": self.show_risk_management,
            "Optimización Avanzada": self.show_advanced_optimization,

            # Descarga de Datos
            "Configuración de APIs": self.show_api_configuration,
            "Descargas Automáticas": self.show_automatic_downloads,
            "Gestión de Progreso": self.show_progress_management,
            "Solución de Problemas": self.show_download_troubleshooting,

            # Configuración
            "Ajustes del Sistema": self.show_system_settings,
            "Preferencias de Usuario": self.show_user_preferences,
            "Configuración de APIs": self.show_api_configuration,
            "Backup y Restauración": self.show_backup_restore,

            # Solución de Problemas
            "Problemas Comunes": self.show_common_issues,
            "Mensajes de Error": self.show_error_messages,
            "Performance Issues": self.show_performance_issues,
            "Soporte Técnico": self.show_technical_support
        }

        if topic_text in content_methods:
            content_methods[topic_text]()

    # Content methods for each topic
    def show_welcome(self):
        """Show welcome content"""
        content = """
        <h1>🎉 ¡Bienvenido a TradingIA!</h1>

        <p><strong>TradingIA</strong> es una plataforma avanzada de trading algorítmico que combina:</p>

        <ul>
        <li>🤖 <strong>A/B Testing Automatizado</strong> - Validación estadística de estrategias</li>
        <li>📊 <strong>Backtesting Robusto</strong> - Análisis histórico con Monte Carlo</li>
        <li>📈 <strong>Paper Trading en Vivo</strong> - Simulación con datos reales</li>
        <li>🔬 <strong>Análisis Avanzado</strong> - Machine Learning y técnicas cuantitativas</li>
        </ul>

        <h2>🚀 ¿Qué puedes hacer?</h2>

        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>Para Principiantes:</h3>
        <ol>
        <li>La plataforma carga automáticamente datos BTC/USD al iniciar</li>
        <li>Ve a la pestaña <strong>"▶️ Backtest"</strong> para probar estrategias predefinidas</li>
        <li>Analiza los resultados en <strong>"📈 Results Analysis"</strong></li>
        <li>Experimenta con diferentes parámetros</li>
        </ol>
        </div>

        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>Para Usuarios Avanzados:</h3>
        <ul>
        <li>Configura estrategias personalizadas en <strong>"⚙️ Strategy Config"</strong></li>
        <li>Realiza A/B testing automatizado en <strong>"🆚 A/B Testing"</strong></li>
        <li>Monitorea trading en vivo en <strong>"📊 Live Monitoring"</strong></li>
        <li>Utiliza análisis avanzado en <strong>"🔬 Advanced Analysis"</strong></li>
        </ul>
        </div>

        <h2>📚 Navegación por la Ayuda</h2>
        <p>Utiliza el panel izquierdo para explorar temas específicos. Cada sección incluye:</p>
        <ul>
        <li>📖 <strong>Explicaciones detalladas</strong> de cada funcionalidad</li>
        <li>🎯 <strong>Guías paso a paso</strong> para completar tareas</li>
        <li>💡 <strong>Consejos y mejores prácticas</strong></li>
        <li>🔧 <strong>Solución de problemas</strong> comunes</li>
        </ul>

        <p><em>¡Comienza explorando las otras secciones para dominar todas las capacidades de TradingIA!</em></p>
        """
        self.content_browser.setHtml(content)

    def show_getting_started(self):
        """Show getting started guide"""
        content = """
        <h1>🚀 Primeros Pasos con TradingIA</h1>

        <h2>1. Verificación Inicial</h2>
        <p>Al iniciar la plataforma:</p>
        <ul>
        <li>✅ Los datos BTC/USD se cargan automáticamente (1 segundo)</li>
        <li>✅ El estado se muestra en la barra inferior</li>
        <li>✅ Todas las pestañas están disponibles</li>
        </ul>

        <h2>2. Tu Primer Backtest</h2>
        <ol>
        <li>Ve a la pestaña <strong>"▶️ Backtest Runner"</strong></li>
        <li>Selecciona una estrategia de la lista</li>
        <li>Haz clic en <strong>"Run Backtest"</strong></li>
        <li>Observa el progreso en tiempo real</li>
        <li>Revisa los resultados en <strong>"📈 Results Analysis"</strong></li>
        </ol>

        <h2>3. Exploración de Funcionalidades</h2>
        <p>Después del primer backtest exitoso:</p>
        <ul>
        <li>📊 <strong>Dashboard</strong> - Vista general del sistema</li>
        <li>📥 <strong>Data Management</strong> - Gestiona tus datos</li>
        <li>⚙️ <strong>Strategy Config</strong> - Personaliza estrategias</li>
        <li>🆚 <strong>A/B Testing</strong> - Compara estrategias automáticamente</li>
        </ul>

        <h2>4. Próximos Pasos</h2>
        <div style="background-color: #0e639c; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>🎯 Objetivos de Aprendizaje:</h3>
        <ul>
        <li>Comprender las métricas de rendimiento</li>
        <li>Aprender a optimizar parámetros</li>
        <li>Configurar paper trading</li>
        <li>Interpretar análisis estadísticos</li>
        </ul>
        </div>

        <p><strong>💡 Tip:</strong> No tengas prisa. Cada pestaña tiene su propia documentación detallada.</p>
        """
        self.content_browser.setHtml(content)

    def show_initial_setup(self):
        """Show initial setup guide"""
        content = """
        <h1>🔧 Configuración Inicial</h1>

        <h2>Requisitos del Sistema</h2>
        <ul>
        <li>✅ <strong>Python 3.8+</strong> - Versión recomendada: 3.11</li>
        <li>✅ <strong>8GB RAM</strong> - Mínimo para análisis complejos</li>
        <li>✅ <strong>Conexión a Internet</strong> - Para descarga de datos</li>
        <li>✅ <strong>Cuenta Alpaca</strong> - Para paper trading (opcional)</li>
        </ul>

        <h2>Archivos de Configuración</h2>

        <h3>.env (Credenciales)</h3>
        <pre style="background-color: #2d2d2d; padding: 10px; border-radius: 5px;">
# Archivo .env en la raíz del proyecto
ALPACA_API_KEY=tu_api_key_aqui
ALPACA_SECRET_KEY=tu_secret_key_aqui
ALPACA_BASE_URL=https://paper-api.alpaca.markets
        </pre>

        <h3>Dependencias</h3>
        <pre style="background-color: #2d2d2d; padding: 10px; border-radius: 5px;">
pip install -r requirements_platform.txt
        </pre>

        <h2>Verificación de Instalación</h2>
        <p>Ejecuta estos comandos para verificar:</p>
        <pre style="background-color: #2d2d2d; padding: 10px; border-radius: 5px;">
python --version              # Python 3.8+
python -c "import PyQt6"      # PyQt6 instalado
python src/main_platform.py   # Plataforma inicia
        </pre>

        <h2>Solución de Problemas Comunes</h2>
        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>❌ "PyQt6 no encontrado"</h3>
        <p>Solución: <code>pip install PyQt6</code></p>
        </div>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>❌ "Datos no se cargan"</h3>
        <p>Solución: Verificar conexión a internet y archivo .env</p>
        </div>

        <p><strong>✅ Una vez completada la configuración, la plataforma estará lista para usar.</strong></p>
        """
        self.content_browser.setHtml(content)

    def show_auto_load(self):
        """Show auto-load feature documentation"""
        content = """
        <h1>⚡ Carga Automática de Datos</h1>

        <h2>¿Cómo Funciona?</h2>
        <p>Al iniciar TradingIA, automáticamente:</p>
        <ol>
        <li>🔍 Verifica si existen datos BTC/USD locales</li>
        <li>📥 Descarga datos de 1 año si no existen</li>
        <li>⚙️ Configura el timeframe de 1 hora por defecto</li>
        <li>✅ Muestra confirmación en la barra de estado</li>
        </ol>

        <h2>Configuración por Defecto</h2>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <tr style="background-color: #2d2d2d;">
            <th style="border: 1px solid #555; padding: 8px;">Parámetro</th>
            <th style="border: 1px solid #555; padding: 8px;">Valor</th>
            <th style="border: 1px solid #555; padding: 8px;">Propósito</th>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Símbolo</td>
            <td style="border: 1px solid #555; padding: 8px;">BTC/USD</td>
            <td style="border: 1px solid #555; padding: 8px;">Par principal de criptomonedas</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Timeframe</td>
            <td style="border: 1px solid #555; padding: 8px;">1 Hora</td>
            <td style="border: 1px solid #555; padding: 8px;">Swing trading</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Período</td>
            <td style="border: 1px solid #555; padding: 8px;">365 días</td>
            <td style="border: 1px solid #555; padding: 8px;">Análisis anual completo</td>
        </tr>
        </table>

        <h2>¿Dónde se Usan los Datos?</h2>
        <ul>
        <li>▶️ <strong>Backtesting</strong> - Estrategias se ejecutan sobre estos datos</li>
        <li>📊 <strong>Análisis</strong> - Gráficos y estadísticas</li>
        <li>🆚 <strong>A/B Testing</strong> - Comparación de estrategias</li>
        <li>📈 <strong>Resultados</strong> - Métricas de rendimiento</li>
        </ul>

        <h2>Personalización</h2>
        <p>Para datos personalizados:</p>
        <ol>
        <li>Ve a <strong>"📥 Data Management"</strong></li>
        <li>Selecciona símbolo y timeframe deseado</li>
        <li>Haz clic en <strong>"Load Data"</strong></li>
        <li>Los datos se agregan al sistema</li>
        </ol>

        <div style="background-color: #28a745; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>✅ Ventajas</h3>
        <ul>
        <li>🚀 <strong>Inicio rápido</strong> - Listo para usar inmediatamente</li>
        <li>🎯 <strong>Optimizado</strong> - Configuración ideal para principiantes</li>
        <li>🔄 <strong>Flexible</strong> - Fácil agregar datos adicionales</li>
        </ul>
        </div>
        """
        self.content_browser.setHtml(content)

    def show_dashboard_overview(self):
        """Show dashboard overview"""
        content = """
        <h1>📊 Dashboard - Vista General</h1>

        <h2>¿Qué es el Dashboard?</h2>
        <p>El Dashboard es tu centro de control principal que proporciona:</p>
        <ul>
        <li>📈 <strong>Métricas del Sistema</strong> - Estado general de la plataforma</li>
        <li>⚡ <strong>Acciones Rápidas</strong> - Atajos para tareas comunes</li>
        <li>🔍 <strong>Estado del Sistema</strong> - Información en tiempo real</li>
        <li>📋 <strong>Actividad Reciente</strong> - Historial de operaciones</li>
        </ul>

        <h2>Componentes Principales</h2>

        <h3>1. Métricas del Sistema</h3>
        <p>Tarjetas que muestran:</p>
        <ul>
        <li><strong>Estrategias Activas:</strong> Número de estrategias configuradas</li>
        <li><strong>Backtests Completados:</strong> Total de simulaciones realizadas</li>
        <li><strong>Datos Cargados:</strong> Cantidad de datos disponibles</li>
        <li><strong>Rendimiento del Sistema:</strong> Estado de salud general</li>
        </ul>

        <h3>2. Acciones Rápidas</h3>
        <p>Botones para tareas comunes:</p>
        <ul>
        <li><strong>🚀 Nuevo Backtest:</strong> Inicia simulación rápida</li>
        <li><strong>📊 Ver Resultados:</strong> Últimos resultados de backtesting</li>
        <li><strong>📥 Cargar Datos:</strong> Importar nuevos datos</li>
        <li><strong>🔧 Configuración:</strong> Ajustes del sistema</li>
        </ul>

        <h3>3. Estado del Sistema</h3>
        <p>Indicadores en tiempo real:</p>
        <ul>
        <li><strong>🟢 Conectado:</strong> Sistema operativo normalmente</li>
        <li><strong>🟡 Procesando:</strong> Operación en curso</li>
        <li><strong>🔴 Error:</strong> Problema que requiere atención</li>
        </ul>

        <h2>¿Cómo Usarlo?</h2>
        <ol>
        <li><strong>Monitorea</strong> las métricas para entender el estado del sistema</li>
        <li><strong>Utiliza</strong> acciones rápidas para tareas comunes</li>
        <li><strong>Revisa</strong> el historial de actividad para seguimiento</li>
        <li><strong>Identifica</strong> problemas mediante indicadores de estado</li>
        </ol>

        <div style="background-color: #0e639c; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>💡 Consejos de Uso</h3>
        <ul>
        <li>El Dashboard se actualiza automáticamente cada 30 segundos</li>
        <li>Las métricas se calculan en tiempo real</li>
        <li>Los botones de acción rápida son accesos directos a otras pestañas</li>
        <li>El historial mantiene los últimos 100 eventos</li>
        </ul>
        </div>
        """
        self.content_browser.setHtml(content)

    def show_data_download(self):
        """Show data download documentation"""
        content = """
        <h1>📥 Descarga de Datos</h1>

        <h2>¿Para Qué Sirve?</h2>
        <p>La pestaña de descarga de datos permite:</p>
        <ul>
        <li>📊 <strong>Ver Estado Actual:</strong> Qué datos tienes disponibles</li>
        <li>📥 <strong>Descargar Nuevos Datos:</strong> Obtener datos históricos</li>
        <li>🔄 <strong>Actualizar Existentes:</strong> Refrescar datos antiguos</li>
        <li>📋 <strong>Monitorear Progreso:</strong> Seguimiento en tiempo real</li>
        </ul>

        <h2>Timeframes Disponibles</h2>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <tr style="background-color: #2d2d2d;">
            <th style="border: 1px solid #555; padding: 8px;">Timeframe</th>
            <th style="border: 1px solid #555; padding: 8px;">Archivo</th>
            <th style="border: 1px solid #555; padding: 8px;">Uso Típico</th>
            <th style="border: 1px solid #555; padding: 8px;">Tamaño Aprox.</th>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">5 Minutos</td>
            <td style="border: 1px solid #555; padding: 8px;">btc_usd_5m.csv</td>
            <td style="border: 1px solid #555; padding: 8px;">Scalping</td>
            <td style="border: 1px solid #555; padding: 8px;">Grande</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">15 Minutos</td>
            <td style="border: 1px solid #555; padding: 8px;">btc_usd_15m.csv</td>
            <td style="border: 1px solid #555; padding: 8px;">Day Trading</td>
            <td style="border: 1px solid #555; padding: 8px;">Mediano</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">1 Hora</td>
            <td style="border: 1px solid #555; padding: 8px;">btc_usd_1h.csv</td>
            <td style="border: 1px solid #555; padding: 8px;">Swing Trading</td>
            <td style="border: 1px solid #555; padding: 8px;">Pequeño</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">4 Horas</td>
            <td style="border: 1px solid #555; padding: 8px;">btc_usd_4h.csv</td>
            <td style="border: 1px solid #555; padding: 8px;">Position Trading</td>
            <td style="border: 1px solid #555; padding: 8px;">Muy Pequeño</td>
        </tr>
        </table>

        <h2>Cómo Descargar Datos</h2>
        <ol>
        <li>Ve a la pestaña <strong>"📥 Data Download"</strong></li>
        <li>Haz clic en <strong>"🔄 Refresh Status"</strong> para ver archivos existentes</li>
        <li>Selecciona un timeframe faltante de la lista</li>
        <li>Haz clic en <strong>"📥 Download Selected"</strong></li>
        <li>Observa el progreso en el panel derecho</li>
        </ol>

        <h3>Descarga Masiva</h3>
        <p>Para descargar todos los timeframes faltantes:</p>
        <ol>
        <li>Haz clic en <strong>"📦 Download All Missing"</strong></li>
        <li>El sistema descargará automáticamente todos los datos necesarios</li>
        <li>El progreso se muestra para cada descarga individual</li>
        </ol>

        <h2>¿Dónde se Guardan los Datos?</h2>
        <pre style="background-color: #2d2d2d; padding: 10px; border-radius: 5px;">
data/
├── processed/
│   ├── btc_usd_1h.csv    # Datos procesados
│   ├── btc_usd_4h.csv
│   └── ...
└── raw/
    ├── btc_usd_5m_raw.csv   # Datos crudos
    ├── btc_usd_15m_raw.csv
    └── ...
        </pre>

        <h2>Solución de Problemas</h2>
        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>❌ "Conexión fallida"</h3>
        <ul>
        <li>Verifica tu conexión a internet</li>
        <li>Comprueba las credenciales de Alpaca en .env</li>
        <li>Intenta descargar un timeframe más pequeño primero</li>
        </ul>
        </div>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>❌ "Descarga lenta"</h3>
        <ul>
        <li>Los timeframes más pequeños toman más tiempo</li>
        <li>Descarga durante horas de menor actividad</li>
        <li>Considera descargar datos históricos por separado</li>
        </ul>
        </div>
        """
        self.content_browser.setHtml(content)

    def show_backtest_execution(self):
        """Show backtest execution guide"""
        content = """
        <h1>▶️ Ejecución de Backtests</h1>

        <h2>¿Qué es un Backtest?</h2>
        <p>Un backtest es una simulación histórica que:</p>
        <ul>
        <li>🎯 <strong>Evalúa Estrategias:</strong> Prueba el rendimiento pasado</li>
        <li>📊 <strong>Calcula Métricas:</strong> Sharpe, Drawdown, Win Rate</li>
        <li>🔍 <strong>Valida Ideas:</strong> Confirma si una estrategia funciona</li>
        <li>⚡ <strong>Optimiza Parámetros:</strong> Encuentra mejores configuraciones</li>
        </ul>

        <h2>Cómo Ejecutar un Backtest</h2>
        <ol>
        <li>Ve a la pestaña <strong>"▶️ Backtest Runner"</strong></li>
        <li>Selecciona una estrategia del menú desplegable</li>
        <li>Configura los parámetros (opcional)</li>
        <li>Selecciona el período de prueba</li>
        <li>Haz clic en <strong>"▶️ Run Backtest"</strong></li>
        </ol>

        <h2>Estrategias Disponibles</h2>
        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>📈 Estrategias de Tendencia</h3>
        <ul>
        <li><strong>Momentum MACD ADX:</strong> Combina momentum con indicadores técnicos</li>
        <li><strong>HFT Momentum VMA:</strong> Alta frecuencia con volume analysis</li>
        </ul>
        </div>

        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>🔄 Estrategias de Reversión</h3>
        <ul>
        <li><strong>Mean Reversion IBS BB:</strong> Reversión a la media con Bollinger Bands</li>
        <li><strong>RSI Mean Reversion:</strong> Usa RSI para identificar reversiones</li>
        </ul>
        </div>

        <h2>Parámetros de Configuración</h2>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <tr style="background-color: #2d2d2d;">
            <th style="border: 1px solid #555; padding: 8px;">Parámetro</th>
            <th style="border: 1px solid #555; padding: 8px;">Descripción</th>
            <th style="border: 1px solid #555; padding: 8px;">Valor Típico</th>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Capital Inicial</td>
            <td style="border: 1px solid #555; padding: 8px;">Dinero para simular</td>
            <td style="border: 1px solid #555; padding: 8px;">$10,000</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Comisión</td>
            <td style="border: 1px solid #555; padding: 8px;">Costo por trade</td>
            <td style="border: 1px solid #555; padding: 8px;">0.1%</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Stop Loss</td>
            <td style="border: 1px solid #555; padding: 8px;">Límite de pérdida</td>
            <td style="border: 1px solid #555; padding: 8px;">2%</td>
        </tr>
        </table>

        <h2>Interpretación de Resultados</h2>
        <p>Después del backtest, revisa:</p>
        <ul>
        <li><strong>📈 Gráfico de Equity:</strong> Curva de crecimiento del capital</li>
        <li><strong>📊 Métricas:</strong> Sharpe ratio, máximo drawdown, win rate</li>
        <li><strong>📋 Trades:</strong> Lista detallada de todas las operaciones</li>
        <li><strong>🔍 Análisis:</strong> Períodos de ganancia vs pérdida</li>
        </ul>

        <h2>Mejores Prácticas</h2>
        <div style="background-color: #28a745; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>✅ Recomendaciones</h3>
        <ul>
        <li><strong>Out-of-Sample:</strong> Prueba en datos no usados para optimización</li>
        <li><strong>Walk-Forward:</strong> Validación temporal robusta</li>
        <li><strong>Realistic Assumptions:</strong> Comisiones y slippage realistas</li>
        <li><strong>Multiple Timeframes:</strong> Prueba en diferentes marcos temporales</li>
        </ul>
        </div>

        <div style="background-color: #ffc107; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>⚠️ Errores Comunes</h3>
        <ul>
        <li><strong>Overfitting:</strong> Optimización excesiva para datos históricos</li>
        <li><strong>Look-ahead Bias:</strong> Uso de información futura</li>
        <li><strong>Survivorship Bias:</strong> Solo considerar activos sobrevivientes</li>
        </ul>
        </div>
        """
        self.content_browser.setHtml(content)

    def show_ab_test_config(self):
        """Show A/B testing configuration"""
        content = """
        <h1>🆚 Configuración de A/B Testing</h1>

        <h2>¿Qué es A/B Testing en Trading?</h2>
        <p>El A/B Testing automatizado compara dos estrategias para determinar:</p>
        <ul>
        <li>📊 <strong>Cual es mejor:</strong> Basado en métricas estadísticas</li>
        <li>🎯 <strong>Significancia:</strong> Si las diferencias son reales o aleatorias</li>
        <li>⚡ <strong>Robustez:</strong> Rendimiento consistente en diferentes condiciones</li>
        <li>🔍 <strong>Recomendaciones:</strong> Sugerencias automáticas de mejora</li>
        </ul>

        <h2>Cómo Configurar un A/B Test</h2>
        <ol>
        <li>Ve a la pestaña <strong>"🆚 A/B Testing"</strong></li>
        <li>Selecciona <strong>"Estrategia A"</strong> del primer menú</li>
        <li>Selecciona <strong>"Estrategia B"</strong> del segundo menú</li>
        <li>Configura parámetros específicos para cada estrategia</li>
        <li>Define el período de prueba</li>
        <li>Haz clic en <strong>"▶️ Run A/B Test"</strong></li>
        </ol>

        <h2>Tipos de Comparaciones</h2>
        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>🔄 Variaciones de Parámetros</h3>
        <p>Compara la misma estrategia con diferentes configuraciones:</p>
        <ul>
        <li>Estrategia A: Stop Loss 1%, Take Profit 2%</li>
        <li>Estrategia B: Stop Loss 2%, Take Profit 4%</li>
        </ul>
        </div>

        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>📈 Estrategias Diferentes</h3>
        <p>Compara enfoques completamente diferentes:</p>
        <ul>
        <li>Estrategia A: Momentum MACD</li>
        <li>Estrategia B: Mean Reversion RSI</li>
        </ul>
        </div>

        <h2>Análisis Estadístico Automático</h2>
        <p>El sistema calcula automáticamente:</p>
        <ul>
        <li><strong>t-test:</strong> Diferencia significativa entre rendimientos</li>
        <li><strong>p-value:</strong> Probabilidad de que el resultado sea aleatorio</li>
        <li><strong>Confidence Intervals:</strong> Rango probable del rendimiento real</li>
        <li><strong>Effect Size:</strong> Magnitud de la diferencia entre estrategias</li>
        </ul>

        <h2>Interpretación de Resultados</h2>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <tr style="background-color: #2d2d2d;">
            <th style="border: 1px solid #555; padding: 8px;">p-value</th>
            <th style="border: 1px solid #555; padding: 8px;">Significancia</th>
            <th style="border: 1px solid #555; padding: 8px;">Conclusión</th>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">p < 0.01</td>
            <td style="border: 1px solid #555; padding: 8px;">Muy Significativa</td>
            <td style="border: 1px solid #555; padding: 8px;">Diferencia real con alta confianza</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">0.01 ≤ p < 0.05</td>
            <td style="border: 1px solid #555; padding: 8px;">Significativa</td>
            <td style="border: 1px solid #555; padding: 8px;">Diferencia probablemente real</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">0.05 ≤ p < 0.10</td>
            <td style="border: 1px solid #555; padding: 8px;">Marginal</td>
            <td style="border: 1px solid #555; padding: 8px;">Diferencia posible pero incierta</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">p ≥ 0.10</td>
            <td style="border: 1px solid #555; padding: 8px;">No Significativa</td>
            <td style="border: 1px solid #555; padding: 8px;">Diferencia probablemente aleatoria</td>
        </tr>
        </table>

        <h2>Recomendaciones Automáticas</h2>
        <p>Basado en el análisis, el sistema recomienda:</p>
        <ul>
        <li><strong>✅ Estrategia Ganadora:</strong> Si hay diferencia significativa</li>
        <li><strong>🔄 Optimización:</strong> Si ninguna es claramente superior</li>
        <li><strong>📊 Más Datos:</strong> Si los resultados son inconclusos</li>
        <li><strong>⚠️ Riesgo:</strong> Si ambas estrategias tienen alto riesgo</li>
        </ul>

        <div style="background-color: #28a745; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>💡 Mejores Prácticas</h3>
        <ul>
        <li><strong>Mismo Período:</strong> Compara estrategias en los mismos datos</li>
        <li><strong>Múltiples Tests:</strong> Repite el test con diferentes subconjuntos</li>
        <li><strong>Robustez:</strong> Prueba en diferentes condiciones de mercado</li>
        <li><strong>Documentación:</strong> Registra todas las configuraciones probadas</li>
        </ul>
        </div>
        """
        self.content_browser.setHtml(content)

    def show_paper_trading(self):
        """Show paper trading documentation"""
        content = """
        <h1>📊 Paper Trading - Simulación en Vivo</h1>

        <h2>¿Qué es Paper Trading?</h2>
        <p>El paper trading simula operaciones reales sin riesgo financiero:</p>
        <ul>
        <li>💰 <strong>Sin Dinero Real:</strong> Usa saldo virtual</li>
        <li>📈 <strong>Datos Reales:</strong> Precios y condiciones del mercado real</li>
        <li>🎯 <strong>Validación:</strong> Prueba estrategias antes de usar dinero real</li>
        <li>📊 <strong>Análisis:</strong> Métricas realistas de rendimiento</li>
        </ul>

        <h2>Cómo Configurar Paper Trading</h2>
        <ol>
        <li>Ve a la pestaña <strong>"📊 Live Monitoring"</strong></li>
        <li>Configura tu conexión con Alpaca (credenciales en .env)</li>
        <li>Selecciona una estrategia para ejecutar</li>
        <li>Define el capital virtual inicial</li>
        <li>Haz clic en <strong>"▶️ Start Paper Trading"</strong></li>
        </ol>

        <h2>Requisitos</h2>
        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>🔑 Credenciales de Alpaca</h3>
        <p>Archivo <code>.env</code> debe contener:</p>
        <pre style="background-color: #1e1e1e; padding: 10px; border-radius: 5px;">
ALPACA_API_KEY=tu_api_key_aqui
ALPACA_SECRET_KEY=tu_secret_key_aqui
ALPACA_BASE_URL=https://paper-api.alpaca.markets
        </pre>
        </div>

        <h2>Características del Paper Trading</h2>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <tr style="background-color: #2d2d2d;">
            <th style="border: 1px solid #555; padding: 8px;">Característica</th>
            <th style="border: 1px solid #555; padding: 8px;">Descripción</th>
            <th style="border: 1px solid #555; padding: 8px;">Beneficio</th>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Tiempo Real</td>
            <td style="border: 1px solid #555; padding: 8px;">Opera con datos en vivo</td>
            <td style="border: 1px solid #555; padding: 8px;">Condiciones realistas</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Sin Riesgo</td>
            <td style="border: 1px solid #555; padding: 8px;">Solo saldo virtual</td>
            <td style="border: 1px solid #555; padding: 8px;">Aprendizaje seguro</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Comisiones</td>
            <td style="border: 1px solid #555; padding: 8px;">Cálculo realista</td>
            <td style="border: 1px solid #555; padding: 8px;">Estimación precisa</td>
        </tr>
        <tr>
            <td style="border: 1px solid #555; padding: 8px;">Monitoreo</td>
            <td style="border: 1px solid #555; padding: 8px;">Dashboard en tiempo real</td>
            <td style="border: 1px solid #555; padding: 8px;">Seguimiento continuo</td>
        </tr>
        </table>

        <h2>Monitoreo en Tiempo Real</h2>
        <p>Durante el paper trading, monitorea:</p>
        <ul>
        <li><strong>📊 Posición Actual:</strong> Estado de la cartera</li>
        <li><strong>💰 P&L:</strong> Ganancias y pérdidas en tiempo real</li>
        <li><strong>📈 Gráfico de Equity:</strong> Evolución del capital</li>
        <li><strong>📋 Historial de Trades:</strong> Todas las operaciones realizadas</li>
        <li><strong>⚠️ Alertas:</strong> Señales importantes del sistema</li>
        </ul>

        <h2>Estrategias para Paper Trading</h2>
        <p>Recomendaciones para empezar:</p>
        <div style="background-color: #0e639c; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>🎯 Estrategias Conservadoras</h3>
        <ul>
        <li><strong>Mean Reversion:</strong> Baja frecuencia, menor riesgo</li>
        <li><strong>Trend Following:</strong> Sigue tendencias establecidas</li>
        <li><strong>Breakout:</strong> Espera confirmación de ruptura</li>
        </ul>
        </div>

        <h2>Transición a Trading Real</h2>
        <p>Antes de usar dinero real:</p>
        <ol>
        <li><strong>✅ Validación:</strong> Estrategia probada en backtesting</li>
        <li><strong>✅ Paper Trading:</strong> Al menos 3 meses de simulación</li>
        <li><strong>✅ Métricas:</strong> Sharpe > 1.5, Drawdown < 10%</li>
        <li><strong>✅ Capital Inicial:</strong> Empieza pequeño (1-5% del capital total)</li>
        <li><strong>✅ Monitoreo:</strong> Sigue todas las operaciones</li>
        </ol>

        <div style="background-color: #ffc107; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>⚠️ Consideraciones Importantes</h3>
        <ul>
        <li><strong>Psicología:</strong> El paper trading no replica el estrés emocional</li>
        <li><strong>Slippage:</strong> Las ejecuciones reales pueden diferir</li>
        <li><strong>Horas de Mercado:</strong> Solo opera durante horarios de trading</li>
        <li><strong>Mantenimiento:</strong> Revisa y ajusta la estrategia regularmente</li>
        </ul>
        </div>
        """
        self.content_browser.setHtml(content)

    def show_common_issues(self):
        """Show common issues and solutions"""
        content = """
        <h1>❓ Problemas Comunes y Soluciones</h1>

        <h2>🔧 Problemas de Instalación</h2>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"PyQt6 no encontrado"</h3>
        <p><strong>Síntomas:</strong> Error al iniciar la aplicación</p>
        <p><strong>Solución:</strong></p>
        <pre style="background-color: #1e1e1e; padding: 10px; border-radius: 5px;">
pip install PyQt6
# O si usas conda:
conda install pyqt
        </pre>
        </div>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Dependencias faltantes"</h3>
        <p><strong>Síntomas:</strong> ImportError en varios módulos</p>
        <p><strong>Solución:</strong></p>
        <pre style="background-color: #1e1e1e; padding: 10px; border-radius: 5px;">
pip install -r requirements_platform.txt
        </pre>
        </div>

        <h2>📊 Problemas de Datos</h2>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Datos no se cargan automáticamente"</h3>
        <p><strong>Síntomas:</strong> Plataforma inicia pero sin datos BTC/USD</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Verificar conexión a internet</li>
        <li>Comprobar archivo .env con credenciales válidas</li>
        <li>Usar la pestaña "📥 Data Download" para descargar manualmente</li>
        </ul>
        </div>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Descarga de datos falla"</h3>
        <p><strong>Síntomas:</strong> Error durante descarga de datos históricos</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Verificar credenciales de Alpaca en .env</li>
        <li>Intentar descargar timeframes más pequeños primero</li>
        <li>Comprobar límites de API de Alpaca</li>
        <li>Esperar y reintentar (posible rate limiting)</li>
        </ul>
        </div>

        <h2>▶️ Problemas de Backtesting</h2>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Backtest no inicia"</h3>
        <p><strong>Síntomas:</strong> Botón "Run Backtest" no responde</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Asegurarse de que hay datos cargados</li>
        <li>Verificar que se seleccionó una estrategia</li>
        <li>Comprobar logs en la consola para errores</li>
        <li>Reiniciar la aplicación si es necesario</li>
        </ul>
        </div>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Resultados de backtest vacíos"</h3>
        <p><strong>Síntomas:</strong> Backtest termina pero sin trades</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Verificar período de datos disponible</li>
        <li>Revisar parámetros de la estrategia</li>
        <li>Comprobar condiciones de entrada/salida</li>
        <li>Usar período más largo de datos históricos</li>
        </ul>
        </div>

        <h2>🖥️ Problemas de Interfaz</h2>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Aplicación no responde"</h3>
        <p><strong>Síntomas:</strong> Interfaz congelada</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Esperar a que termine la operación en curso</li>
        <li>Verificar uso de CPU/memoria</li>
        <li>Reiniciar la aplicación</li>
        <li>Comprobar logs para operaciones largas</li>
        </ul>
        </div>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Gráficos no se muestran"</h3>
        <p><strong>Síntomas:</strong> Pestañas de análisis sin gráficos</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Instalar plotly: <code>pip install plotly</code></li>
        <li>Verificar que hay datos para graficar</li>
        <li>Comprobar configuración de matplotlib</li>
        </ul>
        </div>

        <h2>🔑 Problemas de API</h2>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Error de autenticación Alpaca"</h3>
        <p><strong>Síntomas:</strong> Errores 401/403 de Alpaca</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Verificar API key y secret en .env</li>
        <li>Confirmar que son credenciales de paper trading</li>
        <li>Comprobar expiración de keys</li>
        <li>Regenerar keys en Alpaca si es necesario</li>
        </ul>
        </div>

        <div style="background-color: #8B0000; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>"Rate limiting de Alpaca"</h3>
        <p><strong>Síntomas:</strong> Errores 429, descargas lentas</p>
        <p><strong>Soluciones:</strong></p>
        <ul>
        <li>Esperar entre descargas</li>
        <li>Descargar timeframes más grandes (menos requests)</li>
        <li>Usar datos históricos locales cuando sea posible</li>
        <li>Considerar upgrade del plan Alpaca</li>
        </ul>
        </div>

        <h2>🔧 Solución General de Problemas</h2>

        <div style="background-color: #28a745; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>🐛 Pasos para Diagnosticar</h3>
        <ol>
        <li><strong>Revisar Logs:</strong> Verificar logs/trading.log para errores detallados</li>
        <li><strong>Consola:</strong> Ejecutar desde terminal para ver mensajes de error</li>
        <li><strong>Dependencias:</strong> Verificar instalación con <code>pip list</code></li>
        <li><strong>Entorno Virtual:</strong> Asegurarse de usar el entorno correcto</li>
        <li><strong>Reinicio:</strong> Cerrar y reabrir la aplicación</li>
        </ol>
        </div>

        <h2>📞 Soporte Adicional</h2>
        <p>Si los problemas persisten:</p>
        <ul>
        <li><strong>📧 Comunidad:</strong> Buscar en foros de GitHub</li>
        <li><strong>📋 Issues:</strong> Reportar bugs en el repositorio</li>
        <li><strong>📖 Documentación:</strong> Revisar README completo</li>
        <li><strong>🔄 Actualización:</strong> Verificar última versión</li>
        </ul>

        <div style="background-color: #0e639c; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h3>💡 Prevención</h3>
        <ul>
        <li>Mantener dependencias actualizadas</li>
        <li>Hacer backup regular de configuraciones</li>
        <li>Probar nuevas versiones en entorno separado</li>
        <li>Documentar cambios y configuraciones</li>
        </ul>
        </div>
        """
        self.content_browser.setHtml(content)