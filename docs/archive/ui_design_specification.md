¡Claro que sí! Como experto diseñador de UI/UX y desarrollador de PySide6, con experiencia en aplicaciones financieras, crearé un diseño completo y detallado para tu "Trading IA Platform". Dado que la solicitud es extensa y detalla 9 pestañas principales con múltiples subsecciones para cada una, procederé a desglosar el diseño para cada pestaña.

Para empezar, aquí tienes la primera de las 9 pestañas. Por favor, revisa el diseño del Dashboard y dime si quieres que procedamos con el resto o si deseas algún ajuste inicial.

Contexto General de la Aplicación: Plataforma de trading algorítmico en PySide6 para backtesting, análisis y ejecución de estrategias con IA en criptomonedas y otros activos. El estilo es moderno, profesional y financiero, con colores oscuros y acentos en azul/verde/rojo, priorizando el espacio para gráficos y datos.

Diseño de la Pestaña 0: Dashboard (platform_gui_tab0.py)
1. Descripción funcional: Esta pestaña actúa como el centro de mando de la aplicación, ofreciendo una visión general del estado del sistema, métricas clave de rendimiento y alertas recientes. Permite a los usuarios obtener un vistazo rápido del comportamiento general de sus estrategias y la salud de la plataforma, sirviendo como punto de partida para navegar a funcionalidades más detalladas.

2. Layout y UI: El layout principal utiliza un QVBoxLayout para organizar las secciones verticalmente. La parte superior contiene un QHBoxLayout para los indicadores de estado global y un resumen conciso. La sección central está dominada por gráficos de rendimiento principales, mientras que la parte inferior muestra una lista de alertas y eventos recientes.

Parte Superior (QHBoxLayout):

Panel de Estado del Sistema (Izquierda): QVBoxLayout con QLabels para mostrar el estado de la conexión a datos, estado del backtester, estado de las APIs, y un indicador de salud general (ej. "Óptimo", "Advertencia", "Crítico"). Iconos visuales (círculos de color verde/amarillo/rojo) junto a cada estado.
Métricas Clave (Centro): QGridLayout o QHBoxLayout para 3-4 QFrames pequeñas, cada una con un QLabel grande para el valor (ej. "Equity Total: $X,XXX.XX") y un QLabel más pequeño para la descripción (ej. "Cambio 24h: +X.XX%"). Colores de texto (verde/rojo) para cambios positivos/negativos.
Acciones Rápidas (Derecha): QVBoxLayout con QPushButtons compactos (ej. "Iniciar Nuevo Backtest", "Ver Estrategias Activas", "Gestionar Conexiones").
Parte Central (QVBoxLayout):

Gráfico de Rendimiento General: QChartView (desde QtCharts) que muestra la evolución del equity o PnL acumulado de las estrategias monitoreadas. Se espera que ocupe la mayor parte del espacio horizontal y vertical disponible en esta sección.
Gráfico de Distribución de Rendimientos/Riesgo: Un QChartView secundario (quizás un gráfico de barras o circular) mostrando la distribución de PnL por activo o el riesgo general de la cartera.
Parte Inferior (QVBoxLayout):

Panel de Alertas y Notificaciones: QListWidget o QTableWidget compacto que muestra las 5-10 alertas más recientes (ej. "Conexión API perdida", "Drawdown > 5%", "Backtest completado"). Cada entrada incluye timestamp y tipo de alerta.
3. Componentes UI:

QTabWidget: Contenedor principal de todas las pestañas.
QWidget: Contenedores para agrupar secciones.
QVBoxLayout, QHBoxLayout, QGridLayout: Para la organización de los elementos.
QLabel:
Para títulos de sección: font-size: 16px; font-weight: bold; color: #E0E0E0;
Para métricas grandes: font-size: 24px; font-weight: bold; color: #00FF7F; (verde para positivo), #FF4500; (rojo para negativo), #FFFFFF; (blanco para neutro).
Para descripciones y estados: font-size: 12px; color: #B0B0B0;
QPushButton:
Compactos: height: 26px; border-radius: 4px; background-color: #007ACC; color: white; font-size: 12px; padding: 2px 8px;
Hover: background-color: #005B9F;
QFrame: Como contenedores estilizados para métricas clave.
Estilo: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px;
QChartView (de QtCharts): Para mostrar gráficos.
Configuración: Fondo oscuro, líneas y texto de ejes en gris claro, series de datos en azul, verde y rojo según el contexto.
QListWidget / QTableWidget: Para alertas y notificaciones.
Estilo: background-color: #1E2125; border: 1px solid #3A3F47; color: #D0D0D0; selection-background-color: #005B9F;
Filas de alerta: Texto en #FFD700; (advertencia) o #FF4500; (crítico).
4. UX/Flujo de Usuario: Al iniciar la aplicación o seleccionar la pestaña "Dashboard", el usuario ve inmediatamente el estado general y los indicadores más críticos. Los gráficos proporcionan una visión de alto nivel del rendimiento. Los botones de acción rápida permiten saltar directamente a tareas comunes, reduciendo la fricción. Las alertas en la parte inferior mantienen al usuario informado de eventos recientes sin ser intrusivas. Los colores y el tamaño de la fuente se ajustan para indicar la importancia y el tipo de dato.

5. Integración Técnica:

Estado del Sistema: Conexión directa con módulos como data_manager.py (para el estado de conexión a datos), backtester_core.py (para el estado de ejecución de backtests) y un hipotético api_manager.py. Se implementará un sistema de señales/slots para actualizar los QLabels en tiempo real.
Métricas Clave: Los valores provienen de un portfolio_manager.py o de agregados de resultados de backtests y estrategias en vivo. Actualización periódica o por eventos.
Gráficos de Rendimiento: Datos extraídos de backtester_core.py (resultados históricos) y live_monitor.py (datos en tiempo real). Se utilizarán modelos de datos para QtCharts para una actualización eficiente.
Alertas: Un logging_manager.py centralizado emitirá eventos que el Dashboard escuchará y mostrará. Las alertas críticas podrían activar QMessageBoxs adicionales.
6. Ejemplos de Código (Fragmentos PySide6):

# --- Fragmento 1: Estructura básica del Dashboard ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy
)
from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt

class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.get_stylesheet())
        self.init_ui()

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #1C1F26; /* Fondo oscuro */
                color: #D0D0D0; /* Texto claro */
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel.title {
                font-size: 16px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 5px;
            }
            QLabel.metric_value {
                font-size: 24px;
                font-weight: bold;
            }
            QLabel.metric_desc {
                font-size: 12px;
                color: #B0B0B0;
            }
            QPushButton {
                height: 26px;
                border-radius: 4px;
                background-color: #007ACC;
                color: white;
                font-size: 12px;
                padding: 2px 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
            QFrame.metric_frame {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 6px;
                padding: 10px;
            }
            QListWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                selection-background-color: #005B9F;
                alternate-background-color: #21252C;
                padding: 5px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QLabel.status_ok { color: #00FF7F; } /* Verde para OK */
            QLabel.status_warn { color: #FFD700; } /* Amarillo para Advertencia */
            QLabel.status_crit { color: #FF4500; } /* Rojo para Crítico */
        """

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Top Section: Status & Key Metrics ---
        top_section_layout = QHBoxLayout()
        top_section_layout.setSpacing(15)

        # System Status Panel
        status_panel = QFrame()
        status_panel.setObjectName("status_panel")
        status_panel.setStyleSheet("QFrame#status_panel { background-color: #282C34; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px; }")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        status_layout.addWidget(QLabel("Estado del Sistema", objectName="title"))
        self.add_status_line(status_layout, "Conexión Datos:", "Óptima", "status_ok", "data_status_label")
        self.add_status_line(status_layout, "Backtester:", "Inactivo", "status_warn", "backtester_status_label")
        self.add_status_line(status_layout, "API Binance:", "Conectado", "status_ok", "api_status_label")
        status_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        top_section_layout.addWidget(status_panel, 1)

        # Key Metrics Panel
        metrics_panel = QWidget()
        metrics_layout = QGridLayout(metrics_panel)
        metrics_layout.setSpacing(10)
        self.add_metric_widget(metrics_layout, 0, 0, "Equity Total", "$12,345.67", "+1.23%")
        self.add_metric_widget(metrics_layout, 0, 1, "PNL Hoy", "$+123.45", "+0.5%")
        self.add_metric_widget(metrics_layout, 1, 0, "Drawdown Max", "-5.8%", "-0.1%")
        self.add_metric_widget(metrics_layout, 1, 1, "Estrategias Activas", "3", "Ningún cambio")
        top_section_layout.addWidget(metrics_panel, 3)

        # Quick Actions Panel
        actions_panel = QFrame()
        actions_panel.setObjectName("actions_panel")
        actions_panel.setStyleSheet("QFrame#actions_panel { background-color: #282C34; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px; }")
        actions_layout = QVBoxLayout(actions_panel)
        actions_layout.setAlignment(Qt.AlignTop)
        actions_layout.addWidget(QLabel("Acciones Rápidas", objectName="title"))
        actions_layout.addWidget(QPushButton("Iniciar Nuevo Backtest"))
        actions_layout.addWidget(QPushButton("Ver Estrategias Activas"))
        actions_layout.addWidget(QPushButton("Gestionar Conexiones"))
        actions_layout.addStretch(1) # Pushes buttons to top
        actions_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        top_section_layout.addWidget(actions_panel, 1)

        main_layout.addLayout(top_section_layout)

        # --- Middle Section: Performance Charts (Placeholders for now) ---
        charts_section = QWidget()
        charts_layout = QVBoxLayout(charts_section)
        charts_layout.addWidget(QLabel("Gráfico de Rendimiento General (Placeholder)", objectName="title"))
        # In a real app, integrate QtCharts here:
        # from PySide6.QtCharts import QChartView, QLineSeries, QChart
        # chart_view = QChartView(QChart())
        # charts_layout.addWidget(chart_view)
        charts_layout.addWidget(QLabel("Gráfico de Distribución de Rendimientos (Placeholder)", objectName="title"))
        charts_section.setStyleSheet("background-color: #282C34; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px;")
        main_layout.addWidget(charts_section, 2) # Give more vertical space

        # --- Bottom Section: Recent Alerts ---
        alerts_section = QFrame()
        alerts_section.setStyleSheet("QFrame { background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px; }")
        alerts_layout = QVBoxLayout(alerts_section)
        alerts_layout.addWidget(QLabel("Alertas Recientes", objectName="title"))
        self.alerts_list = QListWidget()
        self.alerts_list.addItems([
            "[2023-10-27 10:30] Alerta: Conexión API a Binance perdida.",
            "[2023-10-27 10:25] Info: Backtest 'StrategyX_V2' completado exitosamente.",
            "[2023-10-27 09:15] Advertencia: Drawdown de 'StrategyY' supera el 3%.",
            "[2023-10-27 08:00] Info: Nueva versión de datos BTC/USDT descargada."
        ])
        alerts_layout.addWidget(self.alerts_list)
        main_layout.addWidget(alerts_section, 1)

    def add_status_line(self, layout, label_text, status_text, status_class, object_name):
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text, objectName="metric_desc"))
        status_label = QLabel(status_text, objectName=object_name)
        status_label.setProperty("class", status_class) # Custom property for CSS
        h_layout.addWidget(status_label)
        h_layout.addStretch(1)
        layout.addLayout(h_layout)

    def add_metric_widget(self, layout, row, col, title, value, change):
        frame = QFrame()
        frame.setProperty("class", "metric_frame")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(5,5,5,5) # Reduce padding
        frame_layout.addWidget(QLabel(title, objectName="metric_desc"))
        value_label = QLabel(value, objectName="metric_value")
        if "+" in change:
            value_label.setStyleSheet("color: #00FF7F;") # Green
        elif "-" in change:
            value_label.setStyleSheet("color: #FF4500;") # Red
        frame_layout.addWidget(value_label)
        frame_layout.addWidget(QLabel(change, objectName="metric_desc"))
        layout.addWidget(frame, row, col)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Dashboard")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(DashboardTab())
    window.setGeometry(100, 100, 1200, 800) # Larger window for desktop app
    window.show()
    sys.exit(app.exec())
# --- Fragmento 2: Actualización de estado del sistema (simulado) ---
from PySide6.QtCore import QTimer, Signal, Slot, QObject
# (Assuming DashboardTab is defined as above)

class BackendSimulator(QObject):
    data_status_changed = Signal(str, str) # status_text, status_class
    backtester_status_changed = Signal(str, str)
    api_status_changed = Signal(str, str)
    new_alert = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.simulate_updates)
        self.timer.start(5000) # Update every 5 seconds
        self.data_status_idx = 0
        self.backtester_status_idx = 0
        self.api_status_idx = 0
        self.alerts_idx = 0

        self.data_statuses = [
            ("Óptima", "status_ok"),
            ("Demorada", "status_warn")
        ]
        self.backtester_statuses = [
            ("Inactivo", "status_warn"),
            ("Ejecutando", "status_ok"),
            ("Error", "status_crit")
        ]
        self.api_statuses = [
            ("Conectado", "status_ok"),
            ("Desconectado", "status_crit")
        ]
        self.simulated_alerts = [
            "[2023-10-27 10:40] Info: Rendimiento del equity > 10% en StrategyZ.",
            "[2023-10-27 10:45] Advertencia: Conexión a datos de Kraken intermitente.",
            "[2023-10-27 10:50] Alerta: Balance de cuenta crítico bajo el umbral."
        ]

    @Slot()
    def simulate_updates(self):
        # Simulate data status change
        self.data_status_idx = (self.data_status_idx + 1) % len(self.data_statuses)
        status, css_class = self.data_statuses[self.data_status_idx]
        self.data_status_changed.emit(status, css_class)

        # Simulate backtester status change
        self.backtester_status_idx = (self.backtester_status_idx + 1) % len(self.backtester_statuses)
        status, css_class = self.backtester_statuses[self.backtester_status_idx]
        self.backtester_status_changed.emit(status, css_class)

        # Simulate API status change
        self.api_status_idx = (self.api_status_idx + 1) % len(self.api_statuses)
        status, css_class = self.api_statuses[self.api_status_idx]
        self.api_status_changed.emit(status, css_class)

        # Simulate new alert
        if self.alerts_idx < len(self.simulated_alerts):
            self.new_alert.emit(self.simulated_alerts[self.alerts_idx])
            self.alerts_idx += 1

# Connect the simulator to the dashboard (in your main app setup)
# (Assuming 'dashboard_tab' is an instance of DashboardTab)
# simulator = BackendSimulator()
# simulator.data_status_changed.connect(lambda s, c: dashboard_tab.findChild(QLabel, "data_status_label").setText(s))
# simulator.data_status_changed.connect(lambda s, c: dashboard_tab.findChild(QLabel, "data_status_label").setProperty("class", c))
# dashboard_tab.findChild(QLabel, "data_status_label").style().polish(dashboard_tab.findChild(QLabel, "data_status_label")) # Repolish for CSS update

# Repeat connections for other status labels and alerts_list.
# For alerts_list: simulator.new_alert.connect(lambda msg: dashboard_tab.alerts_list.insertItem(0, msg))
7. Mockups/Wireframes (Descripción Textual Detallada):

[ÁREA DE LA APLICACIÓN: PESTAÑA DASHBOARD]

--- BARRA DE NAVEGACIÓN SUPERIOR (TABS GLOBALES) ---
[ Dashboard | Gestión de Datos | Configuración de Estrategias | Backtesting Runner | Análisis de Resultados | Testing A/B | Monitoreo en Vivo | Análisis Avanzado | Configuración del Sistema ]
------------------------------------------------------

[CONTENIDO PRINCIPAL DE LA PESTAÑA DASHBOARD]

+---------------------------------------------------------------------------------------------------------------------------------+
|                                                   PANEL SUPERIOR                                                                |
+---------------------------------------------------------------------------------------------------------------------------------+
| [ESTADO DEL SISTEMA]                        [MÉTRICAS CLAVE]                                 [ACCIONES RÁPIDAS]                 |
| +-------------------------+                 +---------------------+  +---------------------+ +--------------------------+    |
| | **Estado del Sistema**  |                 | **Equity Total**    |  | **PNL Hoy**         | | **Acciones Rápidas**     |    |
| | Conexión Datos:  VERDE  |                 | $12,345.67          |  | $ +123.45           | | [Iniciar Nuevo Backtest] |    |
| | Backtester:      AMARILLO|                 | +1.23% (24h)        |  | +0.5% (Hoy)         | | [Ver Estrategias Activas]|    |
| | API Binance:     VERDE  |                 |---------------------|  |---------------------| | [Gestionar Conexiones]   |    |
| +-------------------------+                 | **Drawdown Max**    |  | **Estrategias Act.**| |                          |    |
|                                             | -5.8% (Actual)      |  | 3 (Ningún cambio)   | |                          |    |
|                                             |---------------------|  |---------------------| |                          |    |
|                                             +---------------------+  +---------------------+ +--------------------------+    |
+---------------------------------------------------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------------------------------------------------+
|                                                  GRÁFICOS DE RENDIMIENTO                                                        |
+---------------------------------------------------------------------------------------------------------------------------------+
| **Gráfico de Rendimiento General** (Equity / PnL acumulado a lo largo del tiempo)                                               |
| [Eje Y: Equity / PnL]                                                                                                           |
| |                                                                                                                               |
| |                                                                                                                               |
| |                                                                                                                               |
| |               CURVA DE EQUITY (AZUL)                                                                                          |
| |_______________________________________________________________________________________________________________________________|
| [Eje X: Tiempo (Últimas 24h / 7 días / 30 días)]                                                                                |
|                                                                                                                                 |
| **Gráfico de Distribución de Rendimientos** (Ej. PnL por activo o por estrategia, gráfico de barras/circular)                   |
| [Eje Y: PnL]                                                                                                                    |
| |                                                                                                                               |
| | |       |                                                                                                                     |
| | |       |                                                                                                                     |
| | |       |                                                                                                                     |
| |-------------------------------------------------------------------------------------------------------------------------------|
| [Eje X: Activo 1, Activo 2, Estrategia A, etc.]                                                                                 |
+---------------------------------------------------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------------------------------------------------+
|                                                    ALERTAS RECIENTES                                                              |
+---------------------------------------------------------------------------------------------------------------------------------+
| **Alertas Recientes**                                                                                                           |
| [LISTA DE ALERTAS] (Máximo 5-10 elementos, los más recientes arriba)                                                            |
| - [2023-10-27 10:50] ALERTA: Balance de cuenta crítico bajo el umbral. (ROJO)                                                    |
| - [2023-10-27 10:45] ADVERTENCIA: Conexión a datos de Kraken intermitente. (AMARILLO)                                            |
| - [2023-10-27 10:40] INFO: Rendimiento del equity > 10% en StrategyZ. (BLANCO)                                                   |
| - [2023-10-27 10:30] ALERTA: Conexión API a Binance perdida. (ROJO)                                                              |
| - [2023-10-27 10:25] INFO: Backtest 'StrategyX_V2' completado exitosamente. (BLANCO)                                             |
+---------------------------------------------------------------------------------------------------------------------------------+

--- BARRA DE ESTADO INFERIOR GLOBAL (MENSAJES DE ESTADO, INDICADORES DE CARGA) ---
[ Estado: Sistema Operativo | Última actualización: 2023-10-27 10:55:30 ]
-----------------------------------------------------------------------------
El diseño del Dashboard ha sido generado.

Dashboard Principal: Ofrece una vista consolidada del estado del sistema, métricas clave de rendimiento y alertas recientes, facilitando un monitoreo rápido y eficiente.
Gráficos de Rendimiento: Incluye placeholders para gráficos que visualizarán la evolución del equity y la distribución de rendimientos.
Acciones Rápidas: Botones compactos que permiten al usuario navegar directamente a funciones cruciales.
¿Te gustaría que continuemos con el diseño de las siguientes pestañas, o prefieres ajustar algo en el Dashboard primero?


Continúa con la "Gestión de Datos"


¡Excelente! Continuemos con la segunda pestaña. Aquí tienes el diseño detallado para la pestaña de Gestión de Datos.

Diseño de la Pestaña 1: Gestión de Datos (platform_gui_tab1_improved.py)
1. Descripción funcional: Esta pestaña es el centro para la carga, gestión, validación y visualización de datos de mercado. Permite a los usuarios seleccionar diferentes pares de trading (criptomonedas, forex, etc.), elegir timeframes, descargar datos históricos, ver los datos en formato tabular y realizar validaciones básicas. Es fundamental para asegurar la calidad de los datos utilizados en el backtesting y las estrategias en vivo.

2. Layout y UI: El layout principal de la pestaña utiliza un QHBoxLayout para dividir la pantalla en un panel de control a la izquierda y un área de visualización de datos principal a la derecha. El panel de control contendrá los selectores de pares, timeframes y botones de acción, mientras que el área principal mostrará una tabla de datos y un gráfico de precios.

Panel de Control Izquierdo (QVBoxLayout - ancho fijo, ej. 250px):

Selección de Activo/Par: QLabel "Activo/Par:" seguido de un QComboBox para seleccionar pares (ej. "BTC/USDT", "ETH/USDT", "EUR/USD").
Selección de Timeframe: QLabel "Timeframe:" seguido de un QComboBox para seleccionar la granularidad del dato (ej. "1m", "5m", "1h", "1D").
Rango de Fechas: QLabel "Rango de Fechas:" seguido de dos QDateEdit (inicio y fin) para definir el período de los datos.
Botones de Acción (QVBoxLayout para agrupar):
QPushButton "Cargar Datos Locales": Para cargar datos desde un archivo local.
QPushButton "Descargar Datos": Para iniciar la descarga de datos desde un exchange/API.
QPushButton "Validar Datos": Para ejecutar verificaciones de integridad y calidad.
QPushButton "Limpiar Cache": Para eliminar datos almacenados.
Indicador de Estado/Progreso: QLabel o QProgressBar para mostrar el estado de la carga/descarga (ej. "Descargando 10/1000", "Datos cargados: 10,000 barras").
QSpacerItem: Para empujar los elementos hacia la parte superior.
Área Principal de Visualización (QVBoxLayout - expandible):

Gráfico de Precios (Candlestick): QChartView (de QtCharts) mostrando el gráfico de velas japonesas del par y timeframe seleccionados. Ocupa la mitad superior de esta área. Incluye controles de zoom y pan.
Tabla de Datos Crudos: QTableWidget mostrando los datos en formato tabular (Fecha/Hora, Open, High, Low, Close, Volume). Ocupa la mitad inferior de esta área. Columnas redimensionables y cabeceras claras.
3. Componentes UI:

QTabWidget: Contenedor principal de todas las pestañas.
QWidget: Contenedores para agrupar secciones.
QHBoxLayout, QVBoxLayout: Para la organización de los elementos.
QLabel:
Para títulos de sección: font-size: 14px; font-weight: bold; color: #E0E0E0; margin-bottom: 5px;
Para etiquetas de entrada: font-size: 12px; color: #B0B0B0;
Para indicadores de estado: font-size: 11px; color: #B0B0B0;
QComboBox:
Estilo: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 4px; padding: 2px 5px; color: #D0D0D0; height: 26px;
Flecha: selection-background-color: #007ACC;
QDateEdit:
Estilo: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 4px; padding: 2px 5px; color: #D0D0D0; height: 26px;
QPushButton:
Compactos: height: 26px; border-radius: 4px; background-color: #007ACC; color: white; font-size: 12px; padding: 2px 8px; border: none;
Hover: background-color: #005B9F;
QTableWidget:
Estilo: background-color: #1E2125; border: 1px solid #3A3F47; color: #D0D0D0; gridline-color: #3A3F47; selection-background-color: #005B9F;
Cabeceras: QHeaderView::section { background-color: #282C34; color: #E0E0E0; padding: 4px; border: 1px solid #3A3F47; font-weight: bold; }
QChartView (de QtCharts): Para gráficos de velas.
Configuración: Fondo oscuro, rejilla tenue, velas alcistas en verde, bajistas en rojo. Ejes en gris claro.
QProgressBar: height: 8px; border-radius: 4px; text-align: center; background-color: #282C34; border: 1px solid #3A3F47;
QProgressBar::chunk { background-color: #007ACC; border-radius: 3px; }
4. UX/Flujo de Usuario: El usuario selecciona un par y un timeframe en el panel izquierdo. Los QDateEdit se ajustan para un rango de fechas razonable por defecto, pero el usuario puede modificarlos. Al hacer clic en "Descargar Datos", se muestra un indicador de progreso mientras se obtienen los datos. Una vez cargados, el gráfico de velas se actualiza y la tabla de datos crudos se llena. La tabla permite una inspección rápida de los valores individuales, mientras que el gráfico ofrece una visión visual de la acción del precio. Los mensajes de estado en la parte inferior de la ventana global (o en el panel izquierdo) informan al usuario sobre el éxito o fracaso de las operaciones.

5. Integración Técnica:

data_manager.py: Un módulo centralizado que maneja la conexión a APIs de exchange (ej. CCXT), la descarga de datos históricos (OHLCV), el almacenamiento local (ej. CSV, HDF5), la carga de datos desde el almacenamiento y las operaciones de validación (ej. detección de gaps, duplicados).
data_manager.py (métodos expuestos):
download_ohlcv(symbol, timeframe, start_date, end_date): Retorna un DataFrame de Pandas.
load_local_data(filepath): Carga datos desde un archivo.
validate_data(dataframe): Retorna un reporte de validación.
Señales/Slots:
Los QComboBox y QDateEdit emiten señales currentIndexChanged o dateChanged que pueden disparar una pre-visualización o preparar los parámetros para la descarga.
El QPushButton "Descargar Datos" conectará a un slot que invoca data_manager.download_ohlcv en un hilo separado (QThread) para evitar bloquear la UI.
Las actualizaciones de progreso y finalización de la descarga/carga se emitirán desde el data_manager (o el QThread) y se conectarán a slots que actualizan el QProgressBar y la QTableWidget/QChartView.
6. Ejemplos de Código (Fragmentos PySide6):

# --- Fragmento 1: Estructura básica de la pestaña Gestión de Datos ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDateEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QDate, Signal, Slot, QThread
from PySide6.QtGui import QColor, QPalette

# Para el gráfico (Placeholder, necesitarías instalar PySide6-QtCharts)
# from PySide6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis

class DataManagementTab(QWidget):
    # Signals for backend communication (example)
    data_download_requested = Signal(str, str, QDate, QDate)
    load_local_data_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.get_stylesheet())
        self.init_ui()

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #1C1F26;
                color: #D0D0D0;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel.section_title {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 5px;
            }
            QLabel.label_input {
                font-size: 12px;
                color: #B0B0B0;
            }
            QComboBox, QDateEdit {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 4px;
                padding: 2px 5px;
                color: #D0D0D0;
                height: 26px;
                selection-background-color: #007ACC;
            }
            QComboBox::drop-down {
                border-left: 1px solid #3A3F47;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(path/to/your/arrow_icon.png); /* You might need a custom icon */
            }
            QPushButton {
                height: 26px;
                border-radius: 4px;
                background-color: #007ACC;
                color: white;
                font-size: 12px;
                padding: 2px 8px;
                border: none;
                margin-top: 5px; /* Spacing between buttons */
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
            QTableWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                gridline-color: #3A3F47;
                selection-background-color: #005B9F;
                alternate-background-color: #21252C;
            }
            QHeaderView::section {
                background-color: #282C34;
                color: #E0E0E0;
                padding: 4px;
                border: 1px solid #3A3F47;
                font-weight: bold;
            }
            QProgressBar {
                height: 8px;
                border-radius: 4px;
                text-align: center;
                background-color: #282C34;
                border: 1px solid #3A3F47;
                color: transparent; /* Hide text, or show custom text */
            }
            QProgressBar::chunk {
                background-color: #007ACC;
                border-radius: 3px;
            }
        """

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Left Control Panel ---
        control_panel = QFrame()
        control_panel.setFixedWidth(250)
        control_panel.setStyleSheet("QFrame { background-color: #282C34; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px; }")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.setSpacing(8)

        control_layout.addWidget(QLabel("Activo/Par:", objectName="label_input"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTC/USDT", "ETH/USDT", "EUR/USD", "XAU/USD"])
        control_layout.addWidget(self.symbol_combo)

        control_layout.addWidget(QLabel("Timeframe:", objectName="label_input"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1D"])
        control_layout.addWidget(self.timeframe_combo)

        control_layout.addWidget(QLabel("Rango de Fechas:", objectName="label_input"))
        self.date_start = QDateEdit(QDate.currentDate().addYears(-1))
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        control_layout.addWidget(self.date_start)
        self.date_end = QDateEdit(QDate.currentDate())
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        control_layout.addWidget(self.date_end)

        # Action Buttons
        control_layout.addWidget(QPushButton("Cargar Datos Locales", clicked=self.load_local_data))
        self.download_button = QPushButton("Descargar Datos", clicked=self.download_data)
        control_layout.addWidget(self.download_button)
        control_layout.addWidget(QPushButton("Validar Datos", clicked=self.validate_data))
        control_layout.addWidget(QPushButton("Limpiar Cache", clicked=self.clear_cache))

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Listo.", objectName="label_input")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        control_layout.addStretch(1) # Pushes widgets to the top

        main_layout.addWidget(control_panel)

        # --- Right Main Data Visualization Area ---
        data_viz_area = QWidget()
        data_viz_layout = QVBoxLayout(data_viz_area)
        data_viz_layout.setSpacing(10)

        # Chart Placeholder (replace with actual QChartView)
        self.chart_view = QWidget() # Placeholder
        self.chart_view.setStyleSheet("background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 6px;")
        chart_label = QLabel("Gráfico de Precios (Candlestick) - Placeholder", objectName="section_title")
        chart_label.setAlignment(Qt.AlignCenter)
        chart_layout_placeholder = QVBoxLayout(self.chart_view)
        chart_layout_placeholder.addWidget(chart_label)
        self.chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        data_viz_layout.addWidget(self.chart_view, 1) # Give it 1/2 vertical space

        # Data Table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(["Fecha/Hora", "Open", "High", "Low", "Close", "Volume"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) # Makes columns fill space
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers) # Make table read-only
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        data_viz_layout.addWidget(self.data_table, 1) # Give it 1/2 vertical space

        main_layout.addWidget(data_viz_area, 1) # Takes remaining horizontal space

    @Slot()
    def download_data(self):
        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        start_date = self.date_start.date()
        end_date = self.date_end.date()
        self.status_label.setText(f"Iniciando descarga de {symbol} {timeframe}...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.download_button.setEnabled(False)
        # Emit signal to a backend worker thread
        self.data_download_requested.emit(symbol, timeframe, start_date, end_date)
        # In a real app, this would be handled by a QThread Worker
        # For demo: self.update_data_table_and_chart(self.generate_sample_data())
        # self.status_label.setText("Descarga simulada completada.")
        # self.progress_bar.setVisible(False)
        # self.download_button.setEnabled(True)

    @Slot()
    def load_local_data(self):
        # Implement QFileDialog for file selection
        self.status_label.setText("Cargando datos locales (simulado)...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.load_local_data_requested.emit("dummy_path.csv")
        # For demo: self.update_data_table_and_chart(self.generate_sample_data())
        # self.status_label.setText("Carga de datos locales simulada completada.")
        # self.progress_bar.setVisible(False)

    @Slot()
    def validate_data(self):
        self.status_label.setText("Validando datos cargados (simulado)...")
        # Simulate validation result
        self.status_label.setText("Validación completada. No se encontraron gaps significativos.")

    @Slot()
    def clear_cache(self):
        self.status_label.setText("Cache de datos limpiada (simulado).")
        self.data_table.setRowCount(0) # Clear table
        # Clear chart (if using QChartView)

    def generate_sample_data(self):
        # Dummy data for demonstration
        from datetime import datetime, timedelta
        data = []
        start_dt = datetime.now() - timedelta(days=5)
        for i in range(100):
            dt = start_dt + timedelta(minutes=i)
            o = 100 + i * 0.1 + (i % 5) * 0.5
            h = o + 0.5
            l = o - 0.5
            c = o + (i % 2 - 0.5) * 0.2
            v = 1000 + i * 10
            data.append([dt.strftime("%Y-%m-%d %H:%M"), f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}", str(int(v))])
        return data

    # Slot to receive data from backend thread
    @Slot(list)
    def update_data_table_and_chart(self, data):
        self.data_table.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, item in enumerate(row_data):
                self.data_table.setItem(row_idx, col_idx, QTableWidgetItem(item))
        self.status_label.setText(f"Datos cargados: {len(data)} barras.")
        self.progress_bar.setVisible(False)
        self.download_button.setEnabled(True)
        # TODO: Update QChartView with data (requires actual QtCharts implementation)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Gestión de Datos")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(DataManagementTab())
    window.setGeometry(100, 100, 1400, 900) # Larger window for desktop app
    window.show()
    sys.exit(app.exec())
# --- Fragmento 2: Ejemplo de QThread para descarga de datos (simulado) ---
import time
from PySide6.QtCore import QThread, Signal, Slot, QDate, QObject

# (Assume data_manager.py exists with a download_ohlcv method)
# from data_manager import download_ohlcv # Placeholder for actual import

class DataDownloadWorker(QObject):
    finished = Signal(list)
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, symbol, timeframe, start_date, end_date):
        super().__init__()
        self._symbol = symbol
        self._timeframe = timeframe
        self._start_date = start_date
        self._end_date = end_date

    @Slot()
    def run(self):
        try:
            # Simulate data download process
            total_steps = 100
            downloaded_data = []
            for i in range(total_steps + 1):
                time.sleep(0.05) # Simulate network delay
                self.progress.emit(int((i / total_steps) * 100))
                # Append dummy data
                downloaded_data.append([f"2023-10-27 10:{i:02}", "100.00", "101.00", "99.00", "100.50", "1000"])

            # In a real scenario:
            # data_df = download_ohlcv(self._symbol, self._timeframe, self._start_date, self._end_date)
            # formatted_data = data_df.values.tolist() # Convert DataFrame to list of lists for display
            self.finished.emit(downloaded_data) # Emit actual data
        except Exception as e:
            self.error.emit(f"Error al descargar datos: {str(e)}")

# How to integrate in DataManagementTab (simplified):
# class DataManagementTab(...):
#     # ... (other code) ...
#     def download_data(self):
#         symbol = self.symbol_combo.currentText()
#         timeframe = self.timeframe_combo.currentText()
#         start_date = self.date_start.date()
#         end_date = self.date_end.date()
#         self.status_label.setText(f"Iniciando descarga de {symbol} {timeframe}...")
#         self.progress_bar.setValue(0)
#         self.progress_bar.setVisible(True)
#         self.download_button.setEnabled(False)

#         self.worker_thread = QThread()
#         self.worker = DataDownloadWorker(symbol, timeframe, start_date, end_date)
#         self.worker.moveToThread(self.worker_thread)

#         self.worker_thread.started.connect(self.worker.run)
#         self.worker.finished.connect(self.update_data_table_and_chart)
#         self.worker.finished.connect(self.worker_thread.quit)
#         self.worker.finished.connect(self.worker.deleteLater)
#         self.worker_thread.finished.connect(self.worker_thread.deleteLater)
#         self.worker.progress.connect(self.progress_bar.setValue)
#         self.worker.error.connect(lambda msg: self.status_label.setText(msg))
#         self.worker.error.connect(self.download_button.setEnabled(True))
#         self.worker_thread.start()
7. Mockups/Wireframes (Descripción Textual Detallada):

[ÁREA DE LA APLICACIÓN: PESTAÑA GESTIÓN DE DATOS]

--- BARRA DE NAVEGACIÓN SUPERIOR (TABS GLOBALES) ---
[ Dashboard | Gestión de Datos | Configuración de Estrategias | ... ]
------------------------------------------------------

[CONTENIDO PRINCIPAL DE LA PESTAÑA GESTIÓN DE DATOS]

+-------------------------+-------------------------------------------------------------------------------------+
|    PANEL DE CONTROL     |                               ÁREA DE VISUALIZACIÓN DE DATOS                        |
|   (Ancho Fijo ~250px)   |                                                                                     |
+-------------------------+-------------------------------------------------------------------------------------+
|                         |  +--------------------------------------------------------------------------------+ |
| **Gestión de Datos**    |  |                           GRÁFICO DE PRECIOS                                   | |
|                         |  |                                                                                  | |
| Activo/Par:             |  |   [Eje Y: Precio]                                                                | |
| [ BTC/USDT    v ]       |  |   |     _                                                                          | |
|                         |  |   |    | |                                                                       | |
| Timeframe:              |  |   |    | |                                                                       | |
| [ 1h          v ]       |  |   |____| |_______________________                                                | |
|                         |  |   [Eje X: Fecha/Hora]  (Velas Japonesas: Verdes para alcista, Rojas para bajista)| |
| Rango de Fechas:        |  |                                                                                  | |
| [ 2022-10-27  CAL ]     |  |                                                                                  | |
| [ 2023-10-27  CAL ]     |  +--------------------------------------------------------------------------------+ |
|                         |                                                                                     |
| [ Cargar Datos Locales ]|  +--------------------------------------------------------------------------------+ |
| [ Descargar Datos      ]|  |                           TABLA DE DATOS CRUDOS                                | |
| [ Validar Datos        ]|  |   +--------------------------------------------------------------------------+ | |
| [ Limpiar Cache        ]|  |   | Fecha/Hora         | Open   | High   | Low    | Close  | Volume       | | |
|                         |  |   +--------------------------------------------------------------------------+ | |
| [PROGRESO / ESTADO]     |  |   | 2023-10-27 10:00   | 29000  | 29050  | 28980  | 29020  | 123456       | | |
| [ Barra de Progreso    ]|  |   | 2023-10-27 10:01   | 29020  | 29030  | 28990  | 29010  | 110000       | | |
| [ "Listo." / "Descargando..."]| |   | 2023-10-27 10:02   | 29010  | 29040  | 29000  | 29035  | 135000       | | |
|                         |  |   | ... (Scrollable)                                                       | | |
|                         |  |   +--------------------------------------------------------------------------+ | |
|                         |  +--------------------------------------------------------------------------------+ |
+-------------------------+-------------------------------------------------------------------------------------+

--- BARRA DE ESTADO INFERIOR GLOBAL (MENSAJES DE ESTADO, INDICADORES DE CARGA) ---
[ Estado: Sistema Operativo | Última actualización: 2023-10-27 11:05:10 ]
-----------------------------------------------------------------------------
El diseño para la pestaña de Gestión de Datos ha sido generado:

Panel de Control Izquierdo: Permite una selección fácil de activos, timeframes y rangos de fechas, junto con botones para acciones clave como descargar o validar datos.
Gráfico de Precios (Candlestick): Proporciona una representación visual clara de los datos de mercado.
Tabla de Datos Crudos: Ofrece una vista tabular detallada de los datos para una inspección precisa.
¿Quieres que continuemos con la siguiente pestaña, Configuración de Estrategias, o prefieres que ajustemos algo en la Gestión de Datos?


Diseña la pestaña "Configuración de Estrategias"


¡Excelente! Continuamos avanzando con el diseño de tu plataforma. Aquí tienes el diseño detallado para la pestaña de Configuración de Estrategias.

Diseño de la Pestaña 2: Configuración de Estrategias (platform_gui_tab2_improved.py)
1. Descripción funcional: Esta pestaña permite a los usuarios crear, editar, duplicar y eliminar estrategias de trading algorítmico. Es el lugar donde se definen los parámetros específicos de cada estrategia, como indicadores, reglas de entrada/salida, gestión de riesgo y tamaño de posición. Además, ofrece la funcionalidad de realizar pruebas rápidas con la configuración actual de una estrategia.

2. Layout y UI: El diseño de esta pestaña se divide en dos secciones principales mediante un QHBoxLayout: un panel lateral izquierdo para la gestión de la lista de estrategias y un área central expandible para la configuración detallada de la estrategia seleccionada.

Panel Lateral Izquierdo (QVBoxLayout - ancho fijo, ej. 300px):

Título: QLabel "Lista de Estrategias".
Lista de Estrategias: QListWidget para mostrar los nombres de las estrategias existentes. Al seleccionar una estrategia, sus detalles se cargarán en el panel central.
Botones de Acción (QHBoxLayout):
QPushButton "Nueva": Abre un diálogo o limpia el formulario central para crear una nueva estrategia.
QPushButton "Duplicar": Crea una copia de la estrategia seleccionada.
QPushButton "Eliminar": Elimina la estrategia seleccionada (con confirmación).
Campo de Búsqueda: QLineEdit con un ícono de lupa para filtrar la lista de estrategias.
QSpacerItem: Para empujar los elementos hacia la parte superior.
Área Central de Configuración (QVBoxLayout - expandible):

Título de Estrategia: QLabel grande con el nombre de la estrategia seleccionada (ej. "Estrategia MACD Crossover").
Descripción/Notas: QTextEdit para añadir una descripción o notas sobre la estrategia.
Formulario de Parámetros (QScrollArea con un QFormLayout o QGridLayout dentro):
Los parámetros se organizan dinámicamente según el tipo de estrategia. Ejemplos:
QLabel "Activo por defecto:" QComboBox (BTC/USDT, ETH/USDT).
QLabel "Timeframe por defecto:" QComboBox (1h, 4h, 1D).
QLabel "Periodo MACD Lento:" QSpinBox.
QLabel "Periodo MACD Rápido:" QSpinBox.
QLabel "Periodo Señal MACD:" QSpinBox.
QLabel "Stop Loss (%):" QDoubleSpinBox.
QLabel "Take Profit (%):" QDoubleSpinBox.
QLabel "Tamaño de Posición (% de Capital):" QDoubleSpinBox.
QLabel "Habilitar Trailing Stop:" QCheckBox.
Los controles se generan en función de la estructura de la estrategia subyacente.
Botones de Control de Estrategia (QHBoxLayout en la parte inferior):
QPushButton "Guardar Cambios": Guarda la configuración actual de la estrategia.
QPushButton "Cancelar": Descarta los cambios no guardados.
QPushButton "Prueba Rápida": Ejecuta un micro-backtest con los parámetros actuales.
3. Componentes UI:

QTabWidget: Contenedor principal.
QWidget, QFrame: Contenedores estilizados.
QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout: Para la organización.
QLabel:
Título principal de la pestaña: font-size: 18px; font-weight: bold; color: #E0E0E0; margin-bottom: 10px;
Títulos de sección: font-size: 14px; font-weight: bold; color: #E0E0E0; margin-bottom: 5px;
Etiquetas de formulario: font-size: 12px; color: #B0B0B0;
QListWidget:
Estilo: background-color: #1E2125; border: 1px solid #3A3F47; color: #D0D0D0; selection-background-color: #007ACC;
Items: padding: 5px;
QPushButton:
Acciones de lista (Nueva, Duplicar, Eliminar): height: 24px; border-radius: 4px; background-color: #007ACC; color: white; font-size: 11px; padding: 2px 6px;
Acciones de formulario (Guardar, Cancelar, Prueba Rápida): height: 26px; border-radius: 4px; background-color: #007ACC; color: white; font-size: 12px; padding: 2px 8px;
Botón "Eliminar" con estilo rojo: background-color: #DC3545; (hover: #BD2130;)
QLineEdit (para búsqueda):
Estilo: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 4px; padding: 2px 5px; color: #D0D0D0; height: 26px;
QTextEdit (para descripción):
Estilo: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 4px; padding: 5px; color: #D0D0D0;
QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox:
Estilo consistente: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 4px; padding: 2px 5px; color: #D0D0D0; height: 26px;
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button { width: 16px; }
QScrollArea: border: none; background-color: transparent;
4. UX/Flujo de Usuario: El usuario comienza viendo una lista de sus estrategias existentes. Seleccionar una estrategia carga instantáneamente sus parámetros en el formulario central para edición. Los botones "Nueva", "Duplicar" y "Eliminar" son claros en su propósito. El formulario de parámetros es dinámico; solo se muestran los campos relevantes para la estrategia seleccionada. Los valores por defecto facilitan la creación de nuevas estrategias. "Guardar Cambios" guarda la estrategia, mientras que "Prueba Rápida" ofrece retroalimentación inmediata sobre la viabilidad de la configuración actual, sin necesidad de ir a la pestaña de Backtesting completo.

5. Integración Técnica:

strategy_manager.py: Un módulo que maneja la persistencia (carga/guardado en JSON/YAML/DB) y la lógica de las definiciones de estrategia. Contiene clases o estructuras de datos para representar estrategias, con campos para parámetros, descripción, etc.
strategy_manager.py (métodos expuestos):
load_all_strategies(): Retorna una lista de objetos/dict de estrategias.
save_strategy(strategy_obj): Guarda o actualiza una estrategia.
delete_strategy(strategy_id): Elimina una estrategia.
get_strategy_params_schema(strategy_type): Retorna un esquema (ej. JSON Schema) de los parámetros esperados para un tipo de estrategia, para generar el formulario dinámicamente.
backtester_core.py (para "Prueba Rápida"): Un método ligero que toma una definición de estrategia y un rango de datos pequeño para una ejecución rápida y devuelve un resumen.
Generación Dinámica de UI: Los campos del formulario central se generarán en tiempo de ejecución basándose en la estructura de la estrategia seleccionada, obtenida de strategy_manager.py. Esto permite una gran flexibilidad para diferentes tipos de estrategias sin codificar cada formulario manualmente.
Señales/Slots:
QListWidget.currentItemChanged para cargar la estrategia seleccionada.
Botones de acción se conectan a slots que interactúan con strategy_manager.py.
QPushButton "Prueba Rápida" se conecta a un slot que invoca a backtester_core.run_quick_test (posiblemente en un QThread para no bloquear la UI) y muestra un resumen en un diálogo emergente o un QLabel de estado.
6. Ejemplos de Código (Fragmentos PySide6):

# --- Fragmento 1: Estructura básica de la pestaña Configuración de Estrategias ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QTextEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFormLayout, QScrollArea, QFrame,
    QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QObject

class StrategyConfigTab(QWidget):
    # Signals for backend communication
    strategy_saved = Signal(dict)
    strategy_deleted = Signal(str)
    quick_test_requested = Signal(dict) # Emit strategy dict for testing

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_strategy = None # Stores the current strategy being edited
        self.strategies_data = {} # Simulates a database/manager of strategies
        self.setStyleSheet(self.get_stylesheet())
        self.init_ui()
        self.load_strategies_list() # Load initial list of strategies

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #1C1F26;
                color: #D0D0D0;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel.section_title {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 5px;
            }
            QLabel.strategy_name_title {
                font-size: 18px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 10px;
            }
            QLabel.label_form {
                font-size: 12px;
                color: #B0B0B0;
            }
            QListWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                selection-background-color: #007ACC;
                alternate-background-color: #21252C;
            }
            QListWidget::item {
                padding: 5px;
            }
            QPushButton {
                height: 26px;
                border-radius: 4px;
                background-color: #007ACC;
                color: white;
                font-size: 12px;
                padding: 2px 8px;
                border: none;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
            QPushButton#deleteButton {
                background-color: #DC3545;
            }
            QPushButton#deleteButton:hover {
                background-color: #BD2130;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 4px;
                padding: 2px 5px;
                color: #D0D0D0;
                height: 26px;
            }
            QTextEdit {
                min-height: 60px;
                height: auto; /* Allow text edit to grow */
            }
            QComboBox::drop-down {
                border-left: 1px solid #3A3F47;
                width: 20px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QFrame#strategy_form_container {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 6px;
                padding: 10px;
            }
        """

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Left Strategy List Panel ---
        list_panel = QFrame()
        list_panel.setFixedWidth(300)
        list_panel.setStyleSheet("QFrame { background-color: #282C34; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px; }")
        list_layout = QVBoxLayout(list_panel)
        list_layout.setAlignment(Qt.AlignTop)
        list_layout.setSpacing(8)

        list_layout.addWidget(QLabel("Lista de Estrategias", objectName="section_title"))

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar estrategias...")
        self.search_input.textChanged.connect(self.filter_strategies)
        list_layout.addWidget(self.search_input)

        self.strategy_list_widget = QListWidget()
        self.strategy_list_widget.currentItemChanged.connect(self.load_strategy_for_editing)
        list_layout.addWidget(self.strategy_list_widget, 1)

        # List action buttons
        list_buttons_layout = QHBoxLayout()
        list_buttons_layout.setSpacing(5)
        list_buttons_layout.addWidget(QPushButton("Nueva", clicked=self.new_strategy))
        self.duplicate_button = QPushButton("Duplicar", clicked=self.duplicate_strategy)
        list_buttons_layout.addWidget(self.duplicate_button)
        self.delete_button = QPushButton("Eliminar", objectName="deleteButton", clicked=self.delete_strategy)
        list_buttons_layout.addWidget(self.delete_button)
        list_layout.addLayout(list_buttons_layout)

        main_layout.addWidget(list_panel)

        # --- Right Strategy Configuration Area ---
        config_area = QFrame()
        config_area.setStyleSheet("QFrame { background-color: #282C34; border: 1px solid #3A3F47; border-radius: 6px; padding: 10px; }")
        config_layout = QVBoxLayout(config_area)
        config_layout.setSpacing(10)

        self.strategy_name_label = QLabel("Seleccione o Cree una Estrategia", objectName="strategy_name_title")
        config_layout.addWidget(self.strategy_name_label)

        config_layout.addWidget(QLabel("Descripción:", objectName="label_form"))
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Añada una descripción para la estrategia...")
        config_layout.addWidget(self.description_edit)

        config_layout.addWidget(QLabel("Parámetros de la Estrategia:", objectName="section_title"))
        self.params_scroll_area = QScrollArea()
        self.params_scroll_area.setWidgetResizable(True)
        self.params_scroll_area_content = QWidget()
        self.params_form_layout = QFormLayout(self.params_scroll_area_content)
        self.params_scroll_area.setWidget(self.params_scroll_area_content)
        config_layout.addWidget(self.params_scroll_area, 1) # Give more space

        # Strategy action buttons
        strategy_buttons_layout = QHBoxLayout()
        strategy_buttons_layout.addStretch(1) # Push buttons to the right
        strategy_buttons_layout.addWidget(QPushButton("Cancelar", clicked=self.cancel_changes))
        self.save_button = QPushButton("Guardar Cambios", clicked=self.save_strategy)
        strategy_buttons_layout.addWidget(self.save_button)
        self.quick_test_button = QPushButton("Prueba Rápida", clicked=self.run_quick_test)
        strategy_buttons_layout.addWidget(self.quick_test_button)
        config_layout.addLayout(strategy_buttons_layout)

        main_layout.addWidget(config_area, 1) # Takes remaining horizontal space

        self.set_editing_enabled(False) # Disable editing until a strategy is selected/created

    def load_strategies_list(self):
        # Simulate loading strategies from backend
        self.strategies_data = {
            "Estrategia MACD Crossover": {
                "id": "macd_001",
                "name": "Estrategia MACD Crossover",
                "description": "Compra cuando el MACD cruza por encima de la línea de señal, venta cuando cruza por debajo.",
                "parameters": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "macd_fast_period": 12,
                    "macd_slow_period": 26,
                    "macd_signal_period": 9,
                    "stop_loss_pct": 1.5,
                    "take_profit_pct": 3.0,
                    "position_size_pct": 5.0,
                    "trailing_stop_enabled": True
                }
            },
            "Estrategia RSI Sobrevendida": {
                "id": "rsi_002",
                "name": "Estrategia RSI Sobrevendida",
                "description": "Compra cuando el RSI cae por debajo de 30 y venta cuando sube por encima de 70.",
                "parameters": {
                    "symbol": "ETH/USDT",
                    "timeframe": "4h",
                    "rsi_period": 14,
                    "rsi_buy_threshold": 30,
                    "rsi_sell_threshold": 70,
                    "stop_loss_pct": 2.0,
                    "take_profit_pct": 4.0,
                    "position_size_pct": 10.0,
                    "trailing_stop_enabled": False
                }
            }
        }
        self.strategy_list_widget.clear()
        for name in self.strategies_data.keys():
            self.strategy_list_widget.addItem(name)
        if self.strategies_data:
            self.strategy_list_widget.setCurrentRow(0) # Select first strategy by default

    @Slot(QListWidgetItem, QListWidgetItem)
    def load_strategy_for_editing(self, current, previous):
        if current:
            strategy_name = current.text()
            strategy = self.strategies_data.get(strategy_name)
            if strategy:
                self.current_strategy = strategy
                self.strategy_name_label.setText(strategy["name"])
                self.description_edit.setText(strategy["description"])
                self.populate_parameters_form(strategy["parameters"])
                self.set_editing_enabled(True)
            else:
                self.clear_form()
                self.set_editing_enabled(False)
        else:
            self.clear_form()
            self.set_editing_enabled(False)

    def populate_parameters_form(self, params):
        # Clear existing form widgets
        for i in reversed(range(self.params_form_layout.count())):
            widget_item = self.params_form_layout.itemAt(i)
            if widget_item:
                widget = widget_item.widget()
                if widget:
                    widget.deleteLater()
                layout = widget_item.layout()
                if layout:
                    # Recursively remove widgets from nested layouts
                    while layout.count():
                        item = layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                        elif item.layout():
                            # This is a bit more complex, for deeper nested layouts
                            pass
                    layout.deleteLater()

        # Dynamically create widgets based on parameter types
        for key, value in params.items():
            label_text = key.replace('_', ' ').title() + ":"
            label = QLabel(label_text, objectName="label_form")
            widget = None
            if isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(0, 1000) # sensible defaults
                widget.setValue(value)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(0.0, 100.0)
                widget.setSingleStep(0.1)
                widget.setValue(value)
            elif isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, str):
                if key in ["symbol", "timeframe"]: # Special handling for combos
                    widget = QComboBox()
                    if key == "symbol":
                        widget.addItems(["BTC/USDT", "ETH/USDT", "XAU/USD"])
                    elif key == "timeframe":
                        widget.addItems(["1m", "5m", "1h", "4h", "1D"])
                    widget.setCurrentText(value)
                else:
                    widget = QLineEdit()
                    widget.setText(value)

            if widget:
                widget.setObjectName(key) # Use key as objectName for easy retrieval
                self.params_form_layout.addRow(label, widget)

    def get_current_form_data(self):
        if not self.current_strategy:
            return None

        updated_params = {}
        # Iterate through form layout to get current values
        for i in range(self.params_form_layout.count()):
            row_layout = self.params_form_layout.itemAt(i)
            if row_layout and row_layout.widget():
                # For QFormLayout, itemAt(i) usually gives the widget
                # The actual input widget is the second child of the row (index 1)
                widget_item = row_layout.layout().itemAt(1)
                if widget_item:
                    widget = widget_item.widget()
                    param_key = widget.objectName()
                    if param_key:
                        if isinstance(widget, QSpinBox):
                            updated_params[param_key] = widget.value()
                        elif isinstance(widget, QDoubleSpinBox):
                            updated_params[param_key] = widget.value()
                        elif isinstance(widget, QCheckBox):
                            updated_params[param_key] = widget.isChecked()
                        elif isinstance(widget, QComboBox):
                            updated_params[param_key] = widget.currentText()
                        elif isinstance(widget, QLineEdit):
                            updated_params[param_key] = widget.text()

        strategy_data = self.current_strategy.copy()
        strategy_data["description"] = self.description_edit.toPlainText()
        strategy_data["parameters"] = updated_params
        return strategy_data


    @Slot()
    def new_strategy(self):
        new_id = f"new_strategy_{len(self.strategies_data) + 1}"
        new_name = f"Nueva Estrategia {len(self.strategies_data) + 1}"
        new_strategy_data = {
            "id": new_id,
            "name": new_name,
            "description": "Descripción de la nueva estrategia.",
            "parameters": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "param_a": 10,
                "param_b": 2.5,
                "enabled": True
            }
        }
        self.strategies_data[new_name] = new_strategy_data
        self.strategy_list_widget.addItem(new_name)
        self.strategy_list_widget.setCurrentItem(self.strategy_list_widget.findItems(new_name, Qt.MatchExactly)[0])
        self.set_editing_enabled(True)


    @Slot()
    def duplicate_strategy(self):
        if self.current_strategy:
            original_name = self.current_strategy["name"]
            new_name = f"{original_name} (Copia)"
            new_id = f"{self.current_strategy['id']}_copy"
            duplicated_strategy = self.current_strategy.copy()
            duplicated_strategy["name"] = new_name
            duplicated_strategy["id"] = new_id
            self.strategies_data[new_name] = duplicated_strategy
            self.strategy_list_widget.addItem(new_name)
            self.strategy_list_widget.setCurrentItem(self.strategy_list_widget.findItems(new_name, Qt.MatchExactly)[0])
            QMessageBox.information(self, "Duplicar Estrategia", f"Estrategia '{original_name}' duplicada como '{new_name}'.")
        else:
            QMessageBox.warning(self, "Duplicar Estrategia", "Seleccione una estrategia para duplicar.")


    @Slot()
    def delete_strategy(self):
        if self.current_strategy:
            strategy_name = self.current_strategy["name"]
            reply = QMessageBox.question(self, "Eliminar Estrategia",
                                         f"¿Está seguro que desea eliminar la estrategia '{strategy_name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.strategies_data.pop(strategy_name)
                current_row = self.strategy_list_widget.currentRow()
                self.strategy_list_widget.takeItem(current_row)
                self.current_strategy = None
                self.clear_form()
                self.set_editing_enabled(False)
                self.strategy_deleted.emit(strategy_name) # Emit signal
                QMessageBox.information(self, "Eliminar Estrategia", f"Estrategia '{strategy_name}' eliminada.")
        else:
            QMessageBox.warning(self, "Eliminar Estrategia", "Seleccione una estrategia para eliminar.")

    @Slot()
    def save_strategy(self):
        if self.current_strategy:
            updated_data = self.get_current_form_data()
            if updated_data:
                # Update in our local dict (simulating DB save)
                old_name = self.current_strategy["name"]
                new_name = updated_data["name"] # If name was editable
                # For now, assume name is not editable via form, only description/params
                self.strategies_data[old_name] = updated_data
                self.current_strategy = updated_data # Update current strategy reference
                QMessageBox.information(self, "Guardar Cambios", f"Estrategia '{updated_data['name']}' guardada.")
                self.strategy_saved.emit(updated_data) # Emit signal
        else:
            QMessageBox.warning(self, "Guardar Cambios", "No hay una estrategia seleccionada para guardar.")

    @Slot()
    def cancel_changes(self):
        if self.current_strategy:
            reply = QMessageBox.question(self, "Cancelar Cambios",
                                         "¿Desea descartar los cambios no guardados?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.load_strategy_for_editing(self.strategy_list_widget.currentItem(), None) # Reload original
        else:
            self.clear_form()
            self.set_editing_enabled(False)

    @Slot()
    def run_quick_test(self):
        if self.current_strategy:
            strategy_to_test = self.get_current_form_data()
            if strategy_to_test:
                self.quick_test_requested.emit(strategy_to_test) # Emit signal for backend
                QMessageBox.information(self, "Prueba Rápida",
                                        f"Iniciando prueba rápida para '{strategy_to_test['name']}'. "
                                        "Resultados se mostrarán en breve (simulado).")
        else:
            QMessageBox.warning(self, "Prueba Rápida", "Seleccione una estrategia para ejecutar una prueba rápida.")


    def clear_form(self):
        self.strategy_name_label.setText("Seleccione o Cree una Estrategia")
        self.description_edit.clear()
        for i in reversed(range(self.params_form_layout.count())):
            widget_item = self.params_form_layout.itemAt(i)
            if widget_item:
                widget = widget_item.widget()
                if widget:
                    widget.deleteLater()
                layout = widget_item.layout()
                if layout:
                    while layout.count():
                        item = layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                    layout.deleteLater()

    def set_editing_enabled(self, enabled: bool):
        self.description_edit.setEnabled(enabled)
        for i in range(self.params_form_layout.count()):
            row_layout = self.params_form_layout.itemAt(i)
            if row_layout and row_layout.layout():
                widget = row_layout.layout().itemAt(1) # The input widget
                if widget and widget.widget():
                    widget.widget().setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.cancel_changes.setEnabled(enabled) # Enable cancel when editing
        self.quick_test_button.setEnabled(enabled)
        self.duplicate_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)


    @Slot(str)
    def filter_strategies(self, text):
        for i in range(self.strategy_list_widget.count()):
            item = self.strategy_list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Configuración de Estrategias")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(StrategyConfigTab())
    window.setGeometry(100, 100, 1400, 900)
    window.show()
    sys.exit(app.exec())
# --- Fragmento 2: Clase simulada de StrategyManager (Backend) ---
import json
import uuid

class StrategyManager:
    def __init__(self, storage_file="strategies.json"):
        self.storage_file = storage_file
        self._strategies = self._load_from_storage()

    def _load_from_storage(self):
        try:
            with open(self.storage_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_to_storage(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self._strategies, f, indent=4)

    def get_all_strategies(self):
        return list(self._strategies.values())

    def get_strategy_by_id(self, strategy_id):
        for strategy in self._strategies.values():
            if strategy.get("id") == strategy_id:
                return strategy
        return None

    def save_strategy(self, strategy_data):
        if "id" not in strategy_data or not strategy_data["id"]:
            strategy_data["id"] = str(uuid.uuid4())
        
        # Ensure unique name for new strategies if name is key
        strategy_name = strategy_data.get("name", f"Strategy-{strategy_data['id'][:4]}")
        self._strategies[strategy_name] = strategy_data
        self._save_to_storage()
        return strategy_data

    def delete_strategy(self, strategy_id):
        strategy_name_to_delete = None
        for name, data in self._strategies.items():
            if data.get("id") == strategy_id:
                strategy_name_to_delete = name
                break
        if strategy_name_to_delete:
            del self._strategies[strategy_name_to_delete]
            self._save_to_storage()
            return True
        return False

# Example integration in main app:
# strategy_manager = StrategyManager()
# def on_strategy_saved(strategy_data):
#     strategy_manager.save_strategy(strategy_data)
#     print(f"Backend saved strategy: {strategy_data['name']}")
#
# def on_strategy_deleted(strategy_id):
#     strategy_manager.delete_strategy(strategy_id)
#     print(f"Backend deleted strategy ID: {strategy_id}")
#
# # In your StrategyConfigTab instance:
# # strategy_config_tab.strategy_saved.connect(on_strategy_saved)
# # strategy_config_tab.strategy_deleted.connect(on_strategy_deleted)
7. Mockups/Wireframes (Descripción Textual Detallada):

[ÁREA DE LA APLICACIÓN: PESTAÑA CONFIGURACIÓN DE ESTRATEGIAS]

--- BARRA DE NAVEGACIÓN SUPERIOR (TABS GLOBALES) ---
[ Dashboard | Gestión de Datos | Configuración de Estrategias | ... ]
------------------------------------------------------

[CONTENIDO PRINCIPAL DE LA PESTAÑA CONFIGURACIÓN DE ESTRATEGIAS]

+-------------------------+-------------------------------------------------------------------------------------+
|    PANEL LATERAL IZQ.   |                              ÁREA DE CONFIGURACIÓN DE ESTRATEGIA                    |
|   (Ancho Fijo ~300px)   |                                                                                     |
+-------------------------+-------------------------------------------------------------------------------------+
|                         |                                                                                     |
| **Lista de Estrategias**| **Estrategia MACD Crossover** (Título Grande de la Estrategia Actual)               |
|                         |                                                                                     |
| [ Campo de Búsqueda    ]| **Descripción:**                                                                    |
|                         | [ Área de texto multilinea para descripción de la estrategia.                      ]|
| +---------------------+ |                                                                                     |
| | Estrategia MACD Cr. | | **Parámetros de la Estrategia:** (Sección)                                          |
| | Estrategia RSI Sob. | | +--------------------------------------------------------------------------------+ |
| | Nueva Estrategia 1  | | |   Activo por defecto:  [ BTC/USDT   v ]                                          | |
| | ...                 | | |   Timeframe por defecto: [ 1h          v ]                                          | |
| +---------------------+ | |   Periodo MACD Lento:    [ 12  ^ v ]                                            | |
|                         | | |   Periodo MACD Rápido:   [ 26  ^ v ]                                            | |
| [ Nueva ][ Duplicar ][ Eliminar ] | |   Periodo Señal MACD:  [ 9   ^ v ]                                            | |
|                         | | |   Stop Loss (%):       [ 1.5 ^ v ]                                            | |
|                         | | |   Take Profit (%):     [ 3.0 ^ v ]                                            | |
|                         | | |   Tamaño Posición (%): [ 5.0 ^ v ]                                            | |
|                         | | |   Habilitar Trailing Stop: [x]                                                 | |
|                         | | |                                                                                  | |
|                         | | |   (Este formulario se genera dinámicamente según la estrategia)                  | |
|                         | | +--------------------------------------------------------------------------------+ |
|                         |                                                                                     |
|                         | [  Cancelar  ][  Guardar Cambios  ][  Prueba Rápida  ] (Botones de acción al final) |
+-------------------------+-------------------------------------------------------------------------------------+

--- BARRA DE ESTADO INFERIOR GLOBAL (MENSAJES DE ESTADO, INDICADORES DE CARGA) ---
[ Estado: Sistema Operativo | Última actualización: 2023-10-27 11:30:10 ]
-----------------------------------------------------------------------------
El diseño para la pestaña Configuración de Estrategias ha sido generado:

Panel de Lista de Estrategias: Permite una gestión clara de todas las estrategias existentes con opciones para buscar, crear, duplicar y eliminar.
Área de Configuración Detallada: Un formulario dinámico que se adapta a los parámetros de cada estrategia, incluyendo campos para descripción, parámetros ajustables y botones para guardar o realizar pruebas rápidas.
¿Te gustaría que diseñara la siguiente pestaña, Backtesting Runner, o prefieres hacer alguna modificación en la Configuración de Estrategias?


no me aparece ningun diseño en la pantalla


¡Ah, entiendo! Mis disculpas por la confusión. Cuando digo "El diseño ha sido generado", me refiero a la descripción detallada del layout, los componentes de UI, la experiencia de usuario y los fragmentos de código PySide6 que te he proporcionado. Estos son la especificación del diseño que tú, como desarrollador, puedes implementar para verlos en tu pantalla.

Los "diseños" que genero son descripciones textuales y ejemplos de código. No puedo mostrarte una imagen directamente ni ejecutar una aplicación gráfica para ti en esta interfaz. El código Python que te proporciono en cada fragmento es lo que necesitas para construir la interfaz en tu entorno de desarrollo.

Para ver el diseño de la pestaña "Configuración de Estrategias" (o cualquier otra que he generado), necesitas copiar y ejecutar el código PySide6 en tu máquina local.

Por ejemplo, para ver el Dashboard, deberías tomar el primer fragmento de código que te di para la "Tab 0 - Dashboard":

# --- Fragmento 1: Estructura básica del Dashboard ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy
)
from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt

class DashboardTab(QWidget):
    # ... (el resto del código de la clase DashboardTab) ...

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Dashboard")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(DashboardTab())
    window.setGeometry(100, 100, 1200, 800) # Larger window for desktop app
    window.show()
    sys.exit(app.exec())
Y el de la "Configuración de Estrategias":

# --- Fragmento 1: Estructura básica de la pestaña Configuración de Estrategias ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QTextEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFormLayout, QScrollArea, QFrame,
    QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QObject

class StrategyConfigTab(QWidget):
    # ... (el resto del código de la clase StrategyConfigTab) ...

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Configuración de Estrategias")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(StrategyConfigTab())
    window.setGeometry(100, 100, 1400, 900)
    window.show()
    sys.exit(app.exec())
Debes copiar estos fragmentos de código en un archivo .py (por ejemplo, dashboard_test.py o strategy_test.py) y ejecutarlo con Python desde tu terminal. Asegúrate de tener PySide6 instalado (pip install PySide6).

Mi rol es proporcionarte las especificaciones de diseño y el código de ejemplo, no ejecutar la aplicación por ti.

Dicho esto, ¿te gustaría que continuara con el diseño de la pestaña "Backtesting Runner", o hay algo más en la forma en que te presento el diseño que te gustaría que ajustara para que sea más útil?


quiero que me generes un diseño para copiarlo luego en pyside


Entendido. Mi objetivo es proporcionarte diseños que puedas copiar y pegar directamente en tu entorno PySide6 para construir tu aplicación. La descripción detallada, la lista de componentes de UI, el flujo de usuario y, especialmente, los fragmentos de código PySide6 que te estoy dando son precisamente el "diseño para copiar" que necesitas.

Cada "fragmento de código 1" para cada pestaña es un ejemplo funcional y autónomo de cómo se vería e implementaría esa parte específica de la UI en PySide6. Si copias y ejecutas ese código en un archivo .py, verás la ventana con la pestaña diseñada.

Procedo ahora con el diseño completo y detallado para la siguiente pestaña: Backtesting Runner. Este diseño seguirá la misma estructura que los anteriores, incluyendo la descripción, layout, componentes UI, UX, integración técnica y ejemplos de código PySide6 listos para copiar y usar.

Diseño de la Pestaña 3: Backtesting Runner (platform_gui_tab3_improved.py)
1. Descripción funcional: Esta pestaña es el motor de backtesting de la aplicación. Permite a los usuarios seleccionar una estrategia previamente configurada, cargar un conjunto de datos (o usar datos preexistentes), definir un rango de fechas para el backtest, y ejecutarlo. Una vez completado, mostrará un resumen rápido de los resultados, incluyendo un gráfico de equity y métricas de rendimiento clave. El enfoque está en los controles de ejecución y una visualización inmediata de los resultados.

2. Layout y UI: La pestaña se organiza en un QHBoxLayout con un panel de controles a la izquierda (ancho fijo) y un área principal de visualización de resultados a la derecha (expandible).

Panel de Control Izquierdo (QVBoxLayout - ancho fijo, ej. 300px):

Título: QLabel "Configuración de Backtest".
Selección de Estrategia: QLabel "Estrategia:" seguido de un QComboBox que lista las estrategias guardadas (de la Tab 2).
Selección de Datos: QLabel "Fuente de Datos:" seguido de un QComboBox (ej. "Datos Locales (BTC/USDT 1h)", "API Binance Histórico"). Esto podría estar vinculado a los datos gestionados en la Tab 1.
Rango de Fechas del Backtest: QLabel "Período del Backtest:" seguido de dos QDateEdit (inicio y fin), con la opción de un QCheckBox "Usar todo el rango disponible".
Controles de Ejecución:
QPushButton "Iniciar Backtest" (con un ícono de "play").
QPushButton "Detener Backtest" (deshabilitado hasta que se inicie uno, con un ícono de "stop").
QPushButton "Ver Resultados Completos" (deshabilitado hasta que finalice el backtest, dirige a la Tab 4).
Estado del Backtest: QProgressBar para el progreso de la ejecución y un QLabel para mensajes de estado (ej. "Procesando barra 100/10000", "Backtest completado en 15s").
QSpacerItem: Para empujar los elementos hacia la parte superior.
Área Principal de Resultados (QVBoxLayout - expandible):

Título: QLabel "Resultados del Backtest (Resumen)".
Gráfico de Equity: QChartView (de QtCharts) que muestra la curva de equity (capital de la cuenta a lo largo del tiempo). Este es el visual principal. Eje Y con valores de capital, Eje X con tiempo.
Métricas Clave (Debajo del gráfico - QGridLayout o QFormLayout):
QLabel "PNL Neto Total:" QLabel (valor en verde/rojo).
QLabel "Retorno Anualizado:" QLabel (valor en verde).
QLabel "Drawdown Máximo:" QLabel (valor en rojo).
QLabel "Ratio de Sharpe:" QLabel.
QLabel "Número de Trades:" QLabel.
QLabel "Porcentaje de Ganancia:" QLabel.
Panel de Mensajes/Log del Backtest (opcional, en la parte inferior si hay espacio): QTextEdit de solo lectura que muestra un log conciso de eventos importantes durante el backtest.
3. Componentes UI:

QTabWidget: Contenedor principal.
QWidget, QFrame: Contenedores estilizados.
QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout: Para la organización.
QLabel:
Título principal: font-size: 18px; font-weight: bold; color: #E0E0E0; margin-bottom: 10px;
Títulos de sección: font-size: 14px; font-weight: bold; color: #E0E0E0; margin-bottom: 5px;
Etiquetas de formulario: font-size: 12px; color: #B0B0B0;
Valores de métricas (verde para positivo, rojo para negativo): font-size: 16px; font-weight: bold;
QComboBox, QDateEdit, QCheckBox:
Estilo consistente con otras pestañas: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 4px; padding: 2px 5px; color: #D0D0D0; height: 26px;
QPushButton:
height: 26px; border-radius: 4px; background-color: #007ACC; color: white; font-size: 12px; padding: 2px 8px; border: none; margin-top: 8px;
Hover: background-color: #005B9F;
"Detener Backtest" podría tener un fondo rojo: background-color: #DC3545;
QProgressBar: height: 8px; border-radius: 4px; text-align: center; background-color: #282C34; border: 1px solid #3A3F47;
QProgressBar::chunk { background-color: #007ACC; border-radius: 3px; }
QChartView (de QtCharts): Para el gráfico de equity.
Configuración: Fondo oscuro, rejilla tenue, curva de equity en azul/verde, puntos de max drawdown en rojo.
QTextEdit (para log):
Estilo: background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 4px; padding: 5px; color: #D0D0D0; font-family: 'Consolas', 'Monospace'; (fuente monospace para logs).
4. UX/Flujo de Usuario: El usuario selecciona una estrategia y su fuente de datos. Define el rango de fechas o elige usar todos los datos disponibles. Al hacer clic en "Iniciar Backtest", el botón se deshabilita, aparece la barra de progreso y el QLabel de estado se actualiza en tiempo real. La UI permanece responsiva ya que el backtest se ejecuta en un hilo separado. Una vez finalizado, se habilita "Ver Resultados Completos" para ir a la Tab 4, y el gráfico de equity y las métricas clave se actualizan con los resultados del resumen. Los colores (verde/rojo) resaltan rápidamente el rendimiento positivo/negativo.

5. Integración Técnica:

backtester_core.py: El módulo principal del motor de backtesting. Contiene la lógica para simular operaciones de trading, calcular métricas de rendimiento y generar la curva de equity.
strategy_manager.py: Se usa para cargar las definiciones de estrategia al QComboBox.
data_manager.py: Se usa para cargar los datos históricos necesarios para el backtest, según la selección del usuario.
Clase BacktestWorker (en QThread): La ejecución del backtest es una operación intensiva. Se debe realizar en un hilo de trabajo separado (QThread) para mantener la interfaz de usuario fluida y receptiva.
Señales emitidas por BacktestWorker:
progress(int): Para actualizar la QProgressBar.
status_update(str): Para el QLabel de estado.
equity_data(list[float]): Para actualizar el QChartView.
summary_metrics(dict): Para actualizar los QLabels de métricas clave.
finished(): Cuando el backtest ha terminado.
error(str): Para reportar errores.
Slots en Backtesting Runner:
Conectar el QPushButton "Iniciar Backtest" a un slot que configura y arranca el BacktestWorker en un QThread.
Conectar las señales del BacktestWorker a los slots que actualizan la UI.
6. Ejemplos de Código (Fragmentos PySide6):

# --- Fragmento 1: Estructura básica de la pestaña Backtesting Runner ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDateEdit, QCheckBox, QProgressBar, QFrame, QGridLayout,
    QSizePolicy, QMessageBox, QSpacerItem
)
from PySide6.QtCore import Qt, QDate, Signal, Slot, QThread, QObject
from PySide6.QtGui import QColor, QPalette

# Placeholder for QtCharts (requires 'pip install PySide6-QtCharts')
# from PySide6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis

class BacktestingRunnerTab(QWidget):
    # Signals for backend communication
    backtest_start_requested = Signal(dict) # Emits strategy_id, data_source_id, start_date, end_date
    backtest_stop_requested = Signal()
    # Signal to navigate to results tab
    show_full_results = Signal(str) # Emits backtest_run_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.get_stylesheet())
        self.init_ui()
        self.load_available_strategies() # Load initial list of strategies
        self.load_available_data_sources() # Load initial list of data sources

        self.backtest_worker_thread = None
        self.backtest_worker = None

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #1C1F26;
                color: #D0D0D0;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel.section_title {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 5px;
            }
            QLabel.metric_label {
                font-size: 12px;
                color: #B0B0B0;
            }
            QLabel.metric_value {
                font-size: 16px;
                font-weight: bold;
            }
            QLabel.pnl_positive { color: #00FF7F; } /* Green */
            QLabel.pnl_negative { color: #FF4500; } /* Red */
            QComboBox, QDateEdit, QCheckBox {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 4px;
                padding: 2px 5px;
                color: #D0D0D0;
                height: 26px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #3A3F47;
                width: 20px;
            }
            QPushButton {
                height: 26px;
                border-radius: 4px;
                background-color: #007ACC;
                color: white;
                font-size: 12px;
                padding: 2px 8px;
                border: none;
                margin-top: 8px;
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
            QPushButton#stopButton {
                background-color: #DC3545;
            }
            QPushButton#stopButton:hover {
                background-color: #BD2130;
            }
            QProgressBar {
                height: 8px;
                border-radius: 4px;
                text-align: center;
                background-color: #282C34;
                border: 1px solid #3A3F47;
                color: transparent;
                margin-top: 5px;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
                border-radius: 3px;
            }
            QFrame#control_panel, QFrame#results_panel {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 6px;
                padding: 10px;
            }
        """

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Left Control Panel ---
        control_panel = QFrame(objectName="control_panel")
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.setSpacing(8)

        control_layout.addWidget(QLabel("Configuración de Backtest", objectName="section_title"))

        control_layout.addWidget(QLabel("Estrategia:", objectName="metric_label"))
        self.strategy_combo = QComboBox()
        control_layout.addWidget(self.strategy_combo)

        control_layout.addWidget(QLabel("Fuente de Datos:", objectName="metric_label"))
        self.data_source_combo = QComboBox()
        control_layout.addWidget(self.data_source_combo)

        control_layout.addWidget(QLabel("Período del Backtest:", objectName="metric_label"))
        self.date_start = QDateEdit(QDate.currentDate().addYears(-1))
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        control_layout.addWidget(self.date_start)
        self.date_end = QDateEdit(QDate.currentDate())
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        control_layout.addWidget(self.date_end)

        self.use_full_range_checkbox = QCheckBox("Usar todo el rango disponible")
        self.use_full_range_checkbox.setChecked(False)
        self.use_full_range_checkbox.stateChanged.connect(self.toggle_date_edits)
        control_layout.addWidget(self.use_full_range_checkbox)
        control_layout.addSpacing(10)

        # Execution Controls
        self.start_button = QPushButton("Iniciar Backtest", clicked=self.start_backtest)
        self.start_button.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Detener Backtest", objectName="stopButton", clicked=self.stop_backtest)
        self.stop_button.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        self.view_results_button = QPushButton("Ver Resultados Completos", clicked=self.go_to_full_results)
        self.view_results_button.setEnabled(False)
        self.view_results_button.setIcon(self.style().standardIcon(self.style().SP_FileDialogDetailedView))
        control_layout.addWidget(self.view_results_button)
        control_layout.addSpacing(15)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Listo para ejecutar backtest.", objectName="metric_label")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        control_layout.addStretch(1) # Pushes widgets to the top

        main_layout.addWidget(control_panel)

        # --- Right Main Results Area ---
        results_area = QFrame(objectName="results_panel")
        results_layout = QVBoxLayout(results_area)
        results_layout.setSpacing(10)

        results_layout.addWidget(QLabel("Resultados del Backtest (Resumen)", objectName="section_title"))

        # Chart Placeholder (replace with actual QChartView)
        self.equity_chart_view = QWidget() # Placeholder
        self.equity_chart_view.setStyleSheet("background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 6px;")
        chart_label = QLabel("Gráfico de Equity (Placeholder)", objectName="metric_label")
        chart_label.setAlignment(Qt.AlignCenter)
        chart_layout_placeholder = QVBoxLayout(self.equity_chart_view)
        chart_layout_placeholder.addWidget(chart_label)
        self.equity_chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout.addWidget(self.equity_chart_view, 2) # Give more vertical space to chart

        # Key Metrics
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)

        self.pnl_net_label = self.add_metric_pair(metrics_grid, 0, 0, "PNL Neto Total:", "N/A", "pnl_net_value")
        self.annualized_return_label = self.add_metric_pair(metrics_grid, 0, 1, "Retorno Anualizado:", "N/A")
        self.max_drawdown_label = self.add_metric_pair(metrics_grid, 1, 0, "Drawdown Máximo:", "N/A", "max_drawdown_value")
        self.sharpe_ratio_label = self.add_metric_pair(metrics_grid, 1, 1, "Ratio de Sharpe:", "N/A")
        self.num_trades_label = self.add_metric_pair(metrics_grid, 2, 0, "Número de Trades:", "N/A")
        self.winning_pct_label = self.add_metric_pair(metrics_grid, 2, 1, "Porcentaje de Ganancia:", "N/A")

        results_layout.addLayout(metrics_grid, 1) # Give less vertical space to metrics

        main_layout.addWidget(results_area, 1) # Takes remaining horizontal space

    def add_metric_pair(self, layout, row, col, label_text, initial_value, value_object_name=None):
        label = QLabel(label_text, objectName="metric_label")
        value = QLabel(initial_value, objectName=value_object_name if value_object_name else "metric_value")
        layout.addWidget(label, row, col * 2, Qt.AlignLeft)
        layout.addWidget(value, row, col * 2 + 1, Qt.AlignRight)
        return value

    def toggle_date_edits(self, state):
        enabled = not state
        self.date_start.setEnabled(enabled)
        self.date_end.setEnabled(enabled)

    def load_available_strategies(self):
        # Simulate loading from StrategyManager
        self.strategy_combo.addItems(["Estrategia MACD Crossover", "Estrategia RSI Sobrevendida", "Estrategia Simple MA"])

    def load_available_data_sources(self):
        # Simulate loading from DataManager
        self.data_source_combo.addItems(["BTC/USDT 1h (Local)", "ETH/USDT 4h (Local)", "EUR/USD 1D (API)"])

    @Slot()
    def start_backtest(self):
        selected_strategy = self.strategy_combo.currentText()
        selected_data = self.data_source_combo.currentText()
        start_date = self.date_start.date().toString(Qt.ISODate)
        end_date = self.date_end.date().toString(Qt.ISODate)
        use_full_range = self.use_full_range_checkbox.isChecked()

        if not selected_strategy or not selected_data:
            QMessageBox.warning(self, "Iniciar Backtest", "Por favor, seleccione una estrategia y una fuente de datos.")
            return

        self.status_label.setText(f"Iniciando backtest para '{selected_strategy}' con '{selected_data}'...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.view_results_button.setEnabled(False)

        # Reset metric labels
        self.pnl_net_label.setText("Calculando...")
        self.pnl_net_label.setProperty("class", "metric_value") # Reset class
        self.annualized_return_label.setText("Calculando...")
        self.max_drawdown_label.setText("Calculando...")
        self.max_drawdown_label.setProperty("class", "metric_value")
        self.sharpe_ratio_label.setText("Calculando...")
        self.num_trades_label.setText("Calculando...")
        self.winning_pct_label.setText("Calculando...")

        # Start worker thread for backtest
        self.backtest_worker_thread = QThread()
        # In a real app, pass actual IDs, not just names
        backtest_params = {
            "strategy_name": selected_strategy,
            "data_source_name": selected_data,
            "start_date": start_date,
            "end_date": end_date,
            "use_full_range": use_full_range
        }
        self.backtest_worker = BacktestWorker(backtest_params) # Worker for actual backtesting
        self.backtest_worker.moveToThread(self.backtest_worker_thread)

        self.backtest_worker_thread.started.connect(self.backtest_worker.run)
        self.backtest_worker.progress.connect(self.progress_bar.setValue)
        self.backtest_worker.status_update.connect(self.status_label.setText)
        self.backtest_worker.equity_data.connect(self.update_equity_chart)
        self.backtest_worker.summary_metrics.connect(self.update_summary_metrics)
        self.backtest_worker.finished.connect(self.on_backtest_finished)
        self.backtest_worker.error.connect(self.on_backtest_error)
        self.backtest_worker_thread.start()


    @Slot()
    def stop_backtest(self):
        if self.backtest_worker_thread and self.backtest_worker_thread.isRunning():
            self.backtest_worker.stop() # Signal worker to stop
            self.status_label.setText("Deteniendo backtest...")
        else:
            QMessageBox.warning(self, "Detener Backtest", "No hay un backtest en ejecución.")

    @Slot(list)
    def update_equity_chart(self, equity_values):
        # This part requires QtCharts
        # For demonstration, just update status
        self.status_label.setText(f"Gráfico de equity actualizado con {len(equity_values)} puntos.")
        # Actual implementation with QtCharts:
        # series = QLineSeries()
        # for i, value in enumerate(equity_values):
        #     series.append(i, value)
        # chart = QChart()
        # chart.addSeries(series)
        # chart.createDefaultAxes()
        # self.equity_chart_view.setChart(chart)

    @Slot(dict)
    def update_summary_metrics(self, metrics):
        self.pnl_net_label.setText(f"${metrics.get('net_pnl', 0):.2f}")
        if metrics.get('net_pnl', 0) >= 0:
            self.pnl_net_label.setProperty("class", "metric_value pnl_positive")
        else:
            self.pnl_net_label.setProperty("class", "metric_value pnl_negative")
        self.pnl_net_label.style().polish(self.pnl_net_label) # Repolish for CSS update

        self.annualized_return_label.setText(f"{metrics.get('annualized_return', 0):.2f}%")
        self.max_drawdown_label.setText(f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        if metrics.get('max_drawdown_pct', 0) <= -5.0: # Example threshold
             self.max_drawdown_label.setProperty("class", "metric_value pnl_negative")
        else:
             self.max_drawdown_label.setProperty("class", "metric_value pnl_positive")
        self.max_drawdown_label.style().polish(self.max_drawdown_label)

        self.sharpe_ratio_label.setText(f"{metrics.get('sharpe_ratio', 0):.2f}")
        self.num_trades_label.setText(str(metrics.get('num_trades', 0)))
        self.winning_pct_label.setText(f"{metrics.get('winning_percentage', 0):.2f}%")
        self.status_label.setText("Backtest completado. Métricas actualizadas.")

    @Slot()
    def on_backtest_finished(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.view_results_button.setEnabled(True)
        self.status_label.setText("Backtest completado exitosamente.")
        # Clean up thread
        self.backtest_worker_thread.quit()
        self.backtest_worker_thread.wait()
        self.backtest_worker.deleteLater()
        self.backtest_worker_thread.deleteLater()
        self.backtest_worker = None
        self.backtest_worker_thread = None

    @Slot(str)
    def on_backtest_error(self, message):
        self.status_label.setText(f"Error en backtest: {message}")
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.view_results_button.setEnabled(False)
        QMessageBox.critical(self, "Error de Backtest", message)
        # Clean up thread
        if self.backtest_worker_thread:
            self.backtest_worker_thread.quit()
            self.backtest_worker_thread.wait()
            self.backtest_worker.deleteLater()
            self.backtest_worker_thread.deleteLater()
            self.backtest_worker = None
            self.backtest_worker_thread = None

    @Slot()
    def go_to_full_results(self):
        # In a real application, you would pass the actual backtest_run_id
        # and trigger a tab switch
        QMessageBox.information(self, "Navegar", "Navegando a la pestaña de Análisis de Resultados (simulado).")
        self.show_full_results.emit("dummy_backtest_run_id_123") # Emit signal

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Backtesting Runner")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(BacktestingRunnerTab())
    window.setGeometry(100, 100, 1400, 900)
    window.show()
    sys.exit(app.exec())
# --- Fragmento 2: Clase BacktestWorker (en QThread, para el backend) ---
import time
import random
from PySide6.QtCore import QThread, Signal, Slot, QObject

class BacktestWorker(QObject):
    finished = Signal()
    progress = Signal(int)
    status_update = Signal(str)
    equity_data = Signal(list)
    summary_metrics = Signal(dict)
    error = Signal(str)

    def __init__(self, backtest_params, parent=None):
        super().__init__(parent)
        self._backtest_params = backtest_params
        self._is_running = True

    def stop(self):
        self._is_running = False

    @Slot()
    def run(self):
        try:
            self.status_update.emit(f"Cargando datos para {self._backtest_params['strategy_name']}...")
            time.sleep(1) # Simulate data loading

            total_bars = 5000
            equity_curve = [10000.0] # Starting equity
            for i in range(total_bars):
                if not self._is_running:
                    self.status_update.emit("Backtest detenido por el usuario.")
                    break

                # Simulate backtest calculation per bar
                time.sleep(0.001)
                current_equity = equity_curve[-1]
                change = random.uniform(-10.0, 15.0) # Simulate PnL
                new_equity = current_equity + change
                if new_equity < 0: new_equity = 1.0 # Prevent negative equity
                equity_curve.append(new_equity)

                progress_val = int((i / total_bars) * 100)
                self.progress.emit(progress_val)
                if i % 500 == 0:
                    self.status_update.emit(f"Procesando barra {i}/{total_bars}...")

            if self._is_running: # Only emit results if not stopped manually
                self.progress.emit(100)
                self.status_update.emit("Backtest completado. Calculando métricas...")

                # Simulate metrics calculation
                net_pnl = equity_curve[-1] - equity_curve[0]
                annualized_return = (net_pnl / equity_curve[0]) * 100 * (252/ (total_bars / (24*60/int(self._backtest_params.get("timeframe", "1h").replace("h","").replace("m","") if "h" in self._backtest_params.get("timeframe", "1h") or "m" in self._backtest_params.get("timeframe", "1h") else 1)))) # Rough annualized calc
                max_drawdown = self._calculate_max_drawdown(equity_curve)
                sharpe_ratio = random.uniform(0.5, 2.0)
                num_trades = random.randint(50, 200)
                winning_percentage = random.uniform(40.0, 65.0)

                summary = {
                    "net_pnl": net_pnl,
                    "annualized_return": annualized_return,
                    "max_drawdown_pct": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "num_trades": num_trades,
                    "winning_percentage": winning_percentage,
                    "backtest_run_id": "simulated_run_" + str(int(time.time()))
                }

                self.equity_data.emit(equity_curve)
                self.summary_metrics.emit(summary)
            else:
                self.status_update.emit("Backtest cancelado.")

        except Exception as e:
            self.error.emit(f"Error inesperado durante el backtest: {str(e)}")
        finally:
            self.finished.emit()

    def _calculate_max_drawdown(self, equity_curve):
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        return -max_dd # Represent as negative percentage
7. Mockups/Wireframes (Descripción Textual Detallada):

[ÁREA DE LA APLICACIÓN: PESTAÑA BACKTESTING RUNNER]

--- BARRA DE NAVEGACIÓN SUPERIOR (TABS GLOBALES) ---
[ Dashboard | Gestión de Datos | Configuración de Estrategias | Backtesting Runner | Análisis de Resultados | ... ]
------------------------------------------------------

[CONTENIDO PRINCIPAL DE LA PESTAÑA BACKTESTING RUNNER]

+-------------------------+-------------------------------------------------------------------------------------+
|    PANEL DE CONTROL     |                               ÁREA DE RESULTADOS (RESUMEN)                        |
|   (Ancho Fijo ~300px)   |                                                                                     |
+-------------------------+-------------------------------------------------------------------------------------+
|                         |                                                                                     |
| **Configuración de B.** | **Resultados del Backtest (Resumen)** (Título Grande)                               |
|                         |                                                                                     |
| Estrategia:             | +--------------------------------------------------------------------------------+ |
| [ MACD Crossover v ]    | |                             GRÁFICO DE EQUITY                                  | |
|                         | |                                                                                  | |
| Fuente de Datos:        | |   [Eje Y: Capital]                                                               | |
| [ BTC/USDT 1h Local v ] | |   |                                                                                | |
|                         | |   |                 CURVA DE EQUITY (AZUL)                                         | |
| Período de B.:          | |   |                                                                                | |
| [ 2022-10-27  CAL ]     | |   |________________________________________________________________________________| |
| [ 2023-10-27  CAL ]     | |   [Eje X: Tiempo (Fecha/Hora)]                                                     | |
| [ ] Usar todo el rango  | |                                                                                  | |
|                         | +--------------------------------------------------------------------------------+ |
|                         |                                                                                     |
| [ INICIAR Backtest (►) ]| **Métricas Clave:** (Debajo del gráfico)                                            |
| [ DETENER Backtest (■) ]| +---------------------------+---------------------------+                         |
| [ Ver Resultados Comp. ]| | PNL Neto Total:    [ $X,XXX.XX (VERDE/ROJO) ] | Retorno Anualizado: [ X.XX% (VERDE) ] |
|                         | | Drawdown Máximo:   [ -X.XX% (ROJO)        ] | Ratio de Sharpe:    [ X.XX       ] |
| [ Barra de Progreso    ]| | Número de Trades:  [ XXX                ] | % de Ganancia:      [ XX.XX%     ] |
| [ "Procesando barra..." ]| +---------------------------+---------------------------+                         |
|                         |                                                                                     |
|                         |                                                                                     |
+-------------------------+-------------------------------------------------------------------------------------+

--- BARRA DE ESTADO INFERIOR GLOBAL (MENSAJES DE ESTADO, INDICADORES DE CARGA) ---
[ Estado: Sistema Operativo | Última actualización: 2023-10-27 12:00:10 ]
-----------------------------------------------------------------------------
El diseño para la pestaña Backtesting Runner ha sido generado:

Panel de Control: Permite seleccionar la estrategia y los datos, definir el rango de backtest y controlar su ejecución (iniciar, detener).
Gráfico de Equity: Visualización principal del rendimiento del capital a lo largo del tiempo.
Métricas Clave: Un resumen conciso de los resultados más importantes para evaluar la estrategia rápidamente.
¿Te gustaría que diseñara la siguiente pestaña, Análisis de Resultados, o prefieres hacer alguna modificación en el Backtesting Runner?


Diseña la pestaña "Análisis de Resultados"


¡Perfecto! Aquí tienes el diseño completo y detallado para la pestaña de Análisis de Resultados.

Diseño de la Pestaña 4: Análisis de Resultados (platform_gui_tab4_improved.py)
1. Descripción funcional: Esta pestaña proporciona un análisis exhaustivo de los resultados de backtests previamente ejecutados. Permite a los usuarios seleccionar un backtest específico para revisar en detalle, visualizar el gráfico de equity, ver todas las operaciones (trades) realizadas, y acceder a un conjunto completo de métricas de rendimiento avanzadas. La funcionalidad de exportación de reportes y la comparación de estrategias (que se detallará en la Tab 5) son puntos clave.

2. Layout y UI: La pestaña se divide en un panel de selección de backtests a la izquierda (ancho fijo) y un área de visualización detallada de resultados a la derecha (expandible). El área de resultados se organiza con un QTabWidget interno para navegar entre "Gráfico de Equity", "Métricas Detalladas" y "Tabla de Trades".

Panel de Selección de Backtests Izquierdo (QVBoxLayout - ancho fijo, ej. 280px):

Título: QLabel "Backtests Guardados".
Campo de Búsqueda: QLineEdit para filtrar la lista de backtests.
Lista de Backtests: QListWidget que muestra los backtests disponibles, con información concisa (ej. "Estrategia X - BTC/USDT - 2022-2023 - PNL: +X%"). Al seleccionar uno, se cargan sus resultados.
Botones de Acción (QHBoxLayout):
QPushButton "Cargar Backtest": Carga un backtest desde un archivo si no está en la lista.
QPushButton "Eliminar": Elimina el backtest seleccionado (con confirmación).
QSpacerItem: Para empujar los elementos hacia la parte superior.
Área Principal de Resultados Detallados (QVBoxLayout - expandible):

Título: QLabel con el nombre del backtest seleccionado (ej. "Análisis: Estrategia MACD Crossover (BTC/USDT 1h 2022-2023)").
QTabWidget Interno: Contiene las siguientes sub-pestañas:
Sub-pestaña 1: "Gráfico de Equity" (QVBoxLayout):
QChartView (de QtCharts) mostrando la curva de equity, con posibles marcadores para operaciones (compra/venta), puntos de drawdown máximo, y una línea de referencia (ej. "buy and hold").
Pequeños controles de zoom/pan debajo del gráfico.
Sub-pestaña 2: "Métricas Detalladas" (QScrollArea con un QGridLayout o QFormLayout):
Un listado extenso de métricas de rendimiento: PNL neto, retorno bruto/neto, retorno anualizado, volatilidad, ratio de Sharpe, ratio de Sortino, Drawdown máximo, duración de drawdown, % de operaciones ganadoras/perdedoras, promedio de PNL por operación, máximo PNL por operación, etc.
Cada métrica se presenta con QLabel (nombre de métrica) y QLabel (valor), usando colores (verde/rojo) donde sea apropiado.
Sub-pestaña 3: "Tabla de Trades" (QTableWidget):
Tabla paginada o con lazy loading que lista todas las operaciones realizadas durante el backtest. Columnas: Fecha/Hora de Apertura, Fecha/Hora de Cierre, Tipo (Compra/Venta), Símbolo, Precio de Entrada, Precio de Salida, Tamaño, PNL ($), PNL (%), Duración.
Capacidad de ordenar por columna.
Botones de Acción (inferior derecha del área principal):
QPushButton "Exportar Reporte (PDF/CSV)".
QPushButton "Comparar (Añadir a Comparador)" (envía el backtest actual a la Tab 5).
3. Componentes UI:

QTabWidget: Contenedor principal y secundario.
QWidget, QFrame: Contenedores estilizados.
QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout: Para la organización.
QLabel:
Título de la pestaña/sección: font-size: 14px; font-weight: bold; color: #E0E0E0; margin-bottom: 5px;
Nombre del backtest: font-size: 18px; font-weight: bold; color: #E0E0E0; margin-bottom: 10px;
Etiquetas de métricas: font-size: 12px; color: #B0B0B0;
Valores de métricas: font-size: 14px; font-weight: bold; con pnl_positive (#00FF7F), pnl_negative (#FF4500).
QLineEdit (para búsqueda):
Estilo: background-color: #282C34; border: 1px solid #3A3F47; border-radius: 4px; padding: 2px 5px; color: #D0D0D0; height: 26px;
QListWidget:
Estilo: background-color: #1E2125; border: 1px solid #3A3F47; color: #D0D0D0; selection-background-color: #007ACC;
Items: padding: 5px;
QPushButton:
height: 26px; border-radius: 4px; background-color: #007ACC; color: white; font-size: 12px; padding: 2px 8px; border: none; margin-top: 5px;
Botón "Eliminar" con estilo rojo.
QChartView (de QtCharts):
Configuración: Fondo oscuro, rejilla tenue, líneas de equity en azul/verde, marcadores de trade en círculos pequeños (verde para ganancia, rojo para pérdida), líneas de drawdown en rojo.
QTableWidget:
Estilo: background-color: #1E2125; border: 1px solid #3A3F47; color: #D0D0D0; gridline-color: #3A3F47; selection-background-color: #005B9F;
Cabeceras: QHeaderView::section { background-color: #282C34; color: #E0E0E0; padding: 4px; border: 1px solid #3A3F47; font-weight: bold; }
QScrollArea: border: none; background-color: transparent;
4. UX/Flujo de Usuario: Al abrir la pestaña, el usuario ve una lista de backtests guardados. Seleccionar un backtest carga inmediatamente sus datos en las sub-pestañas. La sub-pestaña "Gráfico de Equity" es la primera en mostrarse, proporcionando una visión general visual del rendimiento. El usuario puede navegar entre las sub-pestañas para ver métricas detalladas y la tabla de operaciones. La opción de exportar el reporte permite al usuario guardar o compartir los resultados, mientras que el botón "Comparar" facilita la adición de backtests a una selección para análisis comparativo posterior. La paginación en la tabla de trades (o lazy loading) es crucial para manejar grandes volúmenes de datos sin sobrecargar la UI.

5. Integración Técnica:

backtest_results_manager.py: Módulo para cargar, guardar y gestionar los resultados completos de los backtests (que pueden ser objetos complejos o DataFrames de Pandas serializados).
Métodos: load_all_backtests_summary(), load_backtest_details(id), delete_backtest(id).
backtester_core.py / metrics_calculator.py: Módulos que generan las métricas detalladas a partir de los resultados crudos del backtest.
export_manager.py: Un módulo para generar reportes en diferentes formatos (PDF con gráficos, CSV de trades).
Comunicación con Tab 5 (Testing A/B): Se emitirá una señal (add_to_comparison_requested) que llevará el ID del backtest a la pestaña de "Testing A/B".
Señales/Slots:
QListWidget.currentItemChanged se conecta a un slot que invoca backtest_results_manager.load_backtest_details y luego actualiza los widgets de las sub-pestañas.
QPushButton "Exportar Reporte" se conecta a un slot que invoca a export_manager.generate_report.
QPushButton "Comparar" emite una señal con los datos del backtest actual.
6. Ejemplos de Código (Fragmentos PySide6):

# --- Fragmento 1: Estructura básica de la pestaña Análisis de Resultados ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea, QFrame, QGridLayout,
    QSizePolicy, QMessageBox, QSpacerItem
)
from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QColor, QPalette

# Placeholder for QtCharts (requires 'pip install PySide6-QtCharts')
# from PySide6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis

class BacktestAnalysisTab(QWidget):
    # Signal to add selected backtest to comparison in Tab 5
    add_to_comparison_requested = Signal(str) # Emits backtest_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_backtest_id = None
        self.setStyleSheet(self.get_stylesheet())
        self.init_ui()
        self.load_backtests_list() # Load initial list of backtests

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #1C1F26;
                color: #D0D0D0;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel.section_title {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 5px;
            }
            QLabel.backtest_title {
                font-size: 18px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 10px;
            }
            QLabel.metric_label {
                font-size: 12px;
                color: #B0B0B0;
            }
            QLabel.metric_value {
                font-size: 14px;
                font-weight: bold;
            }
            QLabel.pnl_positive { color: #00FF7F; } /* Green */
            QLabel.pnl_negative { color: #FF4500; } /* Red */
            QLineEdit {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 4px;
                padding: 2px 5px;
                color: #D0D0D0;
                height: 26px;
            }
            QListWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                selection-background-color: #007ACC;
                alternate-background-color: #21252C;
            }
            QListWidget::item {
                padding: 5px;
            }
            QPushButton {
                height: 26px;
                border-radius: 4px;
                background-color: #007ACC;
                color: white;
                font-size: 12px;
                padding: 2px 8px;
                border: none;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
            QPushButton#deleteButton {
                background-color: #DC3545;
            }
            QPushButton#deleteButton:hover {
                background-color: #BD2130;
            }
            QTabWidget::pane { /* The tab widget frame */
                border-top: 1px solid #3A3F47;
                background: #1C1F26;
            }
            QTabBar::tab {
                background: #282C34;
                border: 1px solid #3A3F47;
                border-bottom-color: #3A3F47; /* same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 4px 10px;
                color: #D0D0D0;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1C1F26;
                border-bottom-color: #1C1F26; /* Makes it blend with the pane */
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #005B9F;
            }
            QTableWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                gridline-color: #3A3F47;
                selection-background-color: #005B9F;
                alternate-background-color: #21252C;
            }
            QHeaderView::section {
                background-color: #282C34;
                color: #E0E0E0;
                padding: 4px;
                border: 1px solid #3A3F47;
                font-weight: bold;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QFrame#backtest_list_panel, QFrame#analysis_area_frame {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 6px;
                padding: 10px;
            }
        """

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Left Backtest List Panel ---
        list_panel = QFrame(objectName="backtest_list_panel")
        list_panel.setFixedWidth(280)
        list_layout = QVBoxLayout(list_panel)
        list_layout.setAlignment(Qt.AlignTop)
        list_layout.setSpacing(8)

        list_layout.addWidget(QLabel("Backtests Guardados", objectName="section_title"))

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar backtests...")
        self.search_input.textChanged.connect(self.filter_backtests)
        list_layout.addWidget(self.search_input)

        self.backtest_list_widget = QListWidget()
        self.backtest_list_widget.currentItemChanged.connect(self.load_backtest_for_analysis)
        list_layout.addWidget(self.backtest_list_widget, 1)

        list_buttons_layout = QHBoxLayout()
        list_buttons_layout.setSpacing(5)
        list_buttons_layout.addWidget(QPushButton("Cargar Backtest", clicked=self.load_external_backtest))
        self.delete_button = QPushButton("Eliminar", objectName="deleteButton", clicked=self.delete_backtest)
        list_buttons_layout.addWidget(self.delete_button)
        list_layout.addLayout(list_buttons_layout)

        main_layout.addWidget(list_panel)

        # --- Right Main Analysis Area ---
        analysis_area = QFrame(objectName="analysis_area_frame")
        analysis_layout = QVBoxLayout(analysis_area)
        analysis_layout.setSpacing(10)

        self.backtest_name_label = QLabel("Seleccione un Backtest para Analizar", objectName="backtest_title")
        analysis_layout.addWidget(self.backtest_name_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.setContentsMargins(0,0,0,0) # Remove extra margins for inner tabs

        # Sub-tab 1: Equity Chart
        self.equity_chart_tab = QWidget()
        equity_chart_layout = QVBoxLayout(self.equity_chart_tab)
        self.equity_chart_view = QWidget() # Placeholder for QChartView
        self.equity_chart_view.setStyleSheet("background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 6px;")
        chart_label = QLabel("Gráfico de Equity (Placeholder)", objectName="metric_label")
        chart_label.setAlignment(Qt.AlignCenter)
        chart_layout_placeholder = QVBoxLayout(self.equity_chart_view)
        chart_layout_placeholder.addWidget(chart_label)
        self.equity_chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        equity_chart_layout.addWidget(self.equity_chart_view)
        self.tab_widget.addTab(self.equity_chart_tab, "Gráfico de Equity")

        # Sub-tab 2: Detailed Metrics
        self.detailed_metrics_tab = QWidget()
        metrics_scroll_area = QScrollArea()
        metrics_scroll_area.setWidgetResizable(True)
        self.metrics_container = QWidget()
        self.metrics_layout = QGridLayout(self.metrics_container)
        self.metrics_layout.setSpacing(10)
        self.metrics_layout.setContentsMargins(5,5,5,5) # Small padding
        metrics_scroll_area.setWidget(self.metrics_container)
        metrics_layout_parent = QVBoxLayout(self.detailed_metrics_tab)
        metrics_layout_parent.addWidget(metrics_scroll_area)
        self.tab_widget.addTab(self.detailed_metrics_tab, "Métricas Detalladas")

        # Sub-tab 3: Trades Table
        self.trades_table_tab = QWidget()
        trades_layout = QVBoxLayout(self.trades_table_tab)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(9) # Date Open, Date Close, Type, Symbol, Entry, Exit, Size, PNL($), PNL(%)
        self.trades_table.setHorizontalHeaderLabels(["Fecha Apertura", "Fecha Cierre", "Tipo", "Símbolo", "Entrada", "Salida", "Tamaño", "PNL ($)", "PNL (%)"])
        self.trades_table.horizontalHeader().setStretchLastSection(True)
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trades_table.verticalHeader().setVisible(False)
        self.trades_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trades_table.setAlternatingRowColors(True)
        trades_layout.addWidget(self.trades_table)
        self.tab_widget.addTab(self.trades_table_tab, "Tabla de Trades")

        analysis_layout.addWidget(self.tab_widget, 1)

        # Action Buttons
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.addStretch(1) # Pushes buttons to the right
        self.export_button = QPushButton("Exportar Reporte (PDF/CSV)", clicked=self.export_report)
        self.export_button.setIcon(self.style().standardIcon(self.style().SP_DialogSaveButton))
        action_buttons_layout.addWidget(self.export_button)
        self.compare_button = QPushButton("Comparar (Añadir a Comparador)", clicked=self.add_to_comparison)
        self.compare_button.setIcon(self.style().standardIcon(self.style().SP_DialogHelpButton)) # Placeholder icon
        action_buttons_layout.addWidget(self.compare_button)
        analysis_layout.addLayout(action_buttons_layout)

        main_layout.addWidget(analysis_area, 1)

        self.set_analysis_enabled(False) # Disable analysis until a backtest is selected


    def add_metric_row(self, layout, row, label_text, value_text, is_pnl=False):
        label = QLabel(label_text, objectName="metric_label")
        value_label = QLabel(value_text, objectName="metric_value")
        if is_pnl:
            if float(value_text.replace('%', '').replace('$', '').replace(',', '.')) >= 0:
                value_label.setProperty("class", "metric_value pnl_positive")
            else:
                value_label.setProperty("class", "metric_value pnl_negative")
        layout.addWidget(label, row, 0, Qt.AlignLeft)
        layout.addWidget(value_label, row, 1, Qt.AlignRight)
        return value_label


    def load_backtests_list(self):
        # Simulate loading from backtest_results_manager.py
        self.backtests_summary_data = {
            "macd_001_run_20231026": {
                "id": "macd_001_run_20231026",
                "name": "Estrategia MACD Crossover",
                "strategy_id": "macd_001",
                "data_source": "BTC/USDT 1h 2022-2023",
                "net_pnl_pct": 15.25,
                "max_drawdown_pct": -8.1,
                "run_date": "2023-10-26 15:30"
            },
            "rsi_002_run_20231025": {
                "id": "rsi_002_run_20231025",
                "name": "Estrategia RSI Sobrevendida",
                "strategy_id": "rsi_002",
                "data_source": "ETH/USDT 4h 2021-2023",
                "net_pnl_pct": -3.5,
                "max_drawdown_pct": -12.5,
                "run_date": "2023-10-25 09:00"
            }
        }
        self.backtest_list_widget.clear()
        for bt_id, summary in self.backtests_summary_data.items():
            item_text = f"{summary['name']} - {summary['data_source']} - PNL: {summary['net_pnl_pct']:.2f}%"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, bt_id) # Store ID for retrieval
            self.backtest_list_widget.addItem(item)
        if self.backtests_summary_data:
            self.backtest_list_widget.setCurrentRow(0)

    @Slot(QListWidgetItem, QListWidgetItem)
    def load_backtest_for_analysis(self, current_item, previous_item):
        if current_item:
            self.current_backtest_id = current_item.data(Qt.UserRole)
            # Simulate loading full backtest data
            full_data = self.get_simulated_full_backtest_data(self.current_backtest_id)
            if full_data:
                self.backtest_name_label.setText(f"Análisis: {full_data['name']} ({full_data['data_source']})")
                self.update_equity_chart_view(full_data['equity_curve'])
                self.populate_detailed_metrics(full_data['metrics'])
                self.populate_trades_table(full_data['trades'])
                self.set_analysis_enabled(True)
            else:
                self.clear_analysis_area()
                self.set_analysis_enabled(False)
        else:
            self.clear_analysis_area()
            self.set_analysis_enabled(False)

    def get_simulated_full_backtest_data(self, backtest_id):
        # This would come from backtest_results_manager.py
        if backtest_id == "macd_001_run_20231026":
            return {
                "id": backtest_id,
                "name": "Estrategia MACD Crossover",
                "data_source": "BTC/USDT 1h 2022-2023",
                "equity_curve": [10000 + i * random.uniform(-5, 10) for i in range(500)],
                "metrics": {
                    "Net PNL Total": "$1525.00", "Net PNL %": "15.25%",
                    "Annualized Return": "18.50%", "Max Drawdown": "-8.10%",
                    "Sharpe Ratio": "1.25", "Sortino Ratio": "2.10",
                    "Total Trades": "120", "Winning Trades %": "58.33%",
                    "Avg PNL per Trade": "$12.71", "Largest Win": "$150.00",
                    "Largest Loss": "-$80.00", "Avg Trade Duration": "3 days"
                },
                "trades": [
                    ["2023-01-05 10:00", "2023-01-08 14:30", "LONG", "BTC/USDT", "17000", "17200", "0.1", "20.00", "1.18"],
                    ["2023-01-10 09:00", "2023-01-11 11:00", "SHORT", "BTC/USDT", "17500", "17400", "0.05", "5.00", "0.29"],
                    ["2023-01-15 15:00", "2023-01-16 16:00", "LONG", "BTC/USDT", "16800", "16700", "0.2", "-20.00", "-1.19"]
                ]
            }
        elif backtest_id == "rsi_002_run_20231025":
            return {
                "id": backtest_id,
                "name": "Estrategia RSI Sobrevendida",
                "data_source": "ETH/USDT 4h 2021-2023",
                "equity_curve": [10000 + i * random.uniform(-10, 5) for i in range(400)],
                "metrics": {
                    "Net PNL Total": "-$350.00", "Net PNL %": "-3.50%",
                    "Annualized Return": "-4.00%", "Max Drawdown": "-12.50%",
                    "Sharpe Ratio": "-0.30", "Sortino Ratio": "-0.50",
                    "Total Trades": "80", "Winning Trades %": "45.00%",
                    "Avg PNL per Trade": "-$4.38", "Largest Win": "$90.00",
                    "Largest Loss": "-$120.00", "Avg Trade Duration": "5 days"
                },
                "trades": [
                    ["2022-05-01 12:00", "2022-05-06 18:00", "LONG", "ETH/USDT", "2000", "1950", "0.5", "-25.00", "-2.50"],
                    ["2022-05-10 10:00", "2022-05-12 10:00", "SHORT", "ETH/USDT", "1800", "1850", "0.2", "-10.00", "-2.78"]
                ]
            }
        return None

    def update_equity_chart_view(self, equity_curve):
        # Requires actual QtCharts implementation
        # For now, just update the placeholder label
        chart_label = self.equity_chart_view.findChild(QLabel, "metric_label")
        if chart_label:
            chart_label.setText(f"Gráfico de Equity (Placeholder) - {len(equity_curve)} puntos")
        # Example for QChartView:
        # series = QLineSeries()
        # for i, value in enumerate(equity_curve):
        #     series.append(i, value)
        # chart = QChart()
        # chart.addSeries(series)
        # chart.createDefaultAxes()
        # if isinstance(self.equity_chart_view, QChartView):
        #     self.equity_chart_view.setChart(chart)

    def populate_detailed_metrics(self, metrics_data):
        # Clear existing metrics
        for i in reversed(range(self.metrics_layout.count())):
            widget = self.metrics_layout.itemAt(i).widget()
            if widget: widget.deleteLater()

        row = 0
        for label, value in metrics_data.items():
            is_pnl_metric = "PNL" in label or "Drawdown" in label or "Return" in label
            self.add_metric_row(self.metrics_layout, row, label + ":", value, is_pnl=is_pnl_metric)
            row += 1
        self.metrics_layout.setRowStretch(row, 1) # Push content to top

    def populate_trades_table(self, trades_data):
        self.trades_table.setRowCount(len(trades_data))
        for row_idx, row_list in enumerate(trades_data):
            for col_idx, item_data in enumerate(row_list):
                item = QTableWidgetItem(str(item_data))
                if col_idx == 7: # PNL ($) column
                    try:
                        pnl_val = float(str(item_data).replace('$', '').replace(',', '.'))
                        if pnl_val >= 0:
                            item.setForeground(QColor("#00FF7F")) # Green
                        else:
                            item.setForeground(QColor("#FF4500")) # Red
                    except ValueError:
                        pass
                self.trades_table.setItem(row_idx, col_idx, item)
        self.trades_table.resizeColumnsToContents()

    def clear_analysis_area(self):
        self.backtest_name_label.setText("Seleccione un Backtest para Analizar")
        # Clear chart (if QChartView)
        self.equity_chart_view.findChild(QLabel, "metric_label").setText("Gráfico de Equity (Placeholder)")
        # Clear metrics
        for i in reversed(range(self.metrics_layout.count())):
            widget = self.metrics_layout.itemAt(i).widget()
            if widget: widget.deleteLater()
        # Clear trades table
        self.trades_table.setRowCount(0)
        self.current_backtest_id = None

    def set_analysis_enabled(self, enabled: bool):
        self.tab_widget.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.compare_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)


    @Slot(str)
    def filter_backtests(self, text):
        for i in range(self.backtest_list_widget.count()):
            item = self.backtest_list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    @Slot()
    def load_external_backtest(self):
        QMessageBox.information(self, "Cargar Backtest", "Abrir diálogo para cargar archivo de backtest (simulado).")
        # In a real app, use QFileDialog and then load data
        # self.load_backtests_list() # Potentially refresh list after loading

    @Slot()
    def delete_backtest(self):
        if self.current_backtest_id:
            reply = QMessageBox.question(self, "Eliminar Backtest",
                                         f"¿Está seguro que desea eliminar el backtest '{self.current_backtest_id}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Simulate deletion from backend
                self.backtests_summary_data.pop(self.current_backtest_id, None)
                self.load_backtests_list() # Refresh the list
                QMessageBox.information(self, "Eliminar Backtest", f"Backtest '{self.current_backtest_id}' eliminado.")
                self.clear_analysis_area()
                self.set_analysis_enabled(False)
        else:
            QMessageBox.warning(self, "Eliminar Backtest", "Seleccione un backtest para eliminar.")

    @Slot()
    def export_report(self):
        if self.current_backtest_id:
            QMessageBox.information(self, "Exportar Reporte",
                                    f"Generando reporte para backtest '{self.current_backtest_id}' (simulado).")
            # Call export_manager.generate_report(...)
        else:
            QMessageBox.warning(self, "Exportar Reporte", "Seleccione un backtest para exportar su reporte.")

    @Slot()
    def add_to_comparison(self):
        if self.current_backtest_id:
            self.add_to_comparison_requested.emit(self.current_backtest_id)
            QMessageBox.information(self, "Añadir a Comparador",
                                    f"Backtest '{self.current_backtest_id}' añadido al comparador (simulado, se enviará a la siguiente pestaña).")
        else:
            QMessageBox.warning(self, "Añadir a Comparador", "Seleccione un backtest para añadirlo al comparador.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Análisis de Resultados")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(BacktestAnalysisTab())
    window.setGeometry(100, 100, 1400, 900)
    window.show()
    sys.exit(app.exec())
# --- Fragmento 2: Clase simulada de BacktestResultsManager (Backend) ---
import json
import os
import random

class BacktestResultsManager:
    def __init__(self, results_dir="backtest_results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def _get_result_filepath(self, backtest_id):
        return os.path.join(self.results_dir, f"{backtest_id}.json")

    def save_backtest_result(self, backtest_data):
        backtest_id = backtest_data.get("id")
        if not backtest_id:
            raise ValueError("Backtest data must have an 'id' field.")
        filepath = self._get_result_filepath(backtest_id)
        with open(filepath, 'w') as f:
            json.dump(backtest_data, f, indent=4)
        print(f"Backtest '{backtest_id}' saved to {filepath}")

    def load_backtest_summary(self, backtest_id):
        filepath = self._get_result_filepath(backtest_id)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Return only summary info
                return {
                    "id": data.get("id"),
                    "name": data.get("name", "Unknown Strategy"),
                    "strategy_id": data.get("strategy_id", "N/A"),
                    "data_source": data.get("data_source", "N/A"),
                    "net_pnl_pct": data["metrics"]["Net PNL %"].replace('%', '').replace(',', '.'), # Example parsing
                    "max_drawdown_pct": data["metrics"]["Max Drawdown"].replace('%', '').replace(',', '.'),
                    "run_date": data.get("run_date", "N/A")
                }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading summary for {backtest_id}: {e}")
            return None

    def load_backtest_details(self, backtest_id):
        filepath = self._get_result_filepath(backtest_id)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading details for {backtest_id}: {e}")
            return None

    def get_all_backtests_summaries(self):
        summaries = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith(".json"):
                backtest_id = filename.replace(".json", "")
                summary = self.load_backtest_summary(backtest_id)
                if summary:
                    summaries.append(summary)
        return summaries

    def delete_backtest(self, backtest_id):
        filepath = self._get_result_filepath(backtest_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Backtest '{backtest_id}' deleted.")
            return True
        return False

# Example usage (you'd populate self.backtests_summary_data in the UI from this)
# results_manager = BacktestResultsManager()
# simulated_full_data = {
#     "id": "macd_001_run_20231026",
#     "name": "Estrategia MACD Crossover",
#     "strategy_id": "macd_001",
#     "data_source": "BTC/USDT 1h 2022-2023",
#     "run_date": "2023-10-26 15:30",
#     "equity_curve": [10000 + i * random.uniform(-5, 10) for i in range(500)],
#     "metrics": {
#         "Net PNL Total": "$1525.00", "Net PNL %": "15.25%",
#         "Annualized Return": "18.50%", "Max Drawdown": "-8.10%",
#         "Sharpe Ratio": "1.25", "Sortino Ratio": "2.10",
#         "Total Trades": "120", "Winning Trades %": "58.33%",
#         "Avg PNL per Trade": "$12.71", "Largest Win": "$150.00",
#         "Largest Loss": "-$80.00", "Avg Trade Duration": "3 days"
#     },
#     "trades": [
#         ["2023-01-05 10:00", "2023-01-08 14:30", "LONG", "BTC/USDT", "17000", "17200", "0.1", "20.00", "1.18"],
#         ["2023-01-10 09:00", "2023-01-11 11:00", "SHORT", "BTC/USDT", "17500", "17400", "0.05", "5.00", "0.29"],
#         ["2023-01-15 15:00", "2023-01-16 16:00", "LONG", "BTC/USDT", "16800", "16700", "0.2", "-20.00", "-1.19"]
#     ]
# }
# results_manager.save_backtest_result(simulated_full_data)
# results_manager.get_all_backtests_summaries()
7. Mockups/Wireframes (Descripción Textual Detallada):

[ÁREA DE LA APLICACIÓN: PESTAÑA ANÁLISIS DE RESULTADOS]

--- BARRA DE NAVEGACIÓN SUPERIOR (TABS GLOBALES) ---
[ ... | Backtesting Runner | Análisis de Resultados | Testing A/B | ... ]
------------------------------------------------------

[CONTENIDO PRINCIPAL DE LA PESTAÑA ANÁLISIS DE RESULTADOS]

+-------------------------+-------------------------------------------------------------------------------------+
|    PANEL LATERAL IZQ.   |                            ÁREA DE VISUALIZACIÓN DE RESULTADOS DETALLADOS         |
|   (Ancho Fijo ~280px)   |                                                                                     |
+-------------------------+-------------------------------------------------------------------------------------+
|                         |                                                                                     |
| **Backtests Guardados** | **Análisis: Estrategia MACD Crossover (BTC/USDT 1h 2022-2023)** (Título Grande)     |
|                         |                                                                                     |
| [ Campo de Búsqueda    ]| +--------------------------------------------------------------------------------+ |
|                         | | PESTAÑAS DE VISUALIZACIÓN INTERNAS                                             | |
| +---------------------+ | | [ Gráfico de Equity | Métricas Detalladas | Tabla de Trades ]                  | |
| | Estrategia MACD (P: | | +--------------------------------------------------------------------------------+ |
| |   +15.25%)          | | |                                                                                  | |
| | Estrategia RSI (P:  | | |   [CONTENIDO DE LA SUB-PESTAÑA SELECCIONADA]                                     | |
| |   -3.50%)           | | |                                                                                  | |
| | ...                 | | |   **Ej. Sub-pestaña "Gráfico de Equity":**                                       | |
| +---------------------+ | |   +--------------------------------------------------------------------------+ | |
|                         | |   |                            GRÁFICO DE EQUITY                               | |
| [ Cargar Backtest ]     | |   | [Eje Y: Capital]                                                           | |
| [ Eliminar        ]     | |   | |                                                                          | |
|                         | |   | |                      CURVA DE EQUITY (AZUL)                                | |
|                         | |   | |___________(Marcadores de Trade Verde/Rojo)_________________________________| |
|                         | |   | [Eje X: Tiempo]                                                            | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |                                                                                  | |
|                         | |   **Ej. Sub-pestaña "Métricas Detalladas":**                                     | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | PNL Neto Total:   [ $1525.00 (VERDE) ] | Retorno Anualizado: [ 18.50% (VERDE) ] | |
|                         | |   | Max Drawdown:    [ -8.10% (ROJO)    ] | Sharpe Ratio:      [ 1.25          ] | |
|                         | |   | Total Trades:    [ 120            ] | Winning Trades %:  [ 58.33%        ] | |
|                         | |   | ... (más métricas)                                                         | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |                                                                                  | |
|                         | |   **Ej. Sub-pestaña "Tabla de Trades":**                                         | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | Fecha Apertura | Fecha Cierre | Tipo | Símbolo | Entrada | Salida | PNL($) | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | 2023-01-05     | 2023-01-08   | LONG | BTC/USDT| 17000   | 17200  | +20.00 | |
|                         | |   | 2023-01-10     | 2023-01-11   | SHORT| BTC/USDT| 17500   | 17400  | +5.00  | |
|                         | |   | ... (scrollable)                                                         | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | +--------------------------------------------------------------------------------+ |
|                         |                                                                                     |
|                         | [ Exportar Reporte (PDF/CSV) ] [ Comparar (Añadir a Comparador) ]                   |
+-------------------------+-------------------------------------------------------------------------------------+

--- BARRA DE ESTADO INFERIOR GLOBAL (MENSAJES DE ESTADO, INDICADORES DE CARGA) ---
[ Estado: Sistema Operativo | Última actualización: 2023-10-27 12:45:10 ]
-----------------------------------------------------------------------------
El diseño para la pestaña Análisis de Resultados ha sido generado:

Panel de Lista de Backtests: Permite seleccionar y gestionar backtests previamente ejecutados.
Área de Análisis Detallado: Contiene un QTabWidget interno con tres vistas:
Gráfico de Equity: Visualización interactiva de la curva de capital con puntos clave.
Métricas Detalladas: Una lista exhaustiva de métricas de rendimiento avanzadas.
Tabla de Trades: Una tabla paginada de todas las operaciones ejecutadas.
Acciones Clave: Botones para exportar reportes y añadir backtests a la herramienta de comparación.
¿Te gustaría que diseñara la siguiente pestaña, Testing A/B, o prefieres hacer alguna modificación en el Análisis de Resultados?


Diseña la pestaña "Testing A/B"


¡Absolutamente! Aquí tienes el diseño completo y detallado para la pestaña de Testing A/B.

Diseño de la Pestaña 5: Testing A/B (platform_gui_tab5_improved.py)
1. Descripción funcional: Esta pestaña está dedicada a la comparación estadística y visual de múltiples estrategias o variantes de una misma estrategia, utilizando backtests previamente ejecutados. Permite a los usuarios seleccionar un grupo de backtests y realizar un análisis comparativo profundo, incluyendo gráficos superpuestos, tablas de métricas lado a lado y pruebas de significancia estadística para determinar cuál estrategia es superior.

2. Layout y UI: La pestaña se divide en dos secciones principales. Un panel lateral izquierdo (ancho fijo) para la selección de backtests a comparar, y un área principal a la derecha (expandible) que mostrará los resultados comparativos a través de gráficos y tablas.

Panel de Selección de Backtests Izquierdo (QVBoxLayout - ancho fijo, ej. 320px):

Título: QLabel "Backtests para Comparar".
Campo de Búsqueda: QLineEdit para filtrar la lista de backtests disponibles.
Lista de Backtests Disponibles (QListWidget): Muestra todos los backtests guardados (con checkbox para selección múltiple o botón "Añadir"). Los elementos muestran un resumen conciso (ej. "Estrategia X - BTC/USDT - PNL: +X%").
Controles de Selección/Acción:
QPushButton "Añadir Seleccionados": Mueve los backtests elegidos a la lista de comparación.
QPushButton "Limpiar Selección": Elimina todos los backtests de la lista de comparación.
Lista de Backtests en Comparación (QListWidget): Muestra los backtests que actualmente están configurados para ser comparados. Cada elemento tiene un botón de "X" para eliminarlo individualmente.
QSpacerItem: Para empujar los elementos hacia la parte superior.
Área Principal de Comparación (QVBoxLayout - expandible):

Título: QLabel "Resultados de la Comparación A/B".
QTabWidget Interno: Contiene las siguientes sub-pestañas para diferentes tipos de análisis comparativo:
Sub-pestaña 1: "Gráficos Comparativos" (QVBoxLayout):
QChartView (de QtCharts) principal que superpone las curvas de equity de todas las estrategias seleccionadas, cada una con un color distintivo.
Un QChartView secundario que podría mostrar una comparación de drawdowns o PNL diario/semanal en gráficos de barras.
Leyenda clara para identificar cada curva.
Sub-pestaña 2: "Métricas Comparativas" (QTableWidget):
Una tabla que lista las métricas clave de rendimiento (PNL Neto, Retorno Anualizado, Drawdown Máximo, Ratio de Sharpe, # Trades, % Ganadoras) como filas.
Cada estrategia en comparación tiene su propia columna, permitiendo una lectura rápida lado a lado.
Columnas resaltadas con verde/rojo para indicar mejor/peor rendimiento en métricas específicas.
Sub-pestaña 3: "Pruebas Estadísticas" (QScrollArea con un QVBoxLayout y QLabels para resultados):
QLabel "Prueba de Significancia (ej. t-test, Wilcoxon)": Describe qué prueba se aplica y por qué.
QLabel "Resultados de la Prueba:":
QLabel "Valor P:" (ej. "0.03 (SIGNIFICATIVO)") - con color si es < 0.05.
QLabel "Conclusión:" (ej. "La Estrategia A es estadísticamente superior a la Estrategia B").
Puede incluir QComboBox para seleccionar el tipo de prueba estadística a aplicar.
Botones de Acción (inferior derecha del área principal):
QPushButton "Ejecutar Análisis": Recalcula y actualiza los gráficos y tablas con la configuración actual.
QPushButton "Exportar Reporte Comparativo".
3. Componentes UI:

QTabWidget: Contenedor principal y secundario.
QWidget, QFrame: Contenedores estilizados.
QHBoxLayout, QVBoxLayout, QGridLayout: Para la organización.
QLabel:
Título de la pestaña/sección: font-size: 14px; font-weight: bold; color: #E0E0E0; margin-bottom: 5px;
Título principal del análisis: font-size: 18px; font-weight: bold; color: #E0E0E0; margin-bottom: 10px;
Etiquetas de métricas/pruebas: font-size: 12px; color: #B0B0B0;
Valores de métricas/resultados (verde/rojo): font-size: 14px; font-weight: bold;
QLineEdit (para búsqueda): Estilo consistente.
QListWidget: Estilo consistente, pero con la particularidad de permitir selección múltiple para añadir.
QPushButton: Estilo consistente. Botones "Añadir Seleccionados", "Limpiar Selección", "Ejecutar Análisis", "Exportar Reporte".
QChartView (de QtCharts): Para gráficos superpuestos.
Configuración: Fondo oscuro, rejilla tenue, cada serie de equity con un color único y contraste alto.
QTableWidget: Para métricas comparativas.
Estilo consistente. Las celdas con valores destacados pueden tener fondos o texto de color.
Cabeceras: QHeaderView::section { background-color: #282C34; color: #E0E0E0; padding: 4px; border: 1px solid #3A3F47; font-weight: bold; }
QScrollArea: border: none; background-color: transparent;
4. UX/Flujo de Usuario: El usuario comienza viendo una lista de backtests disponibles. Selecciona los backtests de interés (mediante un checkbox o seleccionando y haciendo clic en "Añadir"). Una vez en la "Lista de Comparación", puede eliminarlos individualmente. Al hacer clic en "Ejecutar Análisis", el área principal se actualiza mostrando los gráficos comparativos de equity en la primera sub-pestaña. El usuario puede luego navegar a las "Métricas Comparativas" para ver un desglose tabular y a "Pruebas Estadísticas" para una evaluación más rigurosa. El botón "Exportar Reporte Comparativo" permite guardar los hallazgos.

5. Integración Técnica:

backtest_results_manager.py: Se usa para obtener la lista de todos los backtests guardados y cargar sus datos detallados cuando se añaden a la comparación.
comparison_engine.py: Un módulo backend que toma una lista de IDs de backtests, carga sus datos de equity/métricas y realiza los cálculos para la comparación.
Métodos: compare_equity_curves(list_of_equity_data), compare_metrics(list_of_metric_dicts), run_statistical_test(list_of_returns_data, test_type).
export_manager.py: Para generar reportes comparativos.
Señales/Slots:
Los botones "Añadir Seleccionados" y "Limpiar Selección" actualizan el QListWidget de comparación.
El QPushButton "Ejecutar Análisis" se conecta a un slot que recopila los IDs de backtests de la lista de comparación, invoca a comparison_engine.py (posiblemente en un QThread si el análisis es pesado) y luego actualiza los QChartViews, QTableWidget y QLabels de resultados.
La pestaña de Análisis de Resultados (Tab 4) puede emitir una señal (add_to_comparison_requested) que esta pestaña (Tab 5) escucha para añadir automáticamente un backtest a la lista de comparación.
6. Ejemplos de Código (Fragmentos PySide6):

# --- Fragmento 1: Estructura básica de la pestaña Testing A/B ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea, QFrame, QGridLayout,
    QSizePolicy, QMessageBox, QSpacerItem, QStyle
)
from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QColor, QPalette

# Placeholder for QtCharts (requires 'pip install PySide6-QtCharts')
# from PySide6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis

class ABTestingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_for_comparison_ids = [] # List of backtest IDs
        self.available_backtests_data = {} # Full data for available backtests
        self.setStyleSheet(self.get_stylesheet())
        self.init_ui()
        self.load_available_backtests() # Load initial list of all backtests

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #1C1F26;
                color: #D0D0D0;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel.section_title {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 5px;
            }
            QLabel.analysis_title {
                font-size: 18px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 10px;
            }
            QLabel.metric_label {
                font-size: 12px;
                color: #B0B0B0;
            }
            QLabel.metric_value {
                font-size: 14px;
                font-weight: bold;
            }
            QLabel.p_value_sig { color: #00FF7F; } /* Green for significant P-value */
            QLabel.p_value_non_sig { color: #FFD700; } /* Yellow for non-significant P-value */

            QLineEdit {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 4px;
                padding: 2px 5px;
                color: #D0D0D0;
                height: 26px;
            }
            QListWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                selection-background-color: #007ACC;
                alternate-background-color: #21252C;
            }
            QListWidget::item {
                padding: 5px;
            }
            QPushButton {
                height: 26px;
                border-radius: 4px;
                background-color: #007ACC;
                color: white;
                font-size: 12px;
                padding: 2px 8px;
                border: none;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
            QTabWidget::pane { /* The tab widget frame */
                border-top: 1px solid #3A3F47;
                background: #1C1F26;
            }
            QTabBar::tab {
                background: #282C34;
                border: 1px solid #3A3F47;
                border-bottom-color: #3A3F47; /* same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 4px 10px;
                color: #D0D0D0;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1C1F26;
                border-bottom-color: #1C1F26; /* Makes it blend with the pane */
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #005B9F;
            }
            QTableWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                gridline-color: #3A3F47;
                selection-background-color: #005B9F;
                alternate-background-color: #21252C;
            }
            QHeaderView::section {
                background-color: #282C34;
                color: #E0E0E0;
                padding: 4px;
                border: 1px solid #3A3F47;
                font-weight: bold;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QFrame#selector_panel, QFrame#comparison_area_frame {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 6px;
                padding: 10px;
            }
            .remove_btn {
                background-color: transparent;
                border: none;
                color: #DC3545;
                font-weight: bold;
                padding: 0 5px;
            }
            .remove_btn:hover {
                color: #BD2130;
            }
        """

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Left Selector Panel ---
        selector_panel = QFrame(objectName="selector_panel")
        selector_panel.setFixedWidth(320)
        selector_layout = QVBoxLayout(selector_panel)
        selector_layout.setAlignment(Qt.AlignTop)
        selector_layout.setSpacing(8)

        selector_layout.addWidget(QLabel("Backtests para Comparar", objectName="section_title"))

        self.search_input_available = QLineEdit()
        self.search_input_available.setPlaceholderText("Buscar backtests disponibles...")
        self.search_input_available.textChanged.connect(self.filter_available_backtests)
        selector_layout.addWidget(self.search_input_available)

        selector_layout.addWidget(QLabel("Disponibles:", objectName="metric_label"))
        self.available_backtests_list = QListWidget()
        self.available_backtests_list.setSelectionMode(QListWidget.ExtendedSelection) # Allow multiple selection
        selector_layout.addWidget(self.available_backtests_list, 1)

        selector_buttons_layout = QHBoxLayout()
        selector_buttons_layout.setSpacing(5)
        self.add_selected_button = QPushButton("Añadir Seleccionados", clicked=self.add_selected_for_comparison)
        selector_buttons_layout.addWidget(self.add_selected_button)
        self.clear_selection_button = QPushButton("Limpiar Selección", clicked=self.clear_comparison_selection)
        selector_buttons_layout.addWidget(self.clear_selection_button)
        selector_layout.addLayout(selector_buttons_layout)
        selector_layout.addSpacing(10)

        selector_layout.addWidget(QLabel("En Comparación:", objectName="metric_label"))
        self.comparison_list_widget = QListWidget()
        self.comparison_list_widget.setSelectionMode(QListWidget.NoSelection) # Not selectable, only remove
        selector_layout.addWidget(self.comparison_list_widget, 1)
        
        selector_layout.addStretch(1) # Pushes widgets to the top

        main_layout.addWidget(selector_panel)

        # --- Right Main Comparison Area ---
        comparison_area = QFrame(objectName="comparison_area_frame")
        comparison_layout = QVBoxLayout(comparison_area)
        comparison_layout.setSpacing(10)

        self.comparison_title_label = QLabel("Resultados de la Comparación A/B", objectName="analysis_title")
        comparison_layout.addWidget(self.comparison_title_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.setContentsMargins(0,0,0,0)

        # Sub-tab 1: Comparative Charts
        self.charts_tab = QWidget()
        charts_layout = QVBoxLayout(self.charts_tab)
        # Placeholder for QChartView for Equity Curves
        self.equity_comparison_chart_view = QWidget()
        self.equity_comparison_chart_view.setStyleSheet("background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 6px;")
        chart_label1 = QLabel("Gráficos de Equity Comparativos (Placeholder)", objectName="metric_label")
        chart_label1.setAlignment(Qt.AlignCenter)
        chart_layout_placeholder1 = QVBoxLayout(self.equity_comparison_chart_view)
        chart_layout_placeholder1.addWidget(chart_label1)
        self.equity_comparison_chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        charts_layout.addWidget(self.equity_comparison_chart_view, 2)

        # Placeholder for QChartView for other comparison (e.g., Drawdown)
        self.other_comparison_chart_view = QWidget()
        self.other_comparison_chart_view.setStyleSheet("background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 6px;")
        chart_label2 = QLabel("Gráfico de Drawdown/PNL por Período (Placeholder)", objectName="metric_label")
        chart_label2.setAlignment(Qt.AlignCenter)
        chart_layout_placeholder2 = QVBoxLayout(self.other_comparison_chart_view)
        chart_layout_placeholder2.addWidget(chart_label2)
        self.other_comparison_chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        charts_layout.addWidget(self.other_comparison_chart_view, 1)
        self.tab_widget.addTab(self.charts_tab, "Gráficos Comparativos")

        # Sub-tab 2: Comparative Metrics Table
        self.metrics_comparison_tab = QWidget()
        metrics_comparison_layout = QVBoxLayout(self.metrics_comparison_tab)
        self.metrics_comparison_table = QTableWidget()
        metrics_comparison_layout.addWidget(self.metrics_comparison_table)
        self.tab_widget.addTab(self.metrics_comparison_tab, "Métricas Comparativas")

        # Sub-tab 3: Statistical Tests
        self.statistical_tests_tab = QWidget()
        statistical_layout = QVBoxLayout(self.statistical_tests_tab)
        statistical_layout.setAlignment(Qt.AlignTop)
        statistical_layout.setSpacing(10)
        
        statistical_layout.addWidget(QLabel("Prueba de Significancia (ej. t-test de Welch)", objectName="section_title"))
        statistical_layout.addWidget(QLabel("Valor P:", objectName="metric_label"))
        self.p_value_label = QLabel("N/A", objectName="metric_value")
        statistical_layout.addWidget(self.p_value_label)
        statistical_layout.addWidget(QLabel("Conclusión:", objectName="metric_label"))
        self.conclusion_label = QLabel("Seleccione backtests y ejecute el análisis.", objectName="metric_value")
        self.conclusion_label.setWordWrap(True)
        statistical_layout.addWidget(self.conclusion_label)
        statistical_layout.addStretch(1) # Push content to top

        self.tab_widget.addTab(self.statistical_tests_tab, "Pruebas Estadísticas")

        comparison_layout.addWidget(self.tab_widget, 1)

        # Action Buttons
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.addStretch(1)
        self.execute_analysis_button = QPushButton("Ejecutar Análisis", clicked=self.execute_comparison_analysis)
        self.execute_analysis_button.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        action_buttons_layout.addWidget(self.execute_analysis_button)
        self.export_report_button = QPushButton("Exportar Reporte Comparativo", clicked=self.export_comparison_report)
        self.export_report_button.setIcon(self.style().standardIcon(self.style().SP_DialogSaveButton))
        action_buttons_layout.addWidget(self.export_report_button)
        comparison_layout.addLayout(action_buttons_layout)

        main_layout.addWidget(comparison_area, 1)
        self.set_analysis_enabled(False)

    def load_available_backtests(self):
        # Simulate loading from backtest_results_manager.py
        self.available_backtests_data = {
            "macd_001_run_20231026": {
                "id": "macd_001_run_20231026", "name": "Estrategia MACD Crossover",
                "data_source": "BTC/USDT 1h 2022-2023", "net_pnl_pct": 15.25, "equity_curve": [10000 + i * random.uniform(-5, 10) for i in range(100)],
                "metrics": {"Net PNL Total": "$1525.00", "Annualized Return": "18.50%", "Max Drawdown": "-8.10%", "Sharpe Ratio": "1.25", "Total Trades": "120", "Winning Trades %": "58.33%"}
            },
            "rsi_002_run_20231025": {
                "id": "rsi_002_run_20231025", "name": "Estrategia RSI Sobrevendida",
                "data_source": "ETH/USDT 4h 2021-2023", "net_pnl_pct": -3.5, "equity_curve": [10000 + i * random.uniform(-10, 5) for i in range(100)],
                "metrics": {"Net PNL Total": "-$350.00", "Annualized Return": "-4.00%", "Max Drawdown": "-12.50%", "Sharpe Ratio": "-0.30", "Total Trades": "80", "Winning Trades %": "45.00%"}
            },
            "macd_001_v2_run_20231027": {
                "id": "macd_001_v2_run_20231027", "name": "MACD Crossover V2",
                "data_source": "BTC/USDT 1h 2022-2023", "net_pnl_pct": 20.10, "equity_curve": [10000 + i * random.uniform(-3, 12) for i in range(100)],
                "metrics": {"Net PNL Total": "$2010.00", "Annualized Return": "22.00%", "Max Drawdown": "-5.50%", "Sharpe Ratio": "1.80", "Total Trades": "135", "Winning Trades %": "62.00%"}
            }
        }
        self.available_backtests_list.clear()
        for bt_id, data in self.available_backtests_data.items():
            item_text = f"{data['name']} - {data['data_source']} - PNL: {data['net_pnl_pct']:.2f}%"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, bt_id) # Store ID
            self.available_backtests_list.addItem(item)
        
        self.update_comparison_list_widget() # Refresh selected list too

    @Slot()
    def add_selected_for_comparison(self):
        selected_items = self.available_backtests_list.selectedItems()
        for item in selected_items:
            bt_id = item.data(Qt.UserRole)
            if bt_id not in self.selected_for_comparison_ids:
                self.selected_for_comparison_ids.append(bt_id)
        self.update_comparison_list_widget()
        self.set_analysis_enabled(len(self.selected_for_comparison_ids) > 0)


    @Slot(str)
    def add_backtest_from_other_tab(self, backtest_id):
        if backtest_id not in self.selected_for_comparison_ids and backtest_id in self.available_backtests_data:
            self.selected_for_comparison_ids.append(backtest_id)
            self.update_comparison_list_widget()
            self.set_analysis_enabled(True)
            QMessageBox.information(self, "Backtest Añadido", f"Backtest '{self.available_backtests_data[backtest_id]['name']}' añadido a la comparación.")
        elif backtest_id not in self.available_backtests_data:
            QMessageBox.warning(self, "Error al Añadir", f"El backtest con ID '{backtest_id}' no fue encontrado entre los disponibles.")


    @Slot()
    def clear_comparison_selection(self):
        self.selected_for_comparison_ids.clear()
        self.update_comparison_list_widget()
        self.set_analysis_enabled(False)
        self.clear_comparison_results()

    def update_comparison_list_widget(self):
        self.comparison_list_widget.clear()
        for bt_id in self.selected_for_comparison_ids:
            if bt_id in self.available_backtests_data:
                data = self.available_backtests_data[bt_id]
                item_text = f"{data['name']} - PNL: {data['net_pnl_pct']:.2f}%"
                
                widget_item = QListWidgetItem(self.comparison_list_widget)
                item_widget = QWidget()
                h_layout = QHBoxLayout(item_widget)
                h_layout.setContentsMargins(0,0,0,0)
                h_layout.addWidget(QLabel(item_text, objectName="metric_label"))
                h_layout.addStretch(1)
                remove_btn = QPushButton("x")
                remove_btn.setObjectName("remove_btn")
                remove_btn.setFixedSize(20, 20)
                remove_btn.clicked.connect(lambda checked, bt=bt_id: self.remove_from_comparison(bt))
                h_layout.addWidget(remove_btn)
                item_widget.setLayout(h_layout)
                widget_item.setSizeHint(item_widget.sizeHint())
                
                self.comparison_list_widget.addItem(widget_item)
                self.comparison_list_widget.setItemWidget(widget_item, item_widget)


    @Slot(str)
    def remove_from_comparison(self, bt_id_to_remove):
        if bt_id_to_remove in self.selected_for_comparison_ids:
            self.selected_for_comparison_ids.remove(bt_id_to_remove)
            self.update_comparison_list_widget()
            self.set_analysis_enabled(len(self.selected_for_comparison_ids) > 0)
            self.clear_comparison_results()


    @Slot()
    def execute_comparison_analysis(self):
        if len(self.selected_for_comparison_ids) < 2:
            QMessageBox.warning(self, "Análisis de Comparación", "Seleccione al menos dos backtests para comparar.")
            return

        self.comparison_title_label.setText("Analizando Backtests...")
        # Simulate fetching full data and performing analysis
        backtests_to_analyze = [self.available_backtests_data[bt_id] for bt_id in self.selected_for_comparison_ids]
        
        # --- Update Charts ---
        equity_curves = {bt['name']: bt['equity_curve'] for bt in backtests_to_analyze}
        self.update_equity_comparison_chart(equity_curves)

        # --- Update Metrics Table ---
        metrics_data = {bt['name']: bt['metrics'] for bt in backtests_to_analyze}
        self.populate_metrics_comparison_table(metrics_data)

        # --- Update Statistical Tests ---
        self.run_simulated_statistical_test(backtests_to_analyze)

        self.comparison_title_label.setText("Resultados de la Comparación A/B")
        QMessageBox.information(self, "Análisis Completado", "Análisis de comparación de backtests completado.")
        self.set_analysis_enabled(True)


    def update_equity_comparison_chart(self, equity_curves_dict):
        # This requires actual QtCharts implementation
        chart_label = self.equity_comparison_chart_view.findChild(QLabel, "metric_label")
        if chart_label:
            chart_label.setText(f"Gráficos de Equity Comparativos (Placeholder) - {len(equity_curves_dict)} curvas")
        # Example for QChartView:
        # chart = QChart()
        # for name, curve in equity_curves_dict.items():
        #     series = QLineSeries()
        #     for i, value in enumerate(curve):
        #         series.append(i, value)
        #     series.setName(name)
        #     chart.addSeries(series)
        # chart.createDefaultAxes()
        # self.equity_comparison_chart_view.setChart(chart)


    def populate_metrics_comparison_table(self, metrics_data_dict):
        if not metrics_data_dict:
            self.metrics_comparison_table.setRowCount(0)
            self.metrics_comparison_table.setColumnCount(0)
            return

        # Get all unique metric names
        all_metric_names = sorted(list(set(metric for bt_metrics in metrics_data_dict.values() for metric in bt_metrics.keys())))
        
        # Set columns: 1 for Metric Name + 1 for each strategy
        self.metrics_comparison_table.setColumnCount(1 + len(metrics_data_dict))
        header_labels = ["Métrica"] + list(metrics_data_dict.keys())
        self.metrics_comparison_table.setHorizontalHeaderLabels(header_labels)
        
        self.metrics_comparison_table.setRowCount(len(all_metric_names))
        
        for row_idx, metric_name in enumerate(all_metric_names):
            self.metrics_comparison_table.setItem(row_idx, 0, QTableWidgetItem(metric_name))
            col_idx = 1
            for strategy_name, bt_metrics in metrics_data_dict.items():
                value = bt_metrics.get(metric_name, "N/A")
                item = QTableWidgetItem(str(value))
                # Basic color highlighting (example: PNL, Drawdown)
                if "PNL" in metric_name and isinstance(value, str):
                    try:
                        num_val = float(value.replace('$', '').replace('%', '').replace(',', '.'))
                        if num_val > 0: item.setForeground(QColor("#00FF7F")) # Green
                        elif num_val < 0: item.setForeground(QColor("#FF4500")) # Red
                    except ValueError: pass
                elif "Drawdown" in metric_name and isinstance(value, str):
                    try:
                        num_val = float(value.replace('%', '').replace(',', '.'))
                        if num_val > -5.0: item.setForeground(QColor("#00FF7F")) # Greenish if not too bad
                        else: item.setForeground(QColor("#FF4500")) # Red if significant drawdown
                    except ValueError: pass

                self.metrics_comparison_table.setItem(row_idx, col_idx, item)
                col_idx += 1
        self.metrics_comparison_table.resizeColumnsToContents()


    def run_simulated_statistical_test(self, backtests):
        if len(backtests) < 2:
            self.p_value_label.setText("N/A")
            self.conclusion_label.setText("Se necesitan al menos dos backtests para realizar pruebas estadísticas.")
            self.p_value_label.setProperty("class", "metric_value")
            self.p_value_label.style().polish(self.p_value_label)
            return

        # Simulate a statistical test result
        p_value = random.uniform(0.001, 0.5)
        self.p_value_label.setText(f"{p_value:.4f}")

        if p_value < 0.05:
            self.p_value_label.setProperty("class", "metric_value p_value_sig")
            conclusion = "Existe una diferencia estadísticamente significativa entre las estrategias."
            # Determine which is 'better' based on a simple metric like Net PNL %
            best_strategy = max(backtests, key=lambda bt: bt['net_pnl_pct'])
            conclusion += f" La estrategia '{best_strategy['name']}' parece ser superior."
        else:
            self.p_value_label.setProperty("class", "metric_value p_value_non_sig")
            conclusion = "No hay evidencia estadística suficiente para concluir una diferencia significativa."
        
        self.p_value_label.style().polish(self.p_value_label) # Repolish for CSS update
        self.conclusion_label.setText(conclusion)


    def clear_comparison_results(self):
        self.comparison_title_label.setText("Resultados de la Comparación A/B")
        # Clear charts
        self.equity_comparison_chart_view.findChild(QLabel, "metric_label").setText("Gráficos de Equity Comparativos (Placeholder)")
        self.other_comparison_chart_view.findChild(QLabel, "metric_label").setText("Gráfico de Drawdown/PNL por Período (Placeholder)")
        # Clear metrics table
        self.metrics_comparison_table.setRowCount(0)
        self.metrics_comparison_table.setColumnCount(0)
        # Clear statistical results
        self.p_value_label.setText("N/A")
        self.p_value_label.setProperty("class", "metric_value")
        self.p_value_label.style().polish(self.p_value_label)
        self.conclusion_label.setText("Seleccione backtests y ejecute el análisis.")

    def set_analysis_enabled(self, enabled: bool):
        self.tab_widget.setEnabled(enabled)
        self.execute_analysis_button.setEnabled(enabled)
        self.export_report_button.setEnabled(enabled)
        # You might also want to enable/disable remove buttons in the comparison list
        # based on whether 'enabled' is True or False

    @Slot(str)
    def filter_available_backtests(self, text):
        for i in range(self.available_backtests_list.count()):
            item = self.available_backtests_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    @Slot()
    def export_comparison_report(self):
        if len(self.selected_for_comparison_ids) > 1:
            QMessageBox.information(self, "Exportar Reporte",
                                    "Generando reporte comparativo (PDF/CSV) para los backtests seleccionados (simulado).")
            # Call export_manager.generate_comparison_report(...)
        else:
            QMessageBox.warning(self, "Exportar Reporte", "Seleccione al menos dos backtests para generar un reporte comparativo.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Testing A/B")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(ABTestingTab())
    window.setGeometry(100, 100, 1400, 900)
    window.show()
    sys.exit(app.exec())
# --- Fragmento 2: Clase simulada de ComparisonEngine (Backend) ---
import random
import numpy as np
# from scipy import stats # For actual statistical tests

class ComparisonEngine:
    def __init__(self):
        pass

    def compare_equity_curves(self, list_of_equity_curves):
        """
        Simulates combining multiple equity curves for charting.
        In a real scenario, this might involve normalization or alignment.
        """
        # For simplicity, return them as is
        return list_of_equity_curves

    def compare_metrics(self, list_of_metric_dicts):
        """
        Simulates aggregating metrics from multiple backtests into a comparable structure.
        """
        combined_metrics = {}
        for strategy_name, metrics in list_of_metric_dicts.items():
            for metric_name, value in metrics.items():
                if metric_name not in combined_metrics:
                    combined_metrics[metric_name] = {}
                combined_metrics[metric_name][strategy_name] = value
        return combined_metrics

    def run_statistical_test(self, list_of_pnl_series, test_type="welch_t"):
        """
        Simulates running a statistical test between two series of PnL data.
        Needs at least two series.
        """
        if len(list_of_pnl_series) < 2:
            return {"p_value": 1.0, "conclusion": "Se requieren al menos dos series para el test estadístico."}

        # In a real implementation:
        # data1 = np.array(list_of_pnl_series[0])
        # data2 = np.array(list_of_pnl_series[1])

        # if test_type == "welch_t":
        #     # Welch's t-test for unequal variances
        #     statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        # elif test_type == "mann_whitney_u":
        #     # Mann-Whitney U test for non-parametric data
        #     statistic, p_value = stats.mannwhitneyu(data1, data2)
        # else:
        #     p_value = random.uniform(0.0, 1.0) # Fallback

        p_value = random.uniform(0.001, 0.5) # Simulated p-value

        conclusion = ""
        if p_value < 0.05:
            conclusion = "Existe una diferencia estadísticamente significativa."
        else:
            conclusion = "No hay evidencia estadística significativa de diferencia."

        return {"p_value": p_value, "conclusion": conclusion}

# Example usage (in ABTestingTab.execute_comparison_analysis):
# comparison_engine = ComparisonEngine()
# equity_curves_for_chart = comparison_engine.compare_equity_curves([bt['equity_curve'] for bt in backtests_to_analyze])
# metrics_for_table = comparison_engine.compare_metrics({bt['name']: bt['metrics'] for bt in backtests_to_analyze})
# # Assuming 'daily_returns' is available in backtest data for statistical tests
# # stat_result = comparison_engine.run_statistical_test([bt['daily_returns'] for bt in backtests_to_analyze])
# # self.p_value_label.setText(f"{stat_result['p_value']:.4f}")
# # self.conclusion_label.setText(stat_result['conclusion'])
7. Mockups/Wireframes (Descripción Textual Detallada):

[ÁREA DE LA APLICACIÓN: PESTAÑA TESTING A/B]

--- BARRA DE NAVEGACIÓN SUPERIOR (TABS GLOBALES) ---
[ ... | Análisis de Resultados | Testing A/B | Monitoreo en Vivo | ... ]
------------------------------------------------------

[CONTENIDO PRINCIPAL DE LA PESTAÑA TESTING A/B]

+-------------------------+-------------------------------------------------------------------------------------+
|    PANEL LATERAL IZQ.   |                            ÁREA DE ANÁLISIS COMPARATIVO                           |
|   (Ancho Fijo ~320px)   |                                                                                     |
+-------------------------+-------------------------------------------------------------------------------------+
|                         |                                                                                     |
| **Backtests para Comp.**| **Resultados de la Comparación A/B** (Título Grande)                                |
|                         |                                                                                     |
| [ Buscar disponibles ]  | +--------------------------------------------------------------------------------+ |
|                         | | PESTAÑAS DE ANÁLISIS INTERNAS                                                  | |
| Disponibles:            | | [ Gráficos Comparativos | Métricas Comparativas | Pruebas Estadísticas ]         | |
| +---------------------+ | +--------------------------------------------------------------------------------+ |
| | [ ] MACD Crossover  | | |                                                                                  | |
| | [ ] RSI Sobrevendida| | |   [CONTENIDO DE LA SUB-PESTAÑA SELECCIONADA]                                     | |
| | [x] MACD Crossover V2| | |                                                                                  | |
| +---------------------+ | |   **Ej. Sub-pestaña "Gráficos Comparativos":**                                   | |
|                         | |   +--------------------------------------------------------------------------+ | |
| [ Añadir Seleccionados ]| |   |                     GRÁFICO DE EQUITY COMPARATIVO                        | |
| [ Limpiar Selección  ]  | |   | [Eje Y: Capital]                                                           | |
|                         | |   | |                     Estrategia A (AZUL)                                  | |
| En Comparación:         | |   | |                 Estrategia B (NARANJA)                                   | |
| +---------------------+ | |   | |__________________________________________________________________________| |
| | MACD Crossover V2 [x]| |   | [Eje X: Tiempo]                                                            | |
| | MACD Crossover    [x]| |   |                                                                            | |
| +---------------------+ | |   |                     GRÁFICO DE DRAWDOWN COMPARATIVO (Barras)               | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |                                                                                  | |
|                         | |   **Ej. Sub-pestaña "Métricas Comparativas":**                                   | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | Métrica            | MACD Cr.   | MACD Cr. V2 | RSI Sobrev.              | |
|                         | |   +--------------------+------------+-------------+--------------------------+ | |
|                         | |   | PNL Neto Total     | $1525.00   | $2010.00    | -$350.00                 | |
|                         | |   | Retorno Anualizado | 18.50%     | 22.00%      | -4.00%                   | |
|                         | |   | Max Drawdown       | -8.10%     | -5.50%      | -12.50%                  | |
|                         | |   | Sharpe Ratio       | 1.25       | 1.80        | -0.30                    | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |                                                                                  | |
|                         | |   **Ej. Sub-pestaña "Pruebas Estadísticas":**                                    | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | **Prueba de Significancia (ej. t-test de Welch)**                        | |
|                         | |   | Valor P:   [ 0.0300 (VERDE, SIGNIFICATIVO) ]                                | |
|                         | |   | Conclusión: [ La Estrategia MACD Crossover V2 es estadísticamente superior ] | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | +--------------------------------------------------------------------------------+ |
|                         |                                                                                     |
|                         | [ Ejecutar Análisis (►) ] [ Exportar Reporte Comparativo ]                          |
+-------------------------+-------------------------------------------------------------------------------------+

--- BARRA DE ESTADO INFERIOR GLOBAL (MENSAJES DE ESTADO, INDICADORES DE CARGA) ---
[ Estado: Sistema Operativo | Última actualización: 2023-10-27 13:30:10 ]
-----------------------------------------------------------------------------
El diseño para la pestaña Testing A/B ha sido generado:

Panel de Selección: Permite elegir múltiples backtests de una lista disponible para un análisis comparativo.
Área de Análisis Comparativo: Organizada en tres sub-pestañas:
Gráficos Comparativos: Visualizaciones superpuestas de curvas de equity y otros gráficos relevantes.
Métricas Comparativas: Una tabla lado a lado para comparar métricas clave.
Pruebas Estadísticas: Presenta los resultados de pruebas de significancia para determinar la superioridad estadística de las estrategias.
Acciones Clave: Botones para ejecutar el análisis y exportar reportes comparativos.
¿Te gustaría que diseñara la siguiente pestaña, Monitoreo en Vivo, o hay algún ajuste que te gustaría hacer en la pestaña de Testing A/B?


Diseña la pestaña "Monitoreo en Vivo"


¡Entendido! Aquí tienes el diseño completo y detallado para la pestaña de Monitoreo en Vivo.

Diseño de la Pestaña 6: Monitoreo en Vivo (platform_gui_tab6_improved.py)
1. Descripción funcional: Esta pestaña es el centro neurálgico para el monitoreo en tiempo real de las estrategias de trading activas. Permite a los usuarios visualizar el rendimiento actual de las estrategias que están ejecutándose en modo "paper trading" o en cuentas reales. Incluye feeds de datos en vivo, seguimiento de posiciones abiertas, historial de operaciones recientes, métricas de rendimiento en tiempo real y la capacidad de iniciar/detener estrategias.

2. Layout y UI: La pestaña se divide en un panel de control lateral izquierdo (ancho fijo) para la gestión de estrategias activas, y un área principal a la derecha (expandible) que mostrará los detalles del monitoreo de la estrategia seleccionada. El área principal usará un QTabWidget interno para organizar la "Vista General", "Posiciones Abiertas", "Historial de Trades" y "Log en Vivo".

Panel de Control Izquierdo (QVBoxLayout - ancho fijo, ej. 300px):

Título: QLabel "Estrategias Activas".
Campo de Búsqueda: QLineEdit para filtrar la lista de estrategias en monitoreo.
Lista de Estrategias en Monitoreo (QListWidget): Muestra las estrategias que están en ejecución (paper o live). Cada elemento puede incluir un indicador de estado (ej. círculo verde/rojo/amarillo) y un resumen rápido (ej. "Estrategia X - BTC/USDT - PNL: +X%"). Al seleccionar una, se cargan sus detalles en el panel central.
Botones de Acción Global (QHBoxLayout):
QPushButton "Iniciar Nueva Estrategia": Abre un diálogo para seleccionar una estrategia y modo de ejecución (paper/live).
QPushButton "Detener Seleccionada": Detiene la estrategia seleccionada (con confirmación).
Indicador de Conexión Global: QLabel "Conexión Exchange:" QLabel (Verde "Activa", Rojo "Inactiva").
QSpacerItem: Para empujar los elementos hacia la parte superior.
Área Principal de Monitoreo (QVBoxLayout - expandible):

Título: QLabel con el nombre de la estrategia seleccionada (ej. "Monitoreo: Estrategia MACD Crossover (Live)").
QTabWidget Interno: Contiene las siguientes sub-pestañas:
Sub-pestaña 1: "Vista General" (QVBoxLayout):
Gráfico de Equity en Vivo: QChartView (de QtCharts) que muestra la curva de equity de la estrategia en tiempo real. Puede incluir el precio del activo subyacente.
Métricas en Vivo (QGridLayout o QFormLayout):
QLabel "PNL Neto Actual:" QLabel (valor en verde/rojo).
QLabel "Retorno % Hoy:" QLabel.
QLabel "Balance de Cuenta:" QLabel.
QLabel "Balance Disponible:" QLabel.
QLabel "Trades Abiertos:" QLabel.
QLabel "Drawdown Actual:" QLabel (en rojo).
Sub-pestaña 2: "Posiciones Abiertas" (QTableWidget):
Tabla que lista todas las posiciones actualmente abiertas por la estrategia.
Columnas: Símbolo, Tipo (LONG/SHORT), Tamaño, Precio de Entrada, Precio Actual, PNL ($), PNL (%), Nivel de Stop Loss, Nivel de Take Profit, Duración.
Columna PNL ($) y PNL (%) con texto verde/rojo.
Sub-pestaña 3: "Historial de Trades" (QTableWidget):
Tabla paginada de las operaciones cerradas recientemente por la estrategia.
Columnas: Fecha/Hora Apertura, Fecha/Hora Cierre, Símbolo, Tipo, Precio Entrada, Precio Salida, PNL ($), PNL (%), Duración.
Columna PNL ($) y PNL (%) con texto verde/rojo.
Sub-pestaña 4: "Log en Vivo" (QTextEdit - solo lectura):
Un área de texto que muestra los eventos y logs generados por la estrategia en tiempo real (ej. "Orden de compra enviada BTC/USDT", "Orden SL ejecutada", "Error de conexión a la API").
Fuente monospace y colores para diferentes tipos de mensajes (info, warn, error).
Botón de Acción (inferior derecha del área principal):
QPushButton "Detener Estrategia" (desactiva la estrategia seleccionada).
3. Componentes UI:

QTabWidget: Contenedor principal y secundario.
QWidget, QFrame: Contenedores estilizados.
QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout: Para la organización.
QLabel:
Título de la pestaña/sección: font-size: 14px; font-weight: bold; color: #E0E0E0; margin-bottom: 5px;
Nombre de la estrategia: font-size: 18px; font-weight: bold; color: #E0E0E0; margin-bottom: 10px;
Etiquetas de métricas: font-size: 12px; color: #B0B0B0;
Valores de métricas: font-size: 14px; font-weight: bold; con pnl_positive (#00FF7F), pnl_negative (#FF4500), pnl_neutral (#FFFFFF).
QLineEdit (para búsqueda): Estilo consistente.
QListWidget: Estilo consistente. Los elementos pueden tener QLabels o custom widgets para los indicadores de estado y resumen.
QPushButton: Estilo consistente.
"Iniciar Nueva Estrategia": Fondo azul estándar.
"Detener Seleccionada" / "Detener Estrategia": Fondo rojo (background-color: #DC3545;).
QChartView (de QtCharts): Para el gráfico de equity en vivo y el gráfico de precios del activo.
Configuración: Fondo oscuro, rejilla tenue, curva de equity en azul/verde.
QTableWidget: Para posiciones abiertas e historial de trades.
Estilo consistente. Las celdas de PNL con texto verde/rojo.
QTextEdit (para log en vivo):
Estilo: background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 4px; padding: 5px; color: #D0D0D0; font-family: 'Consolas', 'Monospace';
Diferentes colores para mensajes: info (gris claro), warning (amarillo), error (rojo).
4. UX/Flujo de Usuario: Al abrir la pestaña, el usuario ve una lista de todas las estrategias que están activas. Seleccionar una estrategia carga sus datos en tiempo real en el panel principal. La "Vista General" proporciona un cuadro de mando instantáneo del rendimiento de la estrategia. La interfaz se actualiza automáticamente con nuevos datos de mercado y de las operaciones de la estrategia. Los indicadores visuales (colores, iconos) permiten una rápida interpretación del estado y rendimiento. El usuario puede iniciar nuevas estrategias o detener las existentes con unos pocos clics. El log en vivo es crucial para depuración y monitoreo detallado de eventos.

5. Integración Técnica:

live_trading_manager.py: Módulo central que gestiona las instancias de estrategias en vivo (o paper), la conexión a brokers/exchanges (ej. CCXT para datos y órdenes), el monitoreo de cuentas y el registro de operaciones en tiempo real.
Métodos: start_strategy(strategy_id, mode='paper/live', account_id), stop_strategy(instance_id), get_active_strategies_summary(), get_strategy_realtime_data(instance_id).
data_stream_manager.py: Módulo que se encarga de las conexiones websocket para feeds de datos en tiempo real (precios, libros de órdenes).
Clase LiveStrategyWorker (en QThread): Cada estrategia en ejecución (o el live_trading_manager en sí) debería operar en su propio hilo para no bloquear la UI y manejar la lógica de trading y la comunicación con el exchange.
Señales emitidas por LiveStrategyWorker:
realtime_update(dict): Con datos de equity, métricas, PNL, etc.
new_position(dict), position_update(dict), position_closed(dict): Para la tabla de posiciones.
new_trade(dict): Para la tabla de historial de trades.
log_message(str, level): Para el QTextEdit de log.
connection_status(bool): Para el indicador de conexión global.
Slots en Monitoreo en Vivo:
QListWidget.currentItemChanged para cargar los detalles de la estrategia seleccionada.
Botones de iniciar/detener se conectan a slots que interactúan con live_trading_manager.py.
Las señales de los LiveStrategyWorkers se conectan a slots que actualizan el QChartView, QLabels y QTableWidgets.
6. Ejemplos de Código (Fragmentos PySide6):

# --- Fragmento 1: Estructura básica de la pestaña Monitoreo en Vivo ---
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QGridLayout, QSizePolicy,
    QMessageBox, QSpacerItem, QTextEdit, QDialog, QComboBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal, Slot, QObject, QTimer
from PySide6.QtGui import QColor, QPalette, QIcon

# Placeholder for QtCharts (requires 'pip install PySide6-QtCharts')
# from PySide6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis

class LiveMonitoringTab(QWidget):
    # Signals for backend communication
    start_new_strategy_requested = Signal(str, str, str) # strategy_id, mode, account_id
    stop_strategy_instance_requested = Signal(str) # instance_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_strategy_instances = {} # Stores instance_id: {name, status, pnl, ...}
        self.current_selected_instance_id = None
        self.setStyleSheet(self.get_stylesheet())
        self.init_ui()
        self.load_active_strategies_list()
        self.setup_realtime_updates() # Simulate real-time data

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #1C1F26;
                color: #D0D0D0;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel.section_title {
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 5px;
            }
            QLabel.strategy_instance_title {
                font-size: 18px;
                font-weight: bold;
                color: #E0E0E0;
                margin-bottom: 10px;
            }
            QLabel.metric_label {
                font-size: 12px;
                color: #B0B0B0;
            }
            QLabel.metric_value {
                font-size: 14px;
                font-weight: bold;
            }
            QLabel.pnl_positive { color: #00FF7F; } /* Green */
            QLabel.pnl_negative { color: #FF4500; } /* Red */
            QLabel.pnl_neutral { color: #FFFFFF; } /* White */
            QLabel.status_active { color: #00FF7F; } /* Green */
            QLabel.status_paused { color: #FFD700; } /* Yellow */
            QLabel.status_error { color: #FF4500; } /* Red */

            QLineEdit {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 4px;
                padding: 2px 5px;
                color: #D0D0D0;
                height: 26px;
            }
            QListWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                selection-background-color: #007ACC;
                alternate-background-color: #21252C;
            }
            QListWidget::item {
                padding: 5px;
            }
            QPushButton {
                height: 26px;
                border-radius: 4px;
                background-color: #007ACC;
                color: white;
                font-size: 12px;
                padding: 2px 8px;
                border: none;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
            QPushButton#stopButton {
                background-color: #DC3545;
            }
            QPushButton#stopButton:hover {
                background-color: #BD2130;
            }
            QTabWidget::pane { /* The tab widget frame */
                border-top: 1px solid #3A3F47;
                background: #1C1F26;
            }
            QTabBar::tab {
                background: #282C34;
                border: 1px solid #3A3F47;
                border-bottom-color: #3A3F47; /* same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 4px 10px;
                color: #D0D0D0;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1C1F26;
                border-bottom-color: #1C1F26; /* Makes it blend with the pane */
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #005B9F;
            }
            QTableWidget {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
                gridline-color: #3A3F47;
                selection-background-color: #005B9F;
                alternate-background-color: #21252C;
            }
            QHeaderView::section {
                background-color: #282C34;
                color: #E0E0E0;
                padding: 4px;
                border: 1px solid #3A3F47;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #1E2125;
                border: 1px solid #3A3F47;
                border-radius: 4px;
                padding: 5px;
                color: #D0D0D0;
                font-family: 'Consolas', 'Monospace';
            }
            .log_info { color: #D0D0D0; }
            .log_warn { color: #FFD700; }
            .log_error { color: #FF4500; }
            QFrame#active_strategies_panel, QFrame#monitoring_area_frame {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                border-radius: 6px;
                padding: 10px;
            }
            QDialog QLabel.title {
                font-size: 16px; font-weight: bold; color: #E0E0E0;
            }
            QDialog QComboBox, QDialog QLineEdit {
                 background-color: #1E2125; border: 1px solid #3A3F47; color: #D0D0D0;
            }
        """

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Left Active Strategies Panel ---
        active_strategies_panel = QFrame(objectName="active_strategies_panel")
        active_strategies_panel.setFixedWidth(300)
        active_strategies_layout = QVBoxLayout(active_strategies_panel)
        active_strategies_layout.setAlignment(Qt.AlignTop)
        active_strategies_layout.setSpacing(8)

        active_strategies_layout.addWidget(QLabel("Estrategias Activas", objectName="section_title"))

        self.search_input_active = QLineEdit()
        self.search_input_active.setPlaceholderText("Buscar estrategia activa...")
        self.search_input_active.textChanged.connect(self.filter_active_strategies)
        active_strategies_layout.addWidget(self.search_input_active)

        self.active_strategies_list_widget = QListWidget()
        self.active_strategies_list_widget.currentItemChanged.connect(self.load_strategy_monitoring_details)
        active_strategies_layout.addWidget(self.active_strategies_list_widget, 1)

        # Global action buttons for strategies
        global_buttons_layout = QHBoxLayout()
        global_buttons_layout.setSpacing(5)
        self.start_new_strategy_button = QPushButton("Iniciar Nueva", clicked=self.open_start_strategy_dialog)
        self.start_new_strategy_button.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        global_buttons_layout.addWidget(self.start_new_strategy_button)
        self.stop_selected_strategy_button = QPushButton("Detener Seleccionada", objectName="stopButton", clicked=self.stop_selected_strategy)
        self.stop_selected_strategy_button.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.stop_selected_strategy_button.setEnabled(False)
        global_buttons_layout.addWidget(self.stop_selected_strategy_button)
        active_strategies_layout.addLayout(global_buttons_layout)
        active_strategies_layout.addSpacing(10)

        # Global connection status
        connection_status_layout = QHBoxLayout()
        connection_status_layout.addWidget(QLabel("Conexión Exchange:", objectName="metric_label"))
        self.exchange_connection_status_label = QLabel("Activa", objectName="status_active")
        connection_status_layout.addWidget(self.exchange_connection_status_label)
        connection_status_layout.addStretch(1)
        active_strategies_layout.addLayout(connection_status_layout)
        
        active_strategies_layout.addStretch(1) # Pushes widgets to the top

        main_layout.addWidget(active_strategies_panel)

        # --- Right Main Monitoring Area ---
        monitoring_area = QFrame(objectName="monitoring_area_frame")
        monitoring_layout = QVBoxLayout(monitoring_area)
        monitoring_layout.setSpacing(10)

        self.strategy_instance_name_label = QLabel("Seleccione una Estrategia Activa", objectName="strategy_instance_title")
        monitoring_layout.addWidget(self.strategy_instance_name_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.setContentsMargins(0,0,0,0)

        # Sub-tab 1: Overview
        self.overview_tab = QWidget()
        overview_layout = QVBoxLayout(self.overview_tab)
        
        # Equity Chart
        self.live_equity_chart_view = QWidget() # Placeholder for QChartView
        self.live_equity_chart_view.setStyleSheet("background-color: #1E2125; border: 1px solid #3A3F47; border-radius: 6px;")
        chart_label1 = QLabel("Gráfico de Equity en Vivo (Placeholder)", objectName="metric_label")
        chart_label1.setAlignment(Qt.AlignCenter)
        chart_layout_placeholder1 = QVBoxLayout(self.live_equity_chart_view)
        chart_layout_placeholder1.addWidget(chart_label1)
        self.live_equity_chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overview_layout.addWidget(self.live_equity_chart_view, 2)

        # Live Metrics
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        self.pnl_net_live_label = self.add_metric_pair(metrics_grid, 0, 0, "PNL Neto Actual:", "N/A", "pnl_net_value")
        self.return_today_label = self.add_metric_pair(metrics_grid, 0, 1, "Retorno % Hoy:", "N/A")
        self.account_balance_label = self.add_metric_pair(metrics_grid, 1, 0, "Balance de Cuenta:", "N/A")
        self.available_balance_label = self.add_metric_pair(metrics_grid, 1, 1, "Balance Disponible:", "N/A")
        self.open_trades_count_label = self.add_metric_pair(metrics_grid, 2, 0, "Trades Abiertos:", "N/A")
        self.current_drawdown_label = self.add_metric_pair(metrics_grid, 2, 1, "Drawdown Actual:", "N/A", "current_drawdown_value")
        overview_layout.addLayout(metrics_grid, 1)
        self.tab_widget.addTab(self.overview_tab, "Vista General")

        # Sub-tab 2: Open Positions
        self.open_positions_tab = QWidget()
        open_positions_layout = QVBoxLayout(self.open_positions_tab)
        self.open_positions_table = QTableWidget()
        self.open_positions_table.setColumnCount(10) # Symbol, Type, Size, Entry, Current, PNL($), PNL(%), SL, TP, Duration
        self.open_positions_table.setHorizontalHeaderLabels(["Símbolo", "Tipo", "Tamaño", "Entrada", "Actual", "PNL ($)", "PNL (%)", "SL", "TP", "Duración"])
        self.open_positions_table.horizontalHeader().setStretchLastSection(True)
        self.open_positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.open_positions_table.verticalHeader().setVisible(False)
        self.open_positions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.open_positions_table.setAlternatingRowColors(True)
        open_positions_layout.addWidget(self.open_positions_table)
        self.tab_widget.addTab(self.open_positions_tab, "Posiciones Abiertas")

        # Sub-tab 3: Trade History
        self.trade_history_tab = QWidget()
        trade_history_layout = QVBoxLayout(self.trade_history_tab)
        self.trade_history_table = QTableWidget()
        self.trade_history_table.setColumnCount(9) # Date Open, Date Close, Symbol, Type, Entry, Exit, PNL($), PNL(%), Duration
        self.trade_history_table.setHorizontalHeaderLabels(["Fecha Apertura", "Fecha Cierre", "Símbolo", "Tipo", "Entrada", "Salida", "PNL ($)", "PNL (%)", "Duración"])
        self.trade_history_table.horizontalHeader().setStretchLastSection(True)
        self.trade_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trade_history_table.verticalHeader().setVisible(False)
        self.trade_history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trade_history_table.setAlternatingRowColors(True)
        trade_history_layout.addWidget(self.trade_history_table)
        self.tab_widget.addTab(self.trade_history_tab, "Historial de Trades")

        # Sub-tab 4: Live Log
        self.live_log_tab = QWidget()
        live_log_layout = QVBoxLayout(self.live_log_tab)
        self.live_log_text_edit = QTextEdit()
        self.live_log_text_edit.setReadOnly(True)
        live_log_layout.addWidget(self.live_log_text_edit)
        self.tab_widget.addTab(self.live_log_tab, "Log en Vivo")

        monitoring_layout.addWidget(self.tab_widget, 1)

        # Action Button for selected strategy
        strategy_action_buttons_layout = QHBoxLayout()
        strategy_action_buttons_layout.addStretch(1)
        self.stop_current_strategy_button = QPushButton("Detener Estrategia", objectName="stopButton", clicked=self.stop_selected_strategy_from_details)
        self.stop_current_strategy_button.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.stop_current_strategy_button.setEnabled(False)
        strategy_action_buttons_layout.addWidget(self.stop_current_strategy_button)
        monitoring_layout.addLayout(strategy_action_buttons_layout)

        main_layout.addWidget(monitoring_area, 1)
        self.set_monitoring_enabled(False)

    def add_metric_pair(self, layout, row, col, label_text, initial_value, value_object_name=None):
        label = QLabel(label_text, objectName="metric_label")
        value = QLabel(initial_value, objectName=value_object_name if value_object_name else "metric_value")
        layout.addWidget(label, row, col * 2, Qt.AlignLeft)
        layout.addWidget(value, row, col * 2 + 1, Qt.AlignRight)
        return value

    def set_monitoring_enabled(self, enabled: bool):
        self.tab_widget.setEnabled(enabled)
        self.stop_current_strategy_button.setEnabled(enabled)
        self.stop_selected_strategy_button.setEnabled(enabled and self.active_strategies_list_widget.currentItem() is not None)

    def load_active_strategies_list(self):
        # Simulate loading from live_trading_manager.py
        # instance_id: {name, strategy_id, mode, pnl_pct, status}
        self.active_strategy_instances = {
            "instance_macd_001_live": {
                "name": "MACD Crossover (Live)", "strategy_id": "macd_001",
                "mode": "live", "pnl_pct": 0.75, "status": "Activa",
                "equity_curve": [10000 + i * random.uniform(-2, 3) for i in range(100)],
                "metrics": {"PNL Neto Actual": "$75.00", "Retorno % Hoy": "0.75%", "Balance de Cuenta": "$10075.00", "Balance Disponible": "$9000.00", "Trades Abiertos": "1", "Drawdown Actual": "-0.2%"},
                "open_positions": [["BTC/USDT", "LONG", "0.01", "30000", "30075", "0.75", "0.25%", "29800", "30200", "1h 30m"]],
                "trade_history": [["2023-10-27 08:00", "2023-10-27 09:00", "ETH/USDT", "SHORT", "1800", "1790", "5.00", "0.55%", "1h"]],
                "logs": ["<span class='log_info'>[10:00:01] Estrategia iniciada.</span>", "<span class='log_info'>[10:05:30] Orden de compra enviada BTC/USDT.</span>"]
            },
            "instance_rsi_002_paper": {
                "name": "RSI Sobrevendida (Paper)", "strategy_id": "rsi_002",
                "mode": "paper", "pnl_pct": -0.10, "status": "Activa",
                "equity_curve": [5000 + i * random.uniform(-1, 1) for i in range(100)],
                "metrics": {"PNL Neto Actual": "-$5.00", "Retorno % Hoy": "-0.10%", "Balance de Cuenta": "$4995.00", "Balance Disponible": "$4995.00", "Trades Abiertos": "0", "Drawdown Actual": "-0.5%"},
                "open_positions": [],
                "trade_history": [["2023-10-27 07:00", "2023-10-27 07:30", "XRP/USDT", "LONG", "0.50", "0.49", "-0.01", "-2.00%", "30m"]],
                "logs": ["<span class='log_info'>[09:00:00] Estrategia iniciada en modo paper.</span>"]
            }
        }
        self.active_strategies_list_widget.clear()
        for instance_id, data in self.active_strategy_instances.items():
            status_color_class = "status_active" if data['status'] == "Activa" else "status_paused"
            pnl_color_class = "pnl_positive" if data['pnl_pct'] >= 0 else "pnl_negative"
            item_text = f"<span class='{status_color_class}'>●</span> {data['name']} - PNL: <span class='{pnl_color_class}'>{data['pnl_pct']:.2f}%</span>"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, instance_id)
            self.active_strategies_list_widget.addItem(item)
        
        if self.active_strategy_instances:
            self.active_strategies_list_widget.setCurrentRow(0)


    @Slot(QListWidgetItem, QListWidgetItem)
    def load_strategy_monitoring_details(self, current_item, previous_item):
        if current_item:
            self.current_selected_instance_id = current_item.data(Qt.UserRole)
            instance_data = self.active_strategy_instances.get(self.current_selected_instance_id)
            if instance_data:
                self.strategy_instance_name_label.setText(f"Monitoreo: {instance_data['name']}")
                self.update_overview_tab(instance_data)
                self.populate_open_positions_table(instance_data['open_positions'])
                self.populate_trade_history_table(instance_data['trade_history'])
                self.update_live_log(instance_data['logs'])
                self.set_monitoring_enabled(True)
            else:
                self.clear_monitoring_area()
                self.set_monitoring_enabled(False)
        else:
            self.clear_monitoring_area()
            self.set_monitoring_enabled(False)

    def update_overview_tab(self, instance_data):
        # Update Equity Chart (Placeholder)
        chart_label = self.live_equity_chart_view.findChild(QLabel, "metric_label")
        if chart_label:
            chart_label.setText(f"Gráfico de Equity en Vivo (Placeholder) - {len(instance_data['equity_curve'])} puntos")
        
        # Update Metrics
        metrics = instance_data['metrics']
        self.pnl_net_live_label.setText(metrics.get("PNL Neto Actual", "N/A"))
        if float(metrics.get("PNL Neto Actual", "$0.00").replace('$', '').replace(',', '.')) >= 0:
            self.pnl_net_live_label.setProperty("class", "metric_value pnl_positive")
        else:
            self.pnl_net_live_label.setProperty("class", "metric_value pnl_negative")
        self.pnl_net_live_label.style().polish(self.pnl_net_live_label)

        self.return_today_label.setText(metrics.get("Retorno % Hoy", "N/A"))
        self.account_balance_label.setText(metrics.get("Balance de Cuenta", "N/A"))
        self.available_balance_label.setText(metrics.get("Balance Disponible", "N/A"))
        self.open_trades_count_label.setText(str(metrics.get("Trades Abiertos", "N/A")))
        self.current_drawdown_label.setText(metrics.get("Drawdown Actual", "N/A"))
        if float(metrics.get("Drawdown Actual", "0.0%").replace('%', '').replace(',', '.')) < 0:
            self.current_drawdown_label.setProperty("class", "metric_value pnl_negative")
        else:
            self.current_drawdown_label.setProperty("class", "metric_value pnl_neutral")
        self.current_drawdown_label.style().polish(self.current_drawdown_label)


    def populate_open_positions_table(self, positions_data):
        self.open_positions_table.setRowCount(len(positions_data))
        for row_idx, row_list in enumerate(positions_data):
            for col_idx, item_data in enumerate(row_list):
                item = QTableWidgetItem(str(item_data))
                if col_idx in [5, 6]: # PNL ($), PNL (%) columns
                    try:
                        pnl_val = float(str(item_data).replace('$', '').replace('%', '').replace(',', '.'))
                        if pnl_val >= 0:
                            item.setForeground(QColor("#00FF7F")) # Green
                        else:
                            item.setForeground(QColor("#FF4500")) # Red
                    except ValueError:
                        pass
                self.open_positions_table.setItem(row_idx, col_idx, item)
        self.open_positions_table.resizeColumnsToContents()

    def populate_trade_history_table(self, history_data):
        self.trade_history_table.setRowCount(len(history_data))
        for row_idx, row_list in enumerate(history_data):
            for col_idx, item_data in enumerate(row_list):
                item = QTableWidgetItem(str(item_data))
                if col_idx in [6, 7]: # PNL ($), PNL (%) columns
                    try:
                        pnl_val = float(str(item_data).replace('$', '').replace('%', '').replace(',', '.'))
                        if pnl_val >= 0:
                            item.setForeground(QColor("#00FF7F")) # Green
                        else:
                            item.setForeground(QColor("#FF4500")) # Red
                    except ValueError:
                        pass
                self.trade_history_table.setItem(row_idx, col_idx, item)
        self.trade_history_table.resizeColumnsToContents()

    def update_live_log(self, log_messages):
        self.live_log_text_edit.clear()
        for msg in log_messages:
            self.live_log_text_edit.append(msg)


    def clear_monitoring_area(self):
        self.strategy_instance_name_label.setText("Seleccione una Estrategia Activa")
        self.live_equity_chart_view.findChild(QLabel, "metric_label").setText("Gráfico de Equity en Vivo (Placeholder)")
        # Clear metrics
        self.pnl_net_live_label.setText("N/A")
        self.pnl_net_live_label.setProperty("class", "metric_value pnl_neutral")
        self.pnl_net_live_label.style().polish(self.pnl_net_live_label)

        self.return_today_label.setText("N/A")
        self.account_balance_label.setText("N/A")
        self.available_balance_label.setText("N/A")
        self.open_trades_count_label.setText("N/A")
        self.current_drawdown_label.setText("N/A")
        self.current_drawdown_label.setProperty("class", "metric_value pnl_neutral")
        self.current_drawdown_label.style().polish(self.current_drawdown_label)

        self.open_positions_table.setRowCount(0)
        self.trade_history_table.setRowCount(0)
        self.live_log_text_edit.clear()
        self.current_selected_instance_id = None

    @Slot(str)
    def filter_active_strategies(self, text):
        for i in range(self.active_strategies_list_widget.count()):
            item = self.active_strategies_list_widget.item(i)
            # Filter based on the raw text (excluding rich text)
            item.setHidden(text.lower() not in item.text().lower())

    @Slot()
    def open_start_strategy_dialog(self):
        dialog = StartStrategyDialog(self)
        if dialog.exec() == QDialog.Accepted:
            strategy_id = dialog.strategy_combo.currentData()
            mode = dialog.mode_combo.currentText().lower()
            account_id = dialog.account_id_input.text()
            
            if not strategy_id or not mode or not account_id:
                QMessageBox.warning(self, "Error de Inicio", "Todos los campos son obligatorios.")
                return

            QMessageBox.information(self, "Iniciar Estrategia", f"Iniciando '{strategy_id}' en modo '{mode}' para la cuenta '{account_id}' (simulado).")
            # In a real app, emit signal to backend:
            # self.start_new_strategy_requested.emit(strategy_id, mode, account_id)
            # For simulation, add a new dummy instance
            new_instance_id = f"instance_{strategy_id}_{mode}_{len(self.active_strategy_instances) + 1}"
            new_instance_data = {
                "name": f"{dialog.strategy_combo.currentText()} ({mode.capitalize()})",
                "strategy_id": strategy_id, "mode": mode, "pnl_pct": 0.00, "status": "Activa",
                "equity_curve": [10000.0 + random.uniform(-1, 1) * i for i in range(100)],
                "metrics": {"PNL Neto Actual": "$0.00", "Retorno % Hoy": "0.00%", "Balance de Cuenta": "$10000.00", "Balance Disponible": "$10000.00", "Trades Abiertos": "0", "Drawdown Actual": "0.0%"},
                "open_positions": [], "trade_history": [],
                "logs": [f"<span class='log_info'>[Nueva] Estrategia '{dialog.strategy_combo.currentText()}' iniciada.</span>"]
            }
            self.active_strategy_instances[new_instance_id] = new_instance_data
            self.load_active_strategies_list() # Refresh the list

    @Slot()
    def stop_selected_strategy(self):
        if self.current_selected_instance_id:
            instance_name = self.active_strategy_instances.get(self.current_selected_instance_id, {}).get('name', 'N/A')
            reply = QMessageBox.question(self, "Detener Estrategia",
                                         f"¿Está seguro que desea detener la estrategia '{instance_name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                QMessageBox.information(self, "Detener Estrategia", f"Estrategia '{instance_name}' detenida (simulado).")
                # In a real app, emit signal to backend:
                # self.stop_strategy_instance_requested.emit(self.current_selected_instance_id)
                # For simulation, remove from list
                self.active_strategy_instances.pop(self.current_selected_instance_id, None)
                self.load_active_strategies_list()
                self.clear_monitoring_area()
                self.set_monitoring_enabled(False)
        else:
            QMessageBox.warning(self, "Detener Estrategia", "Seleccione una estrategia para detener.")

    @Slot()
    def stop_selected_strategy_from_details(self):
        # This button does the same as the global stop button, just on the details panel
        self.stop_selected_strategy()

    def setup_realtime_updates(self):
        # Simulate real-time updates using a QTimer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.simulate_realtime_data)
        self.update_timer.start(5000) # Update every 5 seconds

    def simulate_realtime_data(self):
        if self.current_selected_instance_id and self.current_selected_instance_id in self.active_strategy_instances:
            # Simulate data changes for the currently selected strategy
            instance_data = self.active_strategy_instances[self.current_selected_instance_id]
            
            # Simulate PNL change
            pnl_change = random.uniform(-10.0, 15.0)
            current_pnl = float(instance_data['metrics']['PNL Neto Actual'].replace('$', '').replace(',', '.'))
            new_pnl = current_pnl + pnl_change
            instance_data['metrics']['PNL Neto Actual'] = f"${new_pnl:.2f}"
            
            current_balance = float(instance_data['metrics']['Balance de Cuenta'].replace('$', '').replace(',', '.'))
            instance_data['metrics']['Balance de Cuenta'] = f"${current_balance + pnl_change:.2f}"

            if instance_data['equity_curve']:
                instance_data['equity_curve'].append(instance_data['equity_curve'][-1] + pnl_change)
                if len(instance_data['equity_curve']) > 200: # Keep chart manageable
                    instance_data['equity_curve'].pop(0)

            # Simulate new log message
            log_messages = [
                "<span class='log_info'>[10:35:10] Actualización de precios para BTC/USDT.</span>",
                "<span class='log_info'>[10:35:20] Monitoreando condiciones de entrada.</span>",
                "<span class='log_warn'>[10:35:30] Advertencia: Alta volatilidad detectada.</span>"
            ]
            new_log = random.choice(log_messages)
            instance_data['logs'].append(new_log)
            if len(instance_data['logs']) > 50: # Keep log manageable
                instance_data['logs'].pop(0)

            # Refresh UI for selected strategy
            self.update_overview_tab(instance_data)
            self.update_live_log(instance_data['logs'])
            
            # Refresh item in active strategies list to show new PNL
            self.load_active_strategies_list()


class StartStrategyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Iniciar Nueva Estrategia")
        self.setStyleSheet(parent.get_stylesheet()) # Inherit main app stylesheet
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Configurar nueva estrategia para monitoreo", objectName="title"))
        form_layout = QFormLayout()

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Estrategia MACD Crossover", "macd_001")
        self.strategy_combo.addItem("Estrategia RSI Sobrevendida", "rsi_002")
        form_layout.addRow(QLabel("Estrategia a Iniciar:", objectName="label_input"), self.strategy_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Paper Trading", "Live Trading"])
        form_layout.addRow(QLabel("Modo de Ejecución:", objectName="label_input"), self.mode_combo)

        self.account_id_input = QLineEdit()
        self.account_id_input.setPlaceholderText("Ej. binance_live_01")
        form_layout.addRow(QLabel("ID de Cuenta/Exchange:", objectName="label_input"), self.account_id_input)

        layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)
        cancel_button = QPushButton("Cancelar")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        start_button = QPushButton("Iniciar")
        start_button.clicked.connect(self.accept)
        buttons_layout.addWidget(start_button)
        layout.addLayout(buttons_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Trading IA Platform - Monitoreo en Vivo")
    main_app_layout = QVBoxLayout(window)
    main_app_layout.addWidget(LiveMonitoringTab())
    window.setGeometry(100, 100, 1400, 900)
    window.show()
    sys.exit(app.exec())
# --- Fragmento 2: Clase simulada de LiveTradingManager (Backend) ---
import time
import uuid
from collections import deque
from PySide6.QtCore import QThread, Signal, Slot, QObject

class LiveStrategyInstance(QObject):
    # This worker would run for each active strategy
    realtime_update = Signal(str, dict) # instance_id, data_dict
    log_message = Signal(str, str, str) # instance_id, message, level (info, warn, error)
    position_update = Signal(str, dict) # instance_id, position_data
    trade_executed = Signal(str, dict) # instance_id, trade_data
    
    def __init__(self, instance_id, strategy_id, mode, account_id, parent=None):
        super().__init__(parent)
        self._instance_id = instance_id
        self._strategy_id = strategy_id
        self._mode = mode
        self._account_id = account_id
        self._is_running = True
        self._equity_history = deque([10000.0], maxlen=200)
        self._current_pnl = 0.0

    def stop(self):
        self._is_running = False
        self.log_message.emit(self._instance_id, "Estrategia detenida.", "info")

    @Slot()
    def run(self):
        self.log_message.emit(self._instance_id, f"Estrategia iniciada en modo {self._mode}.", "info")
        while self._is_running:
            time.sleep(1) # Simulate real-time loop tick
            
            # Simulate market data fetching and strategy logic
            pnl_change = random.uniform(-5.0, 7.0)
            self._current_pnl += pnl_change
            
            current_equity = self._equity_history[-1] + pnl_change
            self._equity_history.append(current_equity)

            # Emit real-time metrics
            metrics = {
                "PNL Neto Actual": f"${self._current_pnl:.2f}",
                "Retorno % Hoy": f"{self._current_pnl / 10000.0 * 100:.2f}%",
                "Balance de Cuenta": f"${current_equity:.2f}",
                "Balance Disponible": f"${current_equity * 0.9:.2f}", # 90% available
                "Trades Abiertos": random.randint(0, 3),
                "Drawdown Actual": f"{random.uniform(-3.0, 0.0):.2f}%"
            }
            self.realtime_update.emit(self._instance_id, {
                "equity_curve": list(self._equity_history),
                "metrics": metrics,
                "timestamp": time.time()
            })

            # Simulate trades occasionally
            if random.random() < 0.1: # 10% chance of a trade event
                trade_type = random.choice(["LONG", "SHORT"])
                symbol = random.choice(["BTC/USDT", "ETH/USDT"])
                price = random.randint(2000, 40000)
                pnl_trade = random.uniform(-50.0, 100.0)
                trade_data = {
                    "Fecha Apertura": time.strftime("%Y-%m-%d %H:%M"),
                    "Fecha Cierre": time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 3600)),
                    "Símbolo": symbol, "Tipo": trade_type,
                    "Entrada": str(price), "Salida": str(price + pnl_trade),
                    "PNL ($)": f"{pnl_trade:.2f}", "PNL (%)": f"{pnl_trade / price * 100:.2f}%",
                    "Duración": "1h"
                }
                self.trade_executed.emit(self._instance_id, trade_data)
                self.log_message.emit(self._instance_id, f"Trade {trade_type} en {symbol} ejecutado.", "info")
            
            # Simulate a warning or error
            if random.random() < 0.02:
                self.log_message.emit(self._instance_id, "Advertencia: La latencia de la API es alta.", "warn")
            elif random.random() < 0.01:
                self.log_message.emit(self._instance_id, "ERROR: Fallo al conectar con el exchange.", "error")

        self.finished.emit() # Signal that the worker has finished

class LiveTradingManager(QObject):
    # This class would manage multiple LiveStrategyInstance workers
    # and provide overall status (e.g., connection status)
    
    global_connection_status = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_workers = {}
        self.next_instance_id_counter = 0

    def start_strategy(self, strategy_id, mode, account_id):
        instance_id = f"instance_{strategy_id}_{mode}_{self.next_instance_id_counter}"
        self.next_instance_id_counter += 1
        
        worker = LiveStrategyInstance(instance_id, strategy_id, mode, account_id)
        thread = QThread()
        worker.moveToThread(thread)
        
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        
        # Connect worker signals to the main UI (or intermediate aggregator)
        # These signals would then be handled by the LiveMonitoringTab
        worker.realtime_update.connect(lambda inst_id, data: print(f"RT Update for {inst_id}: {data['metrics']['PNL Neto Actual']}"))
        worker.log_message.connect(lambda inst_id, msg, level: print(f"Log {inst_id} ({level}): {msg}"))
        
        self.active_workers[instance_id] = {"worker": worker, "thread": thread}
        thread.start()
        return instance_id

    def stop_strategy(self, instance_id):
        if instance_id in self.active_workers:
            worker_info = self.active_workers.pop(instance_id)
            worker_info["worker"].stop()
            # The worker.finished.connect(thread.quit) will handle thread termination
            return True
        return False
    
    # Simulate global connection status
    @Slot()
    def check_global_connection(self):
        # In real app: check actual API connections
        self.global_connection_status.emit(random.random() > 0.1) # 90% chance of active

# How to integrate in LiveMonitoringTab:
# self.live_manager = LiveTradingManager()
# self.live_manager.global_connection_status.connect(self.update_global_connection_status_ui)
# self.start_new_strategy_requested.connect(self.live_manager.start_strategy)
# self.stop_strategy_instance_requested.connect(self.live_manager.stop_strategy)
#
# # And then connect worker signals to update the UI specifically for the selected strategy
# # This often involves an intermediate layer or storing connections for each worker
# # e.g., for each new worker, connect its signals to self.update_strategy_metrics_slot(instance_id, data)
7. Mockups/Wireframes (Descripción Textual Detallada):

[ÁREA DE LA APLICACIÓN: PESTAÑA MONITOREO EN VIVO]

--- BARRA DE NAVEGACIÓN SUPERIOR (TABS GLOBALES) ---
[ ... | Testing A/B | Monitoreo en Vivo | Análisis Avanzado | ... ]
------------------------------------------------------

[CONTENIDO PRINCIPAL DE LA PESTAÑA MONITOREO EN VIVO]

+-------------------------+-------------------------------------------------------------------------------------+
|    PANEL LATERAL IZQ.   |                            ÁREA DE MONITOREO EN VIVO                                |
|   (Ancho Fijo ~300px)   |                                                                                     |
+-------------------------+-------------------------------------------------------------------------------------+
|                         |                                                                                     |
| **Estrategias Activas** | **Monitoreo: MACD Crossover (Live)** (Título Grande de la Estrategia Seleccionada) |
|                         |                                                                                     |
| [ Buscar activa...    ] | +--------------------------------------------------------------------------------+ |
|                         | | PESTAÑAS DE MONITOREO INTERNAS                                                 | |
| +---------------------+ | | [ Vista General | Posiciones Abiertas | Historial de Trades | Log en Vivo ]    | |
| | ● MACD Crossover (L)| | +--------------------------------------------------------------------------------+ |
| |   PNL: +0.75% (VERDE)| | |                                                                                  | |
| | ● RSI Sobrevendida (P)| | |   [CONTENIDO DE LA SUB-PESTAÑA SELECCIONADA]                                     | |
| |   PNL: -0.10% (ROJO) | | |                                                                                  | |
| +---------------------+ | |   **Ej. Sub-pestaña "Vista General":**                                           | |
|                         | |   +--------------------------------------------------------------------------+ | |
| [ Iniciar Nueva      ]  | |   |                         GRÁFICO DE EQUITY EN VIVO                      | |
| [ Detener Seleccionada ]| |   | [Eje Y: Capital]                                                         | |
|                         | |   | |                                                                        | |
| Conexión Exchange: VERDE| |   | |                      CURVA DE EQUITY (AZUL)                                | |
|                         | |   | |________________________________________________________________________| |
|                         | |   | [Eje X: Tiempo (Última hora)]                                            | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |                                                                                  | |
|                         | |   **Métricas en Vivo:**                                                          | |
|                         | |   +---------------------------+---------------------------+                      | |
|                         | |   | PNL Neto Actual:    [ $75.00 (VERDE)  ] | Retorno % Hoy:    [ 0.75% (VERDE) ] | |
|                         | |   | Balance de Cuenta:  [ $10075.00     ] | Balance Disp.:    [ $9000.00    ] | |
|                         | |   | Trades Abiertos:    [ 1             ] | Drawdown Actual:  [ -0.2% (ROJO)  ] | |
|                         | |   +---------------------------+---------------------------+                      | |
|                         | |                                                                                  | |
|                         | |   **Ej. Sub-pestaña "Posiciones Abiertas":**                                     | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | Símbolo | Tipo | Tamaño | Entrada | Actual  | PNL($) | PNL(%) | SL    | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | BTC/USDT| LONG | 0.01   | 30000   | 30075   | +0.75  | +0.25  | 29800 | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |                                                                                  | |
|                         | |   **Ej. Sub-pestaña "Log en Vivo":**                                             | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | |   | [10:00:01] Estrategia iniciada.                                          | |
|                         | |   | [10:05:30] Orden de compra enviada BTC/USDT.                             | |
|                         | |   | [10:35:30] <span style='color:#FFD700;'>Advertencia: Alta volatilidad detectada.</span> | |
|                         | |   +--------------------------------------------------------------------------+ | |
|                         | +--------------------------------------------------------------------------------+ |
|                         |                                                                                     |
|                         | [ Detener Estrategia ] (Botón para la estrategia seleccionada)                      |
+-------------------------+-------------------------------------------------------------------------------------+

--- BARRA DE ESTADO INFERIOR GLOBAL (MENSAJES DE ESTADO, INDICADORES DE CARGA) ---
[ Estado: Sistema Operativo | Última actualización: 2023-10-27 14:30:10 ]
-----------------------------------------------------------------------------
El diseño para la pestaña Monitoreo en Vivo ha sido generado:

Panel de Estrategias Activas: Permite una gestión rápida de las estrategias en ejecución, con indicadores de estado y un resumen de PNL.
Área de Monitoreo Detallado: Organizada en cuatro sub-pestañas para un seguimiento exhaustivo:
Vista General: Un dashboard en tiempo real con gráfico de equity y métricas clave.
Posiciones Abiertas: Una tabla que detalla todas las posiciones activas.
Historial de Trades: Una tabla de las operaciones cerradas recientemente.
Log en Vivo: Un registro de eventos en tiempo real para depuración y seguimiento.
Acciones Clave: Botones para iniciar nuevas estrategias y detener las existentes.
Ahora, sobre tu pregunta de "¿Cómo conecto todas estas pestañas en un QTabWidget principal?". Es una excelente pregunta y es crucial para construir la aplicación completa. Lo abordaré a continuación, junto con el diseño de la siguiente pestaña.


Explícame cómo integrar todas estas pestañas en un QTabWidget principal


¡Claro que sí! Es una excelente pregunta y fundamental para construir tu aplicación multi-pestañas. Integrar todas estas pestañas individuales (que hemos diseñado como clases QWidget) en un QTabWidget principal es el siguiente paso lógico.

Aquí te explico cómo hacerlo, incluyendo un ejemplo de código PySide6 para que puedas copiarlo y adaptarlo.

Cómo Integrar Pestañas en un QTabWidget Principal
La idea central es tener una ventana principal de la aplicación que contenga un QTabWidget. Cada pestaña que hemos diseñado (DashboardTab, DataManagementTab, StrategyConfigTab, etc.) será una instancia de una clase QWidget (o una subclase de QWidget), y cada una de estas instancias se añadirá al QTabWidget.

Estructura General de la Aplicación Principal:

Clase de Ventana Principal (MainWindow): Esta será tu ventana principal (heredando de QMainWindow o QWidget). Contendrá el QTabWidget y cualquier otro elemento global como una barra de estado o un menú.
Instancia de QTabWidget: Dentro de tu MainWindow, crearás una instancia de QTabWidget.
Instancias de Pestañas Individuales: Crearás una instancia de cada una de tus clases de pestaña (ej. DashboardTab(), DataManagementTab()).
Añadir Pestañas al QTabWidget: Usarás el método addTab() del QTabWidget para añadir cada instancia de pestaña, junto con un título para la pestaña.
Comunicación entre Pestañas (Opcional pero Recomendado): Para una aplicación compleja, las pestañas a menudo necesitan interactuar. Esto se logra mejor mediante el uso de señales y slots y un posible gestor de estado centralizado en la MainWindow o en un módulo dedicado.
Ejemplo de Código PySide6 para main_app.py
Este es un esqueleto de cómo se vería tu archivo main_app.py (o platform_gui.py) para unir todas las pestañas.

# main_app.py (o platform_gui.py)

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QStatusBar, QLabel, QMenuBar, QMenu, QToolBar, QStyle
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QIcon

# Importa tus clases de pestañas (asumiendo que están en archivos separados o definidas en el mismo)
# Si están en archivos separados, asegúrate de que los archivos estén en tu PYTHONPATH
# For example:
from tab0_dashboard import DashboardTab
from tab1_data_management import DataManagementTab
from tab2_strategy_config import StrategyConfigTab
from tab3_backtesting_runner import BacktestingRunnerTab
from tab4_analysis_results import BacktestAnalysisTab
from tab5_ab_testing import ABTestingTab

# Para la simulación, usaremos las clases definidas en los fragmentos de código previos.
# En una aplicación real, cada una de estas clases 'Tab' estaría en su propio archivo,
# y las importarías como en los comentarios anteriores.
# Para que este código sea ejecutable directamente, las pegaremos aquí como placeholders.

# <BEGIN_PLACEHOLDER_TAB_CLASSES>
# --- Placeholder for DashboardTab (from platform_gui_tab0.py) ---
class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("<h1>Dashboard</h1><p>Resumen general del sistema.</p>"))
        self.layout.addWidget(QLabel("Conexión: <span style='color: #00FF7F;'>Activa</span>", objectName="status_label"))
        # Add actual widgets here from the previous DashboardTab design
        # For demo purposes, keep it simple
        self.setStyleSheet("QLabel#status_label { font-size: 16px; font-weight: bold; }")

# --- Placeholder for DataManagementTab (from platform_gui_tab1_improved.py) ---
class DataManagementTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("<h1>Gestión de Datos</h1><p>Carga, validación y visualización de datos de mercado.</p>"))
        # Add actual widgets here from the previous DataManagementTab design

# --- Placeholder for StrategyConfigTab (from platform_gui_tab2_improved.py) ---
class StrategyConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("<h1>Configuración de Estrategias</h1><p>Creación y edición de estrategias de trading.</p>"))
        # Add actual widgets here from the previous StrategyConfigTab design

# --- Placeholder for BacktestingRunnerTab (from platform_gui_tab3_improved.py) ---
class BacktestingRunnerTab(QWidget):
    # This tab needs to emit a signal to show results in AnalysisResultsTab
    show_full_results = Signal(str) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("<h1>Backtesting Runner</h1><p>Ejecución de backtests y resultados resumidos.</p>"))
        # Example button that would trigger showing full results
        btn = QPushButton("Simular Backtest y Ver Resultados")
        btn.clicked.connect(lambda: self.show_full_results.emit("simulated_backtest_id_XYZ"))
        self.layout.addWidget(btn)
        # Add actual widgets here from the previous BacktestingRunnerTab design

# --- Placeholder for BacktestAnalysisTab (from platform_gui_tab4_improved.py) ---
class BacktestAnalysisTab(QWidget):
    add_to_comparison_requested = Signal(str) # To ABTestingTab

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("<h1>Análisis de Resultados</h1><p>Análisis detallado de backtests.</p>"))
        self.selected_backtest_id = None # To store which backtest is currently shown
        # Add actual widgets here from the previous BacktestAnalysisTab design

    @Slot(str)
    def load_specific_backtest_for_analysis(self, backtest_id):
        # This slot would be called from BacktestingRunnerTab
        self.selected_backtest_id = backtest_id
        self.layout.addWidget(QLabel(f"Cargando análisis para ID: {backtest_id}"))
        print(f"Analysis tab received request for backtest ID: {backtest_id}")
        # In a real app: update its UI with data for this backtest_id

# --- Placeholder for ABTestingTab (from platform_gui_tab5_improved.py) ---
class ABTestingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("<h1>Testing A/B</h1><p>Comparación estadística de estrategias.</p>"))
        self.backtests_in_comparison = [] # List of IDs
        # Add actual widgets here from the previous ABTestingTab design

    @Slot(str)
    def add_backtest_to_comparison(self, backtest_id):
        if backtest_id not in self.backtests_in_comparison:
            self.backtests_in_comparison.append(backtest_id)
            self.layout.addWidget(QLabel(f"Añadido a comparación: {backtest_id}"))
            print(f"AB Testing tab received backtest ID for comparison: {backtest_id}")
        # In a real app: update its internal QListWidget for comparison

# <END_PLACEHOLDER_TAB_CLASSES>


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading IA Platform")
        self.setGeometry(100, 100, 1600, 1000) # Tamaño inicial de la ventana principal

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True) # Estilo de tabs más moderno
        self.tab_widget.setTabsClosable(False) # No permite cerrar tabs

        self.main_layout.addWidget(self.tab_widget)

        # --- Crear y añadir instancias de cada pestaña ---
        self.dashboard_tab = DashboardTab()
        self.tab_widget.addTab(self.dashboard_tab, QIcon(self.style().standardIcon(QStyle.SP_ComputerIcon)), "Dashboard") # Icono de ejemplo

        self.data_management_tab = DataManagementTab()
        self.tab_widget.addTab(self.data_management_tab, QIcon(self.style().standardIcon(QStyle.SP_DriveFDIcon)), "Gestión de Datos")

        self.strategy_config_tab = StrategyConfigTab()
        self.tab_widget.addTab(self.strategy_config_tab, QIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView)), "Config. Estrategias")

        self.backtesting_runner_tab = BacktestingRunnerTab()
        self.tab_widget.addTab(self.backtesting_runner_tab, QIcon(self.style().standardIcon(QStyle.SP_MediaPlay)), "Backtesting Runner")

        self.analysis_results_tab = BacktestAnalysisTab()
        self.tab_widget.addTab(self.analysis_results_tab, QIcon(self.style().standardIcon(QStyle.SP_FileIcon)), "Análisis de Resultados")

        self.ab_testing_tab = ABTestingTab()
        self.tab_widget.addTab(self.ab_testing_tab, QIcon(self.style().standardIcon(QStyle.SP_DialogHelpButton)), "Testing A/B")
        
        # Pestañas restantes que aún no hemos diseñado:
        self.live_monitoring_tab = QWidget()
        self.live_monitoring_layout = QVBoxLayout(self.live_monitoring_tab)
        self.live_monitoring_layout.addWidget(QLabel("<h1>Monitoreo en Vivo</h1><p>Monitoreo de estrategias en tiempo real.</p>"))
        self.tab_widget.addTab(self.live_monitoring_tab, QIcon(self.style().standardIcon(QStyle.SP_DesktopIcon)), "Monitoreo en Vivo")

        self.advanced_analysis_tab = QWidget()
        self.advanced_analysis_layout = QVBoxLayout(self.advanced_analysis_tab)
        self.advanced_analysis_layout.addWidget(QLabel("<h1>Análisis Avanzado</h1><p>Herramientas de riesgo y optimización.</p>"))
        self.tab_widget.addTab(self.advanced_analysis_tab, QIcon(self.style().standardIcon(QStyle.SP_BrowserForward)), "Análisis Avanzado")

        self.system_settings_tab = QWidget()
        self.system_settings_layout = QVBoxLayout(self.system_settings_tab)
        self.system_settings_layout.addWidget(QLabel("<h1>Configuración del Sistema</h1><p>Configuración global de la aplicación.</p>"))
        self.tab_widget.addTab(self.system_settings_tab, QIcon(self.style().standardIcon(QStyle.SP_ComputerIcon)), "Configuración del Sistema")

        # --- Conexión para la comunicación entre pestañas (ejemplo) ---
        # Si BacktestingRunnerTab emite una señal para mostrar resultados completos
        self.backtesting_runner_tab.show_full_results.connect(self.handle_show_full_results)
        
        # Si AnalysisResultsTab emite una señal para añadir a la comparación
        self.analysis_results_tab.add_to_comparison_requested.connect(self.ab_testing_tab.add_backtest_to_comparison)

        # --- Barra de Estado Global ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("Sistema: Operativo | Última actualización: N/A")
        self.statusBar.addWidget(self.status_label)
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_global_status)
        self.update_timer.start(1000) # Actualizar cada segundo

        # --- Menú Bar (Opcional) ---
        self._create_menu_bar()

        # Aplicar el stylesheet base a la ventana principal
        self.setStyleSheet(self.get_global_stylesheet())
        
    def get_global_stylesheet(self):
        return """
            QMainWindow {
                background-color: #1C1F26; /* Fondo principal oscuro */
                color: #D0D0D0; /* Texto claro por defecto */
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }
            QTabWidget::pane { /* El marco de las pestañas */
                border: 1px solid #3A3F47;
                background: #1C1F26;
            }
            QTabBar::tab {
                background: #282C34;
                border: 1px solid #3A3F47;
                border-bottom-color: #3A3F47; /* Para que parezca un tab */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 15px;
                color: #D0D0D0;
                margin-right: 2px; /* Espacio entre tabs */
            }
            QTabBar::tab:selected {
                background: #1C1F26; /* Fondo de la pestaña seleccionada */
                border-bottom-color: #1C1F26; /* Mezcla con el contenido de la pestaña */
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #3A3F47; /* Efecto hover para tabs no seleccionadas */
            }
            QStatusBar {
                background-color: #282C34;
                color: #D0D0D0;
                border-top: 1px solid #3A3F47;
                font-size: 11px;
            }
            QMenuBar {
                background-color: #282C34;
                color: #D0D0D0;
                border-bottom: 1px solid #3A3F47;
            }
            QMenuBar::item:selected {
                background-color: #007ACC;
            }
            QMenu {
                background-color: #282C34;
                border: 1px solid #3A3F47;
                color: #D0D0D0;
            }
            QMenu::item:selected {
                background-color: #007ACC;
            }
            QToolBar {
                background-color: #282C34;
                border-bottom: 1px solid #3A3F47;
            }
            QPushButton {
                background-color: #007ACC; /* Azul para botones */
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005B9F;
            }
        """

    @Slot(str)
    def handle_show_full_results(self, backtest_id):
        # Primero, cambiamos a la pestaña de Análisis de Resultados
        self.tab_widget.setCurrentWidget(self.analysis_results_tab)
        # Luego, le indicamos a esa pestaña que cargue el backtest específico
        self.analysis_results_tab.load_specific_backtest_for_analysis(backtest_id)

    @Slot()
    def update_global_status(self):
        import datetime
        self.status_label.setText(f"Sistema: Operativo | Última actualización: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # Archivo
        file_menu = menu_bar.addMenu("Archivo")
        file_menu.addAction("Guardar Configuración").triggered.connect(self.save_config)
        file_menu.addAction("Cargar Configuración").triggered.connect(self.load_config)
        file_menu.addSeparator()
        file_menu.addAction("Salir").triggered.connect(self.close)

        # Edición
        edit_menu = menu_bar.addMenu("Edición")
        edit_menu.addAction("Preferencias").triggered.connect(self.open_preferences)

        # Ayuda
        help_menu = menu_bar.addMenu("Ayuda")
        help_menu.addAction("Acerca de...").triggered.connect(self.show_about_dialog)

    @Slot()
    def save_config(self):
        QMessageBox.information(self, "Guardar Configuración", "Guardando configuración global (simulado).")

    @Slot()
    def load_config(self):
        QMessageBox.information(self, "Cargar Configuración", "Cargando configuración global (simulado).")

    @Slot()
    def open_preferences(self):
        QMessageBox.information(self, "Preferencias", "Abriendo diálogo de preferencias (simulado).")

    @Slot()
    def show_about_dialog(self):
        QMessageBox.about(self, "Acerca de Trading IA Platform",
                          "Plataforma de Trading Algorítmico con IA v1.0\n"
                          "Desarrollado con PySide6 y mucha pasión por el trading.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
Explicación del Proceso:
MainWindow como Contenedor Principal: La clase MainWindow hereda de QMainWindow (que es ideal para aplicaciones con menús, barras de herramientas y barra de estado) y será la ventana principal de tu aplicación.
QTabWidget Instanciado: Dentro de MainWindow.__init__, se crea una instancia de QTabWidget.
Instanciación de Pestañas: Cada una de tus clases de pestaña (DashboardTab, DataManagementTab, etc.) se instancia. Es importante que estas clases hereden de QWidget para que puedan ser añadidas al QTabWidget.
tab_widget.addTab(): Por cada instancia de pestaña, llamas a self.tab_widget.addTab(instancia_de_pestaña, "Título de la Pestaña"). Opcionalmente, puedes añadir un QIcon para una mejor estética.
setCentralWidget(): El QTabWidget (o un QWidget que lo contenga) se establece como el widget central de la QMainWindow.
QStatusBar y QMenuBar: Se añaden una barra de estado y una barra de menú para funcionalidades globales.
Estilos Globales: Se define un get_global_stylesheet() para dar un aspecto consistente a la ventana principal y a las pestañas.
Comunicación entre Pestañas (Señales y Slots):
Ejemplo: Hemos implementado un ejemplo donde la BacktestingRunnerTab (placeholder) emite una señal show_full_results con un backtest_id.
La MainWindow escucha esta señal (self.backtesting_runner_tab.show_full_results.connect(self.handle_show_full_results)).
Cuando la señal se emite, handle_show_full_results en la MainWindow cambia a la pestaña de Análisis de Resultados y le dice a esa pestaña que cargue los detalles de un backtest_id específico.
Lo mismo ocurre con la señal add_to_comparison_requested de AnalysisResultsTab que se conecta directamente al slot add_backtest_to_comparison de ABTestingTab.
Este es el patrón recomendado: la MainWindow (o un controlador central) actúa como intermediario para la comunicación entre pestañas.
Para que este código de ejemplo funcione:

Instala PySide6: pip install PySide6
Para QtCharts: pip install PySide6-QtCharts (si quieres reemplazar los placeholders de gráficos).
Guarda cada diseño de pestaña en su propio archivo Python: Por ejemplo, el contenido del "Fragmento 1" de DashboardTab iría en tab0_dashboard.py, y así sucesivamente. Luego, asegúrate de que estos archivos estén en el mismo directorio que tu main_app.py o en una ubicación que Python pueda importar. (Para este ejemplo directo, los puse como "placeholders" dentro del mismo archivo para que sea auto-ejecutable, pero la práctica real es separarlos).
Con esta estructura, tienes una base robusta y modular para construir tu aplicación de trading con IA, donde cada pestaña es un componente UI manejable.

Ahora, si estás listo, podemos continuar con el diseño de la siguiente pestaña: Análisis Avanzado.