# Trading IA - GUI Application

## Aplicación de Escritorio para Pruebas de Estrategias de Trading

Esta aplicación proporciona una interfaz gráfica completa para probar, analizar y comparar estrategias de trading de manera profesional.

## Características Principales

### 🏗️ Arquitectura Modular
- **Dashboard Interactivo**: Interfaz principal con múltiples paneles
- **Controlador Central**: Gestión unificada de datos y operaciones
- **Componentes Modulares**: Paneles independientes para cada funcionalidad

### 📊 Visualización Avanzada
- **Gráficos Interactivos**: Precio, volumen, curvas de equity
- **Análisis de Rendimiento**: Distribuciones de retornos, drawdown
- **Comparación de Estrategias**: Visualización lado a lado

### ⚙️ Configuración de Estrategias
- **Selección Dinámica**: Lista de estrategias disponibles
- **Parámetros Interactivos**: Controles dinámicos para configuración
- **Gestión de Riesgos**: Filtros y límites configurables

### 🚀 Ejecución de Backtests
- **Procesamiento Paralelo**: Ejecución eficiente de múltiples configuraciones
- **Monitoreo en Tiempo Real**: Barra de progreso y estado
- **Resultados Detallados**: Métricas completas y análisis estadístico

### 📈 Análisis de Resultados
- **Métricas Completas**: Sharpe, Sortino, Calmar, y más
- **Análisis de Riesgos**: VaR, drawdown, volatilidad
- **Estadísticas de Trading**: Win rate, profit factor, análisis de trades

## Requisitos del Sistema

- **Python**: 3.11.9 o superior
- **Memoria RAM**: Mínimo 8GB recomendado
- **Espacio en Disco**: 2GB para datos e instalación
- **Sistema Operativo**: Windows 10/11, macOS, Linux

## Instalación

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd tradingIA
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
pip install PySide6 pandas numpy matplotlib
```

### 3. Ejecutar la Aplicación
```bash
python start_gui.py
```

## Estructura de la Aplicación

```
core/ui/
├── main_window.py          # Ventana principal
├── dashboard_controller.py # Controlador de negocio
├── charts_widget.py        # Gráficos interactivos
├── strategy_panel.py       # Configuración de estrategias
├── backtest_panel.py       # Ejecución de backtests
└── results_panel.py        # Análisis de resultados
```

## Uso de la Aplicación

### 1. Carga de Datos
- Usa "File > Load Market Data" para cargar datos históricos
- Soporta archivos CSV con formato OHLCV

### 2. Configuración de Estrategias
- Selecciona una estrategia del panel izquierdo
- Ajusta parámetros usando los controles dinámicos
- Configura filtros de riesgo y límites

### 3. Ejecución de Backtests
- Haz clic en "Run Backtests" para iniciar
- Monitorea el progreso en tiempo real
- Revisa resultados en las pestañas de análisis

### 4. Análisis de Resultados
- **Summary**: Métricas clave y overview
- **Performance**: Curvas de equity y retornos
- **Risk Analysis**: Análisis de drawdown y riesgo
- **Trades**: Historial detallado de operaciones
- **Statistics**: Análisis estadístico avanzado

## Funcionalidades Avanzadas

### Optimización de Parámetros
- Configuración múltiple de parámetros
- Ejecución paralela de combinaciones
- Análisis de sensibilidad

### Análisis Estadístico
- Pruebas de normalidad
- Análisis de correlación serial
- Bootstrap para intervalos de confianza

### Exportación de Resultados
- Reportes HTML completos
- Exportación de gráficos
- Datos de trades en CSV/Excel

## Desarrollo

### Arquitectura
La aplicación sigue el patrón MVC (Model-View-Controller):
- **Model**: Datos y lógica de negocio (core/)
- **View**: Componentes UI (core/ui/)
- **Controller**: Coordinación (DashboardController)

### Extensiones
Para agregar nuevas estrategias:
1. Implementa la estrategia en `strategies/`
2. Regístrala en `config/strategies_registry.json`
3. La UI la detectará automáticamente

### Personalización
Los paneles son modulares y pueden extenderse:
- Agrega nuevos tipos de gráficos en `ChartsWidget`
- Implementa análisis adicionales en `ResultsPanel`
- Extiende controles en `StrategyPanel`

## Solución de Problemas

### Errores Comunes
- **ImportError**: Verifica instalación de PySide6
- **MemoryError**: Reduce tamaño de datos o aumenta RAM
- **Qt Errors**: Actualiza drivers gráficos

### Logs
Los logs se guardan en `logs/trading_ia_gui.log`

### Diagnósticos
Usa "Tools > Run Diagnostics" para verificar el sistema

## Soporte

Para soporte técnico:
1. Revisa los logs en `logs/`
2. Ejecuta diagnósticos desde la aplicación
3. Consulta la documentación en `docs/`

## Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para detalles.