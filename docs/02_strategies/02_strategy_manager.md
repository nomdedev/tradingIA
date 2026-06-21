# Strategy Manager - Sistema de Trading Automatizado

## 📋 Descripción

**Strategy Manager** es un ejecutable interactivo que permite gestionar, analizar y optimizar estrategias de trading de forma sencilla. El sistema integra todos los módulos avanzados de validación estadística, backtesting, análisis de robustez y señales alternativas.

## 🚀 Características Principales

### 1. **Modificación Fácil de Parámetros**
- **Interfaz interactiva** con menús categorizados
- Modificación en tiempo real sin editar código
- Validación automática de tipos de datos
- Guardar/Cargar configuraciones personalizadas

### 2. **Ejecución de Backtests**
- Configuración rápida de período y capital
- Integración con `AdvancedBacktester`
- Métricas completas: Sharpe, Win Rate, Drawdown, Calmar, Sortino
- Análisis de duración de trades y falsos positivos

### 3. **Análisis de Sensibilidad**
- **Un parámetro**: Analiza cómo varía el rendimiento al cambiar un solo parámetro
- **Multi-parámetro**: Grid search automático para encontrar configuración óptima
- Visualización de impacto en métricas clave

### 4. **Sistema de Persistencia**
- **Base de datos JSON** para resultados históricos
- Almacenamiento de configuraciones, métricas y metadatos
- Comparación histórica de estrategias
- Exportación de reportes completos

### 5. **Reportes Detallados**
- Generación automática de reportes en texto
- Resumen de configuración y resultados
- Análisis comparativo de estrategias
- Historial completo de backtests

## 📦 Instalación

```bash
# Clonar repositorio
git clone <repo>
cd tradingIA

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python strategy_manager.py --help
```

## 🎯 Uso Rápido

### Modo Interactivo (Recomendado)

```bash
python strategy_manager.py
```

Esto abre un menú interactivo con las siguientes opciones:

```
STRATEGY MANAGER - Trading IA
=====================================================

1. Ver configuración actual
2. Modificar parámetros
3. Ejecutar backtest
4. Análisis de sensibilidad
5. Ver resultados históricos
6. Comparar estrategias
7. Guardar/Cargar configuración
8. Generar reporte completo
9. Salir
```

### Ejecución Directa de Backtest

```bash
# Con configuración por defecto
python strategy_manager.py --backtest

# Con configuración personalizada
python strategy_manager.py --config configs/my_strategy.json --backtest
```

## 📊 Guía de Uso Detallada

### 1. Ver Configuración Actual

Muestra todos los parámetros organizados por categorías:
- **General**: Symbol, timeframe, fechas
- **Señales**: Confluence threshold, ATR multiplier, R:R
- **Gestión de Riesgo**: Max risk, stop loss, take profit
- **Backtest**: Capital, slippage, comisiones
- **Validación**: Walk-forward periods, Monte Carlo runs

### 2. Modificar Parámetros

**Ejemplo: Cambiar el Risk:Reward Ratio**

1. Seleccionar opción `2. Modificar parámetros`
2. Elegir categoría `1. Señales`
3. Seleccionar `4. risk_reward_ratio`
4. Ingresar nuevo valor (ej: `2.5`)

El sistema valida automáticamente el tipo de dato y actualiza la configuración.

### 3. Ejecutar Backtest

**Flujo:**
1. Confirmar configuración actual
2. Ejecutar backtest con barra de progreso
3. Ver resultados completos:
   ```
   Retorno Total: 45.00%
   Sharpe Ratio: 1.35
   Win Rate: 62.00%
   Total Trades: 150
   Max Drawdown: 18.00%
   Calmar Ratio: 2.50
   Sortino Ratio: 1.80
   Duración Promedio Trade: 4.5 horas
   Tasa Falsos Positivos: 38.00%
   ```
4. Resultado guardado automáticamente con ID único

### 4. Análisis de Sensibilidad

#### Análisis de Un Parámetro

Ejemplo: Analizar `confluence_threshold` en rango [3, 4, 5, 6]

```
Resultados:
   confluence_threshold  sharpe_ratio  total_trades  win_rate  max_drawdown
0                     3          1.12           180      0.58          0.22
1                     4          1.35           150      0.62          0.18
2                     5          1.28           120      0.65          0.20
3                     6          1.15            90      0.68          0.25

Valor óptimo: 4 (sharpe=1.35)
```

#### Análisis Multi-Parámetro

Grid search automático probando todas las combinaciones:

```python
{
    'confluence_threshold': [3, 4, 5],
    'risk_reward_ratio': [2.0, 2.2, 2.5]
}

Total combinaciones: 9
Top 10 configuraciones mostradas por Sharpe Ratio
```

### 5. Ver Resultados Históricos

Lista todos los backtests ejecutados:
```
Total de backtests: 25

Últimos 10 resultados:
ID: 20251112_143022
  Fecha: 2025-11-12T14:30:22
  Sharpe: 1.35
  Win Rate: 62.00%
...
```

### 6. Comparar Estrategias

Selecciona múltiples resultados para comparación lado a lado:

```
COMPARACIÓN DE ESTRATEGIAS
================================================================
ID            Strategy        sharpe_ratio  win_rate  max_drawdown
20251112_143022  IFVG_VP_EMAs       1.35      0.62        0.18
20251111_102015  Alternative_RSI    1.42      0.68        0.15
20251110_154500  Hybrid_VWAP        1.51      0.70        0.14
```

### 7. Guardar/Cargar Configuración

**Guardar:**
```
Nombre del archivo: aggressive_strategy
✓ Configuración guardada en configs/aggressive_strategy.json
```

**Cargar:**
```
Configuraciones disponibles:
1. conservative_strategy
2. aggressive_strategy
3. balanced_strategy

Seleccione archivo: 2
✓ Configuración cargada
```

### 8. Generar Reporte Completo

Crea un archivo `.txt` con:
- Configuración completa
- Resultados del último backtest
- Métricas detalladas
- Timestamp y metadatos

Guardado en: `reports/strategy_report_20251112_150000.txt`

## 🔧 Parámetros Configurables

### Parámetros de Señales

| Parámetro | Descripción | Default | Rango Recomendado |
|-----------|-------------|---------|-------------------|
| `confluence_threshold` | Mínimo score para entry | 4 | 3-6 |
| `htf_ema_period` | Período EMA HTF bias | 210 | 100-300 |
| `atr_multiplier` | Multiplicador ATR para SL | 1.5 | 1.0-3.0 |
| `risk_reward_ratio` | Ratio R:R objetivo | 2.2 | 1.5-3.0 |
| `volume_threshold` | Threshold volumen relativo | 1.5 | 1.0-2.5 |

### Parámetros de Gestión de Riesgo

| Parámetro | Descripción | Default | Rango Recomendado |
|-----------|-------------|---------|-------------------|
| `max_risk_per_trade` | Riesgo máximo por trade | 0.02 (2%) | 0.01-0.05 |
| `max_open_trades` | Trades simultáneos máx | 3 | 1-5 |
| `stop_loss_atr` | SL en ATRs | 1.5 | 1.0-2.5 |
| `take_profit_rr` | TP en R:R | 2.2 | 1.5-3.0 |

### Parámetros de Backtest

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `initial_capital` | Capital inicial | 10000 |
| `slippage` | Slippage estimado | 0.001 (0.1%) |
| `commission` | Comisión por trade | 0.0005 (0.05%) |

## 📈 Métricas Analizadas

### Métricas de Rendimiento

- **Total Return**: Retorno total del período
- **Sharpe Ratio**: Return/riesgo ajustado (>1.0 bueno)
- **Sortino Ratio**: Sharpe considerando solo downside (>1.5 bueno)
- **Calmar Ratio**: Return/Max Drawdown (>2.0 bueno)

### Métricas de Trades

- **Win Rate**: % de trades ganadores (>55% objetivo)
- **Total Trades**: Cantidad total de operaciones
- **Avg Trade Duration**: Duración promedio en horas
- **False Positive Rate**: % de señales falsas

### Métricas de Riesgo

- **Max Drawdown**: Pérdida máxima desde pico (<20% objetivo)
- **VaR 95%**: Value at Risk al 95% confianza
- **Ulcer Index**: "Dolor" sostenido de drawdowns

## 🎓 Flujo de Trabajo Recomendado

### 1. **Configuración Inicial**
```bash
python strategy_manager.py
# Opción 1: Ver configuración
# Opción 2: Ajustar parámetros según preferencias
# Opción 7: Guardar como "base_strategy"
```

### 2. **Backtest Baseline**
```bash
# Opción 3: Ejecutar backtest
# Anotar ID del resultado
```

### 3. **Optimización de Parámetros**
```bash
# Opción 4.1: Análisis sensibilidad de confluence_threshold
# Opción 4.1: Análisis sensibilidad de risk_reward_ratio
# Identificar valores óptimos
```

### 4. **Configuración Optimizada**
```bash
# Opción 2: Actualizar con valores óptimos
# Opción 7: Guardar como "optimized_strategy"
# Opción 3: Ejecutar nuevo backtest
```

### 5. **Validación y Comparación**
```bash
# Opción 6: Comparar "base_strategy" vs "optimized_strategy"
# Verificar mejora en métricas clave
```

### 6. **Análisis Completo**
```bash
# Opción 8: Generar reporte completo
# Revisar reporte en reports/
```

### 7. **Iteración**
```bash
# Probar variaciones adicionales
# Análisis multi-parámetro (Opción 4.2)
# Forward testing con mejor configuración
```

## 🔬 Ejemplo de Sesión Completa

```bash
$ python strategy_manager.py

STRATEGY MANAGER - Trading IA
============================================================

1. Ver configuración actual
...
9. Salir

Seleccione una opción (1-9): 1

CONFIGURACIÓN ACTUAL DE ESTRATEGIA
============================================================

General:
  strategy_name                 : IFVG_VP_EMAs
  symbol                        : BTCUSD
  timeframe                     : 5min
  ...

Seleccione una opción (1-9): 2

MODIFICAR PARÁMETROS
------------------------------------------------------------
Categorías:
1. Señales
2. Gestión de Riesgo
3. Backtest
4. Fechas

Seleccione categoría (1-4): 1

Parámetros en 'Señales':
1. confluence_threshold          = 4
2. htf_ema_period                = 210
3. atr_multiplier                = 1.5
4. risk_reward_ratio             = 2.2

Seleccione parámetro (1-4): 4

Valor actual: 2.2
Nuevo valor: 2.5

✓ Parámetro 'risk_reward_ratio' actualizado a 2.5

Seleccione una opción (1-9): 3

EJECUTAR BACKTEST
------------------------------------------------------------
Configurando backtest...
Símbolo: BTCUSD
Período: 2024-01-01 - 2025-11-12
Capital inicial: $10000

¿Confirmar ejecución? (s/n): s

Ejecutando backtest...
Progreso: 20%
Progreso: 40%
Progreso: 60%
Progreso: 80%
Progreso: 100%

RESULTADOS DEL BACKTEST
============================================================

ID del Resultado: 20251112_153045

Retorno Total: 48.50%
Sharpe Ratio: 1.42
Win Rate: 64.00%
Total Trades: 155
Max Drawdown: 16.50%
Calmar Ratio: 2.94
Sortino Ratio: 1.95
Duración Promedio Trade: 4.2 horas
Tasa Falsos Positivos: 36.00%

============================================================

Seleccione una opción (1-9): 8

GENERAR REPORTE COMPLETO
------------------------------------------------------------
Generando reporte con:
- Configuración actual
- Resultados de backtest
- Análisis de sensibilidad
- Métricas de validación
- Análisis de robustez

¿Continuar? (s/n): s

✓ Reporte generado: reports/strategy_report_20251112_153100.txt

Seleccione una opción (1-9): 9

¡Hasta luego!
```

## 📁 Estructura de Archivos

```
tradingIA/
├── strategy_manager.py         # Ejecutable principal
├── configs/                     # Configuraciones guardadas
│   ├── base_strategy.json
│   ├── aggressive_strategy.json
│   └── conservative_strategy.json
├── results/                     # Base de datos de resultados
│   └── backtest_results.json
├── reports/                     # Reportes generados
│   ├── strategy_report_20251112_150000.txt
│   └── strategy_report_20251111_143000.txt
└── src/                         # Módulos del sistema
    ├── metrics_validation.py
    ├── ab_testing_protocol.py
    ├── robustness_snooping.py
    ├── automated_pipeline.py
    └── alternatives_integration.py
```

## 🐛 Troubleshooting

### Error: "No module named 'src.backtester'"

**Solución:**
```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd tradingIA
python strategy_manager.py
```

### Error: "vectorbt not available"

**No es crítico**. El sistema usa implementaciones alternativas.

**Opcional:**
```bash
pip install vectorbt
```

### Resultados no se guardan

**Verifica:**
```bash
# Permisos de escritura en directorio results/
ls -la results/

# Crear manualmente si no existe
mkdir -p results
```

## 🔄 Integración con Otros Módulos

### Con Automated Pipeline

```python
from strategy_manager import StrategyConfig
from src.automated_pipeline import AutomatedPipeline

# Cargar configuración
config = StrategyConfig.load('configs/my_strategy.json')

# Ejecutar pipeline completo
pipeline = AutomatedPipeline()
pipeline.run_full_pipeline(
    symbol=config.get('symbol'),
    start=config.get('start_date'),
    end=config.get('end_date')
)
```

### Con Metrics Validation

```python
from strategy_manager import ResultsDatabase
from src.metrics_validation import MetricsValidator

# Cargar resultados
db = ResultsDatabase()
latest = db.get_all_results()[-1]

# Validar métricas
validator = MetricsValidator()
validation = validator.validate_metrics(latest['metrics'])
```

## 📚 Recursos Adicionales

- **Documentación completa**: `docs/`
- **Ejemplos**: `examples/`
- **Tests**: `tests/test_strategy_manager.py`

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear branch de feature
3. Commit de cambios
4. Push al branch
5. Crear Pull Request

## 📄 Licencia

MIT License - Ver `LICENSE` para detalles

## 📧 Soporte

Para reportar bugs o solicitar features, crear un issue en GitHub.

---

**Desarrollado con ❤️ para traders cuantitativos**
