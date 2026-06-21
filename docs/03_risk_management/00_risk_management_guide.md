# 🛡️ Guía de Gestión de Riesgo (Risk Management Framework)

Este documento detalla el sistema integral de gestión de riesgo implementado en la plataforma, cubriendo desde el dimensionamiento de posiciones hasta protecciones de ejecución (Kill Switch).

## 1. Arquitectura de Riesgo

El sistema de riesgo se divide en tres componentes principales ubicados en `core/risk/`:

1.  **KellyPositionSizer** (`kelly_sizer.py`): Optimización del tamaño de posición.
2.  **RiskMetricsCalculator** (`risk_metrics.py`): Análisis estadístico (VaR, CVaR, Monte Carlo).
3.  **RiskManager** (`risk_manager.py`): Control de ejecución y Kill Switch.

---

## 2. Dimensionamiento de Posiciones (Kelly Criterion)

El **Criterio de Kelly** se utiliza para determinar el tamaño óptimo de la posición basándose en el rendimiento histórico de la estrategia.

### Configuración
En el Backtester (Tab 3), se puede habilitar "Kelly Position Sizing":
*   **Kelly Fraction**: Fracción del Kelly completo a utilizar (ej. 0.5 = Half Kelly). Recomendado para reducir volatilidad.
*   **Max Position %**: Límite máximo de capital por operación (ej. 10%).

### Lógica
$$ f^* = \frac{p(b+1) - 1}{b} $$
Donde:
*   $f^*$: Fracción del capital a apostar.
*   $p$: Probabilidad de ganancia (Win Rate).
*   $b$: Ratio Ganancia/Pérdida (Win/Loss Ratio).

El sistema ajusta dinámicamente el tamaño de la posición en cada operación del backtest.

---

## 3. Métricas Avanzadas (Risk Dashboard)

El **Tab 11 (Risk Analysis)** proporciona un análisis profundo del riesgo de la estrategia seleccionada.

### Métricas Clave
*   **VaR (Value at Risk)**: Pérdida máxima esperada con un nivel de confianza (95% o 99%).
    *   *Historical*: Basado en retornos pasados.
    *   *Parametric*: Basado en distribución normal.
*   **CVaR (Conditional VaR)**: Pérdida esperada en el peor % de los casos (Expected Shortfall).
*   **Monte Carlo Simulation**: Proyección de 100+ futuros posibles para estimar rangos de Drawdown y Retorno.

---

## 4. Kill Switch y Protecciones (Risk Manager)

El **RiskManager** actúa como un guardián de seguridad durante la ejecución (Backtest y Live).

### Funcionalidad
*   **Max Daily Drawdown**: Si el capital cae más de un X% (ej. 5%) en un solo día, el sistema detiene el trading inmediatamente.
*   **Max Total Drawdown**: Límite global de pérdida.
*   **Kill Switch Manual**: Archivo externo (`kill_switch.json`) para detener el sistema remotamente.

### Integración en Backtest
Durante el backtest, el sistema simula el Kill Switch. Si se viola el límite diario, el backtest se **trunca** en ese punto, reflejando que el trading se habría detenido en la vida real.

---

## 5. Flujo de Trabajo Recomendado

1.  **Backtest Inicial**: Ejecutar estrategia con tamaño fijo.
2.  **Análisis de Riesgo**: Usar Tab 11 para evaluar VaR y estabilidad.
3.  **Optimización**: Activar Kelly Sizing (Tab 3) con fracción conservadora (0.3 - 0.5).
4.  **Validación**: Verificar que el Kill Switch no se active frecuentemente.

---
**Ubicación del Código**: `core/risk/`
