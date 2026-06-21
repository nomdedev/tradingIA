# Plan Maestro de Profesionalización de Infraestructura de Trading
**Fecha:** 15 de Diciembre de 2025
**Versión:** 1.0
**Estado:** En Progreso

## 1. Resumen Ejecutivo
Este documento define la hoja de ruta para transformar el proyecto actual de trading algorítmico (`tradingIA`) de una colección de scripts monolíticos a una infraestructura de nivel institucional. La transformación se guía por la metodología del "Consejo Asesor" (Arquitectura, Quant, Riesgo, Desarrollo), priorizando la robustez, la gestión de riesgo y la validación científica sobre la generación rápida de señales.

## 2. Auditoría de Situación Actual (El Diagnóstico)

### 🏗️ Arquitectura e Infraestructura
*   **Estado:** Ejecución local en Windows, dependencia de archivos CSV planos, scripts síncronos.
*   **Riesgos:** Falta de redundancia, latencia impredecible, corrupción de datos, dificultad para escalar.
*   **Veredicto:** Inadecuado para capital real significativo.

### 📊 Ciencia de Datos (Quant)
*   **Estado:** Backtester avanzado existente pero con riesgo de overfitting (optimización bayesiana en datasets pequeños). Datos de 15min insuficientes para simulación realista de ejecución.
*   **Riesgos:** Lookahead bias, subestimación de slippage, métricas optimistas.
*   **Veredicto:** Necesita validación Out-of-Sample estricta y datos de mayor resolución (tick/1-min).

### 🛡️ Gestión de Riesgo (CRO)
*   **Estado:** Riesgo estático (`risk_per_trade` fijo), falta de controles a nivel de portafolio, sin "Kill Switch" global automatizado.
*   **Riesgos:** Ruina por racha de pérdidas correlacionadas, fallos de software no contenidos.
*   **Veredicto:** Crítico. Se requiere una capa de riesgo independiente que vete órdenes.

### 💻 Ingeniería de Software (Dev)
*   **Estado:** Código funcional pero acoplado (`backtester.py` monolítico). Falta de tests unitarios exhaustivos y logging estructurado.
*   **Riesgos:** Deuda técnica alta, difícil mantenimiento, bugs silenciosos.
*   **Veredicto:** Refactorización modular necesaria hacia arquitectura orientada a eventos.

---

## 3. Visión Estratégica: Arquitectura "The Council"

El sistema evolucionará hacia una **Arquitectura Orientada a Eventos** donde la toma de decisiones es colegiada:

1.  **Data Feed:** Ingesta datos (Tick/Candle) -> Publica Evento `MarketData`.
2.  **Strategy Engine:** Consume `MarketData` -> Consulta `Council` -> Emite `Signal`.
3.  **The Council (Core):** Evalúa la señal contra reglas de expertos (Trend, Volatility, Sentiment).
4.  **Risk Gatekeeper:** Intercepta `Signal` -> Verifica límites (DD, Exposición) -> Aprueba/Rechaza -> Emite `OrderRequest`.
5.  **Execution Handler:** Consume `OrderRequest` -> Ejecuta en Exchange -> Emite `Fill`.

---

## 4. Hoja de Ruta Detallada (Roadmap)

### Fase 1: Cimientos y Refactorización (Prioridad Alta)
*Objetivo: Desacoplar componentes y establecer bases sólidas.*

- [ ] **Refactorización Modular:**
    - Separar `backtester.py` en `Engine`, `Strategy`, `DataHandler`, `Execution`.
    - Estandarizar interfaces (clases base abstractas).
- [ ] **Sistema de Reglas (Council Core):**
    - Implementar `core/rules_loader.py` para cargar reglas desde YAML.
    - Definir estructura de reglas: `DataQuality`, `RiskLimits`, `ExecutionFeasibility`.
- [ ] **Tests Unitarios:**
    - Crear suite de tests para indicadores y lógica de riesgo.

### Fase 2: El "Council" de Riesgo y Gestión (Prioridad Alta)
*Objetivo: Protección de capital.*

- [ ] **Risk Manager Independiente:**
    - Implementar `core/risk/risk_manager.py`.
    - Validaciones: Max Drawdown diario, Max Exposición, Correlación.
- [ ] **Kill Switch:**
    - Mecanismo global para detener trading (archivo flag / Redis key).
- [ ] **Integración en Backtest:**
    - El backtester debe consultar al `RiskManager` antes de cada trade simulado.

### Fase 3: Validación Científica y Backtesting Avanzado (Prioridad Media)
*Objetivo: Realismo y confianza en las estrategias.*

- [ ] **Mejora de Datos:**
    - Migración de CSV a base de datos (TimescaleDB/SQLite para inicio).
    - Soporte para datos de 1-min y simulación de velas mayores.
- [ ] **Simulación Realista:**
    - Modelado de latencia (delay aleatorio).
    - Modelado de Slippage dinámico (basado en volatilidad).
    - Inclusión de costos reales (fees, funding rates).
- [ ] **Reportes Automáticos:**
    - Generación de PDF/HTML con métricas de "Council" (votos por regla, rechazos de riesgo).

### Fase 4: Infraestructura y Operación (Prioridad Baja/Futura)
*Objetivo: Entorno de producción robusto.*

- [ ] **Dockerización:** Contenedores para DB, App y Dashboard.
- [ ] **Logging Estructurado:** JSON logs para ingestión en ELK/Loki.
- [ ] **Dashboard Operativo:** Monitorización de salud del sistema (latencia, memoria, estado de conexión).

---

## 5. Plan de Ejecución Inmediata (Siguientes Pasos)

Comenzaremos ejecutando la **Fase 1 y 2** en paralelo, enfocándonos en la integración del `Council` en el código existente.

1.  **Definición de Reglas (YAML):** Crear estructura de archivos para reglas declarativas.
2.  **Loader de Reglas:** Implementar el cargador para que el sistema lea las reglas.
3.  **Integración en Backtester:** Modificar `AdvancedBacktester` para usar el `Council` en la toma de decisiones.
4.  **Estrategias Simples:** Implementar 3 estrategias base (MA, RSI, Breakout) usando el nuevo sistema para validar.

---
*Este documento servirá como la fuente de verdad para el desarrollo del proyecto.*
