# Análisis de Arquitectura y Plan de Evolución (Consejo de Arquitectura)

## 1. Estado Actual del Sistema

### Componentes Principales
- **Core Engine**: `AdvancedBacktester` (Python) con soporte para Walk-Forward y Monte Carlo.
- **Decision Engine**: `Council` (Python) basado en reglas híbridas (YAML + Funciones).
- **Data Layer**: `SQLDataHandler` (SQLite) con soporte multi-timeframe.
- **Risk Management**: `RiskManager` integrado en el bucle de ejecución.
- **Visualization**: Dashboard en Streamlit (`dashboard/app.py`).

### Fortalezas
- Arquitectura modular (Core, Data, Strategies, Dashboard).
- Sistema de reglas flexible (`Council`) que permite separar la lógica de decisión de la ejecución.
- Optimización robusta (Bayesiana + Walk-Forward).
- Persistencia de datos eficiente (SQLite).

### Debilidades / Áreas Faltantes
- **Testing**: Cobertura de tests unitarios y de integración es desconocida/baja. Faltan tests automáticos en CI.
- **Documentación**: Existe documentación en `docs/`, pero puede estar desactualizada respecto al código reciente.
- **Live Trading**: La infraestructura para ejecución en vivo (conexión a Exchange real) es incipiente (`alpaca` config existe pero no hay `LiveTrader` robusto).
- **Logging/Monitoring**: El dashboard es visual, pero falta un sistema de alertas en tiempo real (Telegram/Email) para producción.
- **Code Quality**: No hay linter/formatter configurado (flake8, black) en el flujo de trabajo.

## 2. Evaluación del Consejo (Nuevos Expertos)

Se han incorporado los siguientes expertos al Consejo para esta evaluación:

1.  **Architect_Prime (System Architect)**: "La estructura es sólida, pero la separación entre `backtester.py` y `live_trader.py` (futuro) debe ser clara. El `Council` debe ser el cerebro compartido."
2.  **Code_Guardian (Quality Assurance)**: "Faltan tests automatizados. Cada regla del Council debe tener su propio test case. Necesitamos un pipeline de CI."
3.  **Data_Oracle (Data Engineer)**: "SQLite es bueno para backtesting, pero para live trading de alta frecuencia necesitamos TimescaleDB o InfluxDB. La limpieza de datos debe ser automática."
4.  **Risk_Warden (Security & Risk)**: "Las reglas de riesgo están en YAML, lo cual es excelente. Necesitamos un 'Kill Switch' global accesible desde el Dashboard."

## 3. Plan de Acción

### Fase 6: Refactorización y Calidad (Inmediato)
- [ ] Implementar `pytest` para `Council` y `RiskManager`.
- [ ] Configurar `pre-commit` hooks (Black, Flake8).
- [ ] Centralizar la definición de tipos (Type Hints) en todo el proyecto.

### Fase 7: Infraestructura de Live Trading (Corto Plazo)
- [ ] Crear clase `LiveTrader` que herede de una interfaz común con `Backtester`.
- [ ] Implementar adaptadores para Exchanges (CCXT o API directa).
- [ ] Sistema de "Paper Trading" continuo en servidor (VPS).

### Fase 8: MLOps y Mejora Continua (Mediano Plazo)
- [ ] Pipeline de re-entrenamiento automático (Airflow o script cron).
- [ ] Versionado de modelos y parámetros (MLflow o JSON history).
- [ ] Dashboard de "Health Check" del sistema.

## 4. Modificaciones al Consejo

Se modificará `core/council.py` para permitir:
1.  Registro de expertos por **Dominio** (Trading, System, Risk).
2.  Capacidad de generar **Reportes** además de decisiones de trading.
