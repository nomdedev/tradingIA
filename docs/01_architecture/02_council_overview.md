# Council de Decisión para Trading

Propósito
- Crear un "Council" (comité) lógico que centralice conocimiento y reglas sobre trading, arquitectura, riesgo, asignación de capital, trading algorítmico, gestión de datos e infraestructura.

Roles recomendados
- Experto en Trading: reglas de entrada/salida, selección de indicadores.
- Arquitecto de Sistema de Trading: diseño de componentes, latencia, resiliencia.
- Risk Manager: sizing, stops dinámicos, límites por estrategia.
- Allocation Manager: asignación de capital entre estrategias/portafolio.
- Ingeniero de Datos: calidad, limpieza, backfill, latencias de ingest.
- Infraestructura/DevOps: despliegue, orquestación, observabilidad.
- Execution/Market Microstructure: slippage model, order types.
- Compliance/Monitor: reglas operacionales, límites regulatorios.

Proceso de toma de decisiones
1. Propuesta: cualquier cambio o nueva estrategia se describe en un RFC (documento corto).
2. Evaluación automática: el Council ejecuta reglas automáticas (checks de calidad, tests, backtests, métricas de riesgo).
3. Votación ponderada: reglas y expertos (o módulos) emiten votos / scores con pesos.
4. Decisión: agregación de scores y reglas de tie-breaker; registra decisión y razones.
5. Documentación: todo cambio debe quedar en el sistema de documentación y en el histórico de backtests.

Ciclo de vida de una estrategia
- Diseñar: definir hipótesis, indicadores, parámetros.
- Implementar: crear la estrategia siguiendo el `BaseStrategy` o plantilla.
- Probar unitariamente: validaciones lógicas y de integridad de datos.
- Backtest: histórico con métricas (retorno, drawdown, sharpe, maxDD, hit-rate).
- Revisar por Council: evaluación automática + revisión humana si procede.
- Deploy: staging → live (paper trading) → producción.
- Monitoreo y retrain: métricas en producción y proceso de retraining.

Automatización de reglas
- Reglas declarativas: YAML/JSON para checks (ej. min_samples, no lookahead, slippage model aplicable)
- Reglas ejecutables: funciones que devuelven `{'signal':..., 'score':..., 'weight':...}`
- Historizar resultados por regla y versión.

Integración y pruebas iniciales
- Implementar `core/council.py` con API para registrar expertos y reglas y obtener decisiones.
- Plantilla de estrategia en `strategies/simple_strategy_template.py`.
- Demo en `scripts/council_demo.py` que muestra la evaluación y un backtest sencillo.

Próximos pasos sugeridos
- Integrar Council con el pipeline de backtesting (`backtesting/advanced_backtest.py`).
- Definir formato de reglas YAML y loader.
- Añadir métricas automáticas y guardado de resultados (`utils/settings_manager.py` ya guarda backtest_results).
- Crear tests automatizados que validen el proceso de decisión.

