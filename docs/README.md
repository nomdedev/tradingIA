# 📚 TradingIA - Documentación

**Sistema de Trading Algorítmico para BTC con Multi-Timeframe Analysis**

> Última actualización: 14 de Enero 2026  
> Estado: ✅ Sistema en producción - Paper Trading Ready

---

## 🗂️ Estructura de Documentación

```
docs/
├── 00_getting_started/     # Guías de inicio rápido
├── 01_architecture/        # Arquitectura del sistema
├── 02_strategies/          # Estrategias de trading
├── 03_risk_management/     # Gestión de riesgo
├── 04_backtesting/         # Backtesting y A/B Testing
├── 05_alerts/              # Sistema de alertas
├── 06_user_guides/         # Guías de usuario completas
├── 07_development/         # Documentación de desarrollo
└── archive/                # Documentación histórica
```

---

## 🚀 Inicio Rápido

| Documento | Descripción |
|-----------|-------------|
| [00_quick_guide.md](00_getting_started/00_quick_guide.md) | Guía de uso básico |
| [01_gui_overview.md](00_getting_started/01_gui_overview.md) | Vista general de la interfaz |
| [02_realistic_execution_quickstart.md](00_getting_started/02_realistic_execution_quickstart.md) | Ejecución realista - inicio rápido |

---

## 🏗️ Arquitectura

| Documento | Descripción |
|-----------|-------------|
| [00_project_context.md](01_architecture/00_project_context.md) | Contexto y estado del proyecto |
| [01_data_flow.md](01_architecture/01_data_flow.md) | Flujo de datos multi-timeframe |
| [02_council_overview.md](01_architecture/02_council_overview.md) | Sistema Council de decisión |
| [03_council_interaction.md](01_architecture/03_council_interaction.md) | Protocolo de interacción |
| [04_logging_system.md](01_architecture/04_logging_system.md) | Sistema de logging |
| [05_ui_architecture.md](01_architecture/05_ui_architecture.md) | Arquitectura de la UI |

---

## 📈 Estrategias

| Documento | Descripción |
|-----------|-------------|
| [00_ifvg_strategy_definition.md](02_strategies/00_ifvg_strategy_definition.md) | Definición IFVG Strategy |
| [01_strategy_framework.md](02_strategies/01_strategy_framework.md) | Framework de estrategias |
| [02_strategy_manager.md](02_strategies/02_strategy_manager.md) | Gestor de estrategias |
| [03_indicators_logic.md](02_strategies/03_indicators_logic.md) | Lógica de indicadores |
| [04_vp_ifvg_ema.md](02_strategies/04_vp_ifvg_ema.md) | VP + IFVG + EMA Docs |
| [05_vp_analysis.md](02_strategies/05_vp_analysis.md) | Análisis Volume Profile |
| [06_conditional_patterns.md](02_strategies/06_conditional_patterns.md) | Patrones condicionales |
| [07_pattern_discovery.md](02_strategies/07_pattern_discovery.md) | Descubrimiento de patrones |

---

## ⚠️ Gestión de Riesgo

| Documento | Descripción |
|-----------|-------------|
| [00_risk_management_guide.md](03_risk_management/00_risk_management_guide.md) | Guía de gestión de riesgo |
| [01_kelly_sizing.md](03_risk_management/01_kelly_sizing.md) | Kelly Criterion Sizing |

---

## 🔬 Backtesting

| Documento | Descripción |
|-----------|-------------|
| [00_backtesting_features.md](04_backtesting/00_backtesting_features.md) | Funcionalidades de backtesting |
| [01_backtest_evaluation.md](04_backtesting/01_backtest_evaluation.md) | Evaluación de backtests |
| [02_ab_base_protocol.md](04_backtesting/02_ab_base_protocol.md) | Protocolo A/B base |
| [03_ab_base.md](04_backtesting/03_ab_base.md) | A/B Testing base |
| [04_ab_advanced.md](04_backtesting/04_ab_advanced.md) | A/B Testing avanzado |
| [05_ab_pipeline.md](04_backtesting/05_ab_pipeline.md) | Pipeline automatizado |
| [06_optimization_guide.md](04_backtesting/06_optimization_guide.md) | Optimización genética |

---

## 🔔 Alertas

| Documento | Descripción |
|-----------|-------------|
| [00_alerts_system.md](05_alerts/00_alerts_system.md) | Sistema de alertas |

---

## 📖 Guías de Usuario

| Documento | Descripción |
|-----------|-------------|
| [00_complete_user_guide.md](06_user_guides/00_complete_user_guide.md) | Guía completa de usuario |
| [01_technical_guide.md](06_user_guides/01_technical_guide.md) | Guía técnica completa |

---

## 🛠️ Desarrollo

| Documento | Descripción |
|-----------|-------------|
| [00_master_checklist.md](07_development/00_master_checklist.md) | Checklist maestro de mejoras |
| [01_features_checklist.md](07_development/01_features_checklist.md) | Checklist de funcionalidades |

---

## 📦 Archivo Histórico

Documentación de fases anteriores y planes completados se encuentra en [archive/](archive/).

---

## 🎯 Orden de Lectura Recomendado

### Para nuevos usuarios:
1. `00_getting_started/00_quick_guide.md`
2. `00_getting_started/01_gui_overview.md`
3. `06_user_guides/00_complete_user_guide.md`

### Para desarrolladores:
1. `01_architecture/00_project_context.md`
2. `01_architecture/01_data_flow.md`
3. `01_architecture/02_council_overview.md`
4. `02_strategies/01_strategy_framework.md`
5. `07_development/00_master_checklist.md`

### Para traders/analistas:
1. `02_strategies/00_ifvg_strategy_definition.md`
2. `03_risk_management/00_risk_management_guide.md`
3. `04_backtesting/00_backtesting_features.md`

---

## 📊 Estado del Sistema

| Componente | Estado | Tests |
|------------|--------|-------|
| Look-Ahead Bias Fix | ✅ | 5/5 |
| WFA Bayesian | ✅ | 7/7 |
| Kelly con Régimen | ✅ | 8/8 |
| Council Integration | ✅ | ✓ |
| Market Impact Crypto | ✅ | 9/9 |
| Risk Manager | ✅ | 13/13 |
| Data Validation | ✅ | 5/5 |
| TradingSignal | ✅ | 18/18 |

**Total: 65+ tests pasando**

---

*Generado automáticamente - TradingIA v2.1*
