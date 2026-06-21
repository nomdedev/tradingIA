# 📚 ÍNDICE MAESTRO - Plan de Mejoras TradingIA

**Creado:** 12 de Enero 2026  
**Versión:** 1.0 - Plan Completo  
**Estado:** ✅ Listo para Ejecución

---

## 🎯 INICIO RÁPIDO (EMPIEZA AQUÍ)

### Primero: Lee el resumen (5 min)
👉 **[RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md)** 
- Qué se va a arreglar
- Por qué importa
- Impacto esperado
- Cronograma general

### Segundo: Plan detallado (30 min)
👉 **[checklist.md](checklist.md)**
- 8 áreas críticas detalladas
- Soluciones pseudocódigo completas
- Roadmap de 4 semanas
- Task checklist por área

### Tercero: Primeras tareas (2 horas)
👉 **[QUICK_START.md](QUICK_START.md)**
- Qué hacer HOY (12 Enero)
- Primeras 3 tareas prioritarias
- Archivos a revisar
- Entregables esperados

---

## 📋 DOCUMENTOS POR PROPÓSITO

### 🚀 EJECUCIÓN E IMPLEMENTACIÓN

| Documento | Propósito | Audiencia | Tiempo |
|-----------|-----------|-----------|--------|
| [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) | Overview de todo el plan | Managers + Developers | 5 min |
| [QUICK_START.md](QUICK_START.md) | Primeras 3 tareas para hoy | Developer (tú) | 2 horas |
| [checklist.md](checklist.md) | Plan maestro detallado | Developers | 30 min |
| [PROGRESS_TRACKING.md](PROGRESS_TRACKING.md) | Seguimiento diario | All | Daily |

### 🧪 TESTING Y VALIDACIÓN

| Documento | Propósito | Audiencia | Tests |
|-----------|-----------|-----------|-------|
| [ESTRATEGIA_TESTING.md](ESTRATEGIA_TESTING.md) | Cómo validar cada fix | QA + Developers | 29+ |
| tests/test_*.py | Tests específicos (por crear) | Developers | 29+ |

### 📚 DOCUMENTACIÓN Y REFERENCIA

| Documento | Propósito | Tipo |
|-----------|-----------|------|
| [ESTRUCTURA_DOCUMENTACION.md](ESTRUCTURA_DOCUMENTACION.md) | Cómo se documentan los fixes | Guide |
| AREA*_ANALYSIS.md | Análisis profundo de cada problema (por crear) | Analysis |
| AREA*_IMPLEMENTATION.md | Documentación de fixes (por crear) | Implementation |

---

## 🔴 8 ÁREAS CRÍTICAS

### SEMANA 1: Fixes Inmediatos

#### 🚨 ÁREA 1: Look-Ahead Bias
**Problema:** Volume Profile calcula valores con datos futuros  
**Impacto:** Sharpe +15-40% inflado  
**Documentos:**
- Plan: [checklist.md - ÁREA 1](checklist.md#área-1-look-ahead-bias)
- Analysis: AREA1_ANALYSIS.md (por crear)
- Implementation: AREA1_FIX.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 1](ESTRATEGIA_TESTING.md#-área-1-look-ahead-bias)
- **Inicio:** Hoy (12 Enero) en [QUICK_START.md - TAREA 1](QUICK_START.md#1--look-ahead-bias---análisis-del-código)

#### 🚨 ÁREA 4: Council Integration
**Problema:** Council implementado pero nunca se consulta en trades  
**Impacto:** Rules definidas pero nunca ejecutadas  
**Documentos:**
- Plan: [checklist.md - ÁREA 4](checklist.md#área-4-council-never-consulted)
- Analysis: AREA4_INTEGRATION_POINTS.md (por crear)
- Implementation: AREA4_IMPLEMENTATION.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 4](ESTRATEGIA_TESTING.md#-área-4-council-integration)
- **Inicio:** Hoy (12 Enero) en [QUICK_START.md - TAREA 2](QUICK_START.md#2--council-integration---dónde-se-llama)

#### 🚨 ÁREA 7: Data Validation Pipeline
**Problema:** DataValidator existe pero nunca se llama automáticamente  
**Impacto:** Datos corruptos pueden pasar sin validar  
**Documentos:**
- Plan: [checklist.md - ÁREA 7](checklist.md#área-7-no-mandatory-data-validation)
- Analysis: AREA7_VALIDATION_PIPELINE.md (por crear)
- Implementation: AREA7_IMPLEMENTATION.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 7](ESTRATEGIA_TESTING.md#-área-7-data-validation-pipeline)
- **Inicio:** Hoy (12 Enero) en [QUICK_START.md - TAREA 3](QUICK_START.md#3--data-validation---pipeline-obligatorio)

---

### SEMANA 2: Validación y Kelly

#### 🚨 ÁREA 2: Walk-Forward Analysis Real
**Problema:** WFA no optimiza parámetros, valida con mismo set siempre  
**Impacto:** Parámetros nunca validados en datos OOS  
**Documentos:**
- Plan: [checklist.md - ÁREA 2](checklist.md#área-2-invalid-walk-forward-analysis)
- Analysis: AREA2_WFA_ANALYSIS.md (por crear)
- Implementation: AREA2_WFA_IMPLEMENTATION.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 2](ESTRATEGIA_TESTING.md#-área-2-walk-forward-analysis)

#### 🚨 ÁREA 3: Kelly Criterion Dinámico
**Problema:** Kelly fijo (50 trades) sin régimen ni correlación serial  
**Impacto:** Over-leverage en bear markets  
**Documentos:**
- Plan: [checklist.md - ÁREA 3](checklist.md#área-3-static-kelly-no-regime-adjustment)
- Analysis: AREA3_KELLY_ANALYSIS.md (por crear)
- Implementation: AREA3_KELLY_IMPLEMENTATION.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 3](ESTRATEGIA_TESTING.md#-área-3-kelly-criterion-dinámico)

---

### SEMANA 3: Market Impact y Risk

#### 🚨 ÁREA 5: Market Impact Crypto
**Problema:** Usa modelo equity Almgren-Chriss, no apto para crypto 24/7  
**Impacto:** Execution costs underestimated  
**Documentos:**
- Plan: [checklist.md - ÁREA 5](checklist.md#área-5-inappropriate-market-impact-model)
- Analysis: AREA5_MARKET_IMPACT_ANALYSIS.md (por crear)
- Implementation: AREA5_MARKET_IMPACT_IMPLEMENTATION.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 5](ESTRATEGIA_TESTING.md#-área-5-market-impact-crypto)

#### 🚨 ÁREA 6: Risk Manager Incompleto
**Problema:** Solo verifica daily DD, sin total DD, correlación, VaR  
**Impacto:** Risk underestimated, drawdowns imprevistos  
**Documentos:**
- Plan: [checklist.md - ÁREA 6](checklist.md#área-6-incomplete-risk-management)
- Analysis: AREA6_RISK_ANALYSIS.md (por crear)
- Implementation: AREA6_RISK_IMPLEMENTATION.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 6](ESTRATEGIA_TESTING.md#-área-6-risk-manager)

---

### SEMANA 4: Limpieza y Estandarización

#### 🚨 ÁREA 8: Signal Format Inconsistency
**Problema:** 3 formatos diferentes para signals (FVGData, Series, BaseStrategy)  
**Impacto:** Código difícil de mantener y extender  
**Documentos:**
- Plan: [checklist.md - ÁREA 8](checklist.md#área-8-inconsistent-signal-formats)
- Analysis: AREA8_SIGNAL_ANALYSIS.md (por crear)
- Implementation: AREA8_SIGNAL_IMPLEMENTATION.md (por crear)
- Tests: [ESTRATEGIA_TESTING.md - ÁREA 8](ESTRATEGIA_TESTING.md#-área-8-trading-signal-standarizado)

---

## 📊 GUÍA DE DOCUMENTOS

### Por Rol

**👨‍💼 Manager / Project Owner**
1. [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) (5 min)
2. [PROGRESS_TRACKING.md](PROGRESS_TRACKING.md) (diario, 2 min)
3. [checklist.md](checklist.md) (opcional, 30 min si quieres detalles)

**👨‍💻 Developer (Implementación)**
1. [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) (5 min)
2. [QUICK_START.md](QUICK_START.md) (2 horas hoy)
3. [checklist.md](checklist.md) (durante desarrollo, 30 min)
4. [ESTRATEGIA_TESTING.md](ESTRATEGIA_TESTING.md) (para tests, consultar por área)
5. [ESTRUCTURA_DOCUMENTACION.md](ESTRUCTURA_DOCUMENTACION.md) (cómo documenta tu fix)

**🧪 QA / Tester**
1. [ESTRATEGIA_TESTING.md](ESTRATEGIA_TESTING.md) (30 min, guía completa)
2. [checklist.md](checklist.md) - ÁREA [N] - Acceptance Criteria
3. AREA*_IMPLEMENTATION.md (cuando se completa cada fix)

**📚 Tech Lead / Reviewer**
1. [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) (5 min)
2. [checklist.md](checklist.md) (30 min, strategy review)
3. AREA*_ANALYSIS.md + AREA*_IMPLEMENTATION.md (durante review de code)

---

## 📈 FLUJO DE DOCUMENTACIÓN

```
┌─────────────────────────────────────────────────┐
│ RESUMEN_EJECUTIVO.md (5 min)                    │
│ - Qué, por qué, impacto, cronograma            │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ QUICK_START.md (2 horas HOY)                    │
│ - Primeras 3 tareas                             │
│ - Dónde buscar, qué documentar                  │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Para CADA ÁREA [1-8]:                           │
│ ┌─────────────────────────────────────┐        │
│ │ 1. ANALYSIS.md (antes)              │        │
│ │    Dónde, qué, impacto             │        │
│ │    ↓                                │        │
│ │ 2. Implementar fix (checklist.md)   │        │
│ │    ↓                                │        │
│ │ 3. IMPLEMENTATION.md (después)      │        │
│ │    Qué cambió, tests, resultados   │        │
│ └─────────────────────────────────────┘        │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ PROGRESS_TRACKING.md (actualizar DIARIO)        │
│ - Checklist de tareas completas                 │
│ - % completado por semana                       │
│ - Blockers y riesgos                            │
└─────────────────────────────────────────────────┘
```

---

## ✅ CHECKLIST PARA COMENZAR

### Hoy (12 Enero)
- [ ] Leer [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) (5 min)
- [ ] Leer [QUICK_START.md](QUICK_START.md) (10 min)
- [ ] **Tarea 1:** Look-Ahead Bias Analysis (45 min)
- [ ] **Tarea 2:** Council Integration Mapping (30 min)
- [ ] **Tarea 3:** Data Validation Pipeline (40 min)
- [ ] Crear rama `feature/fixes-week1`
- [ ] Entregables: 3 análisis .md creados

### Mañana (13 Enero)
- [ ] Comenzar implementación ÁREA 1
- [ ] Crear tests para ÁREA 1
- [ ] Run backtest comparativo Antes/Después

### Semana 1 (Lunes-Viernes)
- [ ] Completar Áreas 1, 4, 7 implementadas
- [ ] 15+ tests creados y pasando
- [ ] Backtest P&L más realista

---

## 🔗 REFERENCIAS RÁPIDAS

### Archivos del Proyecto
- **[core/data/indicators.py](../../core/data/indicators.py)** (707 líneas) - ÁREA 1
- **[core/execution/backtester_core.py](../../core/execution/backtester_core.py)** (1240 líneas) - ÁREAS 2, 4
- **[core/council.py](../../core/council.py)** (332 líneas) - ÁREA 4
- **[core/risk/kelly_sizer.py](../../core/risk/kelly_sizer.py)** (369 líneas) - ÁREA 3
- **[src/execution/market_impact.py](../../src/execution/market_impact.py)** (431 líneas) - ÁREA 5
- **[core/risk/risk_manager.py](../../core/risk/risk_manager.py)** - ÁREA 6
- **[core/data/data_validator.py](../../core/data/data_validator.py)** (413 líneas) - ÁREA 7
- **[strategies/vp_ifvg_ema_strategy.py](../../strategies/vp_ifvg_ema_strategy.py)** (518 líneas) - ÁREA 8

### Documentación Arquitectura
- [docs/COUNCIL.md](COUNCIL.md) - Sistema Council
- [docs/data_flow.md](data_flow.md) - Flujo de datos
- [docs/ARCHITECTURE_REVIEW_AND_PLAN.md](ARCHITECTURE_REVIEW_AND_PLAN.md) - Arquitectura

---

## 💾 ESTRUCTURA DE RAMAS GIT

```bash
# Rama principal para todos los fixes
feature/fixes-week1
  ├─ Semana 1: ÁREAS 1, 4, 7
  ├─ Semana 2: ÁREAS 2, 3
  ├─ Semana 3: ÁREAS 5, 6
  └─ Semana 4: ÁREA 8 + cleanup

# Alternative: Una rama por área
fix/area1-look-ahead-bias
fix/area2-walk-forward-analysis
fix/area3-kelly-dynamic
...
```

---

## 📞 PREGUNTAS FRECUENTES

**P: ¿Por dónde empiezo?**  
R: Lee [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) (5 min), luego [QUICK_START.md](QUICK_START.md) (hoy).

**P: ¿Cuánto tiempo toma?**  
R: 4 semanas full-time (160 horas). Semana 1 es más intensiva.

**P: ¿Qué área es más crítica?**  
R: ÁREA 1 (look-ahead bias) y ÁREA 4 (Council). Empieza con esas.

**P: ¿Necesito testing exhaustivo?**  
R: Sí. 29+ tests creados según [ESTRATEGIA_TESTING.md](ESTRATEGIA_TESTING.md).

**P: ¿Pueden hacerlo dos personas?**  
R: Sí. Una en Semana 1 (Áreas 1,4,7), otra en Semana 2 (Áreas 2,3).

---

## 📝 CONTROL DE VERSIONES

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0 | 12 Ene | Plan maestro completo |
| - | - | - |

---

## 🎓 PARA APRENDER MÁS

**Sobre Look-Ahead Bias:**
- [Datos futuros en indicadores](checklist.md#-area-1-look-ahead-bias)
- [Cómo validar](ESTRATEGIA_TESTING.md#11-unit-test-no-look-ahead-in-volume-profile)

**Sobre Walk-Forward Analysis:**
- [Validación correcta](checklist.md#-area-2-invalid-walk-forward-analysis)
- [Implementación](checklist.md#solution-implementation)

**Sobre Kelly Criterion:**
- [Régimen dinámico](checklist.md#-area-3-static-kelly-no-regime-adjustment)

---

**Última actualización:** 12 Enero 2026, 18:45 UTC  
**Documento:** Master Index v1.0  
**Estado:** ✅ Complete y Listo para Ejecución

👉 **EMPEZAR AHORA:** Lee [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) → [QUICK_START.md](QUICK_START.md)
