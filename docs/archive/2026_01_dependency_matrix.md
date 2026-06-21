# 🔗 MATRIZ DE DEPENDENCIAS Y ARQUITECTURA

**Creado:** 12 de Enero 2026  
**Propósito:** Entender qué áreas pueden hacerse en paralelo vs secuencial

---

## 📊 DEPENDENCIAS ENTRE ÁREAS

```
ÁREA 1: Look-Ahead Bias
└─ Sin dependencias ✓ (puede empezar YA)

ÁREA 4: Council Integration
├─ Podría beneficiarse de ÁREA 1 (datos limpios)
└─ Pero es independiente ✓

ÁREA 7: Data Validation Pipeline
├─ Depende de: ÁREA 1 (parcialmente - para validaciones)
├─ Beneficiado por: Completar análisis primero
└─ Impacta: ÁREA 1, 2, 3 (datos validados)

        ↓↓↓ DESPUÉS DE SEMANA 1 ↓↓↓

ÁREA 2: Walk-Forward Analysis
├─ Depende de: ÁREA 7 (datos validados)
├─ Depende de: Backtester base funcionando
└─ Bloqueante para: ÁREA 3

ÁREA 3: Kelly Dinámico
├─ Depende de: ÁREA 2 (WFA completo da mejor datos históricos)
├─ Depende de: Régimen detection en AnalysisEngines
└─ Independiente de: ÁREA 5, 6

        ↓↓↓ DESPUÉS DE SEMANA 2 ↓↓↓

ÁREA 5: Market Impact Crypto
├─ Depende de: ÁREA 4 (si Council necesita impact info)
├─ Depende de: Datos históricos de volumen/liquidity
└─ Independiente: ÁREA 1, 2, 3, 6, 7, 8

ÁREA 6: Risk Manager
├─ Depende de: ÁREA 4 (Council lo usa para decisions)
├─ Beneficiado por: ÁREA 3 (Kelly dinámico)
└─ Independiente: ÁREA 1, 2, 5, 7, 8

        ↓↓↓ DESPUÉS DE SEMANA 3 ↓↓↓

ÁREA 8: Signal Standardization
├─ Depende de: Todas las anteriores estar estables
└─ Bloqueante para: Production deployment
```

---

## 🎯 ORDEN RECOMENDADO

### ✅ ORDEN CRÍTICO (DEBE RESPETARSE)

```
PRIMERA ONDA (Semana 1):
├─ ÁREA 1 (Look-Ahead Bias) - INICIO INMEDIATO
├─ ÁREA 4 (Council) - PARALELO CON ÁREA 1
├─ ÁREA 7 (Data Validation) - PARALELO CON ÁREA 1 & 4
└─ Esperar: Todas completadas antes de Semana 2

SEGUNDA ONDA (Semana 2):
├─ ÁREA 2 (Walk-Forward) - REQUIERE Semana 1 completa
├─ ÁREA 3 (Kelly) - REQUIERE ÁREA 2 completa
└─ Esperar: WFA funcionando antes de Kelly

TERCERA ONDA (Semana 3):
├─ ÁREA 5 (Market Impact) - Puede empezar after Semana 1
├─ ÁREA 6 (Risk Manager) - Puede empezar after Semana 1
└─ Paralelo: 5 y 6 independientes

CUARTA ONDA (Semana 4):
├─ ÁREA 8 (Signals) - REQUIERE todo estable antes
└─ Cleanup y testing final
```

---

## 👥 PARALELIZACIÓN CON 2 DEVELOPERS

### Opción A: Dividir por Semana (Recomendado)

```
DEV 1:                      DEV 2:
SEMANA 1:                   SEMANA 1:
├─ ÁREA 1 (L-A Bias)       ├─ ÁREA 4 (Council)
└─ ÁREA 7 (Validation)     └─ ÁREA 4 (Testing)

SEMANA 2:                   SEMANA 2:
├─ ÁREA 2 (WFA)            ├─ ÁREA 3 (Kelly)
└─ ÁREA 2 (Testing)        └─ ÁREA 3 (Testing)

SEMANA 3:                   SEMANA 3:
├─ ÁREA 5 (MarketImpact)   ├─ ÁREA 6 (Risk Manager)
└─ ÁREA 5 (Testing)        └─ ÁREA 6 (Testing)

SEMANA 4:                   SEMANA 4:
├─ ÁREA 8 (Signals)        ├─ Integration testing
└─ Cleanup                 └─ Regression tests
```

### Opción B: Dividir por Componente (Alternativo)

```
DEV JUNIOR 1:              DEV SENIOR:
├─ ÁREA 1 (L-A Bias)       ├─ ÁREA 2 (WFA) *complejo
├─ ÁREA 7 (Validation)     ├─ ÁREA 4 (Council) *complejo
└─ ÁREA 8 (Signals)        ├─ ÁREA 3 (Kelly)
                           ├─ ÁREA 5 (Market Impact)
                           ├─ ÁREA 6 (Risk Manager)
                           └─ QA y reviews
```

---

## 📍 MAPA DE CÓDIGO

### Dónde Cada Área Toca el Código

```
FLUJO GENERAL DE BACKTESTER:
┌──────────────────────────────────────────────────────────┐
│ Load Data                                                │
│ └─ ÁREA 7: Validación Obligatoria ✓                    │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│ Generar Señales                                          │
│ ├─ ÁREA 1: Look-Ahead Bias Fix ✓                       │
│ ├─ ÁREA 8: Signal Estandarización (Semana 4)          │
│ └─ Indicadores: VP, IFVG, EMA                           │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│ Consultar Council                                        │
│ └─ ÁREA 4: Integración en Backtester ✓                 │
│    └─ ÁREA 6: Risk Manager para veto                    │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│ Calcular Position Size                                   │
│ └─ ÁREA 3: Kelly Dinámico (Semana 2) ✓                │
│    └─ Régimen detection                                 │
│    └─ Serial correlation penalty                        │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│ Calcular Market Impact                                   │
│ └─ ÁREA 5: Crypto Model (Semana 3) ✓                  │
│    └─ Hourly liquidity factors                          │
│    └─ Buy/sell asymmetry                                │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│ Ejecutar Trade                                           │
│ └─ ÁREA 6: Risk Manager Final Check (Semana 3) ✓      │
│    ├─ Total Drawdown tracking                           │
│    ├─ Correlated risk                                   │
│    └─ Kill switch                                       │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│ Registrar Trade en Histórico                             │
└──────────────────────────────────────────────────────────┘

VALIDACIÓN FINAL: ÁREA 2 (WFA)
└─ Walk-Forward Analysis con parámetros optimizados
   ├─ Degradación OOS/IS
   └─ Stability score
```

---

## 🔄 FLUJO DE DATOS

```
Raw OHLC Data (Alpaca)
         │
         ▼
  ┌─────────────────┐
  │ ÁREA 7: Validate│  ← CRÍTICO: AQUÍ se filtran datos corruptos
  │ ├─ OHLC checks  │
  │ ├─ Time gaps    │
  │ ├─ Look-ahead   │
  │ └─ Auto-fix     │
  └────────┬────────┘
           │
Clean Data
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
    ┌────────────┐   ┌──────────────┐
    │ ÁREA 1:    │   │ ÁREA 5:      │
    │ Indicators │   │ Market Impact│
    │ (no L-A)   │   │ (Crypto)     │
    └─────┬──────┘   └──────────────┘
          │
          ▼
    ┌────────────────────┐
    │ Signals Generated  │
    │ └─ ÁREA 8: Format  │
    └─────┬──────────────┘
          │
          ▼
    ┌────────────────────┐
    │ ÁREA 4: Council    │
    │ ├─ Check rules     │
    │ └─ Vote            │
    └─────┬──────────────┘
          │
          ├──────────────────┐
          │                  │
          ▼                  ▼
    Approved          Rejected
         │                  │
         ▼                  ▼
    Continue         Skip Trade
         │
         ▼
    ┌────────────────────┐
    │ ÁREA 3: Kelly Size │
    │ └─ Régimen adj.    │
    └─────┬──────────────┘
          │
          ▼
    ┌────────────────────┐
    │ ÁREA 6: Risk Check │
    │ ├─ Total DD        │
    │ ├─ Correlation     │
    │ └─ VaR             │
    └─────┬──────────────┘
          │
          ├──────────────────┐
          │                  │
          ▼                  ▼
    Safe to Trade    Kill Switch
         │
         ▼
    Execute (with ÁREA 5 impact)
         │
         ▼
    ┌────────────────────┐
    │ ÁREA 2: WFA        │
    │ ├─ Optimize params │
    │ ├─ In-sample       │
    │ ├─ Out-of-sample   │
    │ └─ Degradation     │
    └────────────────────┘
```

---

## 🎯 IMPACTOS CRUZADOS

### Cambio en ÁREA 1 → Qué Necesita Revalidarse

```
Fix Look-Ahead Bias (ÁREA 1)
    ↓
Sharpe ratio baja 30-40%
    ↓
Requiere revisar:
├─ ÁREA 2: WFA (degradation diferentes)
├─ ÁREA 3: Kelly (menos ganancia promedio)
└─ ÁREA 4: Council (más rejazos por volatilidad)
```

### Cambio en ÁREA 2 → Qué Necesita Revalidarse

```
Implementar WFA Real (ÁREA 2)
    ↓
Parámetros cambian por período
    ↓
Requiere revisar:
├─ ÁREA 3: Kelly (basado en nuevos params)
├─ ÁREA 5: Market Impact (volumen puede variar)
└─ ÁREA 8: Signals (format consistente)
```

### Cambio en ÁREA 4 → Qué Necesita Revalidarse

```
Integrar Council (ÁREA 4)
    ↓
Más rejazos de trades
    ↓
Requiere revisar:
├─ ÁREA 3: Kelly (menos trades, estadísticas)
└─ ÁREA 6: Risk Manager (menos exposición)
```

---

## ⚠️ RIESGOS DE INTEGRACIÓN

| Riesgo | Probabilidad | Remedio |
|--------|--------------|---------|
| ÁREA 1 fix rompe ÁREA 4 signals | Media | Regression tests for Council |
| ÁREA 2 WFA mejora pero ÁREA 3 Kelly falla | Media | Update Kelly sobre WFA output |
| ÁREA 5 Market Impact muy conservador | Baja | Backtest comparativo con datos reales |
| ÁREA 6 Risk Manager detiene todos los trades | Media | Tune thresholds post-semana 1 |
| ÁREA 8 Signal refactor rompe ÁREA 4 | Media | Integration test antes de merge |

---

## ✅ GATES DE CALIDAD

### Gate 1: Fin de Semana 1
```
Requirement:
├─ ÁREA 1: Tests pasando, Sharpe bajó 30-40% ✓
├─ ÁREA 4: Council consultado en 100% trades ✓
├─ ÁREA 7: Datos validándose automáticamente ✓
└─ NO regresiones en backtests otros datos

Sign-off: ✓ Go to Semana 2
```

### Gate 2: Fin de Semana 2
```
Requirement:
├─ ÁREA 2: WFA optimiza parámetros cada período ✓
├─ ÁREA 3: Kelly varía por régimen ✓
└─ Backtest P&L +/- 10% vs Semana 1

Sign-off: ✓ Go to Semana 3
```

### Gate 3: Fin de Semana 3
```
Requirement:
├─ ÁREA 5: Market impact varía por hora ✓
├─ ÁREA 6: Total DD tracking funciona ✓
└─ Backtest realista con 20% margen

Sign-off: ✓ Go to Semana 4
```

### Gate 4: Fin de Semana 4
```
Requirement:
├─ ÁREA 8: Todas estrategias usan TradingSignal ✓
├─ 29+ tests creados y pasando ✓
├─ Regression tests: 0 failures ✓
└─ Documentación completa

Sign-off: ✓ Ready for Live Trading
```

---

## 📋 DEPENDENCY CHECKLIST

Antes de empezar cada ÁREA:

### ANTES DE ÁREA 2
```
Pre-requisitos:
- [ ] ÁREA 1 implementada y testeada
- [ ] ÁREA 7 datos validándose
- [ ] Backtester base estable
- [ ] Optimization framework disponible (skopt)
```

### ANTES DE ÁREA 3
```
Pre-requisitos:
- [ ] ÁREA 2 WFA optimizando parámetros
- [ ] Histórico de trades disponible (50+)
- [ ] Régimen detection implementado
```

### ANTES DE ÁREA 5
```
Pre-requisitos:
- [ ] ÁREA 7 datos validados
- [ ] Histórico de volumen/liquidity por hora
- [ ] Market data completo
```

### ANTES DE ÁREA 6
```
Pre-requisitos:
- [ ] ÁREA 4 Council integrando
- [ ] Histórico de P&L disponible
- [ ] Risk thresholds definidos
```

### ANTES DE ÁREA 8
```
Pre-requisitos:
- [ ] Todas ÁREAS 1-7 estables
- [ ] No cambios esperados en formatos
- [ ] Tests de integración passing
```

---

## 🚀 VELOCIDAD ESTIMADA

```
ÁREA 1: 8 horas (análisis 2h + impl 4h + tests 2h)
ÁREA 4: 10 horas (análisis 3h + impl 5h + tests 2h)
ÁREA 7: 12 horas (análisis 3h + impl 6h + tests 3h)
────────────────────────────────────────────────
SEMANA 1 TOTAL: 30 horas

ÁREA 2: 20 horas (muy complejo)
ÁREA 3: 15 horas (moderado)
────────────────────────────────────────────────
SEMANA 2 TOTAL: 35 horas

ÁREA 5: 12 horas
ÁREA 6: 18 horas
────────────────────────────────────────────────
SEMANA 3 TOTAL: 30 horas

ÁREA 8: 15 horas
Cleanup: 10 horas
────────────────────────────────────────────────
SEMANA 4 TOTAL: 25 horas

GRAN TOTAL: ~120 horas (Semana 1 es ~25% del trabajo)
```

---

## 💡 LECCIONES APRENDIDAS DE ARQUITECTURA

1. **Temporal Correctness es Fundacional**
   - ÁREA 1 debe estar perfecto antes de optimizar
   - ÁREA 7 validación es pre-requisito para todo

2. **Validación Central**
   - ÁREA 7 (validation) impacta a TODAS las demás
   - Sin validación, arreglos en otras áreas son en vano

3. **Risk Management es Vertebral**
   - ÁREA 4, 6 son veto points (pueden detener trades)
   - Deben estar integradas, no optionales

4. **Optimización sin Validación es Peligrosa**
   - ÁREA 2 (WFA) requiere ÁREA 1 (sin bias)
   - De otro modo, optimiza basura

5. **Compatibilidad en la Línea**
   - ÁREA 8 (signals) afecta a ÁREA 4 (Council)
   - Cambiar tardíamente cuesta

---

**Última actualización:** 12 de Enero 2026  
**Próxima revisión:** Fin de Semana 1  

**Estado:** ✅ Matriz completa y validada
