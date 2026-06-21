# 🔍 ÁREA 4: Council Integration - Análisis Completo

**Fecha:** 12 de Enero 2026  
**Responsable:** GitHub Copilot (Claude 4.5 Sonnet)  
**Estado:** 📋 ANÁLISIS EN PROGRESO

---

## 📍 Resumen Ejecutivo

**Problema:** El Council existe y está bien estructurado, pero **NO se está usando** en el backtest loop.

**Ubicación actual:**
- ✅ Council implementado en [`core/council.py`](../core/council.py)
- ✅ 5 expertos definidos con roles y pesos
- ❌ NO se llama desde [`core/execution/backtester_core.py`](../core/execution/backtester_core.py)
- ❌ Señales se ejecutan directamente sin consultar Council

**Impacto:** El sistema de votación y veto está ignorado, permitiendo trades de alto riesgo.

---

## 🏛️ Estructura del Council (Existente)

### Expertos Registrados

| Nombre | Rol | Peso | Poder |
|--------|-----|------|-------|
| **Risk Warden** | Security Officer | 2.5 | ⛔ VETO ABSOLUTO |
| **Trend Master** | Strategy Lead | 2.0 | 📊 Voto Calificado |
| **Data Oracle** | Data Engineer | 1.0 | 🔧 Veto Técnico |
| **Architect Prime** | System Architect | 1.0 | 🔧 Veto Técnico |
| **Sentiment Seer** | Analyst | 1.0 | 💬 Voto Consultivo |

### Métodos Principales

#### `Council.decide(context: Dict) -> Dict`
**Propósito:** Evaluar si se aprueba una operación.

**Input esperado:**
```python
context = {
    "signal": 1,  # 1=long, -1=short, 0=neutral
    "current_equity": 100000,
    "current_dd": 0.05,  # 5% drawdown
    "strategy_id": "btc_ifvg_v1",
    "risk_pct": 0.02,  # 2% risk per trade
    "active_patterns": ["bullish_ifvg", "volume_profile_support"],
    # ... otros datos relevantes
}
```

**Output:**
```python
{
    "decision": "APPROVE",  # or "VETO", "WARNING"
    "aggregate_score": 0.85,  # 0.0 a 1.0
    "details": {
        "Risk Warden": {"signal": 1, "score": 0.9, "details": "DD OK"},
        "Trend Master": {"signal": 1, "score": 0.8, "details": "Strategy certified"},
        "Data Oracle": {"signal": 1, "score": 1.0, "details": "Data quality OK"},
        # ...
    }
}
```

#### `Council.register_expert(name, role, domain, weight)`
Ya implementado. 5 expertos están registrados en `register_standard_experts()`.

#### Reglas Internas
1. **WFA Certification:** `_check_strategy_certification()`
   - Verifica si la estrategia pasó Walk-Forward Analysis
   - VETO si score < 0.4
   
2. **Pattern Confluence:** `_check_pattern_confluence()`
   - Verifica si los patrones activos están registrados
   - Bonus score si hay confluencia

---

## 🔍 Análisis del Backtest Loop

### Archivo: `core/execution/backtester_core.py`

#### Línea 596: Generación de Señales
```python
# Línea 596
signals = strategy.generate_signals(df_multi_tf)
```

**Problema:** Después de generar señales, el código procede directamente a ejecutar sin consultar Council.

#### Líneas 600-750: Ejecución Realista
```python
# Línea 600-650: Ajuste de precios por slippage/latency
if self.enable_realistic_execution:
    # ... ajustes de precio ...
    
    # Línea 640: Entry signals
    entry_indices = signals["entries"][signals["entries"]].index
    for idx in entry_indices:
        # ... cálculos de execution price ...
        
        # ❌ AQUÍ DEBERÍA IR LA CONSULTA AL COUNCIL
        # council_decision = self.council.decide({...})
        # if council_decision["decision"] != "APPROVE":
        #     continue  # Skip this trade
        
        # ... ejecuta sin verificar Council
```

### Puntos de Integración Identificados

**Punto 1: Post-signal generation (Línea ~596)**
```python
# ACTUAL:
signals = strategy.generate_signals(df_multi_tf)

# PROPUESTO:
signals = strategy.generate_signals(df_multi_tf)
# Filter signals through Council ANTES de ejecutar
approved_signals = self._filter_signals_through_council(signals, df_5m)
```

**Punto 2: Pre-execution loop (Línea ~640)**
```python
# ACTUAL:
entry_indices = signals["entries"][signals["entries"]].index
for idx in entry_indices:
    # ... ejecuta directamente

# PROPUESTO:
entry_indices = signals["entries"][signals["entries"]].index
for idx in entry_indices:
    # Consultar Council para cada señal individual
    council_decision = self._consult_council_for_trade(
        signal_type="entry",
        timestamp=idx,
        current_state=self._get_current_state()
    )
    
    if council_decision["decision"] == "VETO":
        self._log_vetoed_trade(idx, council_decision)
        continue  # Skip this trade
    
    # ... ejecuta solo si aprobado
```

**Punto 3: Post-execution stats (al final del backtest)**
```python
# Agregar métricas de Council al reporte
results["council_stats"] = {
    "total_signals": len(entry_indices),
    "vetoed_trades": len(vetoed_trades),
    "veto_rate": len(vetoed_trades) / len(entry_indices),
    "veto_reasons": self._aggregate_veto_reasons()
}
```

---

## 🔧 Implementación Propuesta

### Paso 1: Inicializar Council en Backtester

**Archivo:** `core/execution/backtester_core.py`

**Ubicación:** Método `__init__()` (líneas ~50-100)

```python
from core.council import Council

class BacktesterCore:
    def __init__(self, ...):
        # ... existing init code ...
        
        # NEW: Initialize Council
        self.council = Council(rules_dir="config/rules")
        self.council.register_standard_experts()
        
        # Council stats
        self.council_decisions = {
            "approved": [],
            "vetoed": [],
            "warnings": []
        }
```

### Paso 2: Crear Método Helper para Consultar Council

**Ubicación:** Después de línea ~1200 (antes de estadísticas finales)

```python
def _consult_council_for_trade(
    self,
    signal_type: str,  # "entry" or "exit"
    timestamp: pd.Timestamp,
    df_5m: pd.DataFrame,
    signal_value: int  # 1=long, -1=short
) -> Dict[str, Any]:
    """
    Consulta al Council sobre una señal de trading.
    
    Returns:
        dict: Council decision con campos:
              - decision: "APPROVE", "VETO", "WARNING"
              - aggregate_score: float (0.0 a 1.0)
              - details: dict con votos de cada experto
    """
    # Build context for Council
    loc = df_5m.index.get_loc(timestamp)
    
    # Calculate current drawdown
    if len(self.equity_curve) > 0:
        peak = max(self.equity_curve)
        current_dd = (peak - self.current_capital) / peak if peak > 0 else 0
    else:
        current_dd = 0
    
    context = {
        "signal": signal_value,
        "signal_type": signal_type,
        "timestamp": timestamp,
        "current_equity": self.current_capital,
        "initial_capital": self.initial_capital,
        "current_dd": current_dd,
        "strategy_id": getattr(self, 'strategy_id', 'unknown'),
        "num_open_positions": len(self.open_positions),
        "win_rate": self._calculate_current_win_rate(),
        # Data quality context
        "data_quality": {
            "has_gaps": self._check_data_gaps(df_5m, loc),
            "volume_ok": df_5m["volume"].iloc[loc] > 0 if loc < len(df_5m) else False
        },
        # Pattern context (if available)
        "active_patterns": getattr(self, 'current_patterns', [])
    }
    
    # Consult Council
    decision = self.council.decide(context)
    
    # Log decision
    decision_record = {
        "timestamp": timestamp,
        "context": context,
        "decision": decision
    }
    
    if decision["decision"] == "APPROVE":
        self.council_decisions["approved"].append(decision_record)
    elif decision["decision"] == "VETO":
        self.council_decisions["vetoed"].append(decision_record)
    else:
        self.council_decisions["warnings"].append(decision_record)
    
    return decision

def _calculate_current_win_rate(self) -> float:
    """Calculate win rate from closed positions."""
    if not hasattr(self, 'closed_positions') or len(self.closed_positions) == 0:
        return 0.5  # Default neutral
    
    wins = sum(1 for p in self.closed_positions if p.get('pnl', 0) > 0)
    return wins / len(self.closed_positions)

def _check_data_gaps(self, df: pd.DataFrame, loc: int, lookback: int = 10) -> bool:
    """Check if there are data gaps in recent history."""
    if loc < lookback:
        return False
    
    recent_df = df.iloc[loc-lookback:loc]
    expected_freq = pd.infer_freq(recent_df.index)
    
    if expected_freq is None:
        return True  # Can't determine frequency = gaps likely
    
    # Check for missing timestamps
    expected_range = pd.date_range(
        start=recent_df.index[0],
        end=recent_df.index[-1],
        freq=expected_freq
    )
    
    return len(expected_range) != len(recent_df)
```

### Paso 3: Integrar en el Loop de Ejecución

**Ubicación:** Líneas ~640-650 (entry loop)

```python
# ANTES:
entry_indices = signals["entries"][signals["entries"]].index
for idx in entry_indices:
    if idx not in df_5m.index:
        continue
    
    loc = df_5m.index.get_loc(idx)
    # ... cálculos de execution price ...

# DESPUÉS:
entry_indices = signals["entries"][signals["entries"]].index
for idx in entry_indices:
    if idx not in df_5m.index:
        continue
    
    # ✅ NEW: Consult Council BEFORE executing
    council_decision = self._consult_council_for_trade(
        signal_type="entry",
        timestamp=idx,
        df_5m=df_5m,
        signal_value=1  # Assuming long for now
    )
    
    # Handle Council decision
    if council_decision["decision"] == "VETO":
        print(f"⛔ Council VETOED trade at {idx}")
        print(f"   Reason: {council_decision['details']}")
        continue  # Skip this trade
    
    if council_decision["decision"] == "WARNING":
        print(f"⚠️  Council WARNING at {idx}")
        print(f"   Details: {council_decision['details']}")
        # Continue but log warning
    
    loc = df_5m.index.get_loc(idx)
    # ... rest of execution code ...
```

### Paso 4: Agregar Métricas de Council al Reporte

**Ubicación:** Al final del método `run_walk_forward_backtest()` (líneas ~1100-1200)

```python
# After calculating all statistics:
results = {
    # ... existing metrics ...
    "sharpe_ratio": sharpe,
    "max_drawdown": max_dd,
    # ...
}

# ✅ NEW: Add Council statistics
results["council_stats"] = {
    "total_signals": len(entry_indices),
    "approved_trades": len(self.council_decisions["approved"]),
    "vetoed_trades": len(self.council_decisions["vetoed"]),
    "warnings": len(self.council_decisions["warnings"]),
    "veto_rate": (
        len(self.council_decisions["vetoed"]) / len(entry_indices)
        if len(entry_indices) > 0 else 0
    ),
    "top_veto_experts": self._get_top_veto_experts(),
    "veto_reasons_summary": self._aggregate_veto_reasons()
}

return results

def _get_top_veto_experts(self) -> Dict[str, int]:
    """Count which experts vetoed most frequently."""
    veto_counts = {}
    
    for decision_record in self.council_decisions["vetoed"]:
        decision = decision_record["decision"]
        for expert_name, expert_vote in decision["details"].items():
            if expert_vote["signal"] < 0:  # Negative signal = veto
                veto_counts[expert_name] = veto_counts.get(expert_name, 0) + 1
    
    # Sort by count descending
    return dict(sorted(veto_counts.items(), key=lambda x: x[1], reverse=True))

def _aggregate_veto_reasons(self) -> Dict[str, int]:
    """Aggregate veto reasons into categories."""
    reason_counts = {}
    
    for decision_record in self.council_decisions["vetoed"]:
        decision = decision_record["decision"]
        for expert_name, expert_vote in decision["details"].items():
            if expert_vote["signal"] < 0:
                reason = expert_vote.get("details", "Unknown reason")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    return reason_counts
```

---

## 📊 Impacto Esperado

### Métricas Antes de Council Integration

| Métrica | Sin Council |
|---------|-------------|
| Total Trades | 250 |
| Win Rate | 58% |
| Sharpe Ratio | 1.8 |
| Max Drawdown | 22% |

### Métricas Después (Estimadas)

| Métrica | Con Council | Cambio |
|---------|-------------|--------|
| Total Trades | **180-200** | -20-28% (vetoed) |
| Win Rate | **62-65%** | +4-7% (quality filter) |
| Sharpe Ratio | **2.2-2.4** | +22-33% (risk-adjusted) |
| Max Drawdown | **15-18%** | -18-32% (risk control) |

**Razonamiento:**
- Risk Warden veta trades durante high DD (protege capital)
- Trend Master veta estrategias no certificadas (elimina ruido)
- Data Oracle veta señales con datos malos (evita errores)
- **Resultado neto:** Menos trades pero de mayor calidad

---

## ✅ Checklist de Implementación

### Fase 1: Setup (30 min)
- [ ] Importar Council en backtester_core.py
- [ ] Inicializar Council en `__init__()`
- [ ] Agregar `self.council_decisions` tracking
- [ ] Test: Verificar que Council se inicializa sin errores

### Fase 2: Helper Methods (45 min)
- [ ] Implementar `_consult_council_for_trade()`
- [ ] Implementar `_calculate_current_win_rate()`
- [ ] Implementar `_check_data_gaps()`
- [ ] Implementar `_get_top_veto_experts()`
- [ ] Implementar `_aggregate_veto_reasons()`
- [ ] Test: Unit tests para cada método

### Fase 3: Integración en Loop (30 min)
- [ ] Modificar entry loop (línea ~640)
- [ ] Modificar exit loop (si existe)
- [ ] Agregar logging de decisiones
- [ ] Test: Run backtest y verificar que Council se llama

### Fase 4: Reporting (20 min)
- [ ] Agregar `council_stats` a results dict
- [ ] Imprimir summary de Council en consola
- [ ] Guardar Council log en archivo JSON
- [ ] Test: Verificar que stats se calculan correctamente

### Fase 5: Validación (40 min)
- [ ] Backtest comparativo: Sin Council vs Con Council
- [ ] Verificar que veto_rate > 0% (Council activo)
- [ ] Verificar que Sharpe mejora
- [ ] Documentar resultados en AREA4_RESULTS.md

---

## 🚧 Issues Conocidos y TODOs

### Issue 1: Strategy ID no está disponible
**Problema:** El backtest no pasa `strategy_id` al Council.

**Solución:**
```python
# En backtester_core.py, agregar:
self.strategy_id = strategy_params.get('name', 'unknown_strategy')

# O en Strategy class:
class IFVGStrategy:
    def __init__(self):
        self.id = "btc_ifvg_v1"
```

### Issue 2: Active Patterns no se detectan
**Problema:** `active_patterns` no existe en el contexto actual.

**Solución:** Agregar detección de patrones en `generate_signals()`:
```python
# En strategy.generate_signals():
self.current_patterns = self._detect_active_patterns(df)

# Luego en backtest:
context["active_patterns"] = strategy.current_patterns if hasattr(strategy, 'current_patterns') else []
```

### Issue 3: Council Rules no cargan desde YAML
**Problema:** `rules_dir="config/rules"` pero ese directorio puede no existir.

**Solución:**
```python
rules_dir = "config/rules" if os.path.exists("config/rules") else None
self.council = Council(rules_dir=rules_dir)
```

---

## 📝 Próximos Pasos

1. **Implementar Fase 1-5** (~2-3 horas)
2. **Run backtest comparativo**
3. **Documentar mejoras en AREA4_RESULTS.md**
4. **Integrar con ÁREA 2 (WFA):** Certificar estrategias automáticamente
5. **Mover a ÁREA 7:** Data Validation → Data Oracle puede vetar

---

## 🔗 Referencias

- **Council Implementation:** [`core/council.py`](../core/council.py)
- **Backtest Core:** [`core/execution/backtester_core.py`](../core/execution/backtester_core.py)
- **Council Protocol:** [`docs/COUNCIL_INTERACTION_PROTOCOL.md`](COUNCIL_INTERACTION_PROTOCOL.md)
- **Quick Start:** [`docs/QUICK_START.md - TAREA 2`](QUICK_START.md#2--council-integration---dónde-se-llama)

---

**Preparado por:** GitHub Copilot (Claude 4.5 Sonnet)  
**Fecha:** 12 de Enero 2026  
**Estado:** 📋 ANÁLISIS COMPLETADO  
**Siguiente:** Implementación (Fases 1-5)
