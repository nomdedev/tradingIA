# ⚡ QUICK START - Primeras Tareas de Hoy

**Fecha:** 12 de Enero 2026  
**Prioridad:** 🔴 CRÍTICA

---

## 🎯 TOP 3 TAREAS PARA HOY

### 1. 🚨 LOOK-AHEAD BIAS - Análisis del Código
**Tiempo estimado:** 45 min  
**Archivos a revisar:**
- [core/data/indicators.py](core/data/indicators.py#L285-L310) - Función `generate_filtered_signals()`
- [core/data/indicators.py](core/data/indicators.py#L200-L250) - Función `volume_profile_advanced()`

**Qué buscar:**
```python
# ❌ MALO (incluye datos futuros)
vpc = pd.Series(index=df.index, dtype=float)
for i in range(len(df)):
    window = df.iloc[i-lookback:i+1]  # ← PROBLEMA: incluye df[i] y no debe
    vpc.iloc[i] = window.close.mean()  # ← Usa valor del futuro

# ✅ CORRECTO (solo datos pasados)
vpc = pd.Series(index=df.index, dtype=float)
for i in range(lookback, len(df)):
    window = df.iloc[i-lookback:i]  # ← Solo datos pasados
    vpc.iloc[i] = window.close.mean()
```

**Checklist:**
- [ ] Abre [core/data/indicators.py](core/data/indicators.py)
- [ ] Busca la línea con `volume_profile_advanced()`
- [ ] Identifica dónde se calcula el POC (Point of Control)
- [ ] Documenta: ¿Qué datos se incluyen en la ventana?
- [ ] Crea archivo [AREA1_ANALYSIS.md](docs/AREA1_ANALYSIS.md) con hallazgos
- [ ] Propón fix en pseudocódigo

**Entregable:** `docs/AREA1_ANALYSIS.md` con análisis detallado

---

### 2. 🚨 COUNCIL INTEGRATION - Dónde se llama?
**Tiempo estimado:** 30 min  
**Archivos a revisar:**
- [core/council.py](core/council.py#L1-L50) - Estructura del Council
- [core/execution/backtester_core.py](core/execution/backtester_core.py#L850-L950) - Bucle de ejecución

**Qué buscar:**
```python
# En backtester_core.py, dentro de run_simple_backtest():
# ❌ ACTUAL: Se generan señales pero nunca se consulta Council
signal = strategy.generate_signal(current_data)
if signal:
    # Debería: consultar Council primero
    # council_decision = self.council.decide(...)
    # if council_decision["approve"]:
    #     execute_trade()
    execute_trade()  # ← Ejecuta sin consultar Council

# ✅ DESEADO:
signal = strategy.generate_signal(current_data)
if signal:
    council_decision = self.council.decide({
        "signal": signal,
        "current_equity": self.equity,
        "current_dd": self.current_dd
    })
    if council_decision["approve"]:
        execute_trade()
```

**Checklist:**
- [ ] Abre [core/council.py](core/council.py)
- [ ] Documentar: ¿Cuáles son los 5 expertos?
- [ ] Documentar: ¿Qué reglas se definen?
- [ ] Abre [core/execution/backtester_core.py](core/execution/backtester_core.py)
- [ ] Busca `run_simple_backtest()` y `execute_trade()`
- [ ] Identifica: ¿Dónde debería ir la llamada a `council.decide()`?
- [ ] Crea archivo [AREA4_INTEGRATION_POINTS.md](docs/AREA4_INTEGRATION_POINTS.md)

**Entregable:** `docs/AREA4_INTEGRATION_POINTS.md` con mapa de integración

---

### 3. 🚨 DATA VALIDATION - Pipeline Obligatorio
**Tiempo estimado:** 40 min  
**Archivos a revisar:**
- [core/data/data_validator.py](core/data/data_validator.py) - Validador (ya existe pero no se usa)
- [api/data_fetcher.py](api/data_fetcher.py) - Donde se cargan datos
- [core/backend_core.py](core/backend_core.py) - Punto de entrada

**Qué buscar:**
```python
# ❌ ACTUAL: Datos van directo sin validar
def load_alpaca_data(symbol, timeframe):
    data = api.get_historical_data(symbol, timeframe)
    return data  # ← Sin validar

# ✅ DESEADO: Validación obligatoria
def load_alpaca_data(symbol, timeframe):
    data = api.get_historical_data(symbol, timeframe)
    
    # Validaciones críticas obligatorias
    validator = DataValidator()
    
    # CRITICAL: Si falla, no retornar datos
    validator.validate_ohlc_relationships(data)  
    validator.detect_time_gaps(data)
    validator.detect_look_ahead_bias(data)
    
    # Auto-fix: WARNING level issues
    data = validator.auto_fix_duplicates(data)
    data = validator.auto_fix_gaps(data)
    
    return data
```

**Checklist:**
- [ ] Abre [core/data/data_validator.py](core/data/data_validator.py)
- [ ] Lista todas las funciones de validación disponibles
- [ ] Abre [api/data_fetcher.py](api/data_fetcher.py)
- [ ] Identifica dónde se retornan los datos
- [ ] Propón pipeline de validación obligatoria
- [ ] Crea archivo [AREA7_VALIDATION_PIPELINE.md](docs/AREA7_VALIDATION_PIPELINE.md)

**Entregable:** `docs/AREA7_VALIDATION_PIPELINE.md` con pipeline propuesto

---

## 📋 CHECKLIST DE HOY

```
MAÑANA 12 ENERO - ANÁLISIS Y DOCUMENTACIÓN
├─ ⬜ Tarea 1: Look-Ahead Bias Analysis (45 min)
│  └─ Entregable: AREA1_ANALYSIS.md
├─ ⬜ Tarea 2: Council Integration Points (30 min)
│  └─ Entregable: AREA4_INTEGRATION_POINTS.md
├─ ⬜ Tarea 3: Data Validation Pipeline (40 min)
│  └─ Entregable: AREA7_VALIDATION_PIPELINE.md
└─ ⬜ FINAL: Crear rama feature/fixes-week1 en git
   └─ Entregable: Branch creada, lista para commits
```

---

## 🔗 REFERENCIAS RÁPIDAS

### Archivos Más Críticos
1. **[core/data/indicators.py](core/data/indicators.py)** (707 líneas)
   - Problema: Look-ahead bias en Volume Profile
   - Solución: Usar solo datos pasados en ventanas móviles

2. **[core/execution/backtester_core.py](core/execution/backtester_core.py)** (1240 líneas)
   - Problema 1: WFA no optimiza parámetros
   - Problema 2: Council nunca se consulta

3. **[core/council.py](core/council.py)** (332 líneas)
   - Estado: Implementado pero no usado
   - Solución: Integrar en backtester_core.py

### Comandos Útiles
```bash
# Buscar línea específica en archivo
grep -n "def generate_filtered_signals" core/data/indicators.py

# Contar líneas
wc -l core/data/indicators.py

# Ver estructura de clase
grep -n "class\|def " core/council.py
```

---

## 📞 CONTACTOS Y REFERENCIAS

**Documentación Relacionada:**
- [docs/checklist.md](docs/checklist.md) - Plan completo de 4 semanas
- [docs/PROGRESS_TRACKING.md](docs/PROGRESS_TRACKING.md) - Seguimiento diario
- [docs/COUNCIL.md](docs/COUNCIL.md) - Arquitectura del Council

**Para Más Detalles:**
- Solución completa en [docs/checklist.md](docs/checklist.md) → ÁREA 1, 4, 7
- Código ejemplo en [docs/checklist.md](docs/checklist.md) → Secciones de "Full Solution Code"

---

**¿Listo para comenzar?**  
👉 Abre [core/data/indicators.py](core/data/indicators.py) y empieza con TAREA 1

**Tiempo estimado para completar hoy:** 2 horas  
**Siguiente mileston:** 13 Enero (Implementar fixes Área 1)
