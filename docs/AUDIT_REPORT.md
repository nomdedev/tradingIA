# 📋 INFORME DE AUDITORÍA GENERAL - TradingIA

**Fecha de Auditoría:** 13 de Enero de 2026  
**Versión del Proyecto:** Post-Ronda 10  
**Total de Errores Detectados:** 644 (IDE) + 44 (análisis manual)  

---

## 📊 RESUMEN EJECUTIVO

| Categoría | CRÍTICO | ALTO | MEDIO | BAJO |
|-----------|---------|------|-------|------|
| Complejidad Cognitiva | 2 | 4 | 1 | - |
| Seguridad | - | - | 3 | 1 |
| Imports Faltantes | - | 9 | - | - |
| Malas Prácticas | - | - | 2 | 50+ |
| Variables No Utilizadas | - | - | - | 25+ |
| Inconsistencias Config | - | - | 2 | 1 |

---

## 🔴 PROBLEMAS CRÍTICOS (Requieren atención inmediata)

### 1. Funciones con Complejidad Cognitiva Extrema

| Archivo | Función | Complejidad | Límite |
|---------|---------|-------------|--------|
| `core/execution/backtester_core.py:726` | `run_simple_backtest()` | **71** | 15 |
| `core/council.py:218` | `decide()` | **51** | 15 |
| `core/execution/backtester_core.py:450` | `_process_and_record_trades()` | **32** | 15 |
| `core/data/indicators.py:18` | `calculate_ifvg_enhanced()` | **31** | 15 |
| `core/data/indicators.py:264` | `generate_filtered_signals()` | **27** | 15 |
| `core/data/indicators.py:132` | `volume_profile_advanced_slow()` | **27** | 15 |

**Impacto:** Código difícil de mantener, propenso a errores, difícil de testear.

**Solución recomendada:** Extraer lógica a funciones auxiliares más pequeñas.

---

## 🟠 PROBLEMAS ALTOS (Corregir en próximo sprint)

### 2. Dependencias No Instaladas/Faltantes

Módulos que se importan pero pueden no estar instalados:

```
alpaca-py          - pip install alpaca-py
mlflow             - pip install mlflow
arch               - pip install arch
fastapi            - pip install fastapi
playsound          - pip install playsound
cryptography       - pip install cryptography
pandas-ta          - pip install pandas-ta
python-dotenv      - pip install python-dotenv
```

### 3. Problema con scikit-learn

```
ImportError: No module named 'sklearn.__check_build._check_build'
```

**Causa:** Incompatibilidad entre la versión de Python 3.14 y sklearn compilado para 3.11.

**Solución:**
```bash
pip uninstall scikit-learn
pip install scikit-learn --force-reinstall
```

---

## 🟡 PROBLEMAS MEDIOS (Corregir cuando sea posible)

### 4. Seguridad - Credenciales Expuestas

| Archivo | Línea | Campo Expuesto |
|---------|-------|----------------|
| `config/user_preferences.json` | 17-18 | `alpaca_key`, `alpaca_secret` |
| `config/user_preferences.json` | 113-126 | `api_key` para agentes AI |
| `config/user_preferences.json` | 174 | `telegram_bot_token` |

**Solución:** Mover a variables de entorno y usar `.env`:
```python
import os
ALPACA_KEY = os.getenv("ALPACA_KEY")
```

### 5. Comparaciones Float Incorrectas

| Archivo | Línea | Código Problemático |
|---------|-------|---------------------|
| `core/council.py` | 373 | `if final_score == 0.0` |
| `tests/test_no_lookahead_simple.py` | 135 | `assert ... == 20.0` |

**Solución:**
```python
# Antes
if final_score == 0.0:

# Después
import math
if math.isclose(final_score, 0.0, abs_tol=1e-9):
```

### 6. Literales String Duplicados

| Archivo | Literal | Repeticiones |
|---------|---------|--------------|
| `core/council.py` | `"Trend Master"` | 7 |
| `core/council.py` | `"Risk Warden"` | 4 |
| `core/council.py` | `"Data Oracle"` | 4 |

**Solución:** Definir constantes:
```python
EXPERT_TREND_MASTER = "Trend Master"
EXPERT_RISK_WARDEN = "Risk Warden"
```

### 7. Inconsistencias en Configuración

| Parámetro | Archivo | Valor |
|-----------|---------|-------|
| `end_date` | `backtest_configs.json` | `2025-11-12` |
| `end_date` | otros archivos | `2024-12-31` |

---

## 🟢 PROBLEMAS BAJOS (Mejoras de calidad)

### 8. Variables No Utilizadas (25+ instancias)

Principalmente en archivos de test:

| Archivo | Variables |
|---------|-----------|
| `tests/test_no_look_ahead_bias.py` | `vah_orig`, `val_orig`, `wrong_window`, etc. |
| `core/data/indicators.py:110` | parámetro `params` no usado |

**Solución:** Usar `_` para variables ignoradas:
```python
# Antes
poc_orig, vah_orig, val_orig = volume_profile()

# Después
poc_orig, _, _ = volume_profile()
```

### 9. Uso de `numpy.random.randn()` Deprecado

| Archivo | Líneas |
|---------|--------|
| `tests/test_no_look_ahead_bias.py` | 43, 48, 49, 50 |
| `tests/test_no_lookahead_simple.py` | 64, 97 |

**Solución:**
```python
# Antes
np.random.randn(100)

# Después
rng = np.random.default_rng(42)
rng.standard_normal(100)
```

### 10. Bloques `except Exception` Amplios (~50 instancias)

Archivos con más instancias:
- `backtester_core.py`: 14
- `backend_core.py`: 10
- `app.py`: 8
- `council.py`: 7

**Solución:** Capturar excepciones específicas:
```python
# Antes
except Exception as e:
    pass

# Después
except (ValueError, KeyError) as e:
    logger.error(f"Error específico: {e}")
```

### 11. Strings Concatenados Implícitamente

| Archivo | Líneas |
|---------|--------|
| `backtester_core.py` | 293, 587, 659, 666 |

```python
# Antes
f"Kelly position sizing: ${position_size:.2f} " f"({sizing_result['position_pct']:.1%} of capital)"

# Después
f"Kelly position sizing: ${position_size:.2f} ({sizing_result['position_pct']:.1%} of capital)"
```

### 12. TODOs Pendientes

| Archivo | Línea | TODO |
|---------|-------|------|
| `platform_gui_tab7_improved.py` | 271 | `# TODO: Integrate with actual strategy registry` |
| `platform_gui_tab2_improved.py` | 1257-1288 | Generación de gráficos en thread separado |

---

## ✅ CORRECCIONES IMPLEMENTADAS (Ronda 11)

### Completadas en esta sesión:

1. **✅ `Council.decide()` - Complejidad reducida de 51 → ~15**
   - Extraídos 8 métodos auxiliares:
     - `_evaluate_declarative_rules()`
     - `_gather_expert_evidence()`
     - `_calculate_expert_votes()`
     - `_calculate_single_vote()`
     - `_check_vetos()`
     - `_create_veto_response()`
     - `_calculate_consensus()`
     - `_determine_decision()`
   - Añadidas constantes para nombres de expertos
   - Corregida comparación float `== 0.0` → `math.isclose()`

2. **✅ `run_simple_backtest()` - Complejidad reducida de 71 → ~20**
   - Extraídos 6 métodos auxiliares:
     - `_prepare_backtest_data()`
     - `_calculate_volatility()`
     - `_process_entry_signals()`
     - `_process_exit_signals()`
     - `_execute_backtest()`
     - `_run_realistic_execution()`
     - `_run_simple_execution()`
     - `_process_backtest_results()`
     - `_build_backtest_result()`

3. **✅ Comparaciones float corregidas**
   - `test_extracted_modules.py` - 8 correcciones con `pytest.approx()`
   - `test_no_lookahead_simple.py` - 2 correcciones
   - `test_council_protocol.py` - 4 correcciones con `math.isclose()`
   - `test_council_advanced.py` - 1 corrección
   - `test_backend_core.py` - 3 correcciones
   - `test_backtester_core.py` - 1 corrección
   - `test_critical_corrections.py` - 2 correcciones
   - `test_area3_kelly.py` - 3 correcciones

4. **✅ Variables no utilizadas limpiadas**
   - `test_no_look_ahead_bias.py` - `vah_orig`, `val_orig`, `wrong_window` → `_`

---

## 🛠️ PLAN DE ACCIÓN RECOMENDADO

### Prioridad 1 (Esta semana)
1. [x] ~~Refactorizar `run_simple_backtest()` - dividir en funciones menores~~
2. [x] ~~Refactorizar `Council.decide()` - extraer lógica de votación~~
3. [ ] Reinstalar scikit-learn compatible con Python 3.14

### Prioridad 2 (Próxima semana)
4. [ ] Mover credenciales a variables de entorno
5. [ ] Instalar dependencias faltantes
6. [x] ~~Corregir comparaciones float~~

### Prioridad 3 (Mes siguiente)
7. [x] ~~Limpiar variables no utilizadas en tests~~
8. [ ] Reemplazar `np.random.randn()` deprecado
9. [x] ~~Definir constantes para strings duplicados~~
10. [ ] Refinar bloques except amplios

---

## 📈 MÉTRICAS DE CALIDAD

| Métrica | Pre-Ronda 11 | Post-Ronda 11 | Objetivo |
|---------|--------------|---------------|----------|
| Complejidad cognitiva máx | 71 | **~20** | <15 |
| Bloques except amplios | ~50 | ~50 | <10 |
| Variables no usadas | ~25 | **~20** | 0 |
| Dependencias faltantes | 9 | 9 | 0 |
| Test coverage (estimado) | ~60% | ~60% | 80% |
| Comparaciones float incorrectas | ~24 | **~0** | 0 |

---

## ✅ ARCHIVOS SIN ERRORES

Los siguientes archivos core pasaron la auditoría sin problemas:

- `core/constants.py`
- `core/data/data_validator.py`
- `core/risk/kelly_sizer.py`
- `core/risk/risk_manager.py`
- `core/signals/trading_signal.py`
- `core/strategies/momentum_strategy.py`
- `core/execution/live_trader.py`
- `core/execution/metrics_calculator.py`
- `core/execution/monte_carlo_simulator.py`
- `core/execution/walk_forward_optimizer.py`
- `core/training/retrain_pipeline.py`
- `core/tracking/mlflow_tracker.py`
- `api/data_fetcher.py`
- `dashboard/app.py`

---

*Auditoría generada automáticamente - 13 de Enero 2026*
