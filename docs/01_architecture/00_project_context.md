# Contexto del Proyecto - TradingIA

**Última actualización:** 13 de Enero 2026  
**Estado:** ✅ 8 Áreas Críticas Completadas - Sistema Listo para Paper Trading

---

## 📋 Resumen Ejecutivo

Sistema de backtesting profesional para BTC combinando:
1. **IFVG** (Institutional Fair Value Gaps) - Gaps mitigados
2. **Volume Profile** - POC, VAH, VAL, zonas SD
3. **EMAs Multi-Timeframe** - Cross-TF con interconexiones
4. **Council de Expertos** - Sistema de decisión multi-agente
5. **Risk Management Avanzado** - VaR, CVaR, correlación

---

## 🏗️ Arquitectura del Sistema

```
tradingIA/
├── api/                    # Data fetching (Alpaca)
│   ├── data_fetcher.py     # ✅ DataValidator integrado
│   └── __init__.py
│
├── core/                   # Núcleo del sistema
│   ├── council.py          # ✅ Sistema de decisión multi-experto
│   ├── backend_core.py     # Motor principal
│   ├── data/
│   │   ├── indicators.py   # ✅ ÁREA 1: Look-ahead bias corregido
│   │   └── data_validator.py # ✅ ÁREA 7: Validación de datos
│   ├── execution/
│   │   └── backtester_core.py # ✅ ÁREA 2,4: WFA + Council
│   ├── risk/
│   │   ├── kelly_sizer.py    # ✅ ÁREA 3: Kelly con régimen
│   │   └── risk_manager.py   # ✅ ÁREA 6: HWM, VaR, CVaR
│   ├── signals/
│   │   └── trading_signal.py # ✅ ÁREA 8: TradingSignal dataclass
│   └── strategies/         # Estrategias de trading
│
├── src/                    # Módulos adicionales
│   └── execution/
│       └── market_impact.py # ✅ ÁREA 5: MarketImpactModelCrypto
│
├── dashboard/              # UI Streamlit
│   └── app.py
│
├── tests/                  # Suite de tests (65+)
│   ├── test_no_lookahead_simple.py  # 5 tests
│   ├── test_area2_wfa.py            # 7 tests
│   ├── test_area3_kelly.py          # 8 tests
│   ├── test_area5_market_impact_crypto.py # 9 tests
│   ├── test_area6_risk_manager.py   # 13 tests
│   ├── test_area7_data_validation.py # 5 tests
│   ├── test_area8_trading_signal.py # 18 tests
│   └── run_all_tests.py             # Runner completo
│
└── config/                 # Configuración
    ├── training_config.yaml
    └── strategies/
```

---

## ✅ Áreas Críticas Implementadas (8/8)

### ÁREA 1: Look-Ahead Bias Fix
**Archivo:** `core/data/indicators.py`  
**Problema:** `volume_profile_advanced_slow()` usaba datos futuros  
**Solución:** Cambio de `df.iloc[i-window:i+1]` a `df.iloc[i-window:i]`  
**Tests:** 5/5 pasando

### ÁREA 2: Walk-Forward Analysis Real
**Archivo:** `core/execution/backtester_core.py`  
**Problema:** WFA no optimizaba parámetros realmente  
**Solución:** 
- `_bayesian_optimize()` con skopt
- Degradación: `(IS - OOS) / |IS| * 100`
- Stability Score y criterios de certificación
- Anchored WFA (train_start = 0)  
**Tests:** 7/7 pasando

### ÁREA 3: Kelly Criterion con Régimen
**Archivo:** `core/risk/kelly_sizer.py`  
**Problema:** Kelly fijo sin considerar mercado  
**Solución:**
- `calculate_regime_adjusted_kelly()`
- `REGIME_MULTIPLIERS`: bull=1.0, bear=0.5, chop=0.3
- `STREAK_PENALTIES` para correlación serial
- `calculate_adaptive_lookback()` dinámico  
**Tests:** 8/8 pasando

### ÁREA 4: Council Integration
**Archivo:** `core/council.py` + `core/execution/backtester_core.py`  
**Problema:** Council no se consultaba en trades  
**Solución:**
- Council integrado en backtest loop
- `_build_trade_context()` para contexto
- Tracking de decisiones  
**Tests:** Verificado en comparison backtest

### ÁREA 5: Market Impact Crypto
**Archivo:** `src/execution/market_impact.py`  
**Problema:** Modelo Almgren-Chriss para equities, no crypto  
**Solución:** `MarketImpactModelCrypto` con:
- `liquidity_by_hour`: 0-23 UTC
- `global_daily_volume`: BTC=$30B, ETH=$15B
- `sell_penalty = 1.35` (35% más slippage ventas)
- `get_best_execution_hours()`   
**Tests:** 9/9 pasando

### ÁREA 6: Risk Manager Mejorado
**Archivo:** `core/risk/risk_manager.py`  
**Problema:** Solo daily drawdown, sin correlación  
**Solución:**
- `high_water_mark` para total drawdown
- `calculate_var()` y `calculate_cvar()`
- `calculate_correlated_risk()` 
- `max_consecutive_losses` tracking
- `get_position_size_adjustment()` dinámico  
**Tests:** 13/13 pasando

### ÁREA 7: Data Validation Pipeline
**Archivo:** `core/data/data_validator.py`  
**Problema:** Datos no validados antes de backtest  
**Solución:**
- `DataValidator` con `generate_council_context()`
- Regla `data_quality` en Council
- Validación OHLC, gaps, duplicados  
**Tests:** 5/5 pasando

### ÁREA 8: TradingSignal Standard
**Archivo:** `core/signals/trading_signal.py`  
**Problema:** Formatos de señal inconsistentes  
**Solución:** Dataclass `TradingSignal` con:
- `SignalDirection`: LONG/SHORT/CLOSE
- `SignalStrength`: WEAK/MODERATE/STRONG/VERY_STRONG
- Campos: `reasons`, `council_approved`, `indicators_snapshot`
- Helpers: `create_long_signal()`, `convert_legacy_signal()`  
**Tests:** 18/18 pasando

---

## 🎯 Estrategia Principal: VP + IFVG + EMA Multi-TF

### Jerarquía de Timeframes

```
┌─────────────────────────────────────────────────┐
│         TIMEFRAME HIERARCHY                     │
├─────────────────────────────────────────────────┤
│ 1H  (Trend)     → ALWAYS bias (uptrend/down)   │
│ ↓                 EMA200_1h filter              │
│ 15Min (Momentum) → Confirmation                 │
│ ↓                 EMA50_15m cross               │
│ 5Min (Entry)    → Signals                      │
│                   IFVG + VP + Vol               │
└─────────────────────────────────────────────────┘
```

### Lógica de Señales

```python
# Bull Signal (Filtrado)
bull_filtered = (
    bull_signal_ifvg &           # IFVG gap bull mitigado
    uptrend_1h &                 # HTF filter OBLIGATORIO
    momentum_15m &               # MTF confirmation
    vol_filter &                 # Vol cross-TF
    (close > VAL_5m) &           # VP support
    (abs(close - POC_1h) < 0.5*ATR_1h)  # Near key level
)
```

### Parámetros Optimizables

| Parámetro | Min | Max | Default | Mejor para |
|-----------|-----|-----|---------|------------|
| atr_multi | 0.1 | 0.5 | 0.3 | Alta vol: 0.4-0.5 |
| vol_thresh | 0.8 | 1.5 | 1.2 | Baja vol: 0.8-1.0 |
| ema1_entry | 15 | 25 | 18 | Scalp: 15-18 |
| ema2_entry | 40 | 60 | 48 | Swing: 50-60 |
| tp_rr | 1.8 | 2.5 | 2.2 | Cons: 2.0-2.2 |

---

## 🧪 Ejecutar Tests

```bash
# Todos los tests (65+)
python tests/run_all_tests.py

# Tests individuales por área
python tests/test_no_lookahead_simple.py      # ÁREA 1
python tests/test_area2_wfa.py                # ÁREA 2
python tests/test_area3_kelly.py              # ÁREA 3
python tests/test_area5_market_impact_crypto.py # ÁREA 5
python tests/test_area6_risk_manager.py       # ÁREA 6
python tests/test_area7_data_validation.py    # ÁREA 7
python tests/test_area8_trading_signal.py     # ÁREA 8
```

---

## 📊 Métricas Target

```yaml
Sharpe Ratio: > 1.0
Calmar Ratio: > 2.0
Max Drawdown: < 15%
Win Rate: 55-65%
Profit Factor: > 1.5
WFA Degradation: < 30%
HTF Alignment: > 70%
```

---

## 🚀 Próximos Pasos (Sprint 2+)

### Pendientes de Infraestructura
- [ ] Pre-commit hooks (black, isort, flake8)
- [ ] Eliminar archivos duplicados (_v2, _v3, _debug)
- [ ] Refactorizar imports (eliminar sys.path hacks)
- [ ] Centralizar logging

### Live Trading
- [ ] LiveTrader con interfaz común
- [ ] Reconexión automática API
- [ ] Retry logic para órdenes
- [ ] Rate limiter Alpaca

### Producción
- [ ] Dockerizar aplicación
- [ ] CI/CD completo
- [ ] Deploy paper trading

---

## 📚 Documentación Principal

| Archivo | Descripción |
|---------|-------------|
| [GUIA_USUARIO_COMPLETA.md](GUIA_USUARIO_COMPLETA.md) | Guía completa de usuario |
| [QUICK_START.md](QUICK_START.md) | Inicio rápido |
| [checklist.md](checklist.md) | Estado de implementación |
| [ARCHITECTURE_REVIEW_AND_PLAN.md](ARCHITECTURE_REVIEW_AND_PLAN.md) | Arquitectura |
| [COUNCIL.md](COUNCIL.md) | Sistema Council |
| [RISK_MANAGEMENT_GUIDE.md](RISK_MANAGEMENT_GUIDE.md) | Gestión de riesgo |

---

## � Auditoría de Código (14 de Enero 2026)

### Problemas Encontrados y Corregidos

| # | Severidad | Problema | Archivo | Fix |
|---|-----------|----------|---------|-----|
| 1 | 🔴 CRÍTICO | Código duplicado `_record_trade()` | backtester_core.py | Eliminado duplicado (L412-460) |
| 2 | 🔴 CRÍTICO | `fillna(method="bfill")` deprecated | indicators.py | Cambiado a `.bfill()` |
| 3 | 🔴 CRÍTICO | `fillna(method="ffill")` deprecated | backend_core.py | Cambiado a `.ffill()` |
| 4 | 🟠 ALTO | Lógica de `side` basada en PnL incorrecta | backtester_core.py | Usar `direction` de VectorBT |
| 5 | 🟠 ALTO | Sortino con división por cero si `std==0` | backtester_core.py | Validación `downside_std > 0` |
| 6 | 🟠 ALTO | Information Ratio siempre = 0 | backtester_core.py | Calcular buy-and-hold correcto |
| 7 | 🟠 ALTO | `except:` silencioso en Council | council.py | Capturar `(TypeError, ValueError)` |
| 8 | 🟠 ALTO | Race condition en kill switch | risk_manager.py | Try/except para FileNotFoundError |
| 9 | 🟡 MEDIO | `auto_correct=False` ignorado en OHLC | backend_core.py | Raise ValueError en lugar de auto-corregir |

### Problemas Pendientes (Severidad Media/Baja)

- **BacktesterCore** tiene ~1500 líneas (refactorizar en clases más pequeñas)
- Magic numbers (risk_free_rate=0.04, etc.) deberían ir a config
- Type hints incompletos en varios métodos
- Comentarios mezclados español/inglés

---

## 🔧 Configuración

### Python Environment
```bash
# Activar entorno
.venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Variables de Entorno
```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

---

**Responsable:** Sistema TradingIA  
**Versión:** 2.1 (Post-Auditoría)  
**Última revisión técnica:** 14 de Enero 2026
