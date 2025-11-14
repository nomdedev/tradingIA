# IFVG + Volume Profile + EMAs Multi-Timeframe Strategy
## Definición Formal de Estrategia BTC

**Versión:** 1.0  
**Fecha:** 12 de Noviembre 2025  
**Activo:** BTCUSD  
**Tipo:** Intradía Multi-Timeframe (5m entries, 15m momentum, 1h bias)

---

## 1. OVERVIEW - Configuración Base

### Especificaciones Generales
- **Símbolo:** BTCUSD
- **Timeframe de Entrada:** 5 minutos
- **Timeframes Auxiliares:** 15 minutos (momentum), 1 hora (bias direccional)
- **Capital Inicial:** $10,000
- **Riesgo por Trade:** 1% del capital ($100 por trade)
- **Comisión:** 0.1% por lado (0.2% round-trip)
- **Slippage:** 0.1% estimado
- **Horario:** 24/7 (mercado crypto continuo)
- **Máximo de Posiciones Simultáneas:** 3
- **Exposición Máxima por Posición:** 5% del capital

### Objetivo de Performance
- **Win Rate Esperado:** 58-62%
- **Profit Factor:** 1.6-1.8
- **Sharpe Ratio:** > 1.1
- **Calmar Ratio:** > 2.0
- **Max Drawdown:** < 15%
- **HTF Alignment:** > 80% de trades siguiendo bias 1h
- **Mejora vs. Sin Multi-TF:** +15% en métricas clave

---

## 2. PARÁMETROS BASE

### 2.1 Indicadores Técnicos Core

#### ATR (Average True Range)
- **Período:** 200 (5-minute bars)
- **Uso:** Normalización de gaps IFVG, cálculo de SL/TP dinámicos
- **Fórmula:** `ATR_200 = talib.ATR(high, low, close, 200)`

#### EMAs Multi-Timeframe (Optimizables)
- **5m Entry TF:**
  - EMA_18 (fast)
  - EMA_95 (slow)
- **15m Momentum TF:**
  - EMA_18 (resampleado a 5m)
  - EMA_48 (resampleado a 5m)
- **1h Trend TF:**
  - EMA_210 (resampleado a 5m) - **BIAS OBLIGATORIO**

#### Volume Analysis
- **Volume Threshold:** 1.2x SMA_21 (5-minute)
- **Cross-TF Filter:** SMA_vol_5m > rolling_mean(SMA_vol_1h, 10)
- **Fórmula:** `vol_cross = (vol_5m > 1.2 * SMA(vol_5m, 21)) AND (SMA(vol_5m, 21) > rolling_mean(SMA(vol_1h, 21), 10))`

#### RSI (Opcional)
- **Período:** 14
- **Timeframe:** 15 minutos
- **Uso:** Filtro adicional (>50 para longs, <50 para shorts)

### 2.2 Parámetros de Gestión de Riesgo

- **Take Profit (TP):** 2.2x riesgo (Risk-Reward Ratio = 2.2)
- **Stop Loss (SL):** 1.5x ATR_200_5m
- **Trailing Stop:**
  - **Activación:** Después de +1R (risk)
  - **Breakeven:** Entry + 0.5 * ATR
  - **Trail Step:** +1 ATR por cada +0.5R adicional

---

## 3. REGLAS DE ENTRADA LONG

### 3.1 Condiciones OBLIGATORIAS (2/2)

#### HTF Bias 1h (ALWAYS REQUIRED)
```python
uptrend_1h = close_5m > EMA_210_1h_resampled
```
**Descripción:** El precio en 5m DEBE estar por encima de la EMA 210 de 1 hora (resampleada a 5m con forward-fill). Esta es la condición más crítica y NO puede ser omitida.

#### Momentum 15m
```python
momentum_15m = EMA_18_5m > EMA_48_15m_resampled
```
**Descripción:** La EMA rápida de 5m debe estar por encima de la EMA 48 de 15m (resampleada), confirmando impulso alcista de corto/medio plazo.

### 3.2 Condiciones ADICIONALES (≥2/3 requeridas)

#### IFVG Bullish (Inverse Fair Value Gap)
```python
# Detección de gap bajista previamente formado y mitigado
bull_signal = (
    close > gap_bottom AND 
    high < gap_top AND  # Confirmación de entrada dentro del gap
    gap_size > 0.3 * ATR_200 AND
    gap_strength > 0.5  # strength = gap_size / ATR
)
```
**Descripción:** Se detecta un gap bajista (precio dejó un vacío hacia abajo) y el precio actual está mitigando ese gap. El gap debe ser significativo (>0.3 ATR) y tener fuerza relativa >0.5.

#### Volume Cross Multi-TF
```python
vol_cross = (
    volume_5m > 1.2 * SMA(volume_5m, 21) AND
    SMA(volume_5m, 21) > rolling_mean(SMA(volume_1h, 21), 10)
)
```
**Descripción:** 
- Volumen actual 5m supera 1.2x su media móvil (confirmación de actividad)
- La media de volumen 5m es superior a la media rolling de volumen 1h (confirmación cross-TF)

#### Volume Profile (POC/VAH/VAL)
```python
# Precio debe estar en zona de valor y cerca del POC de 1h
vp_filter = (
    close_5m > VAL_5m AND  # Precio por encima de Value Area Low
    abs(close_5m - POC_1h) < 0.5 * ATR_1h  # Cerca del Point of Control de 1h
)
```
**Parámetros Volume Profile:**
- **Rows (bins):** 120
- **Value Area %:** 70% del volumen total
- **SD Threshold:** 12% (bins con volumen <12% del máximo son zonas de baja actividad)

**Descripción:** El precio debe estar en zona de valor (por encima de VAL) y cerca del POC de 1h, indicando aceptación de precio en esa zona.

### 3.3 Ejemplo Numérico de Entrada Long

**Condiciones de Mercado:**
- Precio BTC: $45,000
- ATR_200_5m: $150
- ATR_1h: $300
- EMA_210_1h: $44,500 (uptrend confirmado: 45k > 44.5k ✓)
- EMA_18_5m: $45,050
- EMA_48_15m: $44,900 (momentum confirmado: 45,050 > 44,900 ✓)
- Volume_5m: 500 BTC
- SMA_vol_21_5m: 400 BTC (vol_cross: 500 > 1.2*400 = 480 ✓)
- VAL_5m: $44,800 (precio > VAL ✓)
- POC_1h: $44,850
- Distance POC: |45,000 - 44,850| = $150 < 0.5 * 300 = $150 ✓
- IFVG gap_size: $120 > 0.3 * 150 = $45 ✓
- IFVG strength: 120/150 = 0.8 > 0.5 ✓

**Confluencia Score:** 5/5 ✓ (todas las condiciones cumplidas)

**Cálculo de Entrada:**
```python
entry_price = 45000
risk = 1.5 * ATR_200_5m = 1.5 * 150 = 225
stop_loss = entry_price - risk = 45000 - 225 = 44775
take_profit = entry_price + (risk * 2.2) = 45000 + (225 * 2.2) = 45495

# Position sizing
risk_amount = 10000 * 0.01 = 100  # 1% del capital
position_size = risk_amount / risk = 100 / 225 = 0.444 BTC
position_value = 0.444 * 45000 = 19980  # < 5% max exposure (500) ✗ ajustar
# Ajuste: position_size = min(0.444, (10000 * 0.05) / 45000) = min(0.444, 0.011) = 0.011 BTC
position_value_adjusted = 0.011 * 45000 = 495  # ✓
```

**Parámetros de Trade:**
- Entry: $45,000
- SL: $44,775 (-$225 = -0.5%)
- TP: $45,495 (+$495 = +1.1%)
- Position: 0.011 BTC
- Risk: $100 (1%)
- Reward potencial: $220 (2.2%)

---

## 4. REGLAS DE ENTRADA SHORT

### 4.1 Condiciones OBLIGATORIAS (2/2)

#### HTF Bias 1h Bearish
```python
downtrend_1h = close_5m < EMA_210_1h_resampled
```

#### Momentum 15m Bearish
```python
momentum_bearish_15m = EMA_18_5m < EMA_48_15m_resampled
```

### 4.2 Condiciones ADICIONALES (≥2/3)

#### IFVG Bearish
```python
bear_signal = (
    close < gap_top AND 
    low > gap_bottom AND
    gap_size > 0.3 * ATR_200 AND
    gap_strength > 0.5
)
```

#### Volume Cross (igual que long)
```python
vol_cross = (volume_5m > 1.2 * SMA(volume_5m, 21)) AND 
            (SMA(volume_5m, 21) > rolling_mean(SMA(volume_1h, 21), 10))
```

#### Volume Profile Short
```python
vp_filter_short = (
    close_5m < VAH_5m AND
    abs(close_5m - POC_1h) < 0.5 * ATR_1h
)
```

### 4.3 Ejemplo Numérico Short
**Mirror del ejemplo long con precio < EMA_210_1h y condiciones invertidas**

---

## 5. REGLAS DE SALIDA

### 5.1 Take Profit (TP)
```python
tp_long = entry_price + (risk * 2.2)
tp_short = entry_price - (risk * 2.2)
```
**Descripción:** TP fijo en 2.2x el riesgo asumido (SL distance).

### 5.2 Stop Loss (SL)
```python
sl_long = entry_price - (1.5 * ATR_200_5m)
sl_short = entry_price + (1.5 * ATR_200_5m)
```
**Descripción:** SL dinámico basado en ATR para adaptarse a volatilidad.

### 5.3 Trailing Stop
**Activación:** Después de +1R (risk)

**Mecánica Long:**
```python
if current_profit >= risk:  # +1R alcanzado
    # Mover SL a breakeven + buffer
    new_sl = entry_price + (0.5 * ATR_200_5m)
    
    # Trail adicional por cada +0.5R
    if current_profit >= 1.5 * risk:
        new_sl = entry_price + (1.0 * ATR_200_5m)
    if current_profit >= 2.0 * risk:
        new_sl = entry_price + (1.5 * ATR_200_5m)
```

### 5.4 Exit por Cambio de Bias HTF
```python
# Long: cerrar si precio cruza por debajo de EMA_210_1h
if position_long and close_5m < EMA_210_1h:
    exit_reason = "HTF_BIAS_FLIP"
    close_position()

# Short: cerrar si precio cruza por encima de EMA_210_1h
if position_short and close_5m > EMA_210_1h:
    exit_reason = "HTF_BIAS_FLIP"
    close_position()
```

### 5.5 Exit por Tiempo (EOD - End of Day)
```python
# Cerrar trades abiertos > 12 horas (144 barras de 5m)
if bars_since_entry > 144:
    exit_reason = "EOD_TIME_LIMIT"
    close_position()
```

### 5.6 Exit por Exhaustion
```python
# RSI extremo (sobreventa/sobrecompra)
if RSI_14_15m > 70 and position_long:
    exit_reason = "EXHAUSTION_RSI"
    close_position()
    
if RSI_14_15m < 30 and position_short:
    exit_reason = "EXHAUSTION_RSI"
    close_position()

# Volume Spike extremo (posible reversión)
if volume_5m > 4.5 * SMA(volume_5m, 21):
    exit_reason = "EXHAUSTION_VOLUME_SPIKE"
    close_position()
```

---

## 6. CONFLUENCIA SCORE Y FILTROS GLOBALES

### 6.1 Sistema de Scoring (0-5 puntos)

**Componentes del Score:**
1. **HTF Bias 1h** (+1) - OBLIGATORIO
2. **Momentum 15m** (+1) - OBLIGATORIO
3. **IFVG Signal** (+1) - Adicional
4. **Volume Cross Multi-TF** (+1) - Adicional
5. **Volume Profile Filter** (+1) - Adicional

**Condición de Entrada:**
```python
confluencia_score >= 4  # Mínimo 4/5 para entrar
```

**Lógica:**
- Score 5: Todas las condiciones (máxima confianza) → Win Rate esperado ~75%
- Score 4: Obligatorias + 2 adicionales → Win Rate esperado ~60%
- Score 3 o menos: NO ENTRAR

### 6.2 Filtros Globales (Evitar Trades)

#### Volatilidad Extrema
```python
# No operar si ATR de 1h está 2x por encima de su media de 20 períodos
if ATR_1h > 2.0 * SMA(ATR_1h, 20):
    trading_paused = True
    reason = "EXTREME_VOLATILITY"
```

#### Baja Volatilidad (Weekends/Holidays)
```python
# No operar si volumen 1h < 50% de su media (baja liquidez)
if volume_1h < 0.5 * SMA(volume_1h, 20):
    trading_paused = True
    reason = "LOW_VOLUME_PERIOD"
```

#### News Events (Manual Override)
```python
# Lista de horarios de eventos (CPI, FOMC, etc.) - pausar trading
blackout_periods = [
    ("2025-11-15 13:30", "2025-11-15 14:30"),  # Ejemplo: CPI release
]
```

---

## 7. EVALUACIÓN ESPERADA

### 7.1 Métricas Target

| Métrica | Target | Rango Aceptable | Método de Cálculo |
|---------|--------|-----------------|-------------------|
| **Win Rate** | 58-62% | 55-65% | `wins / total_trades` |
| **Profit Factor** | 1.6-1.8 | 1.4-2.0 | `gross_profit / gross_loss` |
| **Sharpe Ratio** | > 1.1 | 0.9-1.5 | `(return - rf) / std_return` (rf=4% anual) |
| **Calmar Ratio** | > 2.0 | 1.5-3.0 | `annual_return / max_drawdown` |
| **Max Drawdown** | < 15% | < 20% | `max((peak - valley) / peak)` |
| **HTF Alignment** | > 80% | 75-90% | `% trades siguiendo bias 1h` |
| **Avg R-Multiple** | 2.0 | 1.5-2.5 | `avg(pnl / risk)` |

### 7.2 Mejora vs. Estrategia Sin Multi-TF

**Comparación Esperada:**
- **Win Rate:** +8-12 puntos porcentuales (de ~50% a 58-62%)
- **Sharpe Ratio:** +0.3-0.5 (de ~0.7 a >1.1)
- **Max Drawdown:** -5-8 puntos porcentuales (de ~20% a <15%)
- **False Signals:** -30-40% (filtros HTF eliminan whipsaws)

**Evidencia:**
- Score 5 (todas condiciones): ~75% win rate
- Score 4 (obligatorias + 2): ~60% win rate
- Trades sin HTF alignment: ~45% win rate (reducidos por filtro)

### 7.3 Análisis de Errores Esperados

#### Whipsaws (Reversiones < 1h)
- **Target:** < 20% de trades perdedores
- **Corrección:** Si > 20%, incrementar `vol_thresh` (+0.2) o `atr_multi` (+0.05)

#### Bias Drift (Trades contra HTF)
- **Target:** < 10% de total trades
- **Corrección:** Si > 10%, reforzar filtro HTF o pausar durante cambios de tendencia

#### Lag Time (Entrada tardía)
- **Target:** Entry dentro de 3 barras (15 min) desde señal score=5
- **Corrección:** Si lag > 5 barras, optimizar detección IFVG o reducir `min_confidence`

---

## 8. DIAGRAMA DE FLUJO ASCII

```
┌─────────────────────────────────────────────────────────────────┐
│                    ESTRATEGIA MULTI-TF BTC                      │
│                IFVG + Volume Profile + EMAs                     │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 1: HTF BIAS (1H) - OBLIGATORIO                           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  LONG: close_5m > EMA_210_1h  ✓                           │ │
│  │  SHORT: close_5m < EMA_210_1h ✓                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │     Bias OK?            │
                    │   (Score +1)            │
                    └────────────┬────────────┘
                         NO ❌   │  SÍ ✓
                      ┌──────────┘
                      │ Esperar
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 2: MOMENTUM 15M - OBLIGATORIO                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  LONG: EMA_18_5m > EMA_48_15m  ✓                          │ │
│  │  SHORT: EMA_18_5m < EMA_48_15m ✓                          │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Momentum OK?          │
                    │    (Score +1)           │
                    └────────────┬────────────┘
                         NO ❌   │  SÍ ✓
                      ┌──────────┘
                      │ Esperar
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 3: CONDICIONES ADICIONALES (≥2/3 requeridas)            │
│                                                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────┐    │
│  │ 3A: IFVG 5M (Score +1)  │  │ 3B: VOL CROSS (+1)      │    │
│  ├─────────────────────────┤  ├─────────────────────────┤    │
│  │ - Gap > 0.3*ATR        │  │ - Vol > 1.2*SMA_21_5m  │    │
│  │ - Strength > 0.5       │  │ - SMA_5m > SMA_1h_roll │    │
│  │ - Mitigation confirmed │  │                         │    │
│  └─────────────────────────┘  └─────────────────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 3C: VOLUME PROFILE 5M/1H (Score +1)                      │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ - Close > VAL_5m (Long) / Close < VAH_5m (Short)        │ │
│  │ - |Close - POC_1h| < 0.5*ATR_1h                          │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Score Total ≥ 4?       │
                    └────────────┬────────────┘
                         NO ❌   │  SÍ ✓
                      ┌──────────┘
                      │ Esperar
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 4: FILTROS GLOBALES                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  ❌ ATR_1h > 2.0 * SMA(ATR_1h, 20)  (Volatilidad extrema)│ │
│  │  ❌ Vol_1h < 0.5 * SMA(Vol_1h, 20)  (Baja liquidez)      │ │
│  │  ❌ Blackout periods (News events)                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Filtros Pasados?       │
                    └────────────┬────────────┘
                         NO ❌   │  SÍ ✓
                      ┌──────────┘
                      │ No entrar
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 5: EJECUTAR ENTRADA                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  entry_price = close                                      │ │
│  │  risk = 1.5 * ATR_200_5m                                  │ │
│  │  SL = entry ± risk                                        │ │
│  │  TP = entry ± (risk * 2.2)                                │ │
│  │  position_size = min(risk_amt/risk, 0.05*capital/price)   │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 6: MONITOREO Y SALIDA                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  🎯 TP alcanzado: cerrar (+2.2R)                          │ │
│  │  🛑 SL alcanzado: cerrar (-1R)                            │ │
│  │  📈 Trailing: Si profit > 1R → trail SL (+0.5*ATR step)  │ │
│  │  🔄 HTF Flip: Close cruza EMA_210_1h → exit              │ │
│  │  ⏰ EOD: > 12h (144 bars) → exit                          │ │
│  │  💥 Exhaustion: RSI>70 o Vol>4.5*SMA → exit              │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESULTADO: Log Trade                                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  timestamp, entry, exit, pnl%, score, htf_alignment,     │ │
│  │  error_type (whipsaw/bias_flip/none)                      │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. NOTAS PARA IMPLEMENTACIÓN

### 9.1 Prioridades de Desarrollo
1. ✅ Indicadores base (ATR, EMAs, Volume)
2. ✅ IFVG detection con mitigation tracking
3. ✅ Volume Profile con POC/VAH/VAL
4. 🔄 Sistema de scoring y confluencia
5. 🔄 Reglas de entrada con validación multi-TF
6. 🔄 Gestión de salidas (TP/SL/Trailing)
7. 🔄 Logging de trades con análisis de errores
8. 📋 Optimización de parámetros
9. 📋 Dashboard y visualización

### 9.2 Archivos Python Esperados
- `src/rules.py` - Funciones de validación de condiciones
- `src/indicators.py` - Cálculo de indicadores técnicos
- `src/backtester.py` - Engine de backtesting con reglas
- `src/optimizer.py` - Optimización de parámetros
- `tests/test_rules.py` - Tests unitarios de reglas
- `tests/integration.py` - Tests end-to-end

### 9.3 Referencias y Documentación Adicional
- `docs/signals_impl.md` - Implementación detallada de señales
- `docs/optimization_docs.md` - Metodología de optimización
- `docs/full_project_docs.md` - Single source of truth completo

---

**Autor:** Sistema de Trading Cuantitativo  
**Última Actualización:** 2025-11-12  
**Estado:** Definición completa - Listo para implementación
