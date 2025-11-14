# Data Flow - Multi-Timeframe BTC Trading System

## 🌊 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION LAYER                               │
│  Alpaca API v2 (alpaca-py) → 5Min/15Min/1H BTCUSD bars                  │
│  Rate Limit: 200 req/min → 0.35s delay between calls                    │
│  Caching: CSV files in data/ directory                                  │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     MULTI-TIMEFRAME RESAMPLE                             │
│  1H  → Resample to 5Min (forward-fill)    [EMA200_1h, POC_1h, vol_1h]  │
│  15Min → Resample to 5Min (forward-fill)  [EMA50_15m]                   │
│  5Min → Native resolution                  [OHLCV, indicators]          │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     CROSS-TIMEFRAME FILTERS                              │
│  uptrend_1h = close_5m > EMA200_1h (resampled from 1H)                  │
│  momentum_15m = EMA20_5m > EMA50_15m (resampled from 15Min)             │
│  vol_cross = (vol_5m > 1.2*SMA21_5m) & (vol_5m > SMA_vol_1h)           │
│                                                                          │
│  bull_filter = uptrend_1h & momentum_15m & vol_cross                    │
│  bear_filter = NOT uptrend_1h & vol_cross                               │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     INDICATOR CALCULATION                                │
│  IFVG Enhanced:                                                          │
│    - Gap detection: low > high[2]                                        │
│    - ATR filter: gap_size > atr_multi * ATR                             │
│    - Mitigation tracking (lookback 50)                                  │
│    - Strength scoring: gap_size / ATR                                   │
│                                                                          │
│  Volume Profile Advanced:                                                │
│    - 120 price bins (OHLC range)                                        │
│    - Up/down volume separation                                          │
│    - POC (max volume bin)                                               │
│    - VAH/VAL (70% value area)                                           │
│    - SD zones (threshold 0.12)                                          │
│    - Cross-TF: POC_1h resampled to 5min                                 │
│                                                                          │
│  EMAs Multi-TF:                                                          │
│    - Entry: EMA18, EMA48 on 5min                                        │
│    - Momentum: EMA21, EMA50 on 15min                                    │
│    - Trend: EMA95, EMA200 on 1H                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     SIGNAL GENERATION                                    │
│  Bull Signal (Composite):                                                │
│    ✓ IFVG bull gap (strength > 0.5)                                     │
│    ✓ uptrend_1h (MANDATORY)                                             │
│    ✓ momentum_15m (MANDATORY)                                           │
│    ✓ vol_cross (high volume confirmed)                                  │
│    ✓ close > VAL_5m (above value area low)                              │
│    ✓ abs(close - POC_1h) < 0.5*ATR_1h (near key level)                 │
│                                                                          │
│  Bear Signal (Composite):                                                │
│    ✓ IFVG bear gap (strength > 0.5)                                     │
│    ✓ NOT uptrend_1h (MANDATORY downtrend)                               │
│    ✓ vol_cross (high volume confirmed)                                  │
│    ✓ close < VAH_5m (below value area high)                             │
│                                                                          │
│  Output: bull_filtered, bear_filtered with confidence scores            │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION LAYER                                   │
│  Walk-Forward Analysis:                                                  │
│    - Split data into 6 periods (3 months each)                          │
│    - Train 70%, Test 30% out-of-sample                                  │
│    - Optimize on train using Bayesian                                   │
│    - Validate on test (Calmar > 2.0)                                    │
│                                                                          │
│  Bayesian Optimization (skopt):                                          │
│    - Parameter space: atr_multi, vol_thresh, ema_lengths, tp_rr, etc   │
│    - n_calls = 100 evaluations                                          │
│    - Acquisition: Expected Improvement                                  │
│    - Objective: Maximize Calmar ratio                                   │
│                                                                          │
│  Monte Carlo Simulation:                                                 │
│    - 500 runs with +/-10% price/vol noise                               │
│    - Measure robustness: Sharpe std < 0.1                               │
│                                                                          │
│  Stress Tests:                                                           │
│    - high_vol (+50%), bear (-30%), crash (-20%), low_vol, whipsaw      │
│    - Survival threshold: DD < 20%                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     BACKTESTING ENGINE                                   │
│  Entry Logic:                                                            │
│    - Market order on bull_filtered or bear_filtered                     │
│    - Position size: risk_amt / (SL distance in $)                       │
│    - Max 5% capital exposure per trade                                  │
│                                                                          │
│  Risk Management:                                                        │
│    - Stop Loss: 1.5 * ATR_5m (adjusted by HTF vol)                      │
│    - Take Profit: 2.2 * risk (risk/reward)                              │
│    - Trailing: Start after +1R, delta 0.5R                              │
│                                                                          │
│  Metrics Calculation:                                                    │
│    - Win Rate, Profit Factor, Sharpe (rf=0.04)                          │
│    - Calmar Ratio, Max Drawdown, Recovery Factor                        │
│    - HTF Alignment % (trades following uptrend_1h)                      │
│                                                                          │
│  Output: trades.csv, results.json, equity_curve.png                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     PAPER TRADING ENGINE                                 │
│  Real-Time Data:                                                         │
│    - Alpaca WebSocket for live 5min bars                                │
│    - Fetch 1H/15Min context every 5min                                  │
│                                                                          │
│  Signal Monitoring:                                                      │
│    - Calculate indicators on latest multi-TF data                       │
│    - Generate signals with filters                                      │
│    - Execute market orders with bracket SL/TP                           │
│                                                                          │
│  Emergency Rules:                                                        │
│    - Close all if DD > 10%                                              │
│    - Close position if HTF trend reversal (EMA200_1h cross)             │
│    - Max 3 concurrent positions                                         │
│                                                                          │
│  Logging: paper_trades.csv with HTF_flag column                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     PINE SCRIPT EXPORT                                   │
│  Generate from best_params:                                              │
│    - optimized_indicator.pine (IFVG + VP + EMAs)                        │
│    - optimized_strategy.pine (full system with alerts)                  │
│                                                                          │
│  Multi-TF Implementation:                                                │
│    - request.security() for 1H/15Min data                               │
│    - input() for optimized parameters                                   │
│    - plotshape() for filtered signals                                   │
│                                                                          │
│  Output: scripts_pine/ directory                                        │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     DASHBOARD VISUALIZATION                              │
│  Streamlit Multi-Page App:                                               │
│                                                                          │
│  Page 1 - Backtest Results:                                             │
│    - Equity curve with HTF trend shading                                │
│    - Metrics table (Sharpe, Calmar, DD, etc)                            │
│    - HTF alignment indicator                                            │
│    - Trade distribution by TF bias                                      │
│                                                                          │
│  Page 2 - Optimization:                                                  │
│    - Bayesian parameter heatmap                                         │
│    - Monte Carlo distribution plots                                     │
│    - Walk-forward degradation analysis                                  │
│    - Stress test survival chart                                         │
│                                                                          │
│  Page 3 - Multi-TF Analysis:                                             │
│    - Candlestick chart with indicator overlays                          │
│    - HTF bias indicator (1H trend)                                      │
│    - MTF momentum gauge (15Min)                                         │
│    - Vol cross-TF status                                                │
│                                                                          │
│  Page 4 - Live Paper Trading:                                            │
│    - Real-time PnL chart                                                │
│    - Current signals with confidence                                    │
│    - Alert log with HTF changes                                         │
│    - Emergency rule status                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Structures

### **Multi-TF DataFrame Structure**

```python
# df_5m (Entry Timeframe)
columns = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',  # OHLCV
    'EMA18', 'EMA48',                                        # Entry EMAs
    'ATR', 'ifvg_bull', 'ifvg_bear', 'ifvg_strength',       # IFVG
    'POC', 'VAH', 'VAL', 'vol_up', 'vol_down',              # Volume Profile
    'EMA200_1h', 'POC_1h', 'SMA_vol_1h',                    # 1H resampled
    'EMA50_15m',                                             # 15Min resampled
    'uptrend_1h', 'momentum_15m', 'vol_cross',              # Cross-TF filters
    'bull_filter', 'bear_filter',                           # Combined filters
    'bull_filtered', 'bear_filtered', 'confidence'          # Final signals
]

# df_15m (Momentum Timeframe)
columns = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'EMA21', 'EMA50'                                         # Momentum EMAs
]

# df_1h (Trend Timeframe)
columns = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'EMA95', 'EMA200',                                       # Trend EMAs
    'POC', 'VAH', 'VAL',                                     # VP for key levels
    'SMA_vol'                                                # Vol baseline
]
```

### **Trades CSV Structure**

```csv
timestamp,symbol,direction,entry_price,exit_price,sl,tp,pnl,pnl_pct,duration,uptrend_1h,momentum_15m,vol_cross,ifvg_strength,vp_proximity
2024-01-05 10:30:00,BTCUSD,long,42150.5,42580.2,41900,42700,429.7,1.02,45min,True,True,True,0.62,0.35
2024-01-05 14:15:00,BTCUSD,long,42480.0,42100.0,42200,43100,-380.0,-0.89,30min,True,True,True,0.48,0.55
...
```

### **Results JSON Structure**

```json
{
  "optimization": {
    "method": "bayesian",
    "n_calls": 100,
    "best_params": {
      "atr_multi": 0.35,
      "vol_thresh": 1.15,
      "ema1_entry": 18,
      "ema2_entry": 48,
      "tp_rr": 2.2,
      "va_percent": 0.70
    }
  },
  "backtest_metrics": {
    "sharpe_ratio": 1.15,
    "calmar_ratio": 2.35,
    "max_drawdown": -12.5,
    "win_rate": 0.62,
    "profit_factor": 1.85,
    "htf_alignment": 0.73,
    "total_trades": 145,
    "avg_win": 1.25,
    "avg_loss": -0.95
  },
  "walk_forward": {
    "periods": 6,
    "train_sharpe": [1.20, 1.15, 1.18, 1.22, 1.17, 1.19],
    "test_sharpe": [1.10, 1.08, 1.12, 1.15, 1.11, 1.13],
    "degradation": 0.07
  },
  "monte_carlo": {
    "runs": 500,
    "sharpe_mean": 1.15,
    "sharpe_std": 0.08,
    "calmar_mean": 2.30,
    "calmar_std": 0.15,
    "robustness_score": 0.92
  },
  "stress_tests": {
    "high_vol": {"dd": -18.5, "sharpe": 0.85},
    "bear_market": {"dd": -16.2, "sharpe": 0.95},
    "flash_crash": {"dd": -19.8, "sharpe": 0.78},
    "low_vol": {"dd": -8.5, "sharpe": 0.92},
    "whipsaw": {"dd": -14.2, "sharpe": 0.88}
  }
}
```

---

## 🔄 Interconnections Deep Dive

### **1. HTF Trend → Entry Signals**

```
┌──────────────┐
│ 1H: EMA200   │ Calculate EMA200 on 1H close
└──────┬───────┘
       │ Resample (forward-fill)
       ↓
┌──────────────┐
│ 5Min: close  │ Compare close_5m with EMA200_1h
└──────┬───────┘
       │
       ↓
uptrend_1h = close_5m > EMA200_1h
       │
       ↓ FILTERS
┌──────────────┐
│ Bull Signals │ Only if uptrend_1h == True
└──────────────┘
```

**Why Critical?**
- BTC en 1H uptrend: 70% más probabilidad de longs exitosos
- Reduce contra-trend trades que suelen fallar
- Aligns con flujo institucional (grandes órdenes en HTF)

**Example:**
```python
# Timestamp: 2024-01-05 10:30:00
close_5m = 42150.5
EMA200_1h = 41800.0  # Resampled from 1H to 5Min (forward-fill)
uptrend_1h = True    # 42150.5 > 41800.0

# IFVG bull detected at 10:30
# Without HTF filter: Signal generated
# With HTF filter: Signal + uptrend_1h = TRADE ✓

# Later: 2024-01-05 16:45:00
EMA200_1h crosses above close_5m
uptrend_1h = False   # HTF reversal

# Emergency rule: Close long immediately
# Saves from riding downtrend
```

---

### **2. MTF Momentum → Signal Confirmation**

```
┌──────────────┐         ┌──────────────┐
│ 15Min: EMA50 │         │ 5Min: EMA20  │
└──────┬───────┘         └──────┬───────┘
       │ Resample               │ Native
       ↓                        ↓
┌──────────────────────────────────┐
│ Compare on 5Min resolution       │
│ momentum_15m = EMA20 > EMA50_15m │
└──────┬───────────────────────────┘
       │
       ↓ CONFIRMATION
┌──────────────┐
│ Bull Signals │ Only if momentum_15m == True
└──────────────┘
```

**Why Matters?**
- 5Min puede generar señales en rangos (noise)
- 15Min momentum confirm que hay fuerza sostenida
- Reduce falsas señales en consolidación

**Example:**
```python
# Scenario: BTC ranging 42k-42.5k on 5Min
# 5Min: EMA20 crosses EMA48 → bull signal
# BUT: 15Min still in downtrend (EMA21 < EMA50)

momentum_15m = False  # No confirmation

# Result: Signal filtered out
# Saves from range trade que likely fails

# Later: 15Min confirms uptrend
momentum_15m = True
# Next 5Min signal → TRADES
```

---

### **3. Vol Cross-TF → Genuine Breakouts**

```
┌───────────────┐        ┌───────────────┐
│ 5Min: volume  │        │ 1H: volume    │
└───────┬───────┘        └───────┬───────┘
        │                        │
        ↓ SMA21                  ↓ SMA
┌───────────────┐        ┌───────────────┐
│ vol_sma_5m    │        │ vol_sma_1h    │
└───────┬───────┘        └───────┬───────┘
        │                        │ Resample
        │                        ↓
        └────────┬───────────────┘
                 ↓
vol_cross = (vol_5m > 1.2*vol_sma_5m) AND (vol_5m > vol_sma_1h)
                 │
                 ↓ FILTERS NOISE
┌────────────────────────┐
│ All Signals            │ Only if vol_cross == True
└────────────────────────┘
```

**Why Essential?**
- Vol spike en 5Min puede ser ruido (thin liquidity)
- Vol 1H confirm que es genuino (institucional flow)
- Evita trades en low liquidity moves que reversan rápido

**Example:**
```python
# Timestamp: 2024-01-08 09:35:00 (Asian session, low liquidity)
vol_5m = 150 BTC
vol_sma_5m = 100 BTC  # 150 > 1.2*100 ✓
vol_sma_1h = 180 BTC  # 150 < 180 ✗

vol_cross = False  # No genuine breakout

# IFVG signal detected but vol_cross False
# Result: Filtered out
# Saves from low liquidity fake breakout

# Later: 2024-01-08 14:20:00 (NY session)
vol_5m = 250 BTC
vol_sma_1h = 200 BTC  # 250 > 200 ✓

vol_cross = True  # Genuine institutional flow
# Next IFVG signal → TRADES
```

---

### **4. Volume Profile POC → Key Levels**

```
┌────────────────┐
│ 1H: VP calc    │ POC_1h (price with max volume)
└────────┬───────┘
         │ Resample to 5Min
         ↓
┌────────────────┐
│ 5Min: close    │ abs(close - POC_1h)
└────────┬───────┘
         │
         ↓ PROXIMITY CHECK
┌────────────────────────────┐
│ abs(close - POC_1h) < 0.5*ATR_1h
└────────┬───────────────────┘
         │
         ↓ FILTERS
┌────────────────┐
│ Signal Quality │ Higher confidence near POC_1h
└────────────────┘
```

**Why POC_1h?**
- POC = price con más volumen = zona de balance
- Precio cerca POC tiene alta prob de reacción (support/resistance)
- Cross-TF POC más robusto que 5Min (menos noise)

**Example:**
```python
# Timestamp: 2024-01-10 11:15:00
close_5m = 43200
POC_1h = 43180  # Resampled from 1H VP to 5Min
ATR_1h = 500

proximity = abs(43200 - 43180) = 20
threshold = 0.5 * 500 = 250

# 20 < 250 → Near POC_1h ✓

# IFVG bull signal detected
# Confidence boosted: 0.75 → 0.85
# Higher position size allocated
# Result: Trade with better setup quality
```

---

## 🎯 Signal Flow Example (Full Trace)

### **Timestamp: 2024-02-15 10:30:00**

```
Step 1: DATA ACQUISITION
├─ Alpaca API: Download 5Min/15Min/1H bars
├─ Cache: btcusd_5Min_2024-01-01_2024-03-01.csv
└─ Validation: OHLC valid, volume > 0, nulls filled

Step 2: MULTI-TF RESAMPLE
├─ 1H data:
│   └─ EMA200_1h = 42500, POC_1h = 42650, SMA_vol_1h = 200 BTC
│   └─ Resample to 5Min (forward-fill)
└─ 15Min data:
    └─ EMA50_15m = 42700
    └─ Resample to 5Min (forward-fill)

Step 3: CROSS-TF FILTERS
├─ close_5m = 42850
├─ uptrend_1h = 42850 > 42500 → TRUE ✓
├─ EMA20_5m = 42750, EMA50_15m = 42700
├─ momentum_15m = 42750 > 42700 → TRUE ✓
├─ vol_5m = 250 BTC, SMA21_5m = 180 BTC, SMA_vol_1h = 200 BTC
├─ vol_cross = (250 > 1.2*180) AND (250 > 200) → TRUE ✓
└─ bull_filter = TRUE & TRUE & TRUE = TRUE ✓✓✓

Step 4: INDICATORS
├─ IFVG Detection:
│   ├─ low[i] = 42820, high[i-2] = 42680
│   ├─ gap_size = 42820 - 42680 = 140
│   ├─ ATR = 250
│   ├─ 140 > 0.3*250 = 75 → Gap valid ✓
│   ├─ strength = 140/250 = 0.56 > 0.5 → High strength ✓
│   └─ ifvg_bull = TRUE
├─ Volume Profile:
│   ├─ POC_5m = 42820, VAL_5m = 42650
│   ├─ close > VAL → Above support ✓
│   └─ proximity = abs(42850 - 42650) = 200 < 0.5*500 = 250 → Near POC_1h ✓
└─ EMAs:
    ├─ EMA18_5m = 42780, EMA48_5m = 42650
    └─ EMA18 > EMA48 → Entry uptrend ✓

Step 5: SIGNAL GENERATION
├─ bull_signal_ifvg = TRUE (IFVG + strength)
├─ bull_filter = TRUE (HTF + MTF + Vol)
├─ vp_support = TRUE (close > VAL)
├─ vp_proximity = TRUE (near POC_1h)
└─ bull_filtered = TRUE ✓✓✓✓✓

Step 6: CONFIDENCE SCORING
├─ Base confidence: 0.60
├─ + IFVG strength > 0.5: +0.10 → 0.70
├─ + HTF alignment: +0.05 → 0.75
├─ + Vol cross confirmed: +0.05 → 0.80
├─ + Near POC_1h: +0.10 → 0.90
└─ Final confidence: 0.90 (EXCELLENT)

Step 7: RISK MANAGEMENT
├─ Entry price: 42850
├─ ATR_5m = 250
├─ Stop Loss: 42850 - (1.5 * 250) = 42475
├─ Risk per share: 375
├─ Account: $10,000, risk 1% = $100
├─ Position size: 100 / 375 = 0.27 BTC (rounded to 0.25)
├─ Take Profit: 42850 + (375 * 2.2) = 43675
└─ Trailing: Start at 42850 + 375 = 43225 (+1R)

Step 8: EXECUTION (Backtest)
├─ Buy 0.25 BTC at 42850
├─ Set bracket order: SL 42475, TP 43675
└─ Monitor for trailing activation

Step 9: OUTCOME (Example)
├─ Price reaches 43225 → Trailing activated
├─ Trailing delta: 0.5R = 187.5
├─ New SL: 43225 - 187.5 = 43037.5
├─ Price hits 43650, pulls back
├─ Trailing SL hit at 43465
├─ Exit: 43465
├─ PnL: (43465 - 42850) * 0.25 = 153.75
├─ PnL %: 1.44%
├─ Risk/Reward achieved: 1.64R
└─ uptrend_1h during trade: TRUE ✓

Step 10: LOGGING
├─ trades.csv: timestamp, symbol, long, 42850, 43465, ...
├─ metrics: win, uptrend_1h=TRUE, momentum_15m=TRUE
└─ HTF alignment counter: +1
```

**Result:** Successful trade with high confidence (0.90), followed HTF bias, confirmed by all filters, achieved 1.64R profit.

---

## 🔧 Optimization Data Flow

### **Walk-Forward Analysis**

```
DATA: 18 months (2023-06 to 2024-12)

┌─────────────────────────────────────────────────────┐
│ Split into 6 periods (3 months each)               │
├─────────────────────────────────────────────────────┤
│ Period 1: 2023-06 to 2023-08                       │
│   ├─ Train (70%): 2023-06-01 to 2023-07-21        │
│   └─ Test (30%):  2023-07-22 to 2023-08-31        │
│                                                     │
│ Period 2: 2023-09 to 2023-11                       │
│   ├─ Train: 2023-09-01 to 2023-10-21              │
│   └─ Test:  2023-10-22 to 2023-11-30              │
│ ...                                                 │
│ Period 6: 2024-09 to 2024-11                       │
│   ├─ Train: 2024-09-01 to 2024-10-21              │
│   └─ Test:  2024-10-22 to 2024-11-30              │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ For each period:                                    │
│   1. Optimize on Train using Bayesian              │
│      ├─ n_calls = 100                              │
│      ├─ Objective: Maximize Calmar                 │
│      └─ Output: best_params_train                  │
│                                                     │
│   2. Validate on Test (out-of-sample)              │
│      ├─ Run backtest with best_params_train        │
│      ├─ Measure: Calmar_test, Sharpe_test, DD_test│
│      └─ Check: Calmar_test > 2.0                   │
│                                                     │
│   3. Calculate degradation                         │
│      └─ deg = (metric_train - metric_test) / metric_train
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Aggregate Results:                                  │
│   ├─ Avg degradation across periods: < 15%        │
│   ├─ Min Calmar_test: > 1.8                        │
│   └─ HTF alignment consistency: > 68%              │
│                                                     │
│ Select final params:                                │
│   └─ Period with best Calmar_test (most recent)   │
└─────────────────────────────────────────────────────┘
```

---

## 📈 Live Trading Data Flow

```
┌──────────────────────────────────────────────────┐
│ INITIALIZATION                                   │
│   ├─ Load best_params from results.json         │
│   ├─ Connect Alpaca Paper API                   │
│   └─ Subscribe WebSocket for BTCUSD bars        │
└────────────┬─────────────────────────────────────┘
             │
             ↓ Every 5 minutes
┌──────────────────────────────────────────────────┐
│ DATA UPDATE                                      │
│   ├─ WebSocket: New 5Min bar received           │
│   ├─ API: Fetch latest 15Min/1H bars            │
│   └─ Append to rolling DataFrame (last 500)     │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│ CROSS-TF FILTERS UPDATE                          │
│   ├─ Recalc EMA200_1h, resample to 5Min         │
│   ├─ Check uptrend_1h status                    │
│   ├─ Recalc EMA50_15m, resample to 5Min         │
│   ├─ Check momentum_15m status                  │
│   └─ Check vol_cross status                     │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│ INDICATOR CALCULATION                            │
│   ├─ IFVG detection on latest bars              │
│   ├─ Volume Profile update (rolling 120 bars)   │
│   └─ EMAs update                                 │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│ SIGNAL GENERATION                                │
│   ├─ Generate bull_filtered / bear_filtered     │
│   ├─ Calculate confidence score                 │
│   └─ Check max 3 positions limit                │
└────────────┬─────────────────────────────────────┘
             │
             ↓ If signal and no max positions
┌──────────────────────────────────────────────────┐
│ ORDER EXECUTION                                  │
│   ├─ Calculate position size (1% risk)          │
│   ├─ Submit market order                        │
│   ├─ Submit bracket SL/TP orders                │
│   └─ Log to paper_trades.csv                    │
└────────────┬─────────────────────────────────────┘
             │
             ↓ Continuous monitoring
┌──────────────────────────────────────────────────┐
│ POSITION MANAGEMENT                              │
│   ├─ Check trailing stop activation (+1R)       │
│   ├─ Update trailing SL if price advances       │
│   ├─ Monitor HTF reversal (EMA200_1h cross)     │
│   └─ Check emergency DD > 10%                   │
└────────────┬─────────────────────────────────────┘
             │
             ↓ If emergency condition
┌──────────────────────────────────────────────────┐
│ EMERGENCY CLOSE                                  │
│   ├─ Close all positions market                 │
│   ├─ Log emergency event                        │
│   └─ Alert user via email/Telegram              │
└──────────────────────────────────────────────────┘
```

---

**Última actualización**: 2025-11-12
**Versión**: 1.0
