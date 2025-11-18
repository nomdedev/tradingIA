# Conditional Patterns Evaluation Report
**Dataset:** 100417 bars
**Forward bars:** 10
**Profit threshold:** 1.00%
**Total patterns evaluated:** 13

---

## 🏆 Top Patterns by Expectancy

### 1. IFVG_HighVol_EMA_Bullish

**Pattern Definition:**
```
IFVG_HighVol_EMA_Bullish: ifvg_present({'direction': 'bullish'}) AND volume_high({'multiplier': 1.5}) AND ema_cross({'fast': 9, 'slow': 21, 'direction': 'bullish'})
```

**Performance Metrics:**
- **Occurrences:** 32
- **Win Rate:** 40.62%
- **Expectancy:** 0.0060
- **Profit Factor:** 2.80
- **Avg Profit:** 0.0228
- **Avg Loss:** 0.0056

**Best Parameters:**
- avg_forward_return: 0.0228
- median_forward_return: 0.0200
- min_forward_return: 0.0105
- max_forward_return: 0.0491

---

### 2. LargeMove_2%_Only

**Pattern Definition:**
```
LargeMove_2%_Only: price_movement_large({'threshold_pct': 2.0})
```

**Performance Metrics:**
- **Occurrences:** 8
- **Win Rate:** 37.50%
- **Expectancy:** 0.0038
- **Profit Factor:** 2.30
- **Avg Profit:** 0.0180
- **Avg Loss:** 0.0047

**Best Parameters:**
- avg_forward_return: 0.0180
- median_forward_return: 0.0207
- min_forward_return: 0.0115
- max_forward_return: 0.0218

---

### 3. HighVol_2x_Only

**Pattern Definition:**
```
HighVol_2x_Only: volume_high({'multiplier': 2.0})
```

**Performance Metrics:**
- **Occurrences:** 2
- **Win Rate:** 50.00%
- **Expectancy:** 0.0038
- **Profit Factor:** 1.37
- **Avg Profit:** 0.0276
- **Avg Loss:** 0.0201

**Best Parameters:**
- avg_forward_return: 0.0276
- median_forward_return: 0.0276
- min_forward_return: 0.0276
- max_forward_return: 0.0276

---

### 4. HighVol_1.5x_LargeMove_1.5%

**Pattern Definition:**
```
HighVol_1.5x_LargeMove_1.5%: volume_high({'multiplier': 1.5}) AND price_movement_large({'threshold_pct': 1.5})
```

**Performance Metrics:**
- **Occurrences:** 16
- **Win Rate:** 31.25%
- **Expectancy:** 0.0028
- **Profit Factor:** 1.57
- **Avg Profit:** 0.0250
- **Avg Loss:** 0.0073

**Best Parameters:**
- avg_forward_return: 0.0250
- median_forward_return: 0.0224
- min_forward_return: 0.0200
- max_forward_return: 0.0383

---

### 5. EMA22_Touch_0.5%_SqueezeNeg_BelowPOC

**Pattern Definition:**
```
EMA22_Touch_0.5%_SqueezeNeg_BelowPOC: price_near_ema({'period': 22, 'tolerance_pct': 0.5}) AND squeeze_momentum_slope({'direction': 'negative'}) AND price_vs_poc({'position': 'below'})
```

**Performance Metrics:**
- **Occurrences:** 5657
- **Win Rate:** 31.04%
- **Expectancy:** 0.0017
- **Profit Factor:** 1.36
- **Avg Profit:** 0.0210
- **Avg Loss:** 0.0070

**Best Parameters:**
- avg_forward_return: 0.0210
- median_forward_return: 0.0190
- min_forward_return: 0.0100
- max_forward_return: 0.0659

---

### 6. IFVG_Bearish_HighVol

**Pattern Definition:**
```
IFVG_Bearish_HighVol: ifvg_present({'direction': 'bearish'}) AND volume_high({'multiplier': 1.5})
```

**Performance Metrics:**
- **Occurrences:** 613
- **Win Rate:** 30.02%
- **Expectancy:** 0.0016
- **Profit Factor:** 1.38
- **Avg Profit:** 0.0200
- **Avg Loss:** 0.0062

**Best Parameters:**
- avg_forward_return: 0.0200
- median_forward_return: 0.0180
- min_forward_return: 0.0102
- max_forward_return: 0.0498

---

### 7. IFVG_Any_HighVol

**Pattern Definition:**
```
IFVG_Any_HighVol: ifvg_present({'direction': 'any'}) AND volume_high({'multiplier': 1.5})
```

**Performance Metrics:**
- **Occurrences:** 1270
- **Win Rate:** 29.69%
- **Expectancy:** 0.0015
- **Profit Factor:** 1.34
- **Avg Profit:** 0.0200
- **Avg Loss:** 0.0063

**Best Parameters:**
- avg_forward_return: 0.0200
- median_forward_return: 0.0180
- min_forward_return: 0.0100
- max_forward_return: 0.0498

---

### 8. EMA22_Touch_SqueezeNeg_Only

**Pattern Definition:**
```
EMA22_Touch_SqueezeNeg_Only: price_near_ema({'period': 22, 'tolerance_pct': 0.5}) AND squeeze_momentum_slope({'direction': 'negative'})
```

**Performance Metrics:**
- **Occurrences:** 17077
- **Win Rate:** 30.89%
- **Expectancy:** 0.0015
- **Profit Factor:** 1.30
- **Avg Profit:** 0.0207
- **Avg Loss:** 0.0071

**Best Parameters:**
- avg_forward_return: 0.0207
- median_forward_return: 0.0186
- min_forward_return: 0.0100
- max_forward_return: 0.0659

---

### 9. EMA22_Touch_1.0%_SqueezeNeg_BelowPOC

**Pattern Definition:**
```
EMA22_Touch_1.0%_SqueezeNeg_BelowPOC: price_near_ema({'period': 22, 'tolerance_pct': 1.0}) AND squeeze_momentum_slope({'direction': 'negative'}) AND price_vs_poc({'position': 'below'})
```

**Performance Metrics:**
- **Occurrences:** 12292
- **Win Rate:** 30.04%
- **Expectancy:** 0.0015
- **Profit Factor:** 1.31
- **Avg Profit:** 0.0209
- **Avg Loss:** 0.0068

**Best Parameters:**
- avg_forward_return: 0.0209
- median_forward_return: 0.0189
- min_forward_return: 0.0100
- max_forward_return: 0.0741

---

### 10. IFVG_Only

**Pattern Definition:**
```
IFVG_Only: ifvg_present({'direction': 'any'})
```

**Performance Metrics:**
- **Occurrences:** 41255
- **Win Rate:** 30.21%
- **Expectancy:** 0.0014
- **Profit Factor:** 1.30
- **Avg Profit:** 0.0206
- **Avg Loss:** 0.0068

**Best Parameters:**
- avg_forward_return: 0.0206
- median_forward_return: 0.0185
- min_forward_return: 0.0100
- max_forward_return: 0.0722

---

### 11. IFVG_Bullish_HighVol

**Pattern Definition:**
```
IFVG_Bullish_HighVol: ifvg_present({'direction': 'bullish'}) AND volume_high({'multiplier': 1.5})
```

**Performance Metrics:**
- **Occurrences:** 657
- **Win Rate:** 29.38%
- **Expectancy:** 0.0014
- **Profit Factor:** 1.31
- **Avg Profit:** 0.0200
- **Avg Loss:** 0.0063

**Best Parameters:**
- avg_forward_return: 0.0200
- median_forward_return: 0.0181
- min_forward_return: 0.0100
- max_forward_return: 0.0491

---

### 12. HighVol_2x_LargeMove_2%

**Pattern Definition:**
```
HighVol_2x_LargeMove_2%: volume_high({'multiplier': 2.0}) AND price_movement_large({'threshold_pct': 2.0})
```

**Performance Metrics:**
- **Occurrences:** 0
- **Win Rate:** 0.00%
- **Expectancy:** 0.0000
- **Profit Factor:** 0.00
- **Avg Profit:** 0.0000
- **Avg Loss:** 0.0000

---

### 13. HighVol_3x_LargeMove_3%

**Pattern Definition:**
```
HighVol_3x_LargeMove_3%: volume_high({'multiplier': 3.0}) AND price_movement_large({'threshold_pct': 3.0})
```

**Performance Metrics:**
- **Occurrences:** 0
- **Win Rate:** 0.00%
- **Expectancy:** 0.0000
- **Profit Factor:** 0.00
- **Avg Profit:** 0.0000
- **Avg Loss:** 0.0000

---

