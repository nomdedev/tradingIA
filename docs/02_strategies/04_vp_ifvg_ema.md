# Estrategia VP_IFVG_EMA - Documentación Completa

## 📊 **Descripción General**

La estrategia **VP_IFVG_EMA** es una conversión directa del indicador de Pine Script "Volume Profile + IFVG + EMAs [Combined]" para TradingView. Combina tres indicadores técnicos poderosos:

1. **IFVG (Implied Fair Value Gaps)** - Detecta gaps de valor justo implícitos
2. **Volume Profile** - Análisis de distribución de volumen por niveles de precio
3. **EMAs (Exponential Moving Averages)** - Filtrado de tendencias

## 🎯 **Cómo Funcionan las Señales**

### **Señales Principales: Triángulos Arriba/Abajo**

Las señales principales se generan cuando los **FVGs se invierten**, marcadas con triángulos como en TradingView:

#### **🟢 Triángulo HACIA ARRIBA (Compra)**
- **Cuándo ocurre**: Cuando un **FVG bajista** se invierte al alza
- **Lógica**: Un FVG bajista existe cuando `high < low[2]` y `close[1] < low[2]`
- **Señal**: Se genera cuando el precio rompe por encima del techo del FVG
- **Interpretación**: El mercado está rechazando el gap bajista, señal de fuerza alcista

#### **🔴 Triángulo HACIA ABAJO (Venta)**
- **Cuándo ocurre**: Cuando un **FVG alcista** se invierte a la baja
- **Lógica**: Un FVG alcista existe cuando `low > high[2]` y `close[1] > high[2]`
- **Señal**: Se genera cuando el precio rompe por debajo del piso del FVG
- **Interpretación**: El mercado está rechazando el gap alcista, señal de fuerza bajista

### **Lógica de Detección de FVGs**

```python
# FVG Alcista (Bullish FVG)
if low > high[2] and close[1] > high[2] and abs(low - high[2]) > ATR * multiplier:
    # Crear FVG alcista

# FVG Bajista (Bearish FVG)
if high < low[2] and close[1] < low[2] and abs(low[2] - high) > ATR * multiplier:
    # Crear FVG bajista
```

### **Inversión de FVGs**

```python
# Para FVG Bajista -> Señal Alcista
if precio rompe por encima del techo del FVG:
    generar_triángulo_arriba()

# Para FVG Alcista -> Señal Bajista
if precio rompe por debajo del piso del FVG:
    generar_triángulo_abajo()
```

## 📈 **Componentes Adicionales**

### **Volume Profile**
- **POC (Point of Control)**: Nivel con mayor volumen
- **VAH (Value Area High)**: Techo del área de valor
- **VAL (Value Area Low)**: Piso del área de valor
- **Señales**: Compra cerca de VAL, venta cerca de VAH

### **EMAs para Filtrado**
- **EMA1 (20)** y **EMA2 (50)**: Para cruces de tendencia
- **EMA3 (100)** y **EMA4 (200)**: Para tendencias de largo plazo
- **Función**: Filtra señales contrarias a la tendencia principal

### **Confirmación de Volumen**
- **Volumen Alto**: Confirma fuerza de la señal
- **Volumen Bajo**: Puede indicar falta de convicción

## ⚙️ **Parámetros Configurables**

### **IFVG Settings**
- `disp_num` (5): Número de FVGs recientes a mostrar
- `signal_pref` ("Close"): "Close" o "Wick" para detección de ruptura
- `atr_multi` (0.25): Multiplicador ATR para filtrar FVGs pequeños

### **Volume Profile**
- `vp_length` (360): Período de lookback para VP
- `vp_rows` (100): Número de bins para distribución
- `vp_va` (68): Porcentaje del área de valor
- `vp_polarity` ("Bar Polarity"): Método de polaridad del volumen

### **EMAs**
- `ema1_length` (20): Período EMA rápida
- `ema2_length` (50): Período EMA media
- `ema3_length` (100): Período EMA lenta
- `ema4_length` (200): Período EMA muy lenta

### **Filtros de Señal**
- `use_volume_filter` (True): Usar confirmación de volumen
- `use_ema_filter` (True): Usar filtro de tendencia EMA
- `use_vp_levels` (True): Usar niveles VP para señales
- `min_signal_strength` (1): Fuerza mínima de señal (1-5)

## 📊 **Sistema de Fuerza de Señal**

La estrategia usa un sistema de puntuación para la fuerza de las señales:

- **5**: Señal muy fuerte (FVG + VP + EMA + Volumen alineados)
- **4**: Señal fuerte (3 componentes alineados)
- **3**: Señal moderada (FVG invertido + 1-2 confirmaciones)
- **2**: Señal débil (VP o EMA únicamente)
- **1**: Señal muy débil (confirmación mínima)

## 🔄 **Análisis de Sharpe Ratio**

Para análisis de Sharpe ratio con diferentes parámetros:

1. **Variar `atr_multi`**: 0.1, 0.25, 0.5, 1.0
2. **Variar `vp_length`**: 180, 360, 720, 1440
3. **Variar `ema1_length`**: 10, 20, 30, 50
4. **Variar `ema2_length`**: 25, 50, 75, 100
5. **Variar `min_signal_strength`**: 1, 2, 3, 4, 5

## 📈 **Ejemplo de Uso**

```python
from strategies.vp_ifvg_ema_strategy import VPIFVGEmaStrategy

# Crear estrategia con parámetros personalizados
strategy = VPIFVGEmaStrategy()
strategy.set_parameters({
    'atr_multi': 0.5,
    'vp_length': 720,
    'ema1_length': 15,
    'ema2_length': 45,
    'min_signal_strength': 3
})

# Generar señales
signals = strategy.generate_signals(ohlcv_data)

# Señales: 1=COMPRA, -1=VENTA, 0=HOLD
# signal_strength: 1-5 (fuerza de la señal)
```

## 🎯 **Interpretación de Señales**

### **Señal de Compra (1)**
- Triángulo hacia arriba por inversión de FVG bajista
- Precio cerca de VAL (soporte)
- EMA1 > EMA2 (tendencia alcista)
- Volumen por encima de la media

### **Señal de Venta (-1)**
- Triángulo hacia abajo por inversión de FVG alcista
- Precio cerca de VAH (resistencia)
- EMA1 < EMA2 (tendencia bajista)
- Volumen por encima de la media

### **Sin Señal (0)**
- No hay FVGs activos o condiciones de inversión
- Señales contradictorias entre indicadores
- Fuerza de señal por debajo del mínimo configurado

## 🔧 **Optimización**

Para optimizar la estrategia:

1. **Backtesting** con diferentes activos y timeframes
2. **Walk-forward analysis** para evitar overfitting
3. **Sensitivity analysis** variando parámetros clave
4. **Risk management** basado en fuerza de señal
5. **Portfolio optimization** combinando con otras estrategias

Esta estrategia captura movimientos de precio significativos basados en gaps de valor justo y distribución de volumen, filtrados por tendencias EMA para mayor precisión.</content>
<parameter name="filePath">d:\martin\Proyectos\tradingIA\VP_IFVG_EMA_DOCUMENTATION.md