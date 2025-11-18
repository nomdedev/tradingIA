# 📊 Documentación: Volume Profile + IFVG + EMAs Strategy

## 🎯 Resumen de la Estrategia

Esta estrategia combina tres poderosos indicadores técnicos para identificar oportunidades de trading:

1. **IFVG (Inversion Fair Value Gaps)** - Detecta zonas de desequilibrio de precio
2. **Volume Profile** - Analiza la distribución del volumen por niveles de precio
3. **EMAs (Exponential Moving Averages)** - Identifica tendencias y soportes/resistencias dinámicos

---

## 🔍 Componentes Principales

### 1. **IFVG (Inversion Fair Value Gaps)**

#### ¿Qué son los FVGs?
Los Fair Value Gaps (FVGs) son "huecos" en el precio donde no hubo consenso entre compradores y vendedores, creando zonas de desequilibrio.

#### Detección de FVGs

**FVG Alcista:**
```pinescript
fvg_up = (low > high[2]) and (close[1] > high[2])
```
- Se detecta cuando el mínimo actual está **por encima** del máximo de 2 velas atrás
- Indica que hubo un salto alcista sin trading intermedio

**FVG Bajista:**
```pinescript
fvg_down = (high < low[2]) and (close[1] < low[2])
```
- Se detecta cuando el máximo actual está **por debajo** del mínimo de 2 velas atrás
- Indica que hubo un salto bajista sin trading intermedio

#### Filtro por ATR
```pinescript
atr = nz(ta.atr(200)*atr_multi, ta.cum(high - low) / (bar_index+1))
```
- Solo se consideran FVGs cuyo tamaño sea mayor que `ATR * Multiplicador`
- Esto filtra FVGs insignificantes y reduce falsos positivos

#### Inversiones (Señales de Trading)

**Señal ALCISTA (🔺):**
- Se genera cuando el precio **vuelve a entrar** en un FVG bajista desde abajo
- Condición: `close > bx_top and (wt?low:close[1]) <= bx_top`
- Representa un rechazo de la zona bajista → potencial compra

**Señal BAJISTA (🔻):**
- Se genera cuando el precio **vuelve a entrar** en un FVG alcista desde arriba
- Condición: `close < bx_bot and (wt?high:close[1]) >= bx_bot`
- Representa un rechazo de la zona alcista → potencial venta

#### Visualización
- **Zona verde** (transparente): FVG alcista antes de inversión
- **Zona roja** (transparente): FVG bajista antes de inversión
- **Línea gris punteada**: Punto medio del FVG
- **Cambio de color**: Después de la inversión, el color se invierte
- **Triángulos**: Marcan el punto exacto de entrada de señal

---

### 2. **Volume Profile**

#### ¿Qué es el Volume Profile?
Muestra la cantidad de volumen negociado en cada nivel de precio durante un período determinado.

#### Cálculos Principales

**1. Distribución de Volumen:**
```pinescript
pSTP = (pHST - pLST) / vpNR  // Tamaño de cada fila de precio
```
- Divide el rango de precios en `vpNR` filas (por defecto 100)
- Acumula el volumen en cada nivel de precio

**2. Volumen Alcista vs Bajista:**
```pinescript
// Método 1: Polaridad de la vela
bD.bp.push(ltfBD.get(i).c > ltfBD.get(i).o)

// Método 2: Presión compradora/vendedora
bD.bp.push(ltfBD.get(i).c - ltfBD.get(i).l > ltfBD.get(i).h - ltfBD.get(i).c)
```

**3. Point of Control (POC):**
```pinescript
VP.pcL := vD.vt.indexof(vD.vt.max())
```
- Es el nivel de precio con **mayor volumen negociado**
- Representa el "precio justo" más aceptado por el mercado
- **Línea roja** en el gráfico

**4. Value Area (VA):**
```pinescript
ttV = vD.vt.sum() * vpVA  // 68% del volumen total por defecto
```
- Área que contiene el 68% del volumen negociado (configurable)
- **Value Area High (VAH)**: Límite superior (línea azul)
- **Value Area Low (VAL)**: Límite inferior (línea azul)
- Representa la zona de "valor aceptado" por el mercado

#### Interpretación del Volume Profile

**Nodos de Alto Volumen:**
- Zonas de **consolidación** y precio aceptado
- Actúan como **soporte/resistencia** fuertes
- El POC es el nodo más importante

**Nodos de Bajo Volumen:**
- Zonas de **rechazo** rápido
- Precio se mueve rápidamente a través de estas áreas
- Pueden indicar **zonas de supply/demand**

**Sentiment Profile (Perfil de Sentimiento):**
- **Barras verdes**: Dominio comprador en ese nivel
- **Barras rojas**: Dominio vendedor en ese nivel
- Muestra quién controla cada nivel de precio

**Supply & Demand Zones:**
- Se marcan automáticamente cuando el volumen < 15% del máximo
- **Rojo**: Supply (oferta) - arriba del POC
- **Azul**: Demand (demanda) - debajo del POC

---

### 3. **EMAs (Exponential Moving Averages)**

#### Configuración de EMAs
```pinescript
EMA 1 (roja):    20 períodos  - Tendencia de corto plazo
EMA 2 (naranja): 50 períodos  - Tendencia de medio plazo
EMA 3 (cian):    100 períodos - Tendencia de largo plazo
EMA 4 (azul):    200 períodos - Tendencia de muy largo plazo
```

#### Interpretación

**Cruces de EMAs:**
- **Golden Cross**: EMA rápida cruza por encima de EMA lenta → Señal alcista
- **Death Cross**: EMA rápida cruza por debajo de EMA lenta → Señal bajista

**Soporte/Resistencia Dinámico:**
- En tendencia alcista: EMAs actúan como **soporte**
- En tendencia bajista: EMAs actúan como **resistencia**

**Distancia entre EMAs:**
- EMAs separadas → Tendencia fuerte
- EMAs comprimidas → Consolidación, posible ruptura

---

## 📈 Lógica de Trading Combinada

### Señal de COMPRA Ideal (LONG)

1. **IFVG**: Precio rechaza FVG bajista (triángulo verde 🔺)
2. **Volume Profile**: 
   - Precio cerca de zona de demanda (azul)
   - Precio en o por debajo de VAL
   - Volumen aumentando
3. **EMAs**: 
   - Precio por encima de EMA 20 o rebotando en ella
   - EMAs en orden alcista (20 > 50 > 100 > 200)

### Señal de VENTA Ideal (SHORT)

1. **IFVG**: Precio rechaza FVG alcista (triángulo rojo 🔻)
2. **Volume Profile**: 
   - Precio cerca de zona de supply (rojo)
   - Precio en o por encima de VAH
   - Volumen aumentando
3. **EMAs**: 
   - Precio por debajo de EMA 20 o rebotando en ella
   - EMAs en orden bajista (20 < 50 < 100 < 200)

---

## ⚙️ Parámetros Configurables

### IFVG Settings
- **Show Last**: Cantidad de IFVGs a mostrar (5 por defecto)
- **Signal Preference**: Usar cierre o mechas para señales
- **ATR Multiplier**: Filtro de tamaño mínimo (0.25 por defecto)

### Volume Profile Settings
- **Lookback Length**: Cantidad de velas a analizar (360 por defecto)
- **Number of Rows**: Resolución del perfil (100 por defecto)
- **Value Area %**: Porcentaje de volumen para VA (68% por defecto)
- **Profile Width**: Ancho visual del perfil
- **Polarity Method**: Método para calcular volumen alcista/bajista

### EMA Settings
- **EMA Lengths**: Períodos para cada EMA (20, 50, 100, 200)
- **EMA Colors**: Colores personalizables

---

## 🔔 Alertas Disponibles

1. **Bullish Signal**: Cuando se genera señal de compra (IFVG)
2. **Bearish Signal**: Cuando se genera señal de venta (IFVG)
3. **POC Cross**: Precio cruza el Point of Control
4. **VAH Cross**: Precio cruza Value Area High
5. **VAL Cross**: Precio cruza Value Area Low
6. **High Volume**: Volumen > VolumeMA * Upper Threshold
7. **Volume Spike**: Volumen extremadamente alto (posible agotamiento)

---

## 💡 Mejores Prácticas

### Uso Óptimo de la Estrategia

1. **Confluencia es clave**: 
   - No operar solo con IFVG
   - Buscar confirmación de Volume Profile y EMAs

2. **Contexto de mercado**:
   - En tendencia fuerte: Operar solo a favor de la tendencia (EMAs)
   - En rango: Operar rebotes en extremos del Value Area

3. **Gestión de riesgo**:
   - Stop loss: Por debajo/encima del FVG completo
   - Take profit: En POC o extremos del Value Area

4. **Volumen confirma**:
   - Entradas con volumen creciente son más confiables
   - Cuidado con señales en zonas de bajo volumen

### Timeframes Recomendados

- **Scalping**: 1min - 5min (señales frecuentes)
- **Intraday**: 15min - 1H (señales de calidad)
- **Swing**: 4H - 1D (señales de alta probabilidad)

---

## 📊 Estadísticas del Profile

El indicador muestra automáticamente:
- Profile High/Low
- Value Area High/Low
- Point of Control
- Total Volume en el rango
- Average Volume por barra
- Volume MA actual
- Número de barras analizadas
- Timeframe de datos usado

---

## 🎨 Código de Colores

### IFVG
- 🟢 **Verde**: FVG alcista / Señal de compra
- 🔴 **Rojo**: FVG bajista / Señal de venta
- ⚪ **Gris**: Línea media del FVG

### Volume Profile
- **Gris oscuro**: Volumen alcista
- **Gris claro**: Volumen bajista
- **Azul**: Value Area alcista
- **Amarillo**: Value Area bajista
- **Rojo**: POC line

### Sentiment Profile
- **Verde**: Nodos alcistas (compradores dominan)
- **Rojo**: Nodos bajistas (vendedores dominan)

### Supply & Demand
- 🔴 **Rojo transparente**: Zonas de supply
- 🔵 **Azul transparente**: Zonas de demand

### Volume Histogram
- 🟢 **Verde**: Volumen creciente
- 🔴 **Rojo**: Volumen decreciente
- 🔵 **Azul**: Volume MA

---

## 🔧 Configuración Técnica

### Límites del Indicador
```pinescript
max_boxes_count = 500   // Máximo de cajas para FVGs y VP
max_lines_count = 500   // Máximo de líneas para niveles
max_labels_count = 500  // Máximo de etiquetas para señales
max_bars_back = 5000    // Lookback máximo
```

### Optimización de Datos
- Para lookback ≤ 200 velas: Usa 2 timeframes inferiores
- Para lookback ≤ 700 velas: Usa 1 timeframe inferior
- Para lookback > 700 velas: Usa timeframe del gráfico

---

## 📝 Notas Importantes

1. **Repintado**: Las señales IFVG se confirman en el cierre de vela
2. **Volume Profile**: Se recalcula en cada vela para mostrar desarrollo
3. **POC Developing**: Muestra el movimiento del POC en tiempo real
4. **Performance**: En timeframes muy bajos, considerar reducir lookback length

---

## 🚀 Ejemplo de Setup Completo

### LONG Setup Ideal

```
1. IFVG: Triángulo verde aparece (entrada en FVG bajista)
2. Volume Profile: 
   - Precio en zona de demand (azul)
   - Cerca de VAL o por debajo
   - Volume histogram creciendo (verde)
3. EMAs:
   - Precio rebota en EMA 20 (roja)
   - EMA 20 > EMA 50 > EMA 100 > EMA 200
4. Confirmación: Vela alcista fuerte con volumen

Entry: En el triángulo verde
Stop Loss: Por debajo del FVG
Take Profit 1: POC
Take Profit 2: VAH
```

### SHORT Setup Ideal

```
1. IFVG: Triángulo rojo aparece (entrada en FVG alcista)
2. Volume Profile:
   - Precio en zona de supply (roja)
   - Cerca de VAH o por encima
   - Volume histogram creciendo (verde)
3. EMAs:
   - Precio rechaza EMA 20 (roja)
   - EMA 20 < EMA 50 < EMA 100 < EMA 200
4. Confirmación: Vela bajista fuerte con volumen

Entry: En el triángulo rojo
Stop Loss: Por encima del FVG
Take Profit 1: POC
Take Profit 2: VAL
```

---

## 📚 Recursos Adicionales

### Conceptos Relacionados
- **Smart Money Concepts (SMC)**: Los FVGs son parte de esta metodología
- **Market Profile**: Origen del Volume Profile
- **Order Flow**: El volumen revela el flujo de órdenes institucionales

### Libros Recomendados
- "Trading in the Zone" - Mark Douglas
- "Technical Analysis using Multiple Timeframes" - Brian Shannon
- "Markets in Profile" - James Dalton

---

**Autor**: Estrategia combinada VP + IFVG + EMAs
**Versión**: 5
**Última actualización**: 2025
**Compatibilidad**: TradingView PineScript v5

---

*Esta documentación cubre el funcionamiento técnico de la estrategia. Para resultados óptimos, combina siempre el análisis técnico con una sólida gestión de riesgo.*