# 📚 Documentación de Estrategias de Trading

Esta guía proporciona información detallada sobre todas las estrategias de trading disponibles en la plataforma.

---

## 📋 Índice

1. [Bollinger Bands](#bollinger-bands)
2. [RSI Mean Reversion](#rsi-mean-reversion)
3. [MACD Momentum](#macd-momentum)
4. [Moving Average Crossover](#moving-average-crossover)
5. [Volume Breakout](#volume-breakout)
6. [Oracle Numeris Safeguard](#oracle-numeris-safeguard)
7. [Squeeze ADX TTM](#squeeze-adx-ttm)
8. [VP IFVG EMA](#vp-ifvg-ema)
9. [Cómo Usar las Estrategias](#cómo-usar-las-estrategias)
10. [Presets Disponibles](#presets-disponibles)

---

## Bollinger Bands

### 📝 Descripción
Estrategia de reversión a la media usando Bandas de Bollinger. Opera cuando el precio toca las bandas exteriores, esperando un retorno al centro.

### 🎯 Tipo de Estrategia
**Reversión a la Media** - Conservadora

### 📊 Indicadores Utilizados
- Bollinger Bands (Bandas de Bollinger)
- SMA (Media Móvil Simple)
- Volume MA (Media de Volumen, opcional)

### 📈 Señales de Compra
- El precio toca o cruza la **banda inferior** (sobreventa)
- El precio bajo (low) está por debajo de la banda inferior
- **Opcional**: Volumen superior al promedio (si está activado)

**Interpretación**: Cuando el precio toca la banda inferior, se considera sobreventa y se espera una reversión alcista.

### 📉 Señales de Venta
- El precio toca o cruza la **banda superior** (sobrecompra)
- El precio alto (high) está por encima de la banda superior
- **Opcional**: Volumen superior al promedio (si está activado)

**Interpretación**: Cuando el precio toca la banda superior, se considera sobrecompra y se espera una reversión bajista.

### ⚙️ Parámetros Configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `period` | 20 | Período para la media móvil central |
| `num_std` | 2.0 | Número de desviaciones estándar para las bandas |
| `use_close_for_bands` | True | Usar precio de cierre para cálculo |
| `require_volume_confirmation` | False | Requiere confirmación de volumen alto |
| `volume_ma_period` | 20 | Período para media de volumen |

### 💡 Mejores Condiciones de Mercado
- **Mercados laterales** (rango definido)
- **Volatilidad moderada**
- Evitar en tendencias fuertes

### 📍 Timeframe Recomendado
**5 minutos** (5min)

---

## RSI Mean Reversion

### 📝 Descripción
Estrategia de reversión a la media basada en RSI (Relative Strength Index). Compra en sobreventa y vende en sobrecompra.

### 🎯 Tipo de Estrategia
**Reversión a la Media** - Equilibrada

### 📊 Indicadores Utilizados
- RSI (Relative Strength Index)

### 📈 Señales de Compra
- RSI cruza **por debajo del nivel de sobreventa** (default: 30)
- RSI anterior estaba por encima del nivel de sobreventa
- Indica posible **reversión alcista**

**Interpretación**: Un RSI bajo indica que el activo está sobreventa y puede rebotar.

### 📉 Señales de Venta
- RSI cruza **por encima del nivel de sobrecompra** (default: 70)
- RSI anterior estaba por debajo del nivel de sobrecompra
- Indica posible **reversión bajista**

**Interpretación**: Un RSI alto indica que el activo está sobrecompra y puede corregir.

### ⚙️ Parámetros Configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `rsi_period` | 14 | Período para cálculo del RSI |
| `oversold` | 30 | Nivel de sobreventa (0-100) |
| `overbought` | 70 | Nivel de sobrecompra (0-100) |
| `use_smoothing` | False | Aplicar suavizado al RSI |
| `smooth_period` | 3 | Período de suavizado |

### 💡 Mejores Condiciones de Mercado
- **Mercados laterales** con límites definidos
- **Volatilidad media**
- Evitar en tendencias muy fuertes

### 📍 Timeframe Recomendado
**5 minutos** (5min)

---

## MACD Momentum

### 📝 Descripción
Estrategia de momentum basada en MACD (Moving Average Convergence Divergence). Sigue la tendencia comprando en cruces alcistas y vendiendo en cruces bajistas.

### 🎯 Tipo de Estrategia
**Seguimiento de Tendencia** - Equilibrada

### 📊 Indicadores Utilizados
- MACD (Línea MACD)
- Signal Line (Línea de Señal)
- Histogram (Histograma)

### 📈 Señales de Compra
- Línea MACD cruza **por encima** de la línea de señal (cruce alcista)
- **Opcional**: Histograma debe ser positivo
- **Opcional**: Histograma supera fuerza mínima
- Indica **momentum alcista**

**Interpretación**: El cruce alcista del MACD indica que el momentum se está volviendo positivo.

### 📉 Señales de Venta
- Línea MACD cruza **por debajo** de la línea de señal (cruce bajista)
- **Opcional**: Histograma debe ser negativo
- **Opcional**: Histograma supera fuerza mínima
- Indica **momentum bajista**

**Interpretación**: El cruce bajista del MACD indica que el momentum se está volviendo negativo.

### ⚙️ Parámetros Configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `fast_period` | 12 | Período EMA rápida |
| `slow_period` | 26 | Período EMA lenta |
| `signal_period` | 9 | Período línea de señal |
| `require_histogram_positive` | True | Requiere histograma positivo para compra |
| `min_histogram_strength` | 0.0 | Fuerza mínima del histograma |

### 💡 Mejores Condiciones de Mercado
- **Tendencias claras** (alcistas o bajistas)
- **Volatilidad media a alta**
- Buenos resultados en breakouts

### 📍 Timeframe Recomendado
**5 minutos** (5min)

---

## Moving Average Crossover

### 📝 Descripción
Estrategia clásica de cruce de medias móviles. Compra cuando la MA rápida cruza por encima de la MA lenta (golden cross) y vende cuando cruza por debajo (death cross).

### 🎯 Tipo de Estrategia
**Seguimiento de Tendencia** - Conservadora

### 📊 Indicadores Utilizados
- MA Rápida (SMA o EMA)
- MA Lenta (SMA o EMA)
- Trend MA (opcional, para filtro)

### 📈 Señales de Compra (Golden Cross)
- MA rápida (default: 50) cruza **por encima** de MA lenta (default: 200)
- **Opcional**: Precio debe estar por encima de MA lenta
- **Opcional**: Filtro de tendencia alcista activo
- Indica inicio de **tendencia alcista**

**Interpretación**: El golden cross es una señal clásica de inicio de tendencia alcista.

### 📉 Señales de Venta (Death Cross)
- MA rápida (default: 50) cruza **por debajo** de MA lenta (default: 200)
- **Opcional**: Precio debe estar por debajo de MA lenta
- **Opcional**: Filtro de tendencia bajista activo
- Indica inicio de **tendencia bajista**

**Interpretación**: El death cross es una señal clásica de inicio de tendencia bajista.

### ⚙️ Parámetros Configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `fast_period` | 50 | Período MA rápida |
| `slow_period` | 200 | Período MA lenta |
| `ma_type` | 'EMA' | Tipo de media móvil ('SMA' o 'EMA') |
| `require_price_above` | False | Requiere precio sobre/bajo MA para señal |
| `filter_by_trend` | False | Activar filtro de tendencia adicional |
| `trend_period` | 100 | Período para filtro de tendencia |

### 💡 Mejores Condiciones de Mercado
- **Tendencias sostenidas** de medio/largo plazo
- **Volatilidad baja a media**
- Evitar en mercados muy choppys

### 📍 Timeframe Recomendado
**5 minutos** (5min) o superior

### 🎨 Presets Disponibles
- **Conservative**: 50/200 SMA, requiere precio sobre MA, filtro tendencia
- **Aggressive**: 20/100 EMA, sin filtros adicionales
- **Scalping**: 10/30 EMA, para trading rápido

---

## Volume Breakout

### 📝 Descripción
Estrategia de ruptura confirmada por volumen. Opera cuando el precio rompe niveles clave de soporte/resistencia con volumen alto, indicando movimientos fuertes y sostenidos.

### 🎯 Tipo de Estrategia
**Breakout (Ruptura)** - Agresiva

### 📊 Indicadores Utilizados
- Support/Resistance (Soporte/Resistencia)
- Volume MA (Media de Volumen)
- ATR (Average True Range)

### 📈 Señales de Compra (Breakout Alcista)
- Precio rompe **por encima** de resistencia (+2% default)
- Volumen superior a **1.5x** el promedio
- **Opcional**: Cierre debe estar por encima de resistencia
- Indica fuerte **momentum de compra**

**Interpretación**: La ruptura con volumen alto indica que hay participación real del mercado.

### 📉 Señales de Venta (Breakdown Bajista)
- Precio rompe **por debajo** de soporte (-2% default)
- Volumen superior a **1.5x** el promedio
- **Opcional**: Cierre debe estar por debajo de soporte
- Indica fuerte **momentum de venta**

**Interpretación**: El breakdown con volumen alto valida la caída y sugiere continuación.

### ⚙️ Parámetros Configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `lookback_period` | 20 | Período para detectar soporte/resistencia |
| `volume_ma_period` | 20 | Período para media de volumen |
| `volume_multiplier` | 1.5 | Multiplicador de volumen requerido |
| `breakout_threshold` | 0.02 | Umbral de ruptura (2%) |
| `require_close_beyond` | True | Requiere cierre más allá del nivel |
| `atr_period` | 14 | Período ATR para volatilidad |

### 💡 Mejores Condiciones de Mercado
- **Consolidaciones previas** a ruptura
- **Eventos de noticias** o catalizadores
- **Alta liquidez** para ejecución

### ⚠️ Advertencias
- Puede generar **falsos breakouts**
- Usar **stops ajustados**
- Validar con otros timeframes

### 📍 Timeframe Recomendado
**5 minutos** (5min)

---

## Oracle Numeris Safeguard

### 📝 Descripción
Estrategia avanzada que combina Oracle Numeris (predicción numérica) con Safeguard (gestión de riesgo dinámica). Usa regresión lineal para predecir movimientos y un sistema de puntuación de riesgo basado en ATR y drawdown.

### 🎯 Tipo de Estrategia
**Predicción Cuantitativa + Risk Management** - Equilibrada

### 📊 Indicadores Utilizados
- Linear Regression (Regresión Lineal)
- ATR (Average True Range)
- Drawdown Monitor
- Volume MA (Media de Volumen)
- Trend MA (Media de Tendencia)

### 🧠 Componentes de la Estrategia

#### Oracle Numeris
Sistema de predicción basado en:
- Regresión lineal sobre ventana de precios
- Cálculo de pendiente normalizada
- Suavizado de predicciones

#### Safeguard
Sistema de protección que evalúa:
- Volatilidad actual (ATR)
- Drawdown desde máximo
- Puntuación de riesgo combinada (0-1)

### 📈 Señales de Compra
- Oracle predice movimiento alcista (>+2% default)
- Safeguard: **puntuación de riesgo baja** (<0.7)
- **Opcional**: Volumen 1.2x sobre promedio
- Sistema de predicción basado en regresión lineal

**Interpretación**: Combina predicción de movimiento con análisis de riesgo actual.

### 📉 Señales de Venta
- Oracle predice movimiento bajista (<-2% default)
- Safeguard: **puntuación de riesgo baja** (<0.7)
- **Opcional**: Volumen 1.2x sobre promedio
- Protección contra alta volatilidad y drawdown

**Interpretación**: Solo opera cuando la predicción es clara y el riesgo es aceptable.

### ⚙️ Parámetros Configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `oracle_window` | 20 | Ventana para predicciones Oracle |
| `oracle_threshold` | 0.02 | Umbral de confianza Oracle (2%) |
| `numeris_smoothing` | 5 | Período suavizado Numeris |
| `safeguard_atr_period` | 14 | Período ATR Safeguard |
| `safeguard_stop_mult` | 1.5 | Multiplicador stop loss |
| `safeguard_profit_mult` | 2.0 | Multiplicador take profit |
| `safeguard_max_drawdown` | 0.05 | Drawdown máximo permitido (5%) |
| `require_volume_confirmation` | True | Requiere confirmación de volumen |
| `min_volume_ratio` | 1.2 | Ratio mínimo de volumen |
| `trend_filter_period` | 50 | Período filtro tendencia |

### 💡 Mejores Condiciones de Mercado
- **Tendencias claras** con volatilidad controlada
- Mercados con **liquidez suficiente**
- Evitar en mercados extremadamente caóticos

### 🎨 Presets Disponibles
- **Conservative**: Threshold 3%, menor riesgo, filtros estrictos
- **Balanced**: Threshold 2%, equilibrio riesgo/retorno
- **Aggressive**: Threshold 1.5%, más señales, mayor riesgo

### 📍 Timeframe Recomendado
**5 minutos** (5min)

---

## Squeeze ADX TTM

### 📝 Descripción
Estrategia multi-indicador avanzada que combina Squeeze Momentum (detección de consolidación), ADX (fuerza de tendencia) y TTM Waves (estructura de mercado).

### 🎯 Tipo de Estrategia
**Multi-Indicador Avanzada** - Equilibrada

### 📊 Indicadores Utilizados
- Squeeze Momentum (Bollinger Bands + Keltner Channels)
- ADX (Average Directional Index)
- DI+ / DI- (Directional Indicators)
- TTM Waves (A, B, C)
- Fast MA

### 🧠 Componentes de la Estrategia

#### Squeeze Momentum
- Detecta períodos de **consolidación** (squeeze)
- Identifica **liberación de energía** (expansión)
- Usa BB y KC para determinar compresión

#### ADX
- Mide **fuerza de tendencia**
- Filtra señales en mercados débiles
- Key level para validación

#### TTM Waves
- Analiza **estructura multi-temporal**
- Identifica **ondas de mercado**
- Confirma dirección

### 📈 Señales de Compra
- Squeeze se **libera** en dirección alcista
- ADX > umbral (indica tendencia fuerte)
- Momentum positivo
- Confirmación multi-timeframe

### 📉 Señales de Venta
- Squeeze se **libera** en dirección bajista
- ADX > umbral (indica tendencia fuerte)
- Momentum negativo
- Confirmación multi-timeframe

### ⚙️ Parámetros Clave

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `bb_length` | 20 | Período Bollinger Bands |
| `bb_mult` | 2.0 | Multiplicador BB |
| `kc_length` | 20 | Período Keltner Channels |
| `kc_mult` | 1.5 | Multiplicador KC |
| `adx_length` | 14 | Período ADX |
| `adx_threshold` | 20 | Umbral ADX mínimo |
| `squeeze_threshold` | 0.5 | Sensibilidad squeeze |

### 🎨 Presets Disponibles
- **Conservative**: Filtros más estrictos, menos señales
- **Balanced**: Configuración optimizada para BTC
- **Aggressive**: Más sensible, más señales

### 📍 Timeframe Recomendado
**5 minutos** con confirmación en **15 minutos**

---

## VP IFVG EMA

### 📝 Descripción
Estrategia avanzada basada en Volume Profile, Inversion Fair Value Gaps (IFVG) y EMAs. Identifica zonas de valor y gaps de precio para operaciones de alta probabilidad.

### 🎯 Tipo de Estrategia
**Análisis de Volumen + Price Action** - Avanzada

### 📊 Indicadores Utilizados
- Volume Profile (Perfil de Volumen)
- IFVG (Inversion Fair Value Gaps)
- EMAs (Medias Móviles Exponenciales)
- EMA15m50 (Filtro de proximidad)

### 🧠 Componentes de la Estrategia

#### Volume Profile
- Identifica **zonas de alto volumen** (consolidación)
- Detecta **zonas de bajo volumen** (supply/demand)
- Determina **áreas de valor**

#### IFVG
- Detecta **gaps de inversión**
- Analiza **fair value gaps alcistas y bajistas**
- Filtro por ATR width

#### EMA Proximity Filter
- Filtro adicional basado en EMA 50 de 15 minutos
- Opera solo cuando precio está cerca (±4% default)
- Mejora win rate significativamente

### 📈 Señales de Compra
- Precio en zona de **valor alcista**
- IFVG alcista detectado
- Confirmación de EMAs
- Precio cerca de EMA15m50 (si filtro activo)

### 📉 Señales de Venta
- Precio en zona de **valor bajista**
- IFVG bajista detectado
- Confirmación de EMAs
- Precio cerca de EMA15m50 (si filtro activo)

### ⚙️ Parámetros Clave

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `ema15m50_proximity_pct` | 4.0 | Umbral proximidad a EMA (4%) |
| `use_ema15m50_filter` | True | Activar filtro EMA |
| `ema15m50_period` | 50 | Período EMA |
| `ema15m50_timeframe` | '15T' | Timeframe EMA (15 min) |

### 💡 Resultados de Optimización
- Win Rate: **56.5%** con filtro EMA
- Profit Factor: **1.012**
- Mejor performance cerca de EMA15m50

### 🎨 Presets Disponibles
- **Conservative**: Proximidad 2% (filtro más estricto)
- **Default**: Proximidad 4% (optimizado)
- **Aggressive**: Proximidad 6% (más señales)

### 📍 Timeframe Recomendado
**5 minutos** con referencia a **15 minutos**

---

## Cómo Usar las Estrategias

### 1️⃣ Selección de Estrategia

```python
from strategies.strategy_loader import StrategyLoader

loader = StrategyLoader()
strategies = loader.list_strategies()
print(strategies)  # Ver todas las estrategias disponibles
```

### 2️⃣ Cargar Estrategia con Preset

```python
# Cargar con preset específico
strategy = loader.get_strategy('bollinger_bands', preset='conservative')

# Ver información detallada
info = strategy.get_detailed_info()
print(info['buy_signals'])
print(info['sell_signals'])
```

### 3️⃣ Configurar Parámetros Personalizados

```python
# Obtener parámetros actuales
params = strategy.get_parameters()

# Modificar parámetros
custom_params = {
    'period': 25,
    'num_std': 2.5
}
strategy.set_parameters(custom_params)
```

### 4️⃣ Generar Señales

```python
# Multi-timeframe data
df_multi_tf = {
    '5min': df_5m,
    '15min': df_15m,
    '1h': df_1h
}

# Generar señales
signals = strategy.generate_signals(df_multi_tf)

# signals contiene:
# - 'entries': Señales de entrada (1 = compra, 0 = no)
# - 'exits': Señales de salida (1 = venta, 0 = no)
# - 'signals': Señales combinadas (1 = compra, -1 = venta, 0 = hold)
```

### 5️⃣ Usar en GUI

1. Abrir la plataforma de trading
2. Ir a **Tab 2** (Strategy Configuration)
3. Seleccionar estrategia del dropdown
4. Elegir preset o configurar parámetros manualmente
5. Hacer clic en "View Info" para ver documentación completa

---

## Presets Disponibles

Cada estrategia incluye varios presets predefinidos:

### 🛡️ Conservative (Conservador)
- **Menor riesgo**, **menos señales**
- Filtros más estrictos
- Ideal para: Cuentas pequeñas, bajo riesgo

### ⚖️ Balanced (Equilibrado)
- **Riesgo medio**, **señales moderadas**
- Configuración optimizada
- Ideal para: Mayoría de casos

### 🚀 Aggressive (Agresivo)
- **Mayor riesgo**, **más señales**
- Filtros más laxos
- Ideal para: Cuentas grandes, alta tolerancia al riesgo

### ⚡ Scalping (cuando aplica)
- **Trading muy rápido**
- Períodos cortos
- Ideal para: Day trading intensivo

---

## 📊 Comparación de Estrategias

| Estrategia | Tipo | Riesgo | Mercado Ideal | Complejidad |
|------------|------|--------|---------------|-------------|
| Bollinger Bands | Reversión | Bajo | Lateral | Baja |
| RSI Mean Reversion | Reversión | Medio | Lateral | Baja |
| MACD Momentum | Tendencia | Medio | Trending | Media |
| MA Crossover | Tendencia | Bajo | Trending | Baja |
| Volume Breakout | Breakout | Alto | Consolidación | Media |
| Oracle Numeris | Predicción | Medio | Trending | Alta |
| Squeeze ADX TTM | Multi | Medio | Universal | Alta |
| VP IFVG EMA | Price Action | Medio | Universal | Alta |

---

## 💡 Consejos Generales

### ✅ Mejores Prácticas

1. **Backtesting**: Siempre prueba la estrategia en histórico
2. **Paper Trading**: Practica en modo simulación antes de real
3. **Gestión de Riesgo**: Nunca arriesgues más del 1-2% por operación
4. **Diversificación**: No uses solo una estrategia
5. **Monitoreo**: Revisa el performance regularmente

### ⚠️ Advertencias

- **No hay estrategia perfecta**: Todas tienen períodos de pérdidas
- **Condiciones de mercado**: Las estrategias funcionan mejor en ciertos mercados
- **Optimización**: Evita sobre-optimizar en datos históricos
- **Emociones**: Sigue el plan, no operes por impulso
- **Tamaño de posición**: Ajusta según volatilidad

### 📈 Optimización de Estrategias

1. Usa el **Tab 4** para análisis de estrategias
2. Ejecuta **Walk-Forward Testing** para validación
3. Revisa **métricas de riesgo** en Tab 11
4. Compara estrategias en **Tab 5** (A/B Testing)

---

## 🔧 Desarrollo de Estrategias Personalizadas

Para crear tu propia estrategia:

1. Hereda de `BaseStrategy`
2. Implementa `generate_signals()`
3. Implementa `get_parameters()` y `set_parameters()`
4. **Añade** `get_description()` y `get_detailed_info()`
5. Define `PRESETS` al final del archivo
6. Guarda en `strategies/presets/`

Ejemplo mínimo:

```python
from base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="My Strategy")
        self.parameters = {'period': 20}
    
    def generate_signals(self, df_multi_tf):
        # Tu lógica aquí
        pass
    
    def get_parameters(self):
        return self.parameters.copy()
    
    def set_parameters(self, params):
        self.parameters.update(params)
    
    def get_description(self):
        return "Mi estrategia personalizada"
    
    def get_detailed_info(self):
        return {
            'name': self.name,
            'description': self.get_description(),
            'buy_signals': 'Descripción de compra',
            'sell_signals': 'Descripción de venta',
            'parameters': self.parameters,
            'risk_level': 'Equilibrado',
            'timeframe': '5min',
            'indicators': ['Indicator1', 'Indicator2']
        }
```

---

## 📞 Soporte

Para más información o problemas:
- Revisa la documentación en `/docs`
- Consulta los archivos de estrategia en `/strategies/presets`
- Ejecuta el script directamente para ver ejemplos

---

**Última actualización**: Noviembre 2025
**Versión**: 2.0
