# 📖 Guía Completa de Usuario - BTC Trading Strategy Platform

## 🚀 Índice
1. [Introducción](#introducción)
2. [Instalación y Configuración Inicial](#instalación-y-configuración-inicial)
3. [Guía de Uso por Pestaña](#guía-de-uso-por-pestaña)
4. [Casos de Uso Avanzados](#casos-de-uso-avanzados)
5. [Solución de Problemas](#solución-de-problemas)
6. [Mejores Prácticas](#mejores-prácticas)

---

## 📌 Introducción

La **BTC Trading Strategy Platform** es una aplicación completa de escritorio para diseñar, probar y ejecutar estrategias de trading de criptomonedas. Incluye 7 módulos funcionales que cubren todo el ciclo de desarrollo de estrategias.

### Características Principales
- ✅ Gestión de datos multi-timeframe
- ✅ 5+ estrategias preconfiguradas
- ✅ Backtesting avanzado (Simple, Walk-Forward, Monte Carlo)
- ✅ Análisis estadístico profundo
- ✅ Pruebas A/B entre estrategias
- ✅ Monitoreo en vivo (paper trading)
- ✅ Análisis avanzado (regímenes, stress testing, causalidad)

---

## 🔧 Instalación y Configuración Inicial

### Paso 1: Ejecutar la Aplicación
1. Descarga `main_platform.exe`
2. Haz doble clic para ejecutar
3. La aplicación se abrirá en 1600x900 píxeles

### Paso 2: Configuración de API (Opcional)
Para usar datos reales de Alpaca Markets:

1. Ve a la **Pestaña 1: Data Management**
2. Ingresa tus credenciales:
   - API Key de Alpaca
   - Secret Key de Alpaca
3. Haz clic en **"Connect"**
4. Espera confirmación de conexión exitosa

**Modo Demo**: Si no tienes credenciales, la aplicación funciona con datos demo precargados.

---

## 📊 Guía de Uso por Pestaña

### 📊 Pestaña 1: Data Management

**Propósito**: Cargar y gestionar datos de mercado para backtesting.

#### Cómo Usar:

1. **Configurar API** (si aplica):
   ```
   API Key: [Tu API Key de Alpaca]
   Secret Key: [Tu Secret Key]
   → Clic en "Connect"
   ```

2. **Seleccionar Parámetros de Datos**:
   - **Symbol**: Selecciona `BTCUSD`, `ETHUSD`, etc.
   - **Timeframe**: Elige `5Min`, `15Min`, `1Hour`, `1Day`
   - **Fechas**: Define rango de datos (ej: 2023-01-01 a 2024-01-01)
   - **Multi-Timeframe**: ✅ Activa para análisis en múltiples temporalidades

3. **Cargar Datos**:
   - Clic en **"Load Data"**
   - Observa barra de progreso
   - Verifica preview de datos en tabla inferior

4. **Vista Previa**:
   - La tabla muestra primeras 50 filas
   - Columnas: Date, Open, High, Low, Close, Volume
   - Verifica que no haya valores NaN o incorrectos

#### Casos de Uso:
- **Trading intradiario**: Usa 5Min con 1 mes de datos
- **Swing trading**: Usa 1Hour con 6 meses de datos
- **Análisis de largo plazo**: Usa 1Day con 2+ años de datos

#### ⚠️ Puntos de Atención:
- Si API falla, se usa caché local automáticamente
- Multi-timeframe aumenta tiempo de carga pero mejora análisis
- Datos se guardan en caché para uso posterior

---

### ⚙️ Pestaña 2: Strategy Config

**Propósito**: Configurar y personalizar estrategias de trading.

#### Cómo Usar:

1. **Seleccionar Estrategia**:
   - Abre dropdown "Available Strategies"
   - Opciones disponibles:
     - **IBS_BB**: Mean reversion con Bollinger Bands
     - **MACD_ADX**: Momentum con MACD y ADX
     - **PAIRS_TRADING**: Trading de pares por cointegración
     - **HFT_VMA**: High frequency con VMA
     - **LSTM_ML**: Machine Learning con LSTM

2. **Ver Descripción**:
   - Cada estrategia muestra descripción automáticamente
   - Lee características y mejor uso de la estrategia

3. **Ajustar Parámetros**:
   - Los parámetros aparecen dinámicamente
   - Ejemplo para IBS_BB:
     - `atr_multi`: 0.1 - 0.5 (multiplicador de ATR para stops)
     - `vol_thresh`: 0.8 - 2.0 (umbral de volatilidad)
   - Usa sliders o spinboxes para ajustar

4. **Guardar Presets**:
   - Ingresa nombre del preset: `"Crypto_Conservative"`
   - Clic en **"Save Preset"**
   - Tu configuración se guarda para uso futuro

5. **Cargar Presets**:
   - Selecciona preset del dropdown
   - Clic en **"Load Preset"**
   - Parámetros se cargan automáticamente

6. **Vista Previa de Señales**:
   - La tabla inferior muestra señales simuladas
   - Columnas: Timestamp, Signal Type, Price, Strength, Components
   - Verifica que la lógica sea la esperada

#### Mejores Prácticas:
- Empieza con parámetros por defecto
- Ajusta gradualmente un parámetro a la vez
- Guarda configuraciones exitosas como presets
- Usa nombres descriptivos para presets

#### 💡 Tips de Estrategias:

**IBS_BB** - Mejor para:
- Mercados laterales
- Reversiones a la media
- Timeframes: 5Min - 1Hour

**MACD_ADX** - Mejor para:
- Tendencias fuertes
- Breakouts
- Timeframes: 15Min - 4Hour

**PAIRS_TRADING** - Mejor para:
- Correlaciones estables
- Market neutral
- Timeframes: 1Hour - 1Day

---

### ▶️ Pestaña 3: Backtest Runner

**Propósito**: Ejecutar backtests con diferentes metodologías.

#### Cómo Usar:

1. **Seleccionar Modo de Backtest**:
   
   **Simple Backtest**:
   - Más rápido
   - Ejecuta estrategia sobre todo el dataset
   - Ideal para pruebas iniciales
   
   **Walk-Forward**:
   - Más robusto
   - Divide datos en períodos (train/test)
   - Configura períodos: 3-12 (recomendado: 6-8)
   - Detecta overfitting
   
   **Monte Carlo**:
   - Análisis de robustez
   - Permuta orden de trades
   - Configura runs: 100-2000 (recomendado: 500)
   - Evalúa estabilidad de resultados

2. **Configurar Parámetros**:
   - **Períodos** (Walk-Forward): Más períodos = más conservador
   - **Runs** (Monte Carlo): Más runs = mayor confianza estadística

3. **Ejecutar Backtest**:
   - Asegúrate de tener datos cargados (Pestaña 1)
   - Asegúrate de tener estrategia configurada (Pestaña 2)
   - Clic en **"Run Backtest"**
   - Observa progreso en barra

4. **Interpretar Resultados**:
   - Resultados se guardan automáticamente
   - Tabla de métricas muestra:
     - **Total Return**: Retorno total del período
     - **Sharpe Ratio**: Retorno ajustado por riesgo (>1.5 es bueno)
     - **Max Drawdown**: Pérdida máxima desde pico (menor es mejor)
     - **Win Rate**: % de trades ganadores
     - **Profit Factor**: Ganancias/Pérdidas (>1.5 es bueno)

5. **Análisis Detallado**:
   - Ve a **Pestaña 4: Results Analysis** para gráficos
   - Todos los resultados se transfieren automáticamente

#### Workflow Recomendado:

```
1. Simple Backtest (prueba rápida)
   ↓ Si prometedor
2. Walk-Forward (validación robustez)
   ↓ Si degradation < 30%
3. Monte Carlo (análisis estabilidad)
   ↓ Si std_sharpe < 0.3
4. A/B Testing (comparar con otras estrategias)
```

#### ⚠️ Señales de Advertencia:
- **Sharpe < 0.5**: Estrategia poco rentable
- **Max DD > 30%**: Riesgo excesivo
- **Win Rate < 40%**: Necesita ajustes
- **Degradación > 40%** (WF): Overfitting probable
- **High std_sharpe** (MC): Resultados inestables

---

### 📈 Pestaña 4: Results Analysis

**Propósito**: Visualización profunda de resultados de backtesting.

#### Cómo Usar:

1. **Gráficos Interactivos** (Pestañas superiores):

   **Equity Curve**:
   - Muestra evolución del capital
   - Línea verde: Capital creciendo
   - Línea roja: Drawdown periods
   - Zoom: Arrastra para seleccionar área
   - Pan: Shift + Arrastra

   **Win/Loss Distribution**:
   - Histograma de PnL por trade
   - Verde: Trades ganadores
   - Rojo: Trades perdedores
   - Evalúa simetría de distribución

   **Parameter Sensitivity**:
   - Heatmap de rendimiento vs parámetros
   - Colores cálidos: Mejor performance
   - Identifica rangos óptimos de parámetros

2. **Trade Log** (Tabla inferior izquierda):
   - **Filtrar trades**:
     - ✅ "Score >= 4 only": Muestra solo trades de alta calidad
   - **Columnas**:
     - **Entry/Exit**: Precios de entrada/salida
     - **PnL%**: Ganancia/Pérdida porcentual
     - **Score**: 0-5 (calidad de la señal)
     - **MAE%**: Maximum Adverse Excursion
   - **Doble clic** en trade: Ve detalles completos

3. **Exportar Trades**:
   - Clic en **"Export CSV"**
   - Selecciona ubicación
   - Archivo incluye todos los detalles de trades

4. **Estadísticas** (Panel derecho):

   **Good Entries (Score >= 4)**:
   - Win Rate de entradas de alta calidad
   - PnL promedio
   - Sharpe ratio de buenos trades
   
   **Bad Entries (Score < 4)**:
   - Análisis de entradas de baja calidad
   - Identifica por qué fallan
   
   **Recommendation**:
   - Análisis automático con sugerencias
   - Ejemplos:
     - "Focus on score >= 4 trades only"
     - "Reduce position size on low scores"
     - "Excellent consistency across all entries"

#### Análisis Avanzado:

**Identificar Problemas**:
- Si buenos trades ganan pero malos trades pierden mucho → Filtrar por score
- Si MAE% es alto → Stops muy amplios
- Si distribución sesgada a pérdidas → Revisar parámetros

**Optimización**:
1. Identifica score mínimo rentable (ej: >= 3.5)
2. Ajusta estrategia para generar más señales de ese score
3. Re-backtest con filtro aplicado

---

### 🔄 Pestaña 5: A/B Testing

**Propósito**: Comparar estadísticamente dos estrategias.

#### Cómo Usar:

1. **Seleccionar Estrategias**:
   - **Strategy A**: Estrategia de referencia (ej: IBS_BB)
   - **Strategy B**: Estrategia a comparar (ej: MACD_ADX)

2. **Ejecutar Comparación**:
   - Clic en **"Run A/B Test"**
   - Espera progreso (ejecuta ambas estrategias)

3. **Interpretar Resultados**:

   **Tabla de Métricas Comparativas**:
   ```
   Métrica       | Strategy A | Strategy B | Delta  | % Change
   Sharpe        | 1.8        | 2.1        | +0.3   | +16.7%
   Win Rate      | 65%        | 70%        | +5%    | +7.7%
   Max DD        | 12%        | 8%         | -4%    | -33.3%
   Profit Factor | 1.9        | 2.2        | +0.3   | +15.8%
   ```

   **Pruebas Estadísticas**:
   - **T-Test p-value**: < 0.05 = diferencia significativa
   - **Sharpe Difference**: Magnitud de mejora en risk-adjusted return

4. **Recomendación Automática**:
   - Sistema evalúa resultados y recomienda:
     - "Strategy B is statistically superior (p < 0.05)"
     - "No significant difference detected"
     - "Strategy A more stable despite lower returns"

#### Casos de Uso:

**Optimización de Parámetros**:
- A: Parámetros originales
- B: Parámetros optimizados
- ¿La optimización mejora realmente?

**Comparación de Familias**:
- A: Estrategia mean reversion
- B: Estrategia momentum
- ¿Cuál funciona mejor en datos actuales?

**Validación de Mejoras**:
- A: Versión 1.0 de estrategia
- B: Versión 2.0 con mejoras
- ¿Las mejoras son significativas?

#### 💡 Tips:
- Usa mismos datos para ambas estrategias
- p-value < 0.05 indica confianza 95%
- Delta grande pero p-value alto = azar, no mejora real
- Considera otros factores (drawdown, estabilidad) además de returns

---

### 🔴 Pestaña 6: Live Monitoring

**Propósito**: Monitoreo en tiempo real y paper trading.

#### Cómo Usar:

1. **Iniciar Monitoreo**:
   - Configura API en Pestaña 1 (si aún no)
   - Selecciona estrategia a monitorear
   - Clic en **"Start Monitoring"**
   - Estado cambia a "Monitoring active"

2. **Panel de PnL**:
   - **Gauge circular**: PnL actual en tiempo real
   - Verde: Ganancia
   - Rojo: Pérdida
   - Actualización cada 5 segundos

3. **Métricas en Vivo**:
   - **Sharpe Live**: Sharpe ratio del día actual
   - **Calmar Live**: Calmar ratio en vivo
   - **Win Rate Live**: Win rate de hoy
   - **DD Live**: Drawdown actual
   - **Trades Today**: Número de trades ejecutados

4. **Log de Señales**:
   - Tabla muestra señales detectadas en tiempo real
   - Columnas:
     - **Timestamp**: Hora exacta de señal
     - **Type**: BUY/SELL
     - **Price**: Precio al que se generó señal
     - **Strength**: 1-5 (confianza de la señal)
     - **Reason**: Componente que activó señal

5. **Historial de Trades**:
   - Trades ejecutados hoy
   - Entry/Exit automáticos
   - PnL actualizado

6. **Detener Monitoreo**:
   - Clic en **"Stop Monitoring"**
   - Sistema cierra posiciones abiertas (si aplica)
   - Guarda log del día

#### Modo Demo vs Modo Real:

**Modo Demo** (sin API keys):
- Simula señales y PnL
- No ejecuta trades reales
- Ideal para familiarizarse con interfaz

**Modo Real** (con API keys):
- Conecta a Alpaca Paper Trading
- Ejecuta trades simulados con dinero virtual
- Datos y ejecución reales

#### ⚠️ Alertas Importantes:
- **High DD Alert**: Si drawdown > 15%, considera detener
- **Low Sharpe Alert**: Si Sharpe < 0.5 durante el día, revisa estrategia
- **API Disconnect**: Sistema notifica y guarda estado

#### Workflow Típico:
```
08:00 - Inicio del día
   ↓
09:00 - Start Monitoring
   ↓
Durante el día - Observar señales y métricas
   ↓
16:00 - Revisar performance
   ↓
17:00 - Stop Monitoring
   ↓
Análisis - Comparar con backtest
```

---

### 🔬 Pestaña 7: Advanced Analysis

**Propósito**: Análisis avanzado de robustez y causalidad.

#### Cómo Usar:

1. **Análisis de Regímenes**:

   **Paso a Paso**:
   - Clic en **"Run Regime Analysis"**
   - Sistema detecta regímenes de mercado:
     - **Bull**: Tendencia alcista
     - **Bear**: Tendencia bajista
     - **Sideways**: Mercado lateral
   
   **Interpretar Resultados**:
   - **Distribución de Regímenes**:
     - Bull: 35%, Bear: 25%, Sideways: 40%
   - **Performance por Régimen**:
     - ¿Tu estrategia funciona mejor en qué régimen?
   - **Transiciones**:
     - ¿Cuándo cambia el mercado de régimen?

   **Uso Práctico**:
   - Adapta estrategia según régimen detectado
   - Filtra señales según régimen favorable
   - Reduce exposición en regímenes desfavorables

2. **Stress Testing**:

   **Configurar Escenarios**:
   - ✅ Market Crash (-20%)
   - ✅ Flash Crash (-10% en 1 hora)
   - ✅ High Volatility (vol × 2)
   - ✅ Low Liquidity (spread × 3)
   - ✅ Gap Up/Down (5%)

   **Ejecutar Test**:
   - Selecciona escenarios
   - Clic en **"Run Stress Test"**
   
   **Resultados**:
   ```
   Scenario        | Impact on PnL | Max DD | Recovery Time
   Market Crash    | -18%          | 25%    | 45 days
   Flash Crash     | -8%           | 12%    | 7 days
   High Volatility | +5%           | 18%    | N/A (positive)
   ```

   **Análisis**:
   - Identifica vulnerabilidades
   - Cuantifica pérdidas en eventos extremos
   - Planifica cobertura o ajustes

3. **Causality Testing**:

   **Propósito**:
   - ¿Las señales realmente predicen returns?
   - ¿O es correlación espuria?

   **Pruebas Ejecutadas**:
   - **Granger Causality**:
     - p-value < 0.05: Señales causan returns ✅
     - p-value > 0.05: No hay causalidad ❌
   
   - **Placebo Test**:
     - Compara con señales aleatorias
     - Tu estrategia debe superar placebo

   **Interpretación**:
   ```
   Granger p-value: 0.02 → Causalidad confirmada ✅
   Placebo p-value: 0.78 → Mejor que azar ✅
   
   → Estrategia tiene poder predictivo real
   ```

#### Workflow Avanzado:

```
1. Regime Analysis
   ↓ Identifica régimen actual
2. Ajusta estrategia por régimen
   ↓
3. Stress Testing
   ↓ Evalúa escenarios extremos
4. Implementa protecciones (stops, hedging)
   ↓
5. Causality Testing
   ↓ Valida que señales son predictivas
6. Deploy a Live Monitoring
```

#### 💡 Insights Avanzados:

**Si Granger p-value > 0.05**:
- Señales no predicen returns
- Posible overfitting
- Revisa lógica de estrategia

**Si Stress Test muestra DD > 40%**:
- Implementa circuit breakers
- Reduce tamaño de posición
- Considera hedging dinámico

**Si Régimen Bull tiene mejor performance**:
- Aumenta exposición en Bull
- Reduce o invierte en Bear
- Neutral en Sideways

---

## 🎯 Casos de Uso Avanzados

### Caso 1: Desarrollar Nueva Estrategia desde Cero

```
Día 1-2: Investigación y Diseño
→ Pestaña 2: Crear configuración de parámetros
→ Guardar como preset "Nueva_Estrategia_v1"

Día 3: Testing Inicial
→ Pestaña 1: Cargar 6 meses de datos 5Min
→ Pestaña 3: Simple Backtest
→ Pestaña 4: Analizar resultados
   Si Sharpe > 1.0 → Continuar
   Si Sharpe < 0.5 → Rediseñar

Día 4: Validación de Robustez
→ Pestaña 3: Walk-Forward (8 períodos)
   Si degradación < 30% → Continuar
   Si degradación > 50% → Overfitting, revisar

Día 5: Análisis de Estabilidad
→ Pestaña 3: Monte Carlo (500 runs)
   Si std_sharpe < 0.3 → Estable ✅
→ Pestaña 7: Stress Testing
   Verificar comportamiento en crisis

Día 6: Validación Estadística
→ Pestaña 5: A/B Test vs estrategia benchmark
→ Pestaña 7: Causality Testing
   Si p-value < 0.05 → Causalidad confirmada ✅

Día 7+: Paper Trading
→ Pestaña 6: Live Monitoring (1 semana mínimo)
   Comparar live vs backtest
   Si desviación < 20% → Listo para deployment
```

### Caso 2: Optimizar Estrategia Existente

```
1. Baseline
→ Pestaña 3: Simple Backtest con parámetros actuales
→ Anota métricas baseline

2. Parameter Sweep
→ Pestaña 2: Ajusta parámetros uno a uno
→ Para cada ajuste:
   - Run backtest
   - Compara con baseline via A/B Test
   - Guarda mejores configuraciones

3. Validación Multi-Régimen
→ Pestaña 7: Regime Analysis
→ Verifica performance en cada régimen
→ Ajusta parámetros por régimen si necesario

4. Stress Test Optimización
→ Pestaña 7: Ejecuta escenarios extremos
→ Asegura que optimización no sacrifica robustez

5. Deploy Optimizado
→ Guarda configuración como "Estrategia_v2"
→ Monitor en vivo por 2 semanas
→ Si mejora confirmada → Producción
```

### Caso 3: Portfolio de Estrategias

```
1. Desarrolla 3-5 estrategias diferentes
→ Mean Reversion
→ Momentum
→ Pairs Trading

2. Backtesting Individual
→ Cada estrategia en Pestaña 3
→ Métricas individuales en Pestaña 4

3. Correlation Analysis
→ Pestaña 7: Analiza correlación entre estrategias
→ Objetivo: Baja correlación para diversificación

4. A/B Testing Pairwise
→ Pestaña 5: Compara cada par
→ Identifica complementariedades

5. Regime Allocation
→ Pestaña 7: Regime Analysis
→ Asigna estrategia óptima por régimen:
   - Bull: Momentum
   - Bear: Short Bias / Pairs
   - Sideways: Mean Reversion

6. Live Portfolio Monitoring
→ Pestaña 6: Monitor todas simultáneamente
→ Rebalanceo dinámico según régimen
```

---

## 🔧 Solución de Problemas

### Problema: "No se pueden cargar datos"

**Síntomas**:
- Error en Pestaña 1
- Mensaje "API Error" o "Connection Failed"

**Soluciones**:
1. Verifica credenciales API (copy-paste sin espacios)
2. Verifica conexión a internet
3. Usa modo caché (datos precargados)
4. Reduce rango de fechas (muy amplio puede timeout)

### Problema: "Backtest muy lento"

**Síntomas**:
- Pestaña 3 tarda >5 minutos
- Aplicación no responde

**Soluciones**:
1. Reduce cantidad de datos:
   - Usa menor timeframe (1Hour en vez de 5Min)
   - Reduce rango de fechas
2. Simplifica estrategia:
   - Menos indicadores
   - Lógica más directa
3. Walk-Forward: Reduce períodos a 4-6
4. Monte Carlo: Reduce runs a 100-200

### Problema: "Resultados inconsistentes entre backtests"

**Síntomas**:
- Sharpe varía mucho entre ejecuciones
- Métricas cambian sin cambiar parámetros

**Causas Posibles**:
1. Estrategia usa randomness no seeded
2. Datos cambiaron (recarga desde API)
3. Multi-threading crea race conditions

**Soluciones**:
1. Limpia caché y recarga datos
2. Fija random seed en estrategia
3. Ejecuta Monte Carlo para evaluar variabilidad

### Problema: "Aplicación se cierra inesperadamente"

**Síntomas**:
- Crash sin mensaje de error
- Cierre durante operación

**Soluciones**:
1. Verifica logs en `/logs/platform.log`
2. Ejecuta desde terminal para ver output:
   ```bash
   ./main_platform.exe 2>&1 | tee output.log
   ```
3. Reduce complejidad de operación
4. Aumenta RAM disponible (cierra otras apps)
5. Reinstala Visual C++ Redistributables

### Problema: "Gráficos no se visualizan" (Pestaña 4)

**Síntomas**:
- Pestañas de gráficos vacías
- WebEngine no carga

**Soluciones**:
1. Verifica que resultados de backtest estén disponibles
2. Re-ejecuta backtest en Pestaña 3
3. Cambia de pestaña de gráfico (Equity → Distribution)
4. Reinstala PySide6-WebEngine

---

## ✅ Mejores Prácticas

### Development Workflow

1. **Siempre empieza con datos limpios**:
   - Recarga datos frescos al inicio
   - Verifica preview sin NaN
   - Confirma rango de fechas correcto

2. **Iteración progresiva**:
   ```
   Simple Backtest → Walk-Forward → Monte Carlo → A/B Test → Live
   ```
   No saltes pasos, cada uno valida aspectos diferentes

3. **Documentación de configuraciones**:
   - Usa presets con nombres descriptivos
   - Fecha versiones: "IBS_BB_v1_2024_01"
   - Anota cambios en cada versión

4. **Validación cruzada**:
   - Si backtest excelente (Sharpe > 3):
     - ⚠️ Sospecha de overfitting
     - Valida con Walk-Forward inmediatamente
   - Si Walk-Forward degrada mucho:
     - Simplifica estrategia
     - Reduce parámetros optimizados

### Risk Management

1. **Nunca confíes ciegamente en backtest**:
   - Siempre usa Walk-Forward
   - Siempre usa Monte Carlo
   - Siempre ejecuta Stress Testing

2. **Define límites antes de live**:
   - Max DD aceptable (ej: 15%)
   - Min Sharpe aceptable (ej: 1.0)
   - Max position size
   - Daily loss limit

3. **Monitoreo continuo**:
   - Revisa Live Monitoring diariamente
   - Compara live vs backtest semanalmente
   - Si desviación > 30%, detén y analiza

### Performance Optimization

1. **Datos**:
   - Usa timeframe adecuado al estilo de trading
   - No cargues más datos de los necesarios
   - Limpia caché periódicamente

2. **Backtesting**:
   - Simple: Para iteración rápida
   - Walk-Forward: Para validación final
   - Monte Carlo: Para publicación/deployment

3. **Estrategias**:
   - Menos indicadores = más rápido
   - Vectorización > loops
   - Cache resultados intermedios

### Statistical Rigor

1. **Significancia estadística**:
   - A/B Test p-value < 0.05
   - Monte Carlo con 500+ runs
   - Causality Testing obligatorio antes de live

2. **Out-of-sample testing**:
   - Walk-Forward simula OOS
   - Reserva últimos 20% de datos para test final
   - Nunca optimices sobre datos completos

3. **Multiple testing correction**:
   - Si pruebas 20 configuraciones:
     - 1 probablemente sea buena por azar
     - Usa Bonferroni correction: p-value < 0.05/20 = 0.0025

---

## 📚 Recursos Adicionales

### Logs y Debugging

**Ubicación de logs**:
```
/logs/platform.log          - General platform operations
/logs/data_loading.log      - Data fetching issues
/logs/backtest.log          - Backtest execution
/logs/live_monitor.log      - Live trading activity
```

**Leer logs**:
```bash
tail -f logs/platform.log   # Ver en tiempo real
grep ERROR logs/*.log       # Buscar errores
```

### Exportación de Resultados

**Trades CSV**:
- Pestaña 4 → Export CSV
- Columnas: entry_time, exit_time, pnl, score, etc.
- Compatible con Excel, Python, R

**Configuraciones JSON**:
- Presets se guardan en `/config/presets.json`
- Editable manualmente para batch operations

**Figuras**:
- Gráficos HTML son interactivos
- Se pueden guardar como PNG desde navegador integrado

---

## 🎓 Conclusión

Esta plataforma cubre el ciclo completo de desarrollo de estrategias de trading:

1. **Research** → Pestaña 1, 2
2. **Backtesting** → Pestaña 3, 4
3. **Validation** → Pestaña 5, 7
4. **Deployment** → Pestaña 6

**Recuerda**:
- 🔴 Backtest perfecto = 🚩 Red flag (overfitting)
- ✅ Consistencia > Rendimiento máximo
- 📊 Validación estadística es obligatoria
- 🧪 Paper trading antes de dinero real

Para soporte adicional, consulta:
- README_PLATFORM.md
- Código fuente en `/src`
- Tests en `/tests` para ejemplos de uso

---

**Happy Trading! 🚀📈**
