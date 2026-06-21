# Guía: Sistema de Evaluación de Patrones Condicionales

## 📋 Resumen

Este documento explica cómo usar el **Sistema de Evaluación de Patrones Condicionales** para responder preguntas como:

1. ¿Cuándo el precio toca la EMA22 con squeeze negativo y está debajo del POC, se aleja en dirección contraria?
2. ¿Qué parámetros se repiten cuando hay volumen alto y movimiento importante?
3. ¿Cuál es la probabilidad de éxito de IFVG + confirmación multi-timeframe?

---

## 🎯 Objetivo

**Crear un modelo que evalúe casos específicos y encuentre patrones con mayor probabilidad de éxito.**

En lugar de backtestear estrategias completas, este sistema:
- ✅ Evalúa **condiciones específicas** (ej: "precio cerca EMA22")
- ✅ Mide **probabilidad de éxito** de cada patrón
- ✅ Identifica **parámetros óptimos** en trades ganadores
- ✅ Genera **reglas de trading automáticas**

---

## 🚀 Archivos Creados

### 1. `strategies/vp_ifvg_ema_strategy_v2.py`
**Estrategia refactorizada con gestión de posiciones**

**Mejoras principales:**
- ✅ Tracking de posición actual (long/short/flat)
- ✅ Stop Loss / Take Profit dinámicos (basados en ATR)
- ✅ Risk management (2% capital por trade)
- ✅ Trailing stop opcional
- ✅ Exit logic basada en múltiples condiciones

**Uso:**
```python
from strategies.vp_ifvg_ema_strategy_v2 import VPIFVGEmaStrategyV2

strategy = VPIFVGEmaStrategyV2()
signals = strategy.generate_signals(df_multi_tf)

# Obtener información de trades
info = strategy.get_strategy_info()
print(f"Total trades: {info['total_trades']}")
print(f"Win rate: {info['winning_trades'] / info['total_trades']:.2%}")
```

### 2. `scripts/conditional_pattern_evaluator.py`
**Sistema de evaluación de patrones condicionales**

**Características:**
- 10 tipos de condiciones predefinidas
- Combinación de condiciones (AND/OR)
- Evaluación automática de win rate, expectancy, profit factor
- Auto-discovery de mejores patrones
- Generación de reportes detallados

**Condiciones disponibles:**
1. `PRICE_NEAR_EMA` - Precio cerca de EMA
2. `SQUEEZE_MOMENTUM_SLOPE` - Pendiente del momentum
3. `PRICE_VS_POC` - Precio vs Point of Control
4. `VOLUME_HIGH` - Volumen alto
5. `PRICE_MOVEMENT_LARGE` - Movimiento grande
6. `IFVG_PRESENT` - IFVG presente
7. `EMA_CROSS` - Cruce de EMAs
8. `ATR_EXPANSION` - Expansión de ATR
9. `MULTI_TF_ALIGNED` - Alineación multi-timeframe
10. `VOLATILITY_SPIKE` - Spike de volatilidad

### 3. `scripts/test_specific_patterns.py`
**Script de prueba para casos específicos**

Evalúa los 3 patrones mencionados con múltiples variantes cada uno.

---

## 💻 Uso Básico

### Caso 1: Evaluar Patrón Simple

```python
from scripts.conditional_pattern_evaluator import (
    ConditionalPatternEvaluator, Pattern, Condition, ConditionType
)
import pandas as pd

# Cargar datos
df = pd.read_csv('data/btc_15Min.csv', parse_dates=['timestamp'], index_col='timestamp')

# Crear evaluador
evaluator = ConditionalPatternEvaluator(
    df=df,
    forward_bars=10,  # Evaluar resultado a 10 barras
    profit_threshold=0.01  # 1% profit para considerar ganador
)

# Definir patrón: "Precio toca EMA22"
pattern = Pattern(
    name="EMA22_Touch",
    conditions=[
        Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22, 'tolerance_pct': 0.5})
    ],
    require_all=True
)

# Evaluar
results = evaluator.evaluate_patterns([pattern])

# Ver resultados
for result in results:
    print(f"Pattern: {result.pattern.name}")
    print(f"Occurrences: {result.occurrences}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Expectancy: {result.expectancy:.4f}")
```

### Caso 2: Patrón Compuesto (Múltiples Condiciones)

```python
# Patrón: EMA22 + Squeeze Negativo + Debajo POC
pattern = Pattern(
    name="EMA22_SqueezeNeg_BelowPOC",
    conditions=[
        Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22, 'tolerance_pct': 0.5}),
        Condition(ConditionType.SQUEEZE_MOMENTUM_SLOPE, {'direction': 'negative'}),
        Condition(ConditionType.PRICE_VS_POC, {'position': 'below'})
    ],
    require_all=True  # Requiere TODAS las condiciones (AND)
)

results = evaluator.evaluate_patterns([pattern])
```

### Caso 3: Auto-Discovery (Encuentra Mejores Patrones Automáticamente)

```python
# Descubrir mejores 20 patrones automáticamente
results = evaluator.auto_discover_patterns(max_patterns=20)

# Generar reporte
evaluator.generate_report(results, filename="reports/auto_patterns.md")

# Ver top 5
for i, result in enumerate(results[:5], 1):
    print(f"{i}. {result.pattern.name}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print(f"   Expectancy: {result.expectancy:.4f}")
```

---

## 🧪 Ejecutar Tests

### Test 1: Patrones Específicos del Usuario

```bash
# Activa virtual environment
.\.venv\Scripts\Activate.ps1

# Ejecuta evaluación de patrones específicos
python scripts/test_specific_patterns.py
```

**Este script evalúa:**
1. ✅ EMA22 Touch + Squeeze Negativo + Debajo POC (3 variantes)
2. ✅ Volumen Alto + Movimiento Grande (5 variantes)
3. ✅ IFVG + Volumen Alto (5 variantes)

**Output:**
- Métricas de cada patrón
- Comparación entre variantes
- Insights y recomendaciones
- Reporte completo en `reports/specific_patterns_evaluation.md`

### Test 2: Auto-Discovery

```python
# En Python:
from scripts.conditional_pattern_evaluator import ConditionalPatternEvaluator
import pandas as pd

df = pd.read_csv('data/btc_15Min.csv', parse_dates=['timestamp'], index_col='timestamp')
evaluator = ConditionalPatternEvaluator(df=df, forward_bars=10, profit_threshold=0.01)

# Descubrir mejores patrones
results = evaluator.auto_discover_patterns(max_patterns=20)

# Generar reporte
evaluator.generate_report(results)
```

---

## 📊 Interpretación de Resultados

### Métricas Clave

**1. Occurrences (Ocurrencias)**
- Número de veces que el patrón aparece
- **Mínimo recomendado: 20** para significancia estadística

**2. Win Rate**
- Porcentaje de trades ganadores
- **>55%** = Patrón interesante
- **>60%** = Patrón fuerte
- **>65%** = Patrón excepcional

**3. Expectancy**
- Ganancia esperada por trade
- **>0** = Patrón rentable
- **>0.01** (1%) = Patrón tradeable
- **>0.02** (2%) = Patrón excelente

**4. Profit Factor**
- Ratio gross profit / gross loss
- **>1.0** = Rentable
- **>1.5** = Bueno
- **>2.0** = Excelente

### Ejemplo de Resultado

```
Pattern: EMA22_Touch_SqueezeNeg_BelowPOC
  Occurrences: 45
  Win Rate: 62.22%
  Expectancy: 0.0156
  Profit Factor: 1.67
  Avg Profit: 0.0312 | Avg Loss: 0.0189
```

**Interpretación:**
- ✅ Suficientes casos (45)
- ✅ Win rate fuerte (62%)
- ✅ Expectancy positivo (1.56%)
- ✅ Profit factor bueno (1.67)
- **Conclusión: PATRÓN TRADEABLE**

---

## 🎯 Casos de Uso

### Caso A: Validar Hipótesis

**Pregunta:** "¿Cuando el precio toca EMA22 con squeeze negativo, rebota?"

```python
# Definir patrón
pattern = Pattern(
    name="EMA22_Bounce_Hypothesis",
    conditions=[
        Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22, 'tolerance_pct': 0.5}),
        Condition(ConditionType.SQUEEZE_MOMENTUM_SLOPE, {'direction': 'negative'})
    ]
)

# Evaluar
result = evaluator.evaluate_patterns([pattern])[0]

if result.win_rate > 0.55:
    print("✅ HIPÓTESIS CONFIRMADA")
else:
    print("❌ HIPÓTESIS RECHAZADA")
```

### Caso B: Optimizar Parámetros

**Pregunta:** "¿Qué tolerancia funciona mejor para EMA touch?"

```python
patterns = []
for tolerance in [0.3, 0.5, 0.7, 1.0, 1.5]:
    patterns.append(Pattern(
        name=f"EMA22_Touch_{tolerance}%",
        conditions=[
            Condition(ConditionType.PRICE_NEAR_EMA, 
                     {'period': 22, 'tolerance_pct': tolerance})
        ]
    ))

results = evaluator.evaluate_patterns(patterns)

# Mejor tolerancia
best = max(results, key=lambda x: x.expectancy)
print(f"Mejor tolerancia: {best.pattern.name}")
```

### Caso C: Encontrar Combinaciones Ganadoras

**Pregunta:** "¿Qué combinaciones de volumen + movimiento funcionan?"

```python
patterns = []
for vol_mult in [1.5, 2.0, 2.5]:
    for move_pct in [1.0, 1.5, 2.0]:
        patterns.append(Pattern(
            name=f"Vol{vol_mult}x_Move{move_pct}%",
            conditions=[
                Condition(ConditionType.VOLUME_HIGH, {'multiplier': vol_mult}),
                Condition(ConditionType.PRICE_MOVEMENT_LARGE, {'threshold_pct': move_pct})
            ]
        ))

results = evaluator.evaluate_patterns(patterns)

# Top 3 combinaciones
for i, result in enumerate(results[:3], 1):
    print(f"{i}. {result.pattern.name}: {result.expectancy:.4f}")
```

---

## 🔧 Personalización

### Agregar Nueva Condición

```python
# En conditional_pattern_evaluator.py, agregar al enum:
class ConditionType(Enum):
    # ... existentes ...
    MY_CUSTOM_CONDITION = "my_custom_condition"

# Implementar lógica en _evaluate_condition():
def _evaluate_condition(self, condition: Condition, idx: int) -> bool:
    # ... existentes ...
    
    elif condition.type == ConditionType.MY_CUSTOM_CONDITION:
        # Tu lógica aquí
        threshold = condition.params.get('threshold', 0.5)
        value = self.df.iloc[idx]['my_indicator']
        return value > threshold
```

### Cambiar Definición de "Éxito"

```python
# Cambiar profit_threshold al crear evaluador
evaluator = ConditionalPatternEvaluator(
    df=df,
    forward_bars=10,
    profit_threshold=0.02  # 2% en lugar de 1%
)
```

### Evaluar en Diferentes Timeframes

```python
# Para 1 hora:
evaluator_1h = ConditionalPatternEvaluator(
    df=df_1h,
    forward_bars=5,  # 5 horas
    profit_threshold=0.02
)

# Para 5 minutos:
evaluator_5m = ConditionalPatternEvaluator(
    df=df_5m,
    forward_bars=20,  # 100 minutos
    profit_threshold=0.005  # 0.5%
)
```

---

## 📈 Workflow Recomendado

### Paso 1: Descubrimiento
```python
# Descubrir patrones automáticamente
results = evaluator.auto_discover_patterns(max_patterns=50)

# Filtrar solo los buenos
good_patterns = [r for r in results if r.expectancy > 0.015 and r.occurrences >= 20]
```

### Paso 2: Validación
```python
# Probar variantes de los mejores patrones
# Ajustar parámetros
# Re-evaluar
```

### Paso 3: Implementación
```python
# Integrar mejores patrones en estrategia
# Agregar a VPIFVGEmaStrategyV2
# Backtestear con stops/targets
```

### Paso 4: Monitoreo
```python
# Re-evaluar patrones mensualmente
# Detectar degradación
# Ajustar parámetros si es necesario
```

---

## ⚠️ Consideraciones Importantes

### 1. Data Snooping
- ❌ NO uses el mismo dataset para descubrir Y testear patrones
- ✅ Split: 60% discovery, 20% validation, 20% test
- ✅ O usa walk-forward para validación

### 2. Overfitting
- ❌ NO ajustes parámetros al extremo
- ✅ Busca patrones robustos (funcionan con diferentes params)
- ✅ Valida en out-of-sample data

### 3. Significancia Estadística
- ❌ NO confíes en patrones con <20 ocurrencias
- ✅ Mínimo 20-30 casos para validez
- ✅ Más de 50 casos para alta confianza

### 4. Context Matters
- ❌ NO ignores el régimen de mercado
- ✅ Patrones pueden funcionar diferente en bull/bear
- ✅ Considera evaluar por régimen

---

## 🎓 Ejemplos Avanzados

### Ejemplo 1: Patrón con Peso de Condiciones

```python
pattern = Pattern(
    name="Weighted_Pattern",
    conditions=[
        Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22}, weight=2.0),  # Más importante
        Condition(ConditionType.VOLUME_HIGH, {'multiplier': 1.5}, weight=1.0),
        Condition(ConditionType.SQUEEZE_MOMENTUM_SLOPE, {'direction': 'negative'}, weight=0.5)
    ]
)
```

### Ejemplo 2: Patrón OR (Cualquier Condición)

```python
pattern = Pattern(
    name="Any_IFVG",
    conditions=[
        Condition(ConditionType.IFVG_PRESENT, {'direction': 'bullish'}),
        Condition(ConditionType.IFVG_PRESENT, {'direction': 'bearish'})
    ],
    require_all=False  # OR en lugar de AND
)
```

### Ejemplo 3: Análisis de Subconjuntos

```python
# Evaluar solo en horario específico
df_trading_hours = df.between_time('09:30', '16:00')
evaluator_hours = ConditionalPatternEvaluator(df=df_trading_hours)

# Evaluar solo en alta volatilidad
df_high_vol = df[df['atr'] > df['atr'].quantile(0.75)]
evaluator_vol = ConditionalPatternEvaluator(df=df_high_vol)
```

---

## 📚 Referencias

- **Van Tharp**: "Trade Your Way to Financial Freedom" (SQN, Expectancy)
- **Pardo**: "Design, Testing, and Optimization of Trading Systems" (Walk-forward)
- **Aronson**: "Evidence-Based Technical Analysis" (Data snooping, overfitting)

---

**Fecha:** 14 de noviembre de 2025  
**Versión:** 1.0  
**Status:** ✅ Production Ready

**Próximos pasos:**
1. Ejecutar `python scripts/test_specific_patterns.py`
2. Revisar reporte generado
3. Implementar mejores patrones en estrategia
4. Backtestear con VP+IFVG+EMAs V2
