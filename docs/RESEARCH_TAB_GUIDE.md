# Guía de Usuario - Pestaña Research (Lab de Investigación)

## 📋 ¿Qué es la Pestaña Research?

La pestaña **Research** (también llamada "Lab de Investigación") es un espacio avanzado donde puedes:

1. **Probar hipótesis** sobre estrategias de trading
2. **Analizar importancia de indicadores** técnicos
3. **Detectar regímenes de mercado** (alcista, bajista, lateral)
4. **Estudiar correlaciones** entre indicadores

## ⚠️ IMPORTANTE: Esta pestaña NO es obligatoria

- **Si eres nuevo**: Ignora esta pestaña por ahora. Enfócate en Live, Backtest y Data.
- **Si eres avanzado**: Usa esta pestaña para análisis estadístico profundo.

## 🎯 ¿Para Qué Sirve Cada Herramienta?

### 1. **Test de Hipótesis** 📊

**¿Qué hace?**
- Compara dos estrategias para ver si una es estadísticamente mejor que la otra
- Usa pruebas estadísticas (t-test) para determinar si la diferencia es real o suerte

**¿Cuándo usarlo?**
- Tienes dos estrategias y quieres saber cuál es objetivamente mejor
- Ejemplo: "¿Es RSI mejor que MACD para BTC?"

**¿Cómo interpretarlo?**
- **p-value < 0.05**: La diferencia ES significativa (una estrategia es mejor)
- **p-value > 0.05**: La diferencia NO es significativa (ambas son similares)
- **t-statistic**: Mientras más alto, mayor la diferencia entre estrategias

**Ejemplo práctico:**
```
Hipótesis: "Estrategia RSI tiene mejor Sharpe que MACD"
Resultado: p-value = 0.03, t-stat = 2.4
Conclusión: SÍ, RSI es significativamente mejor con 95% confianza
```

---

### 2. **Importancia de Features** 🔍

**¿Qué hace?**
- Muestra qué indicadores técnicos tienen mayor impacto en tu estrategia
- Identifica cuáles indicadores son "ruido" y cuáles son útiles

**¿Cuándo usarlo?**
- Tu estrategia usa muchos indicadores (RSI, MACD, BB, etc.)
- Quieres simplificar y enfocarte solo en los importantes

**¿Cómo interpretarlo?**
- **Mayor porcentaje = más importante** para predicciones
- Indicadores con <5% de importancia pueden ser eliminados sin pérdida

**Ejemplo práctico:**
```
Resultados:
- RSI_14: 35% (MUY IMPORTANTE)
- MACD: 28% (IMPORTANTE)
- BB_Width: 15% (MODERADO)
- SMA_Cross: 8% (MENOR)
- Volume_Ratio: 5% (DESCARTABLE)

Conclusión: Enfócate en RSI y MACD, elimina Volume_Ratio
```

---

### 3. **Análisis de Correlación** 🔗

**¿Qué hace?**
- Muestra cómo se relacionan diferentes indicadores entre sí
- Detecta redundancia (dos indicadores que miden lo mismo)

**¿Cuándo usarlo?**
- Quieres optimizar tu estrategia eliminando indicadores redundantes
- Buscas diversificar con indicadores no correlacionados

**¿Cómo interpretarlo?**
- **Correlación cercana a +1 o -1**: Indicadores muy relacionados (redundantes)
- **Correlación cercana a 0**: Indicadores independientes (buenos para combinar)

**Ejemplo práctico:**
```
Correlación RSI - MACD: 0.85 (MUY ALTA)
Conclusión: Ambos miden cosas similares, usa solo uno

Correlación RSI - Volume: 0.12 (BAJA)
Conclusión: Son independientes, combinarlos agrega valor
```

---

### 4. **Detección de Regímenes** 📈📉

**¿Qué hace?**
- Clasifica el mercado en diferentes estados (alcista, bajista, lateral)
- Usa modelos estadísticos (HMM - Hidden Markov Models) para detectar cambios

**¿Cuándo usarlo?**
- Quieres estrategias adaptativas (diferentes para cada tipo de mercado)
- Necesitas saber cuándo cambiar de estrategia

**¿Cómo interpretarlo?**
- **Régimen 1 (Verde)**: Típicamente mercado alcista (comprar)
- **Régimen 2 (Rojo)**: Típicamente mercado bajista (vender/evitar)
- **Régimen 3 (Amarillo)**: Mercado lateral (rango)

**Ejemplo práctico:**
```
Detección actual: Régimen 2 (Bajista)
Recomendación: Evita estrategias de momentum, usa mean reversion
Duración promedio: 15 días
Probabilidad de cambio: 25%
```

---

## 🚀 Flujo de Trabajo Recomendado

### Para Principiantes:
1. **Ignora Research por ahora** - enfócate en Backtest y Live
2. Cuando tengas 2-3 estrategias funcionando, vuelve aquí
3. Usa "Test de Hipótesis" para comparar tus estrategias

### Para Intermedios:
1. Haz backtest de tu estrategia
2. Usa "Importancia de Features" para ver qué indicadores son clave
3. Simplifica tu estrategia eliminando indicadores de <10% importancia
4. Re-testea con estrategia simplificada

### Para Avanzados:
1. Usa "Análisis de Correlación" para optimizar tu portfolio de indicadores
2. Implementa "Detección de Regímenes" para estrategias adaptativas
3. Combina todo con "Test de Hipótesis" para validar mejoras

---

## 💡 Consejos Prácticos

### ✅ DO (Hacer):
- Usa Research DESPUÉS de tener resultados de backtest
- Enfócate en 1-2 herramientas a la vez
- Toma notas de tus hallazgos
- Valida con backtest cualquier cambio que hagas

### ❌ DON'T (No Hacer):
- No uses Research si aún no entiendes backtesting básico
- No cambies tu estrategia basándote en UN solo análisis
- No te abrumes intentando usar todas las herramientas a la vez
- No ignores el sentido común por seguir estadísticas ciegamente

---

## 🎓 Glosario de Términos

### Términos Estadísticos:
- **p-value**: Probabilidad de que el resultado sea por azar. <0.05 = significativo
- **t-statistic**: Medida de cuán diferentes son dos grupos
- **Confianza**: Nivel de certeza (95% = muy confiable, 80% = moderado)

### Términos de ML:
- **Feature**: Un indicador técnico (RSI, MACD, etc.)
- **Importancia**: Qué tanto contribuye un indicador a predicciones
- **Correlación**: Cómo se relacionan dos variables (-1 a +1)

### Términos de Trading:
- **Régimen**: Estado o fase del mercado (alcista/bajista/lateral)
- **HMM**: Modelo estadístico que detecta cambios de régimen
- **Sharpe Ratio**: Retorno ajustado por riesgo (>1.5 es bueno)

---

## ❓ Preguntas Frecuentes

### P: ¿Necesito saber estadística para usar Research?
**R:** No obligatorio, pero ayuda. Empieza con "Importancia de Features" que es más intuitivo.

### P: ¿Los resultados de Research garantizan éxito?
**R:** NO. Son herramientas de análisis, no garantías. Siempre valida con backtest.

### P: ¿Qué herramienta es la más útil?
**R:** "Importancia de Features" - te dice qué indicadores realmente importan.

### P: ¿Puedo usar Research sin entender matemáticas?
**R:** Sí, lee las conclusiones y recomendaciones. Ignora los detalles técnicos.

### P: ¿Con qué frecuencia debo usar Research?
**R:** Una vez por semana o después de cambios importantes en tu estrategia.

---

## 📚 Recursos Adicionales

### Si quieres aprender más:
1. **Test de Hipótesis**: Busca "t-test for trading strategies"
2. **Feature Importance**: Busca "Random Forest feature importance"
3. **Correlación**: Busca "correlation in trading indicators"
4. **Regímenes**: Busca "Hidden Markov Models for trading"

### Orden de aprendizaje sugerido:
1. Primero: Domina Backtest (pestaña 3)
2. Segundo: Entiende métricas básicas (Sharpe, Drawdown)
3. Tercero: Usa Feature Importance
4. Cuarto: Explora Test de Hipótesis
5. Quinto: Avanzado - Regímenes y Correlación

---

## 🎯 Ejemplo de Uso Completo

### Escenario: "Tengo una estrategia RSI pero quiero mejorarla"

**Paso 1**: Haz backtest de tu estrategia RSI actual
```
Resultado: Sharpe = 1.2, Win Rate = 52%
```

**Paso 2**: Ve a Research → Feature Importance
```
Descubres que RSI solo aporta 40% de importancia
Hay otros indicadores que podrían ayudar
```

**Paso 3**: Agrega MACD a tu estrategia y re-testea
```
Nueva estrategia: Sharpe = 1.6, Win Rate = 58%
```

**Paso 4**: Ve a Research → Test de Hipótesis
```
Comparas RSI solo vs RSI+MACD
p-value = 0.02 → ¡La combinación es significativamente mejor!
```

**Paso 5**: Ve a Research → Correlación
```
RSI-MACD correlación = 0.65 (moderada)
No son redundantes, combinarlos tiene sentido
```

**Conclusión**: Implementa RSI+MACD en Live Trading

---

**Fecha de creación**: 14 de noviembre de 2025
**Versión**: 1.0 - Guía para usuarios
**Estado**: ✅ Documento de referencia
