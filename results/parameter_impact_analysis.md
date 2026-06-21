# Análisis de Impacto de Parámetros

**Fecha:** 16/11/2025 15:01:06

## Resumen Ejecutivo

Este análisis evalúa cómo cada parámetro afecta el rendimiento de la estrategia de trading Squeeze Momentum + ADX + TTM Waves.

## Tabla de Impacto de Parámetros

| Parámetro | Rango | Corr ME | Corr Win Rate | Corr PF | Mejor Valor |
|-----------|-------|---------|----------------|---------|-------------|
| ADX Min | 14 - 22 | -0.888 | -0.888 | -0.888 | 15 |
| Squeeze Min | 0.12 - 0.3 | -0.811 | -0.811 | -0.811 | 0.15 |
| SL ATR | 0.8 - 1.5 | -0.362 | -0.362 | -0.362 | 1.0 |
| TP ATR | 3.0 - 3.0 | 0.000 | 0.000 | 0.000 | 3.0 |
| Cooldown | 1 - 2 | 0.548 | 0.548 | 0.548 | 1 |
| Trailing Activation | 0.5 - 0.7 | -0.488 | -0.488 | -0.488 | 0.5 |
| EMA Filter | False - True | 1.000 | 1.000 | 1.000 | True |

## Interpretación de Correlaciones

- **Correlación > 0.3**: Parámetro tiene impacto positivo significativo
- **Correlación < -0.3**: Parámetro tiene impacto negativo significativo
- **Correlación entre -0.3 y 0.3**: Impacto neutral o no significativo

## Recomendaciones para Optimización

### Parámetros a Priorizar:
- **EMA Filter**: Enfocarse en valor True (correlación ME: {data['me_correlation']:.3f})
- **ADX Min**: Enfocarse en valor 15 (correlación ME: {data['me_correlation']:.3f})
- **Squeeze Min**: Enfocarse en valor 0.15 (correlación ME: {data['me_correlation']:.3f})

### Parámetros a Eliminar/Reducir:
- **T**: Bajo impacto en rendimiento

### Estrategias para Nuevos Parámetros:
- Considerar rangos más amplios para parámetros con correlación baja
- Probar combinaciones no lineales de parámetros correlacionados
- Implementar validación cruzada para evitar sobre-optimización
