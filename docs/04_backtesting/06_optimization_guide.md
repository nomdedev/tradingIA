# Optimización de Parámetros con Algoritmos Genéticos

## Visión General

El módulo de optimización implementa algoritmos genéticos avanzados para encontrar automáticamente los mejores parámetros de estrategias de trading. Esta funcionalidad es crucial para maximizar el rendimiento de las estrategias sin intervención manual exhaustiva.

## Características Principales

### 🧬 Algoritmos Genéticos Avanzados
- **Selección por Torneo**: Selección de padres más aptos
- **Crossover Adaptativo**: Recombinación inteligente de parámetros
- **Mutación Controlada**: Exploración del espacio de parámetros
- **Elitismo**: Preservación de las mejores soluciones

### ⚡ Optimización Paralela
- **Procesamiento Multihilo**: Ejecución simultánea de evaluaciones
- **Escalabilidad**: Configurable para diferentes tamaños de población
- **Eficiencia**: Optimización del uso de recursos del sistema

### 📊 Funciones de Fitness Múltiples
- **Sharpe Ratio**: Medida de riesgo-ajustado al rendimiento
- **Total Return**: Rendimiento absoluto
- **Calmar Ratio**: Ratio de recuperación
- **Sortino Ratio**: Ratio de Sharpe con downside deviation

### 🎯 Restricciones y Penalizaciones
- **Límites de Drawdown**: Penalización automática por alto riesgo
- **Validación de Parámetros**: Verificación de límites y tipos
- **Análisis de Sensibilidad**: Evaluación de robustez

## Arquitectura del Sistema

```
core/optimization/
├── genetic_optimizer.py      # Algoritmos genéticos principales
├── optimization_panel.py     # UI para optimización
└── __init__.py              # Módulo exports
```

### GeneticOptimizer
Clase principal que implementa el algoritmo genético:
- Configuración flexible de parámetros
- Evaluación paralela de fitness
- Evolución generacional
- Análisis de convergencia

### OptimizationController
Controlador que integra optimización con el dashboard:
- Gestión de workers en background
- Comunicación con UI
- Manejo de resultados

### OptimizationPanel
Interfaz gráfica para configuración y monitoreo:
- Configuración de parámetros GA
- Monitoreo en tiempo real
- Visualización de resultados

## Uso del Sistema

### 1. Configuración de Estrategia
```python
from core.optimization import OptimizationController

# Crear controlador
controller = OptimizationController()

# Agregar estrategia para optimización
controller.add_strategy("RSIStrategy", RSIStrategy)
```

### 2. Configuración de Parámetros
```python
from core.optimization import OptimizationConfig

config = OptimizationConfig(
    population_size=100,
    generations=50,
    mutation_rate=0.15,
    fitness_function="sharpe_ratio"
)

# Definir rangos de parámetros
param_bounds = {
    'rsi_period': (5, 30),
    'overbought': (65, 85),
    'oversold': (15, 35)
}
```

### 3. Ejecución de Optimización
```python
# Función de backtest
def backtest_function(**params):
    return run_strategy_backtest("RSIStrategy", params)

# Iniciar optimización
controller.start_optimization("RSIStrategy", backtest_function)
```

### 4. Monitoreo de Resultados
```python
# Obtener resultados
results = controller.get_optimization_results("RSIStrategy")

print(f"Mejores parámetros: {results['best_parameters']}")
print(f"Mejor fitness: {results['best_fitness']}")
print(f"Tiempo de optimización: {results['optimization_time']:.2f}s")
```

## Parámetros de Configuración

### OptimizationConfig
- `population_size`: Tamaño de la población (50-200 recomendado)
- `generations`: Número de generaciones (30-100 recomendado)
- `mutation_rate`: Tasa de mutación (0.05-0.2 recomendado)
- `crossover_rate`: Tasa de crossover (0.7-0.9 recomendado)
- `elitism_count`: Número de élites preservados (3-10 recomendado)
- `tournament_size`: Tamaño del torneo (3-7 recomendado)
- `fitness_function`: Función objetivo
- `max_workers`: Workers paralelos (igual al número de CPUs)

## Funciones de Fitness

### Sharpe Ratio
```python
fitness = mean(returns) / std(returns) * sqrt(252)
```
Mejor para estrategias con retornos consistentes.

### Total Return
```python
fitness = (final_value - initial_value) / initial_value
```
Mejor para maximizar ganancias absolutas.

### Calmar Ratio
```python
fitness = mean(returns) / max_drawdown
```
Mejor para estrategias con bajo drawdown.

### Sortino Ratio
```python
fitness = mean(returns) / downside_deviation
```
Mejor para penalizar pérdidas negativas.

## Estrategias de Optimización

### Exploración vs Explotación
- **Alta mutación**: Más exploración del espacio
- **Alto elitismo**: Más explotación de buenas soluciones
- **Gran población**: Mejor cobertura del espacio

### Convergencia
- Monitorear el historial de fitness
- Ajustar generaciones basado en convergencia
- Usar validación cruzada para evitar overfitting

## Resultados y Análisis

### Métricas de Resultados
- **Best Fitness**: Mejor valor de la función objetivo
- **Best Parameters**: Parámetros óptimos encontrados
- **Convergence Generation**: Generación donde se encontró la mejor solución
- **Optimization Time**: Tiempo total de optimización

### Validación de Resultados
- **Walk-Forward Analysis**: Validación fuera de muestra
- **Bootstrap Analysis**: Robustez estadística
- **Sensitivity Analysis**: Estabilidad de parámetros

## Integración con Dashboard

### Panel de Optimización
El panel de optimización en la UI permite:
- Selección de estrategias disponibles
- Configuración de parámetros GA
- Monitoreo de progreso en tiempo real
- Visualización de mejores parámetros
- Tabla de resultados comparativos

### Flujo de Trabajo
1. **Configurar**: Seleccionar estrategia y parámetros GA
2. **Ejecutar**: Iniciar optimización en background
3. **Monitorear**: Ver progreso y mejores resultados
4. **Analizar**: Revisar parámetros óptimos y métricas
5. **Validar**: Probar parámetros en datos fuera de muestra

## Ejemplos Prácticos

### Optimización de RSI Strategy
```python
# Configuración
param_bounds = {
    'rsi_period': (10, 25),
    'overbought_level': (70, 80),
    'oversold_level': (20, 30),
    'stop_loss': (0.05, 0.15)
}

# Optimización
optimizer = GeneticOptimizer(OptimizationConfig(
    population_size=100,
    generations=50,
    fitness_function="sharpe_ratio"
))

results = optimizer.optimize(backtest_func)
```

### Optimización de Momentum Strategy
```python
param_bounds = {
    'lookback_period': (20, 60),
    'momentum_threshold': (0.02, 0.10),
    'holding_period': (5, 20)
}
```

## Consideraciones de Rendimiento

### Optimización de Recursos
- **Población**: 50-100 para estrategias simples, 100-200 para complejas
- **Generaciones**: 30-50 para convergencia típica
- **Workers**: Igual al número de CPUs disponibles

### Tiempo de Ejecución
- Estrategias simples: 5-15 minutos
- Estrategias complejas: 30-60 minutos
- Optimización completa: 2-4 horas

## Troubleshooting

### Problemas Comunes
- **Fitness siempre cero**: Verificar función de backtest
- **No convergencia**: Aumentar generaciones o población
- **Overfitting**: Usar validación cruzada
- **Memoria insuficiente**: Reducir población o workers

### Debugging
- Logs detallados en `logs/trading_ia_gui.log`
- Monitoreo de progreso en tiempo real
- Análisis de parámetros en cada generación

## Futuras Extensiones

### Algoritmos Avanzados
- **PSO (Particle Swarm Optimization)**
- **DE (Differential Evolution)**
- **NSGA-II (Multi-objective)**

### Features Adicionales
- **Optimización multi-objetivo**
- **Constraints avanzadas**
- **Auto-tuning de hiperparámetros**

Esta implementación proporciona una base sólida para optimización automática de estrategias, permitiendo a los traders encontrar parámetros óptimos de manera eficiente y sistemática. 🎯