# Trading Strategies

## 📚 Biblioteca de Estrategias Pre-configuradas

Esta carpeta contiene estrategias de trading listas para usar en la plataforma.

## 🚀 Estrategias Disponibles

### 1. **RSI Mean Reversion** (`rsi_mean_reversion.py`)
- **Descripción**: Reversión a la media basada en RSI
- **Señales**: 
  - COMPRA cuando RSI < nivel de sobreventa (default: 30)
  - VENTA cuando RSI > nivel de sobrecompra (default: 70)
- **Presets**: `conservative`, `aggressive`, `default`
- **Mejor para**: Mercados laterales/ranging

### 2. **MACD Momentum** (`macd_momentum.py`)
- **Descripción**: Momentum basado en cruces MACD
- **Señales**:
  - COMPRA cuando MACD cruza por encima de línea de señal
  - VENTA cuando MACD cruza por debajo de línea de señal
- **Presets**: `conservative`, `aggressive`, `default`
- **Mejor para**: Mercados con tendencia

### 3. **Bollinger Bands** (`bollinger_bands.py`)
- **Descripción**: Reversión en toques de bandas
- **Señales**:
  - COMPRA cuando precio toca banda inferior
  - VENTA cuando precio toca banda superior
- **Presets**: `conservative`, `aggressive`, `default`
- **Mejor para**: Volatilidad media

### 4. **Moving Average Crossover** (`ma_crossover.py`)
- **Descripción**: Cruce de medias móviles
- **Señales**:
  - COMPRA cuando MA rápida cruza por encima de MA lenta (Golden Cross)
  - VENTA cuando MA rápida cruza por debajo de MA lenta (Death Cross)
- **Presets**: `conservative`, `aggressive`, `default`, `scalping`
- **Mejor para**: Tendencias fuertes

### 5. **Volume Breakout** (`volume_breakout.py`)
- **Descripción**: Breakouts confirmados por volumen
- **Señales**:
  - COMPRA en breakout alcista con volumen alto
  - VENTA en breakout bajista con volumen alto
- **Presets**: `conservative`, `aggressive`, `default`
- **Mejor para**: Activos con alto volumen

## 📖 Uso

### En código Python:

```python
from strategies import load_strategy

# Cargar estrategia con configuración default
strategy = load_strategy('rsi_mean_reversion')

# Cargar con preset específico
strategy = load_strategy('macd_momentum', preset='aggressive')

# Generar señales
import pandas as pd
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

signals = strategy.generate_signals(df)
print(signals[signals['signal'] != 0])  # Ver señales generadas
```

### En la plataforma GUI:

1. Ir al **Tab 2 (Strategy Builder)**
2. Modo **Research**: Seleccionar estrategia del dropdown
3. Modo **Production**: Cargar estrategia guardada
4. Ejecutar backtesting o trading en vivo

## 🔧 Crear Estrategia Personalizada

### Paso 1: Crear archivo en `strategies/presets/`

```python
# mi_estrategia.py
from strategies.base_strategy import BaseStrategy
import pandas as pd

class MiEstrategia(BaseStrategy):
    def __init__(self):
        super().__init__(name="Mi Estrategia")
        self.parameters = {
            'param1': 10,
            'param2': 20
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Tu lógica aquí
        df['signal'] = 0  # 1=BUY, -1=SELL, 0=HOLD
        return df
    
    def get_parameters(self):
        return self.parameters.copy()
    
    def set_parameters(self, params):
        self.parameters.update(params)

# Opcional: definir presets
PRESETS = {
    'default': {'param1': 10, 'param2': 20},
    'aggressive': {'param1': 5, 'param2': 15}
}
```

### Paso 2: La estrategia se cargará automáticamente

```python
from strategies import list_available_strategies, load_strategy

# Tu estrategia aparecerá en la lista
print(list_available_strategies())
# ['rsi_mean_reversion', 'macd_momentum', ..., 'mi_estrategia']

# Cargarla
strategy = load_strategy('mi_estrategia')
```

## 📊 Estructura de Señales

Todas las estrategias devuelven un DataFrame con:

- `signal`: 1 (COMPRA), -1 (VENTA), 0 (HOLD)
- `signal_strength`: 1-5 (fuerza de la señal)
- Columnas adicionales con indicadores

## 🎯 Presets Explicados

### Conservative
- Parámetros más estrictos
- Menos señales, mayor calidad
- Menor riesgo

### Aggressive  
- Parámetros más permisivos
- Más señales, menor calidad
- Mayor riesgo/recompensa

### Default
- Balance entre conservative y aggressive
- Configuración estándar probada

## 📈 Backtesting

Todas las estrategias son compatibles con el motor de backtesting:

```python
from backtesting import BacktestEngine
from strategies import load_strategy

strategy = load_strategy('rsi_mean_reversion', preset='aggressive')
engine = BacktestEngine(strategy)

results = engine.run(df, initial_capital=10000)
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_dd']:.2f}%")
```

## 🔄 Actualizar Estrategia

Para modificar una estrategia existente:

1. Editar archivo en `strategies/presets/`
2. Guardar cambios
3. Recargar estrategia:
   ```python
   from strategies import get_loader
   loader = get_loader()
   loader._scan_strategies()  # Recargar
   ```

## ⚙️ Parámetros Configurables

Cada estrategia tiene parámetros ajustables:

```python
strategy = load_strategy('rsi_mean_reversion')

# Ver parámetros actuales
print(strategy.get_parameters())

# Modificar parámetros
strategy.set_parameters({
    'rsi_period': 10,
    'oversold': 25,
    'overbought': 75
})
```

## 📝 Validación de Datos

Todas las estrategias validan que el DataFrame tenga:
- `open`, `high`, `low`, `close`, `volume`

Si falta alguna columna, se lanzará `ValueError`.

## 🐛 Debugging

Para probar una estrategia:

```bash
cd strategies/presets
python rsi_mean_reversion.py
```

Cada estrategia tiene un `if __name__ == "__main__"` con ejemplo.

## 📚 Recursos

- **Documentación Base**: `base_strategy.py`
- **Strategy Loader**: `strategy_loader.py`
- **Ejemplos**: `presets/*.py`

## 🤝 Contribuir

Para agregar nuevas estrategias al repositorio:

1. Seguir estructura de `BaseStrategy`
2. Incluir docstrings
3. Definir al menos preset `default`
4. Agregar tests en `if __name__ == "__main__"`
5. Documentar en este README
