# Strategy Framework Documentation

## Overview

The Strategy Framework provides an extensible architecture for implementing and managing trading strategies in the TradingIA platform. It features a plugin-based system that allows developers to easily create, register, and use different trading strategies.

## Architecture

### Core Components

1. **BaseStrategy**: Abstract base class defining the strategy interface
2. **StrategyConfig**: Configuration dataclass for strategy parameters
3. **StrategyRegistry**: Central registry for strategy management
4. **Concrete Strategies**: Implementations of specific trading strategies

### Directory Structure

```
core/strategies/
├── __init__.py                 # Module exports
├── base_strategy.py           # Abstract base class and configuration
├── strategy_registry.py       # Strategy registration and management
├── momentum_strategy.py       # Momentum-based strategy
├── mean_reversion_strategy.py # Mean reversion strategy
└── breakout_strategy.py       # Breakout strategy
```

## Usage

### Basic Strategy Creation

```python
from core.strategies import StrategyRegistry, StrategyConfig, MomentumStrategy

# Create registry
registry = StrategyRegistry()
registry.register_strategy(MomentumStrategy)

# Configure strategy
config = StrategyConfig(
    name="MomentumStrategy",
    description="Custom momentum strategy",
    parameters={
        'fast_period': 10,
        'slow_period': 20,
        'momentum_threshold': 0.02
    },
    risk_management={
        'max_drawdown_threshold': 0.15
    },
    filters=['max_drawdown_filter', 'trend_filter']
)

# Create strategy instance
strategy = registry.create_strategy(config)
```

### Strategy Implementation

To create a new strategy, inherit from `BaseStrategy` and implement the required methods:

```python
from core.strategies import BaseStrategy, StrategyConfig
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def get_required_parameters(self) -> List[str]:
        return ['my_param1', 'my_param2']

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add custom indicators to data
        df = data.copy()
        df['my_indicator'] = df['close'].rolling(10).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Generate trading signals (-1, 0, 1)
        signals = pd.Series(0, index=data.index)
        # Your signal logic here
        return signals
```

### Risk Management and Filters

Strategies support built-in risk management filters:

- **max_drawdown_filter**: Prevents trading when drawdown exceeds threshold
- **volatility_filter**: Filters based on market volatility
- **trend_filter**: Only allows trades in trending markets

Configure filters in the strategy config:

```python
config = StrategyConfig(
    # ... other config ...
    risk_management={
        'max_drawdown_threshold': 0.10,
        'volatility_threshold': 0.03
    },
    filters=['max_drawdown_filter', 'trend_filter']
)
```

## Available Strategies

### 1. MomentumStrategy

**Description**: Trades based on price momentum and trend following.

**Parameters**:
- `fast_period`: Short-term moving average period
- `slow_period`: Long-term moving average period
- `momentum_threshold`: Minimum momentum for signal generation

**Indicators**: Moving averages, momentum, RSI, MACD

### 2. MeanReversionStrategy

**Description**: Trades against the trend, expecting prices to revert to mean.

**Parameters**:
- `lookback_period`: Period for calculating mean/std
- `entry_threshold`: Z-score threshold for entry
- `exit_threshold`: Z-score threshold for exit

**Indicators**: Rolling mean/std, Z-score, Bollinger Bands, RSI

### 3. BreakoutStrategy

**Description**: Trades breakouts above resistance or below support levels.

**Parameters**:
- `lookback_period`: Period for calculating support/resistance
- `breakout_threshold`: Minimum breakout percentage
- `consolidation_period`: Period for consolidation detection

**Indicators**: Rolling highs/lows, consolidation metrics, volume confirmation

## Integration with Backtesting

The strategy framework integrates seamlessly with the optimized backtester:

```python
from core.execution.optimized_backtester import OptimizedBacktester
from core.strategies import StrategyRegistry

# Setup
registry = StrategyRegistry()
registry.register_strategy(MomentumStrategy)
backtester = OptimizedBacktester()

# Create strategy configurations for optimization
configs = [
    StrategyConfig(name="MomentumStrategy", parameters={'fast_period': 5, 'slow_period': 15, 'momentum_threshold': 0.03}),
    StrategyConfig(name="MomentumStrategy", parameters={'fast_period': 10, 'slow_period': 20, 'momentum_threshold': 0.02}),
]

# Run parallel backtests
results = backtester.run_parallel_backtests(
    data=market_data,
    strategy_func=lambda data, **params: registry.create_strategy(
        StrategyConfig(name="MomentumStrategy", parameters=params)
    ).generate_signals(data),
    parameter_sets=[config.parameters for config in configs]
)
```

## Testing

Run the strategy framework tests:

```bash
pytest tests/test_strategy_framework.py -v
```

Run the demonstration script:

```bash
python scripts/strategy_framework_demo.py
```

## Extending the Framework

### Adding New Strategies

1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement required abstract methods
3. Register the strategy in the registry
4. Add appropriate tests

### Custom Indicators

Override the `calculate_indicators` method to add custom technical indicators:

```python
def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    df = super().calculate_indicators(data)  # Call parent for common indicators

    # Add custom indicators
    df['custom_indicator'] = custom_calculation(df['close'])

    return df
```

### Custom Filters

Add custom risk filters by overriding `apply_filters`:

```python
def apply_filters(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
    filtered_signals = super().apply_filters(data, signals)

    # Apply custom filter
    custom_condition = data['custom_indicator'] > threshold
    filtered_signals = filtered_signals.where(custom_condition, 0)

    return filtered_signals
```

## Best Practices

1. **Parameter Validation**: Always validate strategy parameters in `get_required_parameters`
2. **Data Validation**: Use `validate_data()` to ensure input data quality
3. **Error Handling**: Implement proper error handling in signal generation
4. **Documentation**: Document strategy logic and parameters clearly
5. **Testing**: Write comprehensive tests for each strategy implementation
6. **Performance**: Optimize indicator calculations for large datasets

## Performance Considerations

- Strategies are designed to work with the parallel backtesting engine
- Indicator calculations are cached within each backtest run
- Memory usage scales with data size and indicator complexity
- Consider using Numba for computationally intensive calculations