# Base A/B Testing Protocol

## Overview

The `BaseABTesting` class provides fundamental A/B testing functionality for trading strategies, implementing statistical comparison methods with proper hypothesis testing and effect size calculations.

## Features

### Core Statistical Testing
- **t-test**: Parametric comparison for normally distributed metrics
- **Mann-Whitney U**: Non-parametric comparison for ordinal data
- **Bootstrap Confidence Intervals**: Distribution-free confidence estimation
- **Effect Size Calculation**: Cohen's d for practical significance

### Trading-Specific Metrics
- **Sharpe Ratio**: Risk-adjusted returns comparison
- **Maximum Drawdown**: Risk metric comparison
- **Win Rate**: Trade success rate comparison
- **Profit Factor**: Gross profit/net loss ratio

### Hypothesis Testing Framework
- **Null Hypothesis**: No difference between strategy variants
- **Alternative Hypothesis**: One strategy outperforms the other
- **Significance Levels**: Configurable alpha levels (default 0.05)
- **One-tailed Tests**: Directional testing for superiority

## Usage

### Basic Usage

```python
from src.ab_base_protocol import BaseABTesting

# Initialize A/B tester
ab_tester = BaseABTesting()

# Compare two strategy results
results_a = {'sharpe_ratio': 1.2, 'max_drawdown': 0.15, 'total_return': 0.45}
results_b = {'sharpe_ratio': 1.5, 'max_drawdown': 0.12, 'total_return': 0.52}

# Run statistical comparison
comparison = ab_tester.compare_strategies(results_a, results_b)

print(f"P-value: {comparison['p_value']:.4f}")
print(f"Significant: {comparison['significant']}")
print(f"Winner: {comparison['winner']}")
```

### Signal-Based Testing

```python
import pandas as pd

# Test based on trading signals
signals_a = pd.Series([1, -1, 1, -1, 1, 1, -1, ...])  # Long/Short signals
signals_b = pd.Series([1, 1, -1, -1, 1, -1, 1, ...])

market_data = pd.DataFrame({
    'close': [100, 101, 99, 102, 98, ...],
    'timestamp': pd.date_range('2020-01-01', periods=len(signals_a), freq='5min')
})

# Run signal-based A/B test
results = ab_tester.test_signals(signals_a, signals_b, market_data)

print(f"Strategy A Return: {results['variant_a']['total_return']:.2%}")
print(f"Strategy B Return: {results['variant_b']['total_return']:.2%}")
print(f"Better Strategy: {results['comparison']['winner']}")
```

## API Reference

### `BaseABTesting` Class

#### Constructor

```python
BaseABTesting(
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    random_state: int = 42
)
```

**Parameters:**
- `alpha`: Significance level (default: 0.05)
- `alternative`: Test type - 'two-sided', 'less', 'greater' (default: 'two-sided')
- `random_state`: Random seed for reproducibility (default: 42)

#### Methods

##### `compare_strategies(results_a: Dict, results_b: Dict, metric: str = 'sharpe_ratio') -> Dict`
Compares two strategy result dictionaries for a specific metric.

**Parameters:**
- `results_a`: Dictionary with strategy A results
- `results_b`: Dictionary with strategy B results
- `metric`: Metric to compare (default: 'sharpe_ratio')

**Returns:**
- Dictionary with comparison results

##### `test_signals(signals_a: pd.Series, signals_b: pd.Series, df: pd.DataFrame) -> Dict`
Runs complete A/B test based on trading signals and market data.

**Parameters:**
- `signals_a`: Trading signals for variant A (-1, 0, 1)
- `signals_b`: Trading signals for variant B (-1, 0, 1)
- `df`: Market data DataFrame with price data

**Returns:**
- Complete A/B test results dictionary

##### `calculate_metrics(signals: pd.Series, df: pd.DataFrame) -> Dict`
Calculates trading performance metrics from signals and market data.

**Parameters:**
- `signals`: Trading signals series
- `df`: Market data DataFrame

**Returns:**
- Dictionary with calculated metrics

##### `statistical_test(data_a, data_b, test_type: str = 't-test') -> Dict`
Performs statistical comparison between two data samples.

**Parameters:**
- `data_a`: First data sample (array-like)
- `data_b`: Second data sample (array-like)
- `test_type`: Type of statistical test ('t-test', 'mann-whitney')

**Returns:**
- Statistical test results

##### `bootstrap_confidence_interval(data, n_boot: int = 1000, confidence: float = 0.95) -> Tuple`
Calculates bootstrap confidence interval for a statistic.

**Parameters:**
- `data`: Data sample
- `n_boot`: Number of bootstrap resamples (default: 1000)
- `confidence`: Confidence level (default: 0.95)

**Returns:**
- Tuple of (lower_bound, upper_bound)

## Statistical Methods

### Supported Tests

#### Student's t-test
- **Assumptions**: Normal distribution, equal variances
- **Use Case**: Comparing means of normally distributed metrics
- **Output**: t-statistic, p-value, degrees of freedom

#### Mann-Whitney U Test
- **Assumptions**: Ordinal data, independent samples
- **Use Case**: Non-parametric comparison of distributions
- **Output**: U-statistic, p-value, effect size

#### Bootstrap Confidence Intervals
- **Assumptions**: None (distribution-free)
- **Use Case**: Robust confidence estimation
- **Output**: Percentile confidence intervals

### Effect Size Measures

#### Cohen's d
- **Formula**: `d = (mean_a - mean_b) / pooled_std`
- **Interpretation**:
  - Small: 0.2
  - Medium: 0.5
  - Large: 0.8

#### Percentage Superiority
- **Formula**: Percentage of time variant B outperforms variant A
- **Range**: 0-100%
- **Interpretation**: >50% indicates B is generally better

## Trading Metrics

### Performance Metrics
- **Total Return**: Cumulative percentage return
- **Annualized Return**: Compounded annual growth rate
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return (return / volatility)

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Expected Shortfall**: Average loss beyond VaR
- **Ulcer Index**: Measure of drawdown severity

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Average Win/Loss**: Mean profit/loss per trade
- **Calmar Ratio**: Return / maximum drawdown

## Output Formats

### Comparison Results

```python
{
    'metric': 'sharpe_ratio',
    'variant_a_value': 1.23,
    'variant_b_value': 1.45,
    'difference': 0.22,
    'p_value': 0.034,
    'significant': True,
    'effect_size': 0.67,
    'effect_size_interpretation': 'medium',
    'confidence_interval': [0.08, 0.36],
    'winner': 'B',
    'percentage_superiority': 62.5
}
```

### Complete Test Results

```python
{
    'timestamp': '2024-01-01T12:00:00',
    'test_duration': 45.2,
    'variant_a': {
        'name': 'Strategy A',
        'signals_count': 1250,
        'metrics': {
            'total_return': 0.45,
            'sharpe_ratio': 1.23,
            'max_drawdown': 0.15,
            'win_rate': 0.52,
            'profit_factor': 1.35
        }
    },
    'variant_b': {
        'name': 'Strategy B',
        'signals_count': 1180,
        'metrics': {
            'total_return': 0.52,
            'sharpe_ratio': 1.45,
            'max_drawdown': 0.12,
            'win_rate': 0.55,
            'profit_factor': 1.42
        }
    },
    'comparison': {
        'sharpe_ratio': {...},  # As above
        'max_drawdown': {...},
        'total_return': {...},
        'win_rate': {...}
    },
    'overall_winner': 'B',
    'confidence_level': 'high'
}
```

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and time series
- `scipy.stats`: Statistical functions
- `sklearn.metrics`: Additional metrics (optional)

## Best Practices

### Data Preparation
1. **Signal Alignment**: Ensure signals and market data are properly aligned
2. **Data Quality**: Handle missing data and outliers appropriately
3. **Time Zones**: Ensure consistent timezone handling
4. **Data Frequency**: Match signal frequency with market data

### Testing Setup
1. **Sample Size**: Ensure adequate sample size for statistical power
2. **Test Period**: Use representative market conditions
3. **Transaction Costs**: Include realistic trading costs
4. **Slippage**: Account for price impact in backtests

### Interpretation Guidelines
1. **Statistical vs Practical Significance**: Consider both p-values and effect sizes
2. **Risk-Adjusted Returns**: Focus on Sharpe ratio over raw returns
3. **Drawdown Analysis**: Maximum drawdown is critical for risk management
4. **Out-of-Sample Testing**: Validate results on unseen data

## Troubleshooting

### Common Issues

1. **Low Statistical Power**: Increase sample size or effect size
2. **Data Mismatch**: Check signal and market data alignment
3. **Distribution Assumptions**: Use non-parametric tests if normality violated
4. **Multiple Testing**: Apply correction for multiple comparisons

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug output in tests
results = ab_tester.test_signals(signals_a, signals_b, df, debug=True)
```

### Performance Optimization

- Use vectorized operations for signal processing
- Cache calculated metrics when possible
- Parallelize bootstrap operations for large samples
- Use approximate methods for very large datasets

## Integration Examples

### With Backtesting Engine

```python
from backtesting.backtest_engine import BacktestEngine
from src.ab_base_protocol import BaseABTesting

# Run backtests
engine = BacktestEngine()
results_a = engine.run_backtest(strategy_a, market_data)
results_b = engine.run_backtest(strategy_b, market_data)

# Statistical comparison
ab_tester = BaseABTesting()
comparison = ab_tester.compare_strategies(results_a, results_b)

print(f"Winner: {comparison['winner']} (p={comparison['p_value']:.3f})")
```

### With Signal Generators

```python
from strategies.signal_generator import SignalGenerator

# Generate signals for A/B testing
signal_gen = SignalGenerator()
signals_a = signal_gen.generate_signals(market_data, params_a)
signals_b = signal_gen.generate_signals(market_data, params_b)

# Run A/B test
ab_tester = BaseABTesting()
results = ab_tester.test_signals(signals_a, signals_b, market_data)
```

## Extensions

The base protocol serves as foundation for:

- **Advanced A/B Testing**: Robustness and anti-snooping analysis
- **Multi-Armed Bandits**: Dynamic strategy allocation
- **Bayesian A/B Testing**: Probabilistic decision making
- **Time Series A/B Testing**: Temporal pattern analysis

## Future Enhancements

### Planned Features
- **Bayesian Methods**: Probabilistic A/B testing
- **Sequential Testing**: Real-time A/B testing
- **Multi-Variant Testing**: A/B/C/D... testing
- **Network Effects**: Social trading A/B testing
- **Survival Analysis**: Time-to-failure analysis