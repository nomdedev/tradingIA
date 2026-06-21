# Advanced A/B Testing Framework

## Overview

The `AdvancedABTesting` class provides a comprehensive framework for statistical validation of trading strategies with advanced features including robustness analysis, anti-snooping bias detection, and automated decision making.

## Features

### Statistical Validation
- **Multiple Testing Correction**: Bonferroni and Holm-Bonferroni methods
- **Effect Size Analysis**: Cohen's d and Hedges' g calculations
- **Confidence Intervals**: Bootstrap and parametric confidence intervals
- **Power Analysis**: Statistical power calculations for sample size planning

### Robustness Analysis
- **Out-of-Sample Testing**: Forward-looking validation
- **Subsample Stability**: Analysis across different market conditions
- **Parameter Sensitivity**: Robustness to parameter changes
- **Monte Carlo Simulation**: Distribution of possible outcomes

### Anti-Snooping Bias Detection
- **False Discovery Rate Control**: Benjamini-Hochberg procedure
- **Snooping Bias Metrics**: Detection of data mining effects
- **Multiple Hypothesis Testing**: Correction for multiple comparisons
- **Overfitting Detection**: Identification of spurious results

### Automated Decision Making
- **Multi-Factor Scoring**: Comprehensive decision criteria
- **Risk Assessment**: Risk-adjusted performance evaluation
- **Confidence Scoring**: Statistical confidence in results
- **Actionable Recommendations**: Clear deployment guidance

## Usage

### Basic Usage

```python
from src.ab_advanced import AdvancedABTesting

# Initialize A/B tester
ab_tester = AdvancedABTesting()

# Run comprehensive analysis
results_a = {'sharpe_ratio': 1.2, 'max_drawdown': 0.15, 'total_return': 0.45}
results_b = {'sharpe_ratio': 1.5, 'max_drawdown': 0.12, 'total_return': 0.52}

analysis = ab_tester.run_comprehensive_analysis(results_a, results_b)
decision = ab_tester.generate_automated_decision(analysis)

print(f"Decision: {decision['automated_action']}")
print(f"Confidence: {decision['confidence_score']:.2f}")
```

### Advanced Usage with Custom Metrics

```python
# Define custom metrics
custom_metrics = {
    'sharpe_ratio': {'higher_is_better': True, 'threshold': 1.0},
    'max_drawdown': {'higher_is_better': False, 'threshold': 0.20},
    'win_rate': {'higher_is_better': True, 'threshold': 0.55}
}

analysis = ab_tester.run_comprehensive_analysis(
    results_a, results_b,
    custom_metrics=custom_metrics
)
```

## API Reference

### `AdvancedABTesting` Class

#### Constructor

```python
AdvancedABTesting(
    confidence_level: float = 0.95,
    min_sample_size: int = 30,
    random_state: int = 42
)
```

**Parameters:**
- `confidence_level`: Statistical confidence level (default: 0.95)
- `min_sample_size`: Minimum sample size for analysis (default: 30)
- `random_state`: Random seed for reproducibility (default: 42)

#### Methods

##### `run_comprehensive_analysis(results_a, results_b, custom_metrics=None) -> Dict`
Performs complete A/B analysis including statistical tests, robustness checks, and bias detection.

**Parameters:**
- `results_a`: Dictionary of metrics for variant A
- `results_b`: Dictionary of metrics for variant B
- `custom_metrics`: Optional custom metric definitions

**Returns:**
- Comprehensive analysis dictionary with all results

##### `calculate_statistical_significance(metric_a, metric_b, metric_name) -> Dict`
Calculates statistical significance for a single metric.

##### `calculate_robustness_metrics(results_a, results_b) -> Dict`
Calculates robustness metrics including out-of-sample performance.

##### `detect_snooping_bias(results_a, results_b) -> Dict`
Detects potential snooping bias in the results.

##### `generate_automated_decision(analysis) -> Dict`
Generates automated decision based on comprehensive analysis.

##### `calculate_effect_size(metric_a, metric_b) -> float`
Calculates Cohen's d effect size.

##### `calculate_confidence_intervals(data, confidence_level) -> Tuple`
Calculates bootstrap confidence intervals.

## Analysis Components

### Statistical Significance Testing

#### Supported Tests
- **t-test**: Parametric test for normally distributed data
- **Mann-Whitney U**: Non-parametric test for ordinal data
- **Bootstrap**: Distribution-free confidence intervals
- **Permutation Test**: Exact test for small samples

#### Multiple Testing Correction
- **Bonferroni**: Conservative correction for multiple comparisons
- **Holm-Bonferroni**: Step-down procedure for better power
- **Benjamini-Hochberg**: FDR control for large numbers of tests

### Robustness Metrics

#### Out-of-Sample Performance
- **Forward Testing**: Performance on unseen data
- **Walk-Forward Analysis**: Rolling window validation
- **Cross-Validation**: K-fold validation for time series

#### Stability Analysis
- **Parameter Sensitivity**: Performance across parameter ranges
- **Market Regime Stability**: Performance in different market conditions
- **Subsample Analysis**: Performance in data subsets

### Anti-Snooping Detection

#### Bias Detection Methods
- **False Discovery Rate**: Control of type I errors
- **Data Mining Bias**: Detection of overfitting effects
- **Multiple Hypothesis Testing**: Correction for selection bias
- **Spurious Correlation**: Identification of false positives

#### Risk Assessment
- **Overfitting Risk**: Probability of false positive results
- **Selection Bias**: Impact of strategy selection on results
- **Survivorship Bias**: Impact of backtest period selection

## Decision Logic

The automated decision making uses a multi-factor approach:

### Decision Categories

1. **Strong Superiority** (Confidence > 0.85)
   - Clear statistical significance
   - Robust performance across conditions
   - Low snooping bias risk
   - Action: Deploy immediately

2. **Moderate Superiority** (0.70 < Confidence ≤ 0.85)
   - Statistical significance present
   - Some robustness concerns
   - Moderate snooping risk
   - Action: Deploy with monitoring

3. **Low Risk Superiority** (0.55 < Confidence ≤ 0.70)
   - Marginal statistical significance
   - Limited robustness evidence
   - Higher snooping risk
   - Action: Deploy hybrid approach

4. **No Advantage** (Confidence ≤ 0.55)
   - No statistical significance
   - Poor robustness
   - High snooping risk
   - Action: Maintain current strategy

5. **Snooping Detected** (High bias risk)
   - Strong evidence of data mining
   - Results likely spurious
   - Action: Investigate further

### Confidence Scoring

Confidence scores are calculated based on:
- Statistical significance (p-values)
- Effect sizes (Cohen's d)
- Robustness metrics
- Snooping bias indicators
- Sample size adequacy

## Output Formats

### Analysis Results

```python
{
    'statistical_tests': {
        'sharpe_ratio': {
            'p_value': 0.023,
            'significant': True,
            'effect_size': 0.45,
            'confidence_interval': [0.12, 0.78]
        },
        'max_drawdown': {
            'p_value': 0.156,
            'significant': False,
            'effect_size': -0.23,
            'confidence_interval': [-0.56, 0.10]
        }
    },
    'robustness_analysis': {
        'out_of_sample_performance': 0.78,
        'parameter_stability': 0.82,
        'market_regime_stability': 0.71
    },
    'snooping_detection': {
        'bias_risk': 'low',
        'fdr_adjusted_p': 0.045,
        'overfitting_probability': 0.12
    },
    'overall_confidence': 0.76
}
```

### Decision Results

```python
{
    'automated_action': 'deploy_with_monitoring',
    'confidence_score': 0.78,
    'risk_level': 'medium',
    'rationale': [
        'Statistical significance achieved for Sharpe ratio',
        'Robustness metrics show moderate stability',
        'Low risk of snooping bias detected'
    ],
    'recommendations': [
        'Deploy variant B to 25% of capital initially',
        'Monitor performance for first 30 trading days',
        'Prepare rollback plan if drawdown exceeds 5%'
    ],
    'next_steps': [
        'Set up performance monitoring dashboard',
        'Schedule weekly performance reviews',
        'Plan full deployment after successful monitoring period'
    ]
}
```

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `statsmodels`: Advanced statistical models

## Best Practices

### Data Preparation
1. **Sample Size**: Ensure adequate sample size (minimum 30 observations)
2. **Data Quality**: Clean data and handle outliers appropriately
3. **Time Series**: Account for autocorrelation in financial data
4. **Stationarity**: Test for stationarity when appropriate

### Analysis Setup
1. **Metric Selection**: Choose metrics relevant to trading objectives
2. **Benchmarking**: Compare against appropriate benchmarks
3. **Risk Adjustment**: Use risk-adjusted performance metrics
4. **Multiple Perspectives**: Analyze from different angles

### Interpretation Guidelines
1. **Context Matters**: Consider market conditions and strategy type
2. **Practical Significance**: Focus on economically meaningful differences
3. **Risk Management**: Never ignore risk metrics for return metrics
4. **Long-term Focus**: Consider long-term sustainability over short-term gains

## Troubleshooting

### Common Issues

1. **Low Statistical Power**: Increase sample size or effect size
2. **High Snooping Risk**: Use out-of-sample testing and cross-validation
3. **Unstable Results**: Check for data quality issues or parameter sensitivity
4. **Conflicting Metrics**: Use composite scoring or risk-adjusted metrics

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicLevel(level=logging.DEBUG)

# Enable debug mode in analysis
analysis = ab_tester.run_comprehensive_analysis(
    results_a, results_b,
    debug=True
)
```

### Performance Optimization

- Use vectorized operations for large datasets
- Cache bootstrap results for repeated analyses
- Parallelize Monte Carlo simulations
- Use approximate methods for very large samples

## Integration Examples

### With Backtesting Framework

```python
from backtesting.backtest_engine import BacktestEngine
from src.ab_advanced import AdvancedABTesting

# Run backtests for both strategies
engine = BacktestEngine()
results_a = engine.run_backtest(strategy_a, data)
results_b = engine.run_backtest(strategy_b, data)

# Analyze results
ab_tester = AdvancedABTesting()
analysis = ab_tester.run_comprehensive_analysis(results_a, results_b)
decision = ab_tester.generate_automated_decision(analysis)

# Log decision
logger.info(f"A/B Decision: {decision['automated_action']}")
```

### With Pipeline Automation

```python
from src.ab_pipeline import ABPipeline

# Pipeline automatically handles A/B analysis
pipeline = ABPipeline()
results = pipeline.run_full_pipeline()

# Results include comprehensive A/B analysis
analysis = results['analysis']
decision = results['decision']
```

## Future Enhancements

### Planned Features
- **Bayesian A/B Testing**: Probabilistic approach to decision making
- **Multi-Armed Bandit**: Dynamic strategy allocation
- **Time Series Analysis**: Advanced temporal pattern analysis
- **Machine Learning Integration**: ML-based effect size estimation
- **Real-time Monitoring**: Live A/B testing capabilities

### Research Directions
- **Non-stationary Markets**: Adaptive testing for changing market conditions
- **High-frequency Trading**: Microsecond-level A/B testing
- **Portfolio-level Testing**: Multi-asset strategy validation
- **Risk Parity Integration**: Risk-adjusted A/B testing frameworks