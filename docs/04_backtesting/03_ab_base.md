# A/B Testing Base Protocol

## Overview
Base protocol for A/B testing trading signals comparing IFVG base strategy vs RSI/BB alternative.

## Features
- Random data split (50/50 by periods, seed=42)
- Parallel backtesting with vectorbt or simplified implementation
- Statistical significance testing (paired t-test)
- Superiority analysis (% periods B > A)
- Minimum trade requirements validation
- A/A test for bias detection

## Usage
```python
from src.ab_base_protocol import ab_protocol

# Run A/B test
results = ab_protocol(
    hypothesis="RSI/BB performs better in chop markets",
    variant_a_signals=ifvg_signals,
    variant_b_signals=rsi_bb_signals,
    df_5m=btc_5min_data
)

print(f"Decision: {results['decision']['recommendation']}")
print(f"Confidence: {results['decision']['confidence']:.1%}")
```

## Key Components

### Data Split
- Random split by trading days (not sequential)
- 50/50 default ratio
- Seed=42 for reproducibility

### Backtesting
- VectorBT integration (if available)
- Fallback simplified implementation
- Realistic slippage (0.1%) and commissions (0.05%)

### Statistical Analysis
- Paired t-test for returns difference
- Superiority percentage (B > A periods)
- Cohen's d effect size
- Significance threshold p < 0.05

### Decision Making
- Minimum 100 trades requirement
- Multi-criteria decision logic
- Confidence scoring

### Bias Detection
- A/A test with multiple runs
- False positive rate monitoring
- Framework validation

## Output Structure
```python
{
    'hypothesis': str,
    'timestamp': pd.Timestamp,
    'variant_a': {'name': str, 'results': dict},
    'variant_b': {'name': str, 'results': dict},
    'statistical_analysis': {
        't_statistic': float,
        'p_value': float,
        'significant': bool,
        'superiority_percentage': float,
        'cohens_d': float
    },
    'aa_test': {
        'false_positive_rate': float,
        'bias_detected': bool
    },
    'decision': {
        'recommendation': str,  # 'ADOPT_B', 'HYBRID_TEST', 'KEEP_A', 'INSUFFICIENT_DATA'
        'reason': str,
        'confidence': float
    }
}
```

## Decision Criteria
- **ADOPT_B**: p < 0.05, superiority > 60%, ΔSharpe > 0.2
- **HYBRID_TEST**: p < 0.1, ΔSharpe > 0.1 (needs more testing)
- **KEEP_A**: No significant improvement
- **INSUFFICIENT_DATA**: < 100 trades per variant

## Example Results
```
Hypothesis: RSI/BB > IFVG in chop markets
Decision: ADOPT_B (confidence: 78%)
Reason: B significantly better (p=0.023, superiority=68%, ΔSharpe=0.31)
```

## Integration
Used by `ab_advanced.py` and `ab_pipeline.py` for enhanced A/B testing with robustness metrics and automation.