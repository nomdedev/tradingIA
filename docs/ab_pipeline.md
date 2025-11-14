# Automated A/B Testing Pipeline

## Overview

The `ABPipeline` class provides a complete automated pipeline for A/B testing of trading strategies, integrating data fetching, signal generation, backtesting, analysis, and reporting with version control and CI/CD support.

## Features

### Complete Pipeline Automation
- **Data Management**: Automated fetching and validation of market data
- **Signal Generation**: Parallel generation of A/B test signals
- **Backtesting**: Automated parallel backtests with comprehensive metrics
- **Analysis**: Full A/B analysis with robustness and anti-snooping
- **Decision Making**: Automated decision recommendations
- **Reporting**: Comprehensive markdown and JSON reports
- **Version Control**: DVC and Git integration for reproducibility

### Version Control Integration
- **DVC Pipeline**: Data and model versioning
- **Git Integration**: Automatic commits of results
- **Docker Support**: Containerized execution
- **CI/CD Ready**: GitHub Actions integration

### Enterprise Features
- **Comprehensive Logging**: Detailed execution logs
- **Error Handling**: Robust error handling and recovery
- **Modular Design**: Easy extension and customization
- **Configuration Management**: JSON-based configuration

## Usage

### Basic Usage

```python
from src.ab_pipeline import ABPipeline

# Initialize pipeline
pipeline = ABPipeline(
    symbol='BTCUSD',
    start_date='2020-01-01',
    end_date='2024-01-01',
    capital=10000
)

# Run full pipeline
results = pipeline.run_full_pipeline()

print(f"Status: {results['status']}")
print(f"Decision: {results['decision']['automated_action']}")
```

### Command Line Usage

```bash
# Run full pipeline
python src/ab_pipeline.py --symbol BTCUSD --start 2020-01-01 --end 2024-01-01

# Run specific stage
python src/ab_pipeline.py --stage data_fetch

# Create DVC pipeline config
python src/ab_pipeline.py --create-dvc
```

### DVC Pipeline Usage

```bash
# Initialize DVC
dvc init

# Create pipeline config
python src/ab_pipeline.py --create-dvc

# Run pipeline
dvc repro

# Or run specific stages
dvc repro data_fetch
dvc repro signals_generation
dvc repro ab_testing
```

## API Reference

### `ABPipeline` Class

#### Constructor

```python
ABPipeline(
    symbol: str = 'BTCUSD',
    start_date: str = '2018-01-01',
    end_date: Optional[str] = None,
    capital: float = 10000
)
```

**Parameters:**
- `symbol`: Trading symbol (e.g., 'BTCUSD')
- `start_date`: Start date for data (YYYY-MM-DD)
- `end_date`: End date for data (YYYY-MM-DD), defaults to today
- `capital`: Starting capital for backtests

#### Methods

##### `run_full_pipeline() -> Dict[str, Any]`
Runs the complete A/B testing pipeline.

**Returns:**
- Dictionary with pipeline results and metadata

##### `_fetch_and_validate_data() -> pd.DataFrame`
Fetches and validates market data with integrity checks.

##### `_generate_ab_signals(df_5m: pd.DataFrame)`
Generates A/B test signals using different strategies.

##### `_run_parallel_backtests(df_5m, signals_a, signals_b)`
Runs parallel backtests for both variants.

##### `_run_comprehensive_analysis(results_a, results_b) -> Dict`
Performs comprehensive A/B analysis with robustness and anti-snooping.

##### `_generate_automated_decision(analysis) -> Dict`
Generates automated decision based on analysis results.

##### `_generate_reports(results_a, results_b, analysis, decision) -> Path`
Creates comprehensive markdown and JSON reports.

##### `_version_and_commit()`
Versions results and commits to Git/DVC.

## Pipeline Stages

### 1. Data Fetch & Validation
- Fetches market data from APIs
- Validates data integrity (missing values, date ranges)
- Saves data with DVC versioning
- Handles data quality issues

### 2. Signal Generation
- Generates signals for variant A (IFVG base)
- Generates signals for variant B (alternative strategy)
- Parallel processing for efficiency
- DVC tracking of signal files

### 3. Parallel Backtesting
- Runs backtests for both variants simultaneously
- Comprehensive metrics calculation
- Performance logging and monitoring

### 4. Comprehensive Analysis
- Statistical significance testing
- Robustness metrics calculation
- Anti-snooping bias detection
- Confidence interval analysis

### 5. Automated Decision Making
- Multi-factor decision logic
- Confidence scoring
- Risk assessment
- Actionable recommendations

### 6. Reporting & Versioning
- Markdown reports with executive summaries
- JSON data for programmatic access
- Git/DVC versioning
- CI/CD integration

## Configuration

### Directory Structure
```
project/
├── data/           # Market data (DVC tracked)
├── signals/        # Generated signals (DVC tracked)
├── results/        # Analysis results
├── reports/        # Generated reports
├── logs/          # Execution logs
└── config/        # Configuration files
```

### DVC Pipeline Configuration

The pipeline creates a `dvc.yaml` with the following stages:

```yaml
stages:
  data_fetch:
    cmd: python src/ab_pipeline.py --stage=data_fetch
    outs: [data/]
  signals_generation:
    cmd: python src/ab_pipeline.py --stage=signals
    deps: [data/]
    outs: [signals/]
  ab_testing:
    cmd: python src/ab_pipeline.py --stage=ab_test
    deps: [signals/]
    outs: [results/, reports/]
```

## Decision Logic

The automated decision making follows this hierarchy:

1. **Snooping Detected**: Investigate further (high risk)
2. **Strong Superiority**: Deploy variant B immediately (low risk)
3. **Moderate Superiority**: Deploy with monitoring (medium risk)
4. **Low Risk Superiority**: Deploy hybrid approach (low risk)
5. **No Advantage**: Keep current strategy (no risk)

## Output Formats

### Markdown Report
- Executive summary with key metrics
- Detailed statistical analysis
- Decision rationale and next steps
- Performance visualizations

### JSON Results
```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "symbol": "BTCUSD",
    "period": "2020-01-01_to_2024-01-01",
    "pipeline_version": "1.0.0"
  },
  "results_a": {...},
  "results_b": {...},
  "analysis": {...},
  "decision": {...}
}
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `src.ab_advanced`: Advanced A/B testing functionality
- `dvc`: Data version control (optional)
- `git`: Version control (optional)
- `docker`: Containerization (optional)

## Docker Integration

### Dockerfile
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/ab_pipeline.py"]
```

### Docker Usage
```bash
# Build image
docker build -t ab-pipeline .

# Run pipeline
docker run ab-pipeline --symbol BTCUSD --start 2020-01-01

# Run with volume mounting
docker run -v $(pwd)/data:/app/data ab-pipeline
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: AB Pipeline
on: [push, schedule]
jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run pipeline
      run: python src/ab_pipeline.py
    - name: Commit results
      run: |
        git add results/ reports/
        git commit -m "Pipeline results $(date +%Y%m%d)" || true
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Data Issues**: Automatic data validation and cleaning
- **API Failures**: Retry logic and fallback mechanisms
- **Analysis Errors**: Graceful degradation with warnings
- **Version Control**: Non-blocking version control operations
- **Logging**: Detailed error logging for debugging

## Best Practices

1. **Data Quality**: Always validate data integrity before analysis
2. **Version Control**: Use DVC for data and Git for code
3. **Testing**: Run pipeline on historical data before live deployment
4. **Monitoring**: Set up alerts for pipeline failures
5. **Documentation**: Keep detailed logs of all pipeline runs
6. **Reproducibility**: Pin all dependencies and use fixed random seeds

## Troubleshooting

### Common Issues

1. **Data Fetch Failures**: Check API credentials and network connectivity
2. **Memory Issues**: Reduce date ranges or use data sampling
3. **DVC Errors**: Ensure DVC is properly initialized
4. **Analysis Timeouts**: Check for infinite loops in signal generation

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- Use data sampling for large datasets
- Parallelize backtesting operations
- Cache intermediate results
- Use SSD storage for data operations