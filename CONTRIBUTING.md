# Contributing to Trading IA

Thank you for your interest in contributing to this project!

## Getting Started

### 1. Clone and Setup

```bash
git clone <repository-url>
cd tradingIA
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will automatically check code formatting and quality before each commit.

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Development Workflow

### Code Style

- **Formatter**: Black with 120 character line length
- **Imports**: isort with black profile
- **Linter**: flake8
- **Type Hints**: Optional but recommended for new code

### Running Pre-commit Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on specific file
pre-commit run --files path/to/file.py
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_specific.py

# With coverage
pytest --cov=core --cov-report=html
```

### Project Structure

```
tradingIA/
├── api/              # External API integrations
├── backtesting/      # Backtesting engine
├── config/           # Configuration files
├── core/             # Core trading logic
│   ├── brokers/      # Broker integrations
│   ├── data/         # Data processing
│   ├── execution/    # Trade execution
│   ├── risk/         # Risk management
│   └── strategies/   # Trading strategies
├── dashboard/        # Web dashboard
├── data/             # Data files
├── docs/             # Documentation
├── tests/            # Test files
└── utils/            # Utility functions
```

## Pull Request Guidelines

### Before Submitting

1. ✅ Run pre-commit hooks: `pre-commit run --all-files`
2. ✅ Run tests: `pytest`
3. ✅ Update documentation if needed
4. ✅ Add tests for new features
5. ✅ Update CHANGELOG.md

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] Tests pass locally
- [ ] New code has appropriate tests
- [ ] Documentation has been updated
- [ ] CHANGELOG.md has been updated
- [ ] No sensitive data (API keys, secrets) in code

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add position sizing optimization
fix: correct Kelly criterion calculation
docs: update README with new features
refactor: simplify backtest loop
test: add tests for risk manager
```

## Issue Guidelines

### Bug Reports

Include:
- Python version
- OS (Windows/Mac/Linux)
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative solutions considered

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

### What We Look For

1. **Correctness**: Does the code do what it's supposed to?
2. **Tests**: Are there appropriate tests?
3. **Performance**: Is the code efficient?
4. **Readability**: Is the code easy to understand?
5. **Security**: Are there any security concerns?

## Contact

For questions or discussions, please open an issue on GitHub.

---

Thank you for contributing! 🚀
