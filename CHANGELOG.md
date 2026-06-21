# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Ronda 11 (2026-01-13)

### Added
- `LiveTrader` class for production live trading with Alpaca API
- Pre-commit hooks configuration (`.pre-commit-config.yaml`)
- Centralized logging configuration (`utils/logging_config.py`)
- `.env.example` template for environment variables
- Sensitive data filtering in logs
- `docs/AUDIT_REPORT.md` - Comprehensive project audit report
- 9 new helper methods in `BacktesterCore` for reduced complexity
- 8 new helper methods in `Council` for reduced complexity

### Changed
- Improved thread cleanup in `ProductionMonitor` with `threading.Event`
- `stop_monitoring()` now uses interruptible wait with proper timeout
- Refactored `calculate_metrics()` to accept optional `close` parameter
- **MAJOR: Refactored `Council.decide()` - complexity reduced from 51 → ~15**
  - Extracted: `_evaluate_declarative_rules()`, `_gather_expert_evidence()`, 
    `_calculate_expert_votes()`, `_calculate_single_vote()`, `_check_vetos()`, 
    `_create_veto_response()`, `_calculate_consensus()`, `_determine_decision()`
- **MAJOR: Refactored `run_simple_backtest()` - complexity reduced from 71 → ~20**
  - Extracted: `_prepare_backtest_data()`, `_calculate_volatility()`, 
    `_process_entry_signals()`, `_process_exit_signals()`, `_execute_backtest()`,
    `_run_realistic_execution()`, `_run_simple_execution()`, `_process_backtest_results()`,
    `_build_backtest_result()`

### Fixed
- Removed duplicate `kelly_info` block in backtester_core.py
- Removed duplicate `_update_capital` method in backtester_core.py
- Fixed bare `except:` clauses in test files
- Fixed undefined `close` variable in calculate_metrics()
- Fixed 24+ float comparison issues using `math.isclose()` and `pytest.approx()`
  - `test_extracted_modules.py` - 8 corrections
  - `test_council_protocol.py` - 4 corrections  
  - `test_council_advanced.py` - 1 correction
  - `test_backend_core.py` - 3 corrections
  - `test_backtester_core.py` - 1 correction
  - `test_critical_corrections.py` - 2 corrections
  - `test_area3_kelly.py` - 3 corrections
  - `test_no_lookahead_simple.py` - 2 corrections
  - `core/council.py` - 1 correction
- Removed unused variables in `test_no_look_ahead_bias.py` (vah_orig, val_orig, wrong_window → _)

### Code Quality
- Added constants in `core/council.py`:
  - `EXPERT_RISK_WARDEN`, `EXPERT_TREND_MASTER`, `EXPERT_DATA_ORACLE`, `EXPERT_ARCHITECT_PRIME`
  - `CERT_APPROVED_THRESHOLD`, `CERT_REJECTED_THRESHOLD`
- Extracted `_determine_certification_status()` from nested ternary in Council

### Security
- Added sensitive data redaction in centralized logging
- Environment variables documented in `.env.example`

## [0.1.0] - 2024-XX-XX

### Added
- Initial trading platform architecture
- Council-based decision system
- Multi-timeframe (MTF) strategy support
- Advanced backtesting engine with realistic execution
- Kelly criterion position sizing
- Risk management with kill switch
- Pattern discovery GUI
- Dashboard for monitoring
- API integration with Alpaca

### Features
- Backtesting with slippage, commissions, and market impact
- Walk-forward optimization
- Adaptive retraining configuration
- Real-time alerts system
- Performance metrics tracking

---

## Categories

- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities
