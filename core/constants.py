"""
Global Constants for TradingIA.

Centraliza valores que antes estaban hardcodeados en múltiples archivos.
AUDITORÍA FIX: Eliminar magic numbers.

Author: TradingIA Team
Date: 14 de Enero, 2026
"""

# ==============================================================================
# FINANCIAL CONSTANTS
# ==============================================================================

# Risk-free rate (proxy para crypto, basado en staking rates)
RISK_FREE_RATE_ANNUAL = 0.04  # 4% anual
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / 252
RISK_FREE_RATE_HOURLY = RISK_FREE_RATE_ANNUAL / (252 * 24)
RISK_FREE_RATE_5MIN = RISK_FREE_RATE_ANNUAL / (252 * 24 * 12)

# Trading days per year
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 24  # Crypto es 24/7
BARS_5MIN_PER_DAY = 288  # 24 * 12

# ==============================================================================
# RISK MANAGEMENT DEFAULTS
# ==============================================================================

# Drawdown limits
DEFAULT_MAX_DAILY_DRAWDOWN = 0.05  # 5%
DEFAULT_MAX_TOTAL_DRAWDOWN = 0.15  # 15%
DEFAULT_MAX_POSITION_PCT = 0.10  # 10% max position size

# VaR/CVaR defaults
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_CVAR_CONFIDENCE = 0.95
DEFAULT_MAX_VAR_PCT = 0.02  # 2% del equity

# Consecutive losses
DEFAULT_MAX_CONSECUTIVE_LOSSES = 5

# ==============================================================================
# KELLY CRITERION DEFAULTS
# ==============================================================================

DEFAULT_KELLY_FRACTION = 0.5  # Half Kelly (conservador)
DEFAULT_MIN_KELLY_FRACTION = 0.1
DEFAULT_MAX_KELLY_FRACTION = 1.0

# ==============================================================================
# MARKET IMPACT - CRYPTO
# ==============================================================================

# Global daily volumes (USD billions)
CRYPTO_DAILY_VOLUME = {
    "BTC": 30_000_000_000,  # $30B
    "ETH": 15_000_000_000,  # $15B
    "SOL": 3_000_000_000,   # $3B
    "DEFAULT": 1_000_000_000,  # $1B
}

# Sell penalty (ventas tienen más impacto)
CRYPTO_SELL_PENALTY = 1.35  # 35% más slippage en ventas

# ==============================================================================
# BACKTEST DEFAULTS
# ==============================================================================

DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_COMMISSION_PCT = 0.001  # 0.1% (típico crypto)
DEFAULT_SLIPPAGE_PCT = 0.0005  # 0.05%

# Walk-Forward Analysis
DEFAULT_WFA_WINDOWS = 6
DEFAULT_WFA_TRAIN_RATIO = 0.7
DEFAULT_WFA_STABILITY_THRESHOLD = 0.6

# ==============================================================================
# COUNCIL THRESHOLDS
# ==============================================================================

COUNCIL_APPROVAL_THRESHOLD = 0.6
COUNCIL_REJECTION_THRESHOLD = 0.4
COUNCIL_VETO_SCORE = -1.0

# ==============================================================================
# STRATEGY PARAMETERS
# ==============================================================================

# EMA periods
EMA_FAST_DEFAULT = 8
EMA_MEDIUM_DEFAULT = 21
EMA_SLOW_DEFAULT = 55
EMA_TREND_DEFAULT = 210

# ATR for stops
ATR_PERIOD_DEFAULT = 14
ATR_SL_MULTIPLIER_DEFAULT = 1.5
ATR_TP_MULTIPLIER_DEFAULT = 2.2

# Volume Profile
VP_WINDOW_DEFAULT = 20
VP_NUM_BINS_DEFAULT = 50

# ==============================================================================
# LOGGING
# ==============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# ==============================================================================
# PROJECT PATHS (Relative to PROJECT_ROOT)
# ==============================================================================

from pathlib import Path

# Project root - detectado automáticamente basado en la ubicación de este archivo
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_CACHE_DIR = DATA_DIR / "cache"

# Config paths
CONFIG_DIR = PROJECT_ROOT / "config"

# Logs paths
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PLATFORM = LOGS_DIR / "platform.log"
LOG_FILE_BACKTEST = LOGS_DIR / "backtest.log"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Rules paths
RULES_DIR = PROJECT_ROOT / "core" / "rules"

# Default data files
DEFAULT_BTC_5MIN = DATA_DIR / "btc_5Min.csv"
DEFAULT_BTC_15MIN = DATA_DIR / "btc_15Min.csv"
DEFAULT_BTC_1H = DATA_DIR / "btc_1H.csv"

# ==============================================================================
# FILE PATHS - JSON/YAML CONFIGS
# ==============================================================================

CONFIG_APP = CONFIG_DIR / "app_config.json"
CONFIG_BACKTEST = CONFIG_DIR / "backtest_configs.json"
CONFIG_COSTS = CONFIG_DIR / "costs_params.json"
CONFIG_TRAINING = CONFIG_DIR / "training_config.yaml"
CONFIG_STRATEGIES = CONFIG_DIR / "strategies_registry.json"