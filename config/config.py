"""
Configuration Module
Centralized configuration for BTC IFVG backtesting system
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
CONFIG_DIR = PROJECT_ROOT / 'config'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Alpaca API Configuration
ALPACA_CONFIG = {
    'api_key': os.getenv('ALPACA_API_KEY', ''),
    'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
    'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
}

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'BTCUSD',
    'timeframes': ['5Min', '15Min', '1H'],
    'start_date': '2023-01-01',
    'end_date': '2025-11-12',
    'timezone': 'UTC',  # Alpaca uses UTC
    'local_timezone': 'America/Argentina/Buenos_Aires',  # UTC-3
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'initial_capital': 10000,
    'risk_per_trade': 0.01,  # 1% of capital
    'commission': 0.001,  # 0.1% round-trip
    'slippage': 0.001,  # 0.1% slippage
    'max_positions': 3,
    'position_size': 0.01,  # BTC per trade
}

# Strategy Parameters - IFVG
IFVG_CONFIG = {
    'atr_period': 200,
    'atr_multiplier': 0.25,
    'wick_based': True,
    'use_close_mitigation': True,
}

# Strategy Parameters - Volume Profile
VP_CONFIG = {
    'rows': 100,
    'value_area_percent': 0.68,  # 68% of volume
    'sd_threshold_percent': 0.15,  # 15% of max volume for S/D
}

# Strategy Parameters - EMAs
EMA_CONFIG = {
    'periods': [20, 50, 100, 200],
    'use_multi_tf': True,
    'ema_5min': [20, 50],
    'ema_15min': [50, 200],
    'ema_1h': [200],
}

# Signal Generation Config
SIGNAL_CONFIG = {
    'volume_threshold': 1.0,  # Multiple of SMA21
    'use_volume_profile': True,
    'use_ema_filter': True,
    'min_confidence': 0.6,
    'risk_reward_ratio': 2.0,
    'sl_atr_multiplier': 2.0,
}

# Paper Trading Config
PAPER_TRADING_CONFIG = {
    'initial_capital': 10000.0,
    'risk_per_trade': 0.01,  # 1% risk per trade
    'check_interval': 300,  # 5 minutes in seconds
    'run_hours': (7, 23),  # UTC-3: 7 AM to 11 PM
    'emergency_drawdown': 0.10,  # 10% max drawdown emergency stop
    'trailing_stop': True,
    'trailing_stop_activation': 1.0,  # After 1R profit
    'sl_atr_multiplier': 1.5,
    'tp_risk_reward': 2.0,
}

# Optimization Config
OPTIMIZATION_CONFIG = {
    'walk_forward_periods': 6,
    'train_test_split': 0.6,  # 60% train, 40% test
    'param_grid': {
        'atr_multiplier': [0.1, 0.25, 0.5],
        'volume_threshold': [0.8, 1.0, 1.5],
        'risk_reward_ratio': [1.5, 2.0, 2.5],
    },
    'monte_carlo_runs': 100,
    'noise_level': 0.10,  # 10% noise in MC simulation
}

# Logging Config
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / 'logs' / 'trading.log',
}

# Data Fetching Config
DATA_FETCH_CONFIG = {
    'rate_limit_delay': 1.0,  # 1 second between requests
    'max_retries': 3,
    'cache_data': True,
    'validate_data': True,
    'min_volume_threshold': 0,  # Minimum volume to keep bar
}

def get_config(section=None):
    """
    Get configuration dictionary
    
    Args:
        section: Specific config section or None for all
        
    Returns:
        Configuration dictionary
    """
    all_config = {
        'alpaca': ALPACA_CONFIG,
        'trading': TRADING_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'ifvg': IFVG_CONFIG,
        'volume_profile': VP_CONFIG,
        'ema': EMA_CONFIG,
        'signals': SIGNAL_CONFIG,
        'paper_trading': PAPER_TRADING_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'data_fetch': DATA_FETCH_CONFIG,
    }
    
    if section:
        return all_config.get(section, {})
    return all_config

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check Alpaca credentials
    if not ALPACA_CONFIG['api_key'] or not ALPACA_CONFIG['secret_key']:
        errors.append("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
    
    # Check date format
    try:
        from datetime import datetime
        datetime.strptime(TRADING_CONFIG['start_date'], '%Y-%m-%d')
        datetime.strptime(TRADING_CONFIG['end_date'], '%Y-%m-%d')
    except ValueError:
        errors.append("start_date and end_date must be in YYYY-MM-DD format")
    
    # Check risk parameters
    if not 0 < BACKTEST_CONFIG['risk_per_trade'] <= 0.05:
        errors.append("risk_per_trade should be between 0 and 0.05 (5%)")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))
    
    return True

if __name__ == "__main__":
    # Test configuration
    print("🔧 Configuration Loaded")
    print("=" * 50)
    
    try:
        validate_config()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration errors:\n{e}")
    
    print(f"\nAlpaca Base URL: {ALPACA_CONFIG['base_url']}")
    print(f"Symbol: {TRADING_CONFIG['symbol']}")
    print(f"Timeframes: {TRADING_CONFIG['timeframes']}")
    print(f"Initial Capital: ${BACKTEST_CONFIG['initial_capital']:,}")
    print(f"Risk per Trade: {BACKTEST_CONFIG['risk_per_trade']*100}%")
