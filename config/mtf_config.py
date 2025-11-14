"""
Multi-Timeframe Configuration - BTC IFVG Strategy
=================================================
Configuración optimizada para estrategia BTC con:
- Multi-TF: 5Min (entries), 15Min (momentum), 1H (trend ALWAYS)
- Parámetros interconectados y optimizables
- Walk-forward, Bayes, Monte Carlo setup
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'
MODELS_DIR = PROJECT_ROOT / 'models'
DOCS_DIR = PROJECT_ROOT / 'docs'
PINE_DIR = PROJECT_ROOT / 'scripts_pine'

for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR, DOCS_DIR, PINE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ALPACA API
# ============================================================================
ALPACA_CONFIG = {
    'api_key': os.getenv('ALPACA_API_KEY', ''),
    'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
    'base_url': 'https://paper-api.alpaca.markets',
    'data_url': 'https://data.alpaca.markets'
}

# ============================================================================
# MULTI-TIMEFRAME SETUP (CRITICAL)
# ============================================================================
MTF_CONFIG = {
    'timeframes': {
        'entry': '5Min',      # Entry signals
        'momentum': '15Min',  # Momentum confirmation  
        'trend': '1H'         # Major trend (ALWAYS filter)
    },
    'alpaca_map': {'5Min': '5Min', '15Min': '15Min', '1H': '1Hour'},
    'resample': {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'},
    'rate_limit': 200,
    'rate_delay': 0.35,
    'timezone': 'America/Argentina/Buenos_Aires'
}

# ============================================================================
# DATA FETCH CONFIG
# ============================================================================
DATA_FETCH_CONFIG = {
    'rate_limit_delay': 0.35,
    'max_retries': 3,
    'min_volume_threshold': 1000,
    'cache_data': True
}

# ============================================================================
# TRADING CONFIG
# ============================================================================
TRADING_CONFIG = {
    'symbol': 'BTCUSD',
    'start_date': '2023-01-01',
    'end_date': '2025-11-12',
    'initial_capital': 10000,
    'risk_per_trade': 0.01,
    'max_positions': 3,
    'commission': 0.001,
    'slippage': 0.001,
    'htf_filter_enabled': True  # ALWAYS True - trend filter mandatory
}

# ============================================================================
# INDICATOR PARAMETERS (Optimizable ranges)
# ============================================================================
INDICATOR_PARAMS = {
    # IFVG Enhanced
    'ifvg': {
        'atr_multi': 0.3,        # Range: 0.1-0.5
        'min_gap_size': 0.0015,  # 0.15% minimum
        'strength_thresh': 0.5,   # gap_size/ATR > 0.5
        'lookback': 50           # Mitigation lookback
    },
    
    # Volume Profile Advanced
    'volume_profile': {
        'rows': 120,             # Price bins
        'va_percent': 0.70,      # Value Area: 65-75%
        'sd_thresh': 0.12,       # SD threshold
        'poc_proximity': 0.5     # ATR multiplier for POC proximity
    },
    
    # EMAs Multi-TF (Optimizable lengths)
    'emas': {
        'entry_tf': [18, 48],    # 5Min: Range 15-25, 40-60
        'momentum_tf': [21, 50], # 15Min
        'trend_tf': [95, 200]    # 1H: Range 90-100, 195-210
    },
    
    # Volume Filters Cross-TF
    'volume': {
        'vol_thresh': 1.2,       # Range: 0.8-1.5
        'sma_period': 21,
        'cross_tf_multi': 1.0    # vol_5m > SMA_vol_1h
    },
    
    # Additional Filters
    'filters': {
        'rsi_period': 14,
        'rsi_thresh_long': 50,   # RSI_15m > 50 for longs
        'rsi_thresh_short': 50,
        'atr_period': 14
    }
}

# ============================================================================
# SIGNAL GENERATION (Multi-TF interconnected)
# ============================================================================
SIGNAL_CONFIG = {
    # HTF Trend Filter (ALWAYS applies)
    'htf_trend': {
        'enabled': True,  # MANDATORY
        'ema_period': 200,
        'uptrend_rule': 'close > EMA200_1h',  # Longs only
        'downtrend_rule': 'close < EMA200_1h'  # Shorts only
    },
    
    # MTF Momentum Confirmation
    'mtf_momentum': {
        'enabled': True,
        'cross_rule': 'EMA20_5m > EMA50_15m',  # Resampled
        'alignment_pct': 0.7  # 70% time aligned
    },
    
    # Vol Cross-TF
    'vol_cross': {
        'enabled': True,
        'rule': '(vol_5m > thresh*SMA21_5m) AND (vol_5m > SMA_vol_1h)',
        'rolling_mean_period': 10
    },
    
    # VP Proximity
    'vp_proximity': {
        'enabled': True,
        'poc_distance': 'abs(close - POC_1h) < 0.5*ATR_1h',
        'val_filter': 'close > VAL_5m for longs'
    },
    
    # Final Signal Combination
    'combination': {
        'bull': 'bull_signal AND uptrend_1h AND momentum_15m AND vol_cross AND vp_filters',
        'bear': 'bear_signal AND NOT uptrend_1h AND vol_cross',
        'confidence_min': 0.6
    }
}

# ============================================================================
# BACKTEST CONFIG
# ============================================================================
BACKTEST_CONFIG = {
    # Entry/Exit Rules
    'entry': {
        'order_type': 'market',
        'max_slippage': 0.001
    },
    
    'exit': {
        'sl_atr_multi': 1.5,     # SL = 1.5*ATR_5m (adjusted by HTF vol)
        'tp_risk_reward': 2.2,   # TP = risk * 2.2
        'trailing_start': 1.0,   # Start trailing after +1R
        'trailing_offset': 0.5   # Trail 0.5*ATR
    },
    
    # Position Sizing
    'position': {
        'risk_formula': 'risk_amt / (ATR_5m / close)',
        'max_exposure': 0.05,    # Max 5% capital per position
        'scale_by_confidence': True
    },
    
    # Metrics Targets
    'targets': {
        'sharpe_min': 1.0,
        'calmar_min': 2.0,
        'max_drawdown': 0.15,    # 15%
        'win_rate_min': 0.55,
        'profit_factor_min': 1.5,
        'htf_alignment_min': 0.70  # 70% trades following HTF
    }
}

# ============================================================================
# OPTIMIZATION CONFIG
# ============================================================================
OPTIMIZATION_CONFIG = {
    # Walk-Forward
    'walk_forward': {
        'n_periods': 6,
        'train_split': 0.7,      # 70% train, 30% test
        'period_months': 3,
        'optimize_metric': 'calmar',
        'min_trades_period': 50
    },
    
    # Bayesian Optimization (skopt)
    'bayesian': {
        'n_calls': 100,
        'n_random_starts': 20,
        'param_space': {
            'atr_multi': (0.1, 0.5),
            'vol_thresh': (0.8, 1.5),
            'ema1_entry': (15, 25),
            'ema2_entry': (40, 60),
            'tp_rr': (1.8, 2.5),
            'va_percent': (0.65, 0.75)
        },
        'acq_func': 'EI',  # Expected Improvement
        'random_state': 42
    },
    
    # Monte Carlo
    'monte_carlo': {
        'n_simulations': 500,
        'noise_pct': 10,         # +/-10% price/vol noise
        'confidence_level': 0.95,
        'metrics_to_test': ['sharpe', 'calmar', 'max_dd'],
        'robustness_threshold': {
            'sharpe_std': 0.1,
            'calmar_std': 0.2
        }
    },
    
    # Stress Testing
    'stress_tests': {
        'scenarios': [
            {'name': 'high_vol', 'vol_multi': 1.5, 'expected_dd': 0.20},
            {'name': 'bear_market', 'price_shift': -0.30, 'expected_dd': 0.25},
            {'name': 'flash_crash', 'price_shift': -0.20, 'duration': '1d'},
            {'name': 'low_vol', 'vol_multi': 0.5, 'expected_sharpe': 0.8},
            {'name': 'whipsaw', 'reverse_freq': 'high', 'expected_wr': 0.45}
        ],
        'survival_threshold': 0.10  # Max 10% DD in stress
    }
}

# ============================================================================
# PAPER TRADING CONFIG
# ============================================================================
PAPER_TRADING_CONFIG = {
    'enabled': True,
    'symbol': 'BTCUSD',
    'position_size': 0.01,  # BTC
    'max_positions': 3,
    'monitor_interval': 300,  # 5 minutes
    'websocket_enabled': True,
    
    # Emergency Rules
    'emergency': {
        'max_dd_stop': 0.10,     # Stop all if 10% DD
        'htf_reversal_close': True,  # Close on HTF trend change
        'max_daily_trades': 10
    },
    
    # Logging
    'log_trades': True,
    'log_file': LOGS_DIR / 'paper_trades.csv',
    'alert_webhook': os.getenv('DISCORD_WEBHOOK', '')
}

# ============================================================================
# PINE SCRIPT EXPORT CONFIG
# ============================================================================
PINE_CONFIG = {
    'version': 'v5',
    'indicator_name': 'MTF IFVG Optimized',
    'strategy_name': 'MTF IFVG Strategy',
    'output_dir': PINE_DIR,
    
    # Template paths
    'indicator_template': PINE_DIR / 'template_indicator.pine',
    'strategy_template': PINE_DIR / 'template_strategy.pine',
    
    # TradingView settings
    'capital': 10000,
    'commission_type': 'percent',
    'commission_value': 0.1,
    'slippage': 1,  # ticks
    'currency': 'USD'
}

# ============================================================================
# VALIDATION & HELPER FUNCTIONS
# ============================================================================

def validate_config():
    """Validate configuration and dependencies"""
    errors = []
    
    # Check API keys
    if not ALPACA_CONFIG['api_key']:
        errors.append("ALPACA_API_KEY not set in .env")
    
    # Check HTF filter is enabled (CRITICAL)
    if not TRADING_CONFIG['htf_filter_enabled']:
        errors.append("HTF filter MUST be enabled - это fundamental requirement")
    
    # Check param correlations
    if INDICATOR_PARAMS['ifvg']['atr_multi'] > 0.4 and INDICATOR_PARAMS['volume']['vol_thresh'] > 1.3:
        errors.append("High atr_multi + high vol_thresh = too restrictive")
    
    # Check targets are realistic
    if BACKTEST_CONFIG['targets']['calmar_min'] > 3.0:
        errors.append("Calmar > 3.0 unrealistic for BTC")
    
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(errors))
    
    return True


def get_param_space():
    """Get optimization parameter space for skopt"""
    from skopt.space import Real, Integer
    
    return [
        Real(0.1, 0.5, name='atr_multi'),
        Real(0.8, 1.5, name='vol_thresh'),
        Integer(15, 25, name='ema1_entry'),
        Integer(40, 60, name='ema2_entry'),
        Integer(90, 100, name='ema1_trend'),
        Integer(195, 210, name='ema2_trend'),
        Real(1.8, 2.5, name='tp_rr'),
        Real(0.65, 0.75, name='va_percent')
    ]


def validate_params(params_dict):
    """Check parameter correlations before optimization"""
    atr_multi = params_dict.get('atr_multi', 0.3)
    vol_thresh = params_dict.get('vol_thresh', 1.2)
    
    # Inverse correlation: high ATR multi → lower vol thresh
    if atr_multi > 0.4 and vol_thresh > 1.3:
        return False, "Params too restrictive together"
    
    # EMA order
    if params_dict.get('ema1_entry', 18) >= params_dict.get('ema2_entry', 48):
        return False, "EMA1 must < EMA2"
    
    return True, "OK"


def get_config():
    """Get full config as dict"""
    return {
        'alpaca': ALPACA_CONFIG,
        'mtf': MTF_CONFIG,
        'trading': TRADING_CONFIG,
        'indicators': INDICATOR_PARAMS,
        'signals': SIGNAL_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'paper': PAPER_TRADING_CONFIG,
        'pine': PINE_CONFIG
    }


if __name__ == '__main__':
    validate_config()
    print("✅ Configuration validated successfully")
    print(f"HTF Filter: {'ENABLED' if TRADING_CONFIG['htf_filter_enabled'] else 'DISABLED'}")
    print(f"Timeframes: {MTF_CONFIG['timeframes']}")
    print(f"Optimization: Walk-forward {OPTIMIZATION_CONFIG['walk_forward']['n_periods']} periods")
