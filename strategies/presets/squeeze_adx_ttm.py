"""
Squeeze Momentum + ADX + TTM Strategy Preset
Pre-configured parameters for the advanced multi-indicator strategy
"""

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy


def create_squeeze_adx_ttm_strategy():
    """
    Create Squeeze Momentum + ADX + TTM strategy with optimized parameters

    Returns:
        SqueezeMomentumADXTTMStrategy: Configured strategy instance
    """
    strategy = SqueezeMomentumADXTTMStrategy()

    # Optimized parameters based on parameter importance analysis
    optimized_params = {
        # Squeeze Momentum - optimized for BTC volatility
        'bb_length': 20,
        'bb_mult': 2.0,
        'kc_length': 20,
        'kc_mult': 1.5,
        'linear_momentum': 20,
        'use_true_range': True,

        # ADX - tuned for trend detection
        'adx_length': 14,
        'di_length': 14,
        'key_level': 23,

        # TTM Waves - market structure analysis
        'wave_a_length': 55,
        'wave_b_length': 144,
        'wave_c_length': 233,
        'fast_ma_period': 8,

        # Multi-timeframe - higher timeframe confirmation
        'higher_tf_weight': 0.3,
        'lower_tf_weight': 0.2,

        # Signal thresholds - balanced sensitivity
        'squeeze_threshold': 0.5,
        'adx_threshold': 20,
        'momentum_threshold': 0.1,

        # Risk management - conservative approach
        'stop_loss_atr_mult': 1.5,
        'take_profit_rr': 2.0
    }

    strategy.set_parameters(optimized_params)
    return strategy


# Conservative preset - lower risk, fewer signals
def create_squeeze_adx_ttm_conservative():
    """Create conservative version with stricter filters"""
    strategy = SqueezeMomentumADXTTMStrategy()

    conservative_params = {
        'bb_length': 25,
        'bb_mult': 2.2,
        'kc_length': 25,
        'kc_mult': 1.8,
        'adx_threshold': 25,
        'key_level': 25,
        'squeeze_threshold': 0.7,
        'momentum_threshold': 0.15,
        'higher_tf_weight': 0.4,
        'stop_loss_atr_mult': 1.2,
        'take_profit_rr': 1.8
    }

    strategy.set_parameters(conservative_params)
    return strategy


# Aggressive preset - higher risk, more signals
def create_squeeze_adx_ttm_aggressive():
    """Create aggressive version with looser filters"""
    strategy = SqueezeMomentumADXTTMStrategy()

    aggressive_params = {
        'bb_length': 15,
        'bb_mult': 1.8,
        'kc_length': 15,
        'kc_mult': 1.2,
        'adx_threshold': 18,
        'key_level': 20,
        'squeeze_threshold': 0.3,
        'momentum_threshold': 0.08,
        'higher_tf_weight': 0.2,
        'stop_loss_atr_mult': 1.8,
        'take_profit_rr': 2.5
    }

    strategy.set_parameters(aggressive_params)
    return strategy


# Preset configurations for easy access
PRESETS = {
    'squeeze_adx_ttm': create_squeeze_adx_ttm_strategy,
    'squeeze_adx_ttm_conservative': create_squeeze_adx_ttm_conservative,
    'squeeze_adx_ttm_aggressive': create_squeeze_adx_ttm_aggressive
}


def get_preset_names():
    """Get list of available preset names"""
    return list(PRESETS.keys())


def create_strategy_from_preset(preset_name: str):
    """
    Create strategy instance from preset name

    Args:
        preset_name: Name of the preset configuration

    Returns:
        SqueezeMomentumADXTTMStrategy: Configured strategy instance

    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in PRESETS:
        available = get_preset_names()
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available}")

    return PRESETS[preset_name]()