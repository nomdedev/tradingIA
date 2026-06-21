"""
Squeeze Momentum + ADX + TTM + EMA15m50 Proximity Filter Strategy Preset
Enhanced version with 15-minute EMA50 proximity filter (4% threshold)

Based on optimization results showing 56.5% WR and 1.012 PF near this EMA.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from strategies.squeeze_adx_ttm_ema15m50_strategy import SqueezeMomentumADXTTMEMA15m50Strategy


class SqueezeMomentumADXTTMEMA15m50Preset(SqueezeMomentumADXTTMEMA15m50Strategy):
    """
    Squeeze+ADX+TTM strategy with EMA15m50 proximity filter preset
    """

    def __init__(self):
        super().__init__()

        # Optimized parameters based on backtest results
        self.set_parameters({
            'ema15m50_proximity_pct': 4.0,  # 4% threshold for optimal win rate
            'use_ema15m50_filter': True,
            'ema15m50_period': 50,
            'ema15m50_timeframe': '15T'
        })


# Preset configurations
PRESETS = {
    'default': {
        'ema15m50_proximity_pct': 4.0,
        'use_ema15m50_filter': True,
        'ema15m50_period': 50,
        'ema15m50_timeframe': '15T'
    },
    'conservative': {
        'ema15m50_proximity_pct': 2.0,  # Tighter filter
        'use_ema15m50_filter': True,
        'ema15m50_period': 50,
        'ema15m50_timeframe': '15T'
    },
    'aggressive': {
        'ema15m50_proximity_pct': 6.0,  # Looser filter
        'use_ema15m50_filter': True,
        'ema15m50_period': 50,
        'ema15m50_timeframe': '15T'
    }
}


def create_strategy(preset='default'):
    """Create strategy instance with preset parameters"""
    strategy = SqueezeMomentumADXTTMEMA15m50Preset()

    if preset in PRESETS:
        strategy.set_parameters(**PRESETS[preset])

    return strategy