"""
Moving Average Crossover Strategy
Classic dual MA crossover: fast MA crosses slow MA
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_strategy import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover strategy
    
    Signals:
    - BUY: Fast MA crosses above slow MA (golden cross)
    - SELL: Fast MA crosses below slow MA (death cross)
    - HOLD: Otherwise
    """
    
    def __init__(self):
        super().__init__(name="MA Crossover")
        self.parameters = {
            'fast_period': 50,
            'slow_period': 200,
            'ma_type': 'EMA',  # 'SMA' or 'EMA'
            'require_price_above': False,
            'filter_by_trend': False,
            'trend_period': 100
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        df = df.copy()
        
        # Calculate moving averages
        df = self._calculate_moving_averages(df)
        
        # Optional: trend filter
        if self.parameters['filter_by_trend']:
            df['trend_ma'] = self._get_ma(
                df['close'], 
                self.parameters['trend_period']
            )
            uptrend = df['close'] > df['trend_ma']
        else:
            uptrend = True
        
        # Generate signals
        df['signal'] = 0
        
        # BUY: Golden Cross (fast MA crosses above slow MA)
        golden_cross = (
            (df['fast_ma'] > df['slow_ma']) & 
            (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        )
        
        if self.parameters['require_price_above']:
            golden_cross = golden_cross & (df['close'] > df['slow_ma'])
        
        if self.parameters['filter_by_trend']:
            golden_cross = golden_cross & uptrend
        
        df.loc[golden_cross, 'signal'] = 1
        
        # SELL: Death Cross (fast MA crosses below slow MA)
        death_cross = (
            (df['fast_ma'] < df['slow_ma']) & 
            (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
        )
        
        if self.parameters['require_price_above']:
            death_cross = death_cross & (df['close'] < df['slow_ma'])
        
        if self.parameters['filter_by_trend']:
            death_cross = death_cross & ~uptrend
        
        df.loc[death_cross, 'signal'] = -1
        
        # Calculate signal strength based on MA separation
        df['signal_strength'] = 0
        
        signal_mask = df['signal'] != 0
        if signal_mask.any():
            ma_separation = abs(
                df.loc[signal_mask, 'fast_ma'] - df.loc[signal_mask, 'slow_ma']
            ) / df.loc[signal_mask, 'close']
            
            df.loc[signal_mask, 'signal_strength'] = (
                ma_separation * 100
            ).clip(1, 5)
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fast and slow moving averages"""
        df['fast_ma'] = self._get_ma(df['close'], self.parameters['fast_period'])
        df['slow_ma'] = self._get_ma(df['close'], self.parameters['slow_period'])
        return df
    
    def _get_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate moving average (SMA or EMA)"""
        if self.parameters['ma_type'] == 'SMA':
            return series.rolling(window=period).mean()
        else:  # EMA
            return series.ewm(span=period, adjust=False).mean()
    
    def get_parameters(self) -> Dict:
        """Get strategy parameters"""
        return self.parameters.copy()
    
    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        for key, value in params.items():
            if key in self.parameters:
                self.parameters[key] = value


# Create preset configurations
PRESETS = {
    'conservative': {
        'fast_period': 50,
        'slow_period': 200,
        'ma_type': 'SMA',
        'require_price_above': True,
        'filter_by_trend': True,
        'trend_period': 100
    },
    'aggressive': {
        'fast_period': 20,
        'slow_period': 100,
        'ma_type': 'EMA',
        'require_price_above': False,
        'filter_by_trend': False,
        'trend_period': 100
    },
    'default': {
        'fast_period': 50,
        'slow_period': 200,
        'ma_type': 'EMA',
        'require_price_above': False,
        'filter_by_trend': False,
        'trend_period': 100
    },
    'scalping': {
        'fast_period': 10,
        'slow_period': 30,
        'ma_type': 'EMA',
        'require_price_above': False,
        'filter_by_trend': False,
        'trend_period': 100
    }
}


if __name__ == "__main__":
    # Test strategy
    print("Moving Average Crossover Strategy")
    print("=" * 50)
    
    strategy = MovingAverageCrossoverStrategy()
    print(f"\nStrategy: {strategy}")
    print(f"Parameters: {strategy.get_parameters()}")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=300, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(300).cumsum() + 100,
        'high': np.random.randn(300).cumsum() + 102,
        'low': np.random.randn(300).cumsum() + 98,
        'close': np.random.randn(300).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 300)
    }, index=dates)
    
    # Generate signals
    result = strategy.generate_signals(df)
    
    print(f"\nGenerated {len(result[result['signal'] != 0])} signals")
    print(f"BUY signals: {len(result[result['signal'] == 1])}")
    print(f"SELL signals: {len(result[result['signal'] == -1])}")
