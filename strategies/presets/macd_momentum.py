"""
MACD Momentum Strategy
Buy on bullish MACD crossover, Sell on bearish crossover
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_strategy import BaseStrategy


class MACDMomentumStrategy(BaseStrategy):
    """
    MACD-based momentum strategy
    
    Signals:
    - BUY: MACD crosses above signal line
    - SELL: MACD crosses below signal line
    - HOLD: Otherwise
    """
    
    def __init__(self):
        super().__init__(name="MACD Momentum")
        self.parameters = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'require_histogram_positive': True,
            'min_histogram_strength': 0.0
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD-based signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        df = df.copy()
        
        # Calculate MACD
        df = self._calculate_macd(df)
        
        # Generate signals
        df['signal'] = 0
        
        # BUY when MACD crosses above signal
        bullish_cross = (
            (df['macd'] > df['signal_line']) & 
            (df['macd'].shift(1) <= df['signal_line'].shift(1))
        )
        
        # Optional: require positive histogram
        if self.parameters['require_histogram_positive']:
            bullish_cross = bullish_cross & (df['histogram'] > 0)
        
        # Optional: minimum histogram strength
        if self.parameters['min_histogram_strength'] > 0:
            bullish_cross = bullish_cross & (
                abs(df['histogram']) >= self.parameters['min_histogram_strength']
            )
        
        df.loc[bullish_cross, 'signal'] = 1
        
        # SELL when MACD crosses below signal
        bearish_cross = (
            (df['macd'] < df['signal_line']) & 
            (df['macd'].shift(1) >= df['signal_line'].shift(1))
        )
        
        if self.parameters['require_histogram_positive']:
            bearish_cross = bearish_cross & (df['histogram'] < 0)
        
        if self.parameters['min_histogram_strength'] > 0:
            bearish_cross = bearish_cross & (
                abs(df['histogram']) >= self.parameters['min_histogram_strength']
            )
        
        df.loc[bearish_cross, 'signal'] = -1
        
        # Calculate signal strength based on histogram magnitude
        df['signal_strength'] = 0
        
        signal_mask = df['signal'] != 0
        df.loc[signal_mask, 'signal_strength'] = (
            abs(df.loc[signal_mask, 'histogram']) * 10
        ).clip(1, 5)
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        # Calculate EMAs
        ema_fast = df['close'].ewm(
            span=self.parameters['fast_period'], 
            adjust=False
        ).mean()
        ema_slow = df['close'].ewm(
            span=self.parameters['slow_period'], 
            adjust=False
        ).mean()
        
        # MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Signal line
        df['signal_line'] = df['macd'].ewm(
            span=self.parameters['signal_period'], 
            adjust=False
        ).mean()
        
        # Histogram
        df['histogram'] = df['macd'] - df['signal_line']
        
        return df
    
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
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'require_histogram_positive': True,
        'min_histogram_strength': 0.5
    },
    'aggressive': {
        'fast_period': 8,
        'slow_period': 21,
        'signal_period': 7,
        'require_histogram_positive': False,
        'min_histogram_strength': 0.0
    },
    'default': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'require_histogram_positive': True,
        'min_histogram_strength': 0.0
    }
}


if __name__ == "__main__":
    # Test strategy
    print("MACD Momentum Strategy")
    print("=" * 50)
    
    strategy = MACDMomentumStrategy()
    print(f"\nStrategy: {strategy}")
    print(f"Parameters: {strategy.get_parameters()}")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Generate signals
    result = strategy.generate_signals(df)
    
    print(f"\nGenerated {len(result[result['signal'] != 0])} signals")
    print(f"BUY signals: {len(result[result['signal'] == 1])}")
    print(f"SELL signals: {len(result[result['signal'] == -1])}")
