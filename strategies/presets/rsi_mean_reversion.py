"""
RSI Mean Reversion Strategy
Buy when RSI < oversold, Sell when RSI > overbought
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_strategy import BaseStrategy


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy
    
    Signals:
    - BUY: RSI crosses below oversold level
    - SELL: RSI crosses above overbought level
    - HOLD: Otherwise
    """
    
    def __init__(self):
        super().__init__(name="RSI Mean Reversion")
        self.parameters = {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'use_smoothing': False,
            'smooth_period': 3
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        df = df.copy()
        
        # Calculate RSI
        df = self._calculate_rsi(df, self.parameters['rsi_period'])
        
        # Optional smoothing
        if self.parameters['use_smoothing']:
            df['rsi'] = df['rsi'].rolling(
                window=self.parameters['smooth_period']
            ).mean()
        
        # Generate signals
        df['signal'] = 0
        
        # BUY when RSI crosses below oversold
        df.loc[
            (df['rsi'] < self.parameters['oversold']) & 
            (df['rsi'].shift(1) >= self.parameters['oversold']),
            'signal'
        ] = 1
        
        # SELL when RSI crosses above overbought
        df.loc[
            (df['rsi'] > self.parameters['overbought']) & 
            (df['rsi'].shift(1) <= self.parameters['overbought']),
            'signal'
        ] = -1
        
        # Calculate signal strength based on RSI distance from threshold
        df['signal_strength'] = 0
        
        # Buy strength: lower RSI = stronger signal
        buy_mask = df['signal'] == 1
        df.loc[buy_mask, 'signal_strength'] = (
            (self.parameters['oversold'] - df.loc[buy_mask, 'rsi']) / 10
        ).clip(1, 5)
        
        # Sell strength: higher RSI = stronger signal
        sell_mask = df['signal'] == -1
        df.loc[sell_mask, 'signal_strength'] = (
            (df.loc[sell_mask, 'rsi'] - self.parameters['overbought']) / 10
        ).clip(1, 5)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
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
        'rsi_period': 14,
        'oversold': 25,
        'overbought': 75,
        'use_smoothing': True,
        'smooth_period': 3
    },
    'aggressive': {
        'rsi_period': 10,
        'oversold': 35,
        'overbought': 65,
        'use_smoothing': False,
        'smooth_period': 3
    },
    'default': {
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
        'use_smoothing': False,
        'smooth_period': 3
    }
}


if __name__ == "__main__":
    # Test strategy
    print("RSI Mean Reversion Strategy")
    print("=" * 50)
    
    strategy = RSIMeanReversionStrategy()
    print(f"\nStrategy: {strategy}")
    print(f"Parameters: {strategy.get_parameters()}")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Generate signals
    result = strategy.generate_signals(df)
    
    print(f"\nGenerated {len(result[result['signal'] != 0])} signals")
    print(f"BUY signals: {len(result[result['signal'] == 1])}")
    print(f"SELL signals: {len(result[result['signal'] == -1])}")
    
    print("\nPresets available:")
    for preset_name in PRESETS.keys():
        print(f"  - {preset_name}")
