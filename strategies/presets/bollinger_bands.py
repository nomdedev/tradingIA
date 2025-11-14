"""
Bollinger Bands Breakout Strategy
Buy on lower band bounce, Sell on upper band touch
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands breakout/reversion strategy
    
    Signals:
    - BUY: Price touches lower band (oversold)
    - SELL: Price touches upper band (overbought)
    - HOLD: Otherwise
    """
    
    def __init__(self):
        super().__init__(name="Bollinger Bands")
        self.parameters = {
            'period': 20,
            'num_std': 2.0,
            'use_close_for_bands': True,
            'require_volume_confirmation': False,
            'volume_ma_period': 20
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        df = df.copy()
        
        # Calculate Bollinger Bands
        df = self._calculate_bollinger_bands(df)
        
        # Optional: volume confirmation
        if self.parameters['require_volume_confirmation']:
            df['volume_ma'] = df['volume'].rolling(
                window=self.parameters['volume_ma_period']
            ).mean()
            high_volume = df['volume'] > df['volume_ma']
        else:
            high_volume = True
        
        # Generate signals
        df['signal'] = 0
        
        # BUY when price touches or crosses below lower band
        buy_condition = (
            (df['close'] <= df['bb_lower']) | 
            (df['low'] <= df['bb_lower'])
        )
        
        if self.parameters['require_volume_confirmation']:
            buy_condition = buy_condition & high_volume
        
        df.loc[buy_condition, 'signal'] = 1
        
        # SELL when price touches or crosses above upper band
        sell_condition = (
            (df['close'] >= df['bb_upper']) | 
            (df['high'] >= df['bb_upper'])
        )
        
        if self.parameters['require_volume_confirmation']:
            sell_condition = sell_condition & high_volume
        
        df.loc[sell_condition, 'signal'] = -1
        
        # Calculate signal strength based on distance from band
        df['signal_strength'] = 0
        
        # Buy strength: further below lower band = stronger
        buy_mask = df['signal'] == 1
        if buy_mask.any():
            df.loc[buy_mask, 'signal_strength'] = (
                (df.loc[buy_mask, 'bb_lower'] - df.loc[buy_mask, 'close']) / 
                df.loc[buy_mask, 'bb_width'] * 10
            ).clip(1, 5)
        
        # Sell strength: further above upper band = stronger
        sell_mask = df['signal'] == -1
        if sell_mask.any():
            df.loc[sell_mask, 'signal_strength'] = (
                (df.loc[sell_mask, 'close'] - df.loc[sell_mask, 'bb_upper']) / 
                df.loc[sell_mask, 'bb_width'] * 10
            ).clip(1, 5)
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        # Middle band (SMA)
        df['bb_middle'] = df['close'].rolling(
            window=self.parameters['period']
        ).mean()
        
        # Standard deviation
        df['bb_std'] = df['close'].rolling(
            window=self.parameters['period']
        ).std()
        
        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (
            self.parameters['num_std'] * df['bb_std']
        )
        df['bb_lower'] = df['bb_middle'] - (
            self.parameters['num_std'] * df['bb_std']
        )
        
        # Band width (for signal strength calculation)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
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
        'period': 20,
        'num_std': 2.5,
        'use_close_for_bands': True,
        'require_volume_confirmation': True,
        'volume_ma_period': 20
    },
    'aggressive': {
        'period': 15,
        'num_std': 1.5,
        'use_close_for_bands': True,
        'require_volume_confirmation': False,
        'volume_ma_period': 20
    },
    'default': {
        'period': 20,
        'num_std': 2.0,
        'use_close_for_bands': True,
        'require_volume_confirmation': False,
        'volume_ma_period': 20
    }
}


if __name__ == "__main__":
    # Test strategy
    print("Bollinger Bands Strategy")
    print("=" * 50)
    
    strategy = BollingerBandsStrategy()
    print(f"\nStrategy: {strategy}")
    print(f"Parameters: {strategy.get_parameters()}")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=150, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(150).cumsum() + 100,
        'high': np.random.randn(150).cumsum() + 102,
        'low': np.random.randn(150).cumsum() + 98,
        'close': np.random.randn(150).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 150)
    }, index=dates)
    
    # Generate signals
    result = strategy.generate_signals(df)
    
    print(f"\nGenerated {len(result[result['signal'] != 0])} signals")
    print(f"BUY signals: {len(result[result['signal'] == 1])}")
    print(f"SELL signals: {len(result[result['signal'] == -1])}")
