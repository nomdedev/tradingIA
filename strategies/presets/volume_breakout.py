"""
Volume Breakout Strategy
Trade breakouts confirmed by volume spikes
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_strategy import BaseStrategy


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume-confirmed breakout strategy
    
    Signals:
    - BUY: Price breaks above resistance with high volume
    - SELL: Price breaks below support with high volume
    - HOLD: Otherwise
    """
    
    def __init__(self):
        super().__init__(name="Volume Breakout")
        self.parameters = {
            'lookback_period': 20,
            'volume_ma_period': 20,
            'volume_multiplier': 1.5,
            'breakout_threshold': 0.02,  # 2% breakout
            'require_close_beyond': True,
            'atr_period': 14
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume breakout signals"""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        df = df.copy()
        
        # Calculate indicators
        df = self._calculate_support_resistance(df)
        df = self._calculate_volume_indicators(df)
        df = self._calculate_atr(df)
        
        # Generate signals
        df['signal'] = 0
        
        # High volume condition
        high_volume = (
            df['volume'] > df['volume_ma'] * self.parameters['volume_multiplier']
        )
        
        # BUY: Breakout above resistance with high volume
        breakout_above = df['close'] > df['resistance'] * (
            1 + self.parameters['breakout_threshold']
        )
        
        if self.parameters['require_close_beyond']:
            breakout_above = breakout_above & (df['close'] > df['resistance'])
        
        buy_condition = breakout_above & high_volume
        df.loc[buy_condition, 'signal'] = 1
        
        # SELL: Breakdown below support with high volume
        breakdown_below = df['close'] < df['support'] * (
            1 - self.parameters['breakout_threshold']
        )
        
        if self.parameters['require_close_beyond']:
            breakdown_below = breakdown_below & (df['close'] < df['support'])
        
        sell_condition = breakdown_below & high_volume
        df.loc[sell_condition, 'signal'] = -1
        
        # Calculate signal strength based on volume ratio and breakout size
        df['signal_strength'] = 0
        
        # Buy strength
        buy_mask = df['signal'] == 1
        if buy_mask.any():
            volume_ratio = (
                df.loc[buy_mask, 'volume'] / df.loc[buy_mask, 'volume_ma']
            )
            breakout_size = (
                (df.loc[buy_mask, 'close'] - df.loc[buy_mask, 'resistance']) / 
                df.loc[buy_mask, 'atr']
            )
            df.loc[buy_mask, 'signal_strength'] = (
                (volume_ratio * breakout_size * 2)
            ).clip(1, 5)
        
        # Sell strength
        sell_mask = df['signal'] == -1
        if sell_mask.any():
            volume_ratio = (
                df.loc[sell_mask, 'volume'] / df.loc[sell_mask, 'volume_ma']
            )
            breakdown_size = (
                (df.loc[sell_mask, 'support'] - df.loc[sell_mask, 'close']) / 
                df.loc[sell_mask, 'atr']
            )
            df.loc[sell_mask, 'signal_strength'] = (
                (volume_ratio * breakdown_size * 2)
            ).clip(1, 5)
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels"""
        # Rolling high/low for lookback period
        df['resistance'] = df['high'].rolling(
            window=self.parameters['lookback_period']
        ).max()
        
        df['support'] = df['low'].rolling(
            window=self.parameters['lookback_period']
        ).min()
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume moving average"""
        df['volume_ma'] = df['volume'].rolling(
            window=self.parameters['volume_ma_period']
        ).mean()
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range for volatility adjustment"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(
            window=self.parameters['atr_period']
        ).mean()
        
        return df
    
    def get_parameters(self) -> Dict:
        """Get strategy parameters"""
        return self.parameters.copy()
    
    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        for key, value in params.items():
            if key in self.parameters:
                self.parameters[key] = value
    
    def get_description(self) -> str:
        """Get strategy description"""
        return (
            "Estrategia de ruptura confirmada por volumen. "
            "Opera cuando el precio rompe niveles clave de soporte/resistencia con volumen alto, "
            "indicando movimientos fuertes y sostenidos."
        )
    
    def get_detailed_info(self) -> Dict:
        """Get detailed strategy information"""
        return {
            'name': self.name,
            'description': self.get_description(),
            'buy_signals': (
                "📈 COMPRA cuando (Breakout alcista):\n"
                f"  • Precio rompe por encima de resistencia (+{self.parameters['breakout_threshold']*100}%)\n"
                f"  • Volumen superior a {self.parameters['volume_multiplier']}x el promedio\n"
                "  • Opcionalmente: cierre debe estar por encima de resistencia\n"
                "  • Indica fuerte momentum de compra"
            ),
            'sell_signals': (
                "📉 VENTA cuando (Breakdown bajista):\n"
                f"  • Precio rompe por debajo de soporte (-{self.parameters['breakout_threshold']*100}%)\n"
                f"  • Volumen superior a {self.parameters['volume_multiplier']}x el promedio\n"
                "  • Opcionalmente: cierre debe estar por debajo de soporte\n"
                "  • Indica fuerte momentum de venta"
            ),
            'parameters': {
                'lookback_period': f"{self.parameters['lookback_period']} - Período para detectar soporte/resistencia",
                'volume_ma_period': f"{self.parameters['volume_ma_period']} - Período para media de volumen",
                'volume_multiplier': f"{self.parameters['volume_multiplier']}x - Multiplicador de volumen requerido",
                'breakout_threshold': f"{self.parameters['breakout_threshold']*100}% - Umbral de ruptura",
                'require_close_beyond': f"{self.parameters['require_close_beyond']} - Requiere cierre más allá del nivel",
                'atr_period': f"{self.parameters['atr_period']} - Período ATR para volatilidad"
            },
            'risk_level': 'Agresivo',
            'timeframe': '5min',
            'indicators': ['Support/Resistance', 'Volume MA', 'ATR']
        }


# Create preset configurations
PRESETS = {
    'conservative': {
        'lookback_period': 30,
        'volume_ma_period': 30,
        'volume_multiplier': 2.0,
        'breakout_threshold': 0.03,
        'require_close_beyond': True,
        'atr_period': 14
    },
    'aggressive': {
        'lookback_period': 10,
        'volume_ma_period': 10,
        'volume_multiplier': 1.2,
        'breakout_threshold': 0.01,
        'require_close_beyond': False,
        'atr_period': 10
    },
    'default': {
        'lookback_period': 20,
        'volume_ma_period': 20,
        'volume_multiplier': 1.5,
        'breakout_threshold': 0.02,
        'require_close_beyond': True,
        'atr_period': 14
    }
}


if __name__ == "__main__":
    # Test strategy
    print("Volume Breakout Strategy")
    print("=" * 50)
    
    strategy = VolumeBreakoutStrategy()
    print(f"\nStrategy: {strategy}")
    print(f"Parameters: {strategy.get_parameters()}")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=150, freq='1h')
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
