"""
Momentum Trading Strategy Implementation
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, StrategyConfig

class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.

    This strategy generates signals based on price momentum, entering long positions
    when momentum is positive and strong, and exiting when momentum weakens.
    """

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for momentum strategy."""
        return ['fast_period', 'slow_period', 'momentum_threshold']

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        df = data.copy()

        # Calculate moving averages
        fast_period = self.config.parameters['fast_period']
        slow_period = self.config.parameters['slow_period']

        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()

        # Calculate momentum (rate of change)
        df['momentum'] = df['close'].pct_change(periods=fast_period)

        # Calculate RSI for additional momentum confirmation
        df['rsi'] = self._calculate_rsi(df['close'], 14)

        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on momentum."""
        signals = pd.Series(0, index=data.index)

        fast_period = self.config.parameters['fast_period']
        slow_period = self.config.parameters['slow_period']
        momentum_threshold = self.config.parameters['momentum_threshold']

        # Skip initial rows where indicators are NaN
        start_idx = max(fast_period, slow_period, 14, 26)  # Max of all periods used

        for i in range(start_idx, len(data)):
            # Momentum conditions
            momentum_up = data.iloc[i]['momentum'] > momentum_threshold
            ma_crossover_up = (data.iloc[i]['fast_ma'] > data.iloc[i]['slow_ma'] and
                             data.iloc[i-1]['fast_ma'] <= data.iloc[i-1]['slow_ma'])
            rsi_oversold = data.iloc[i]['rsi'] < 70  # Not overbought
            macd_positive = data.iloc[i]['macd_hist'] > 0

            # Exit conditions
            momentum_down = data.iloc[i]['momentum'] < -momentum_threshold
            ma_crossover_down = (data.iloc[i]['fast_ma'] < data.iloc[i]['slow_ma'] and
                               data.iloc[i-1]['fast_ma'] >= data.iloc[i-1]['slow_ma'])
            rsi_overbought = data.iloc[i]['rsi'] > 70

            # Generate signals
            if momentum_up and ma_crossover_up and rsi_oversold and macd_positive:
                signals.iloc[i] = 1  # Buy signal
            elif momentum_down or ma_crossover_down or rsi_overbought:
                signals.iloc[i] = -1  # Sell signal

        return signals

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist