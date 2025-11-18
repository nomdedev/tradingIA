"""
Mean Reversion Trading Strategy Implementation
"""

from typing import List
import pandas as pd
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.

    This strategy identifies overbought/oversold conditions and trades against
    the trend, expecting prices to revert to their mean.
    """

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for mean reversion strategy."""
        return ['lookback_period', 'entry_threshold', 'exit_threshold']

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        df = data.copy()

        lookback = self.config.parameters['lookback_period']

        # Calculate rolling mean and standard deviation
        df['rolling_mean'] = df['close'].rolling(window=lookback).mean()
        df['rolling_std'] = df['close'].rolling(window=lookback).std()

        # Calculate z-score (standardized deviation from mean)
        df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']

        # Calculate Bollinger Bands
        df['bb_upper'] = df['rolling_mean'] + (df['rolling_std'] * 2)
        df['bb_lower'] = df['rolling_mean'] - (df['rolling_std'] * 2)
        df['bb_middle'] = df['rolling_mean']

        # Calculate RSI for additional confirmation
        df['rsi'] = self._calculate_rsi(df['close'], 14)

        # Calculate volume confirmation
        df['volume_ma'] = df.get('volume', pd.Series(1, index=df.index)).rolling(window=lookback).mean()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on mean reversion."""
        signals = pd.Series(0, index=data.index)

        entry_threshold = self.config.parameters['entry_threshold']
        exit_threshold = self.config.parameters['exit_threshold']
        lookback = self.config.parameters['lookback_period']

        # Skip initial rows where indicators are NaN
        start_idx = lookback

        for i in range(start_idx, len(data)):
            z_score = data.iloc[i]['z_score']
            rsi = data.iloc[i]['rsi']

            # Entry conditions for long (buy when oversold)
            oversold_z = z_score < -entry_threshold
            oversold_rsi = rsi < 30
            volume_confirm = (data.iloc[i].get('volume', 1) >
                            data.iloc[i]['volume_ma'] * 0.8)  # 80% of average volume

            # Entry conditions for short (sell when overbought)
            overbought_z = z_score > entry_threshold
            overbought_rsi = rsi > 70

            # Exit conditions (when price reverts to mean)
            near_mean = abs(z_score) < exit_threshold

            # Generate signals
            if oversold_z and oversold_rsi and volume_confirm:
                signals.iloc[i] = 1  # Buy signal (expect reversion up)
            elif overbought_z and overbought_rsi:
                signals.iloc[i] = -1  # Sell signal (expect reversion down)
            elif near_mean:
                signals.iloc[i] = 0  # Close position (reverted to mean)

        return signals

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi