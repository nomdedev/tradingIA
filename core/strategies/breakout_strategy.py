"""
Breakout Trading Strategy Implementation
"""

from typing import List
import pandas as pd
from .base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    """
    Breakout trading strategy.

    This strategy identifies key support/resistance levels and trades breakouts
    above resistance or below support, with momentum confirmation.
    """

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for breakout strategy."""
        return ['lookback_period', 'breakout_threshold', 'consolidation_period']

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout indicators."""
        df = data.copy()

        lookback = self.config.parameters['lookback_period']
        consolidation = self.config.parameters['consolidation_period']

        # Calculate rolling high/low for breakout levels
        df['rolling_high'] = df['high'].rolling(window=lookback).max()
        df['rolling_low'] = df['low'].rolling(window=lookback).min()

        # Calculate consolidation (low volatility period)
        df['high_low_range'] = df['high'] - df['low']
        df['avg_range'] = df['high_low_range'].rolling(window=consolidation).mean()
        df['range_std'] = df['high_low_range'].rolling(window=consolidation).std()

        # Consolidation filter: price action within narrow range
        df['is_consolidating'] = df['range_std'] < (df['avg_range'] * 0.5)

        # Volume confirmation
        df['volume_ma'] = df.get('volume', pd.Series(1, index=df.index)).rolling(window=lookback).mean()
        df['volume_ratio'] = df.get('volume', 1) / df['volume_ma']

        # Momentum indicators
        df['momentum'] = df['close'].pct_change(periods=5)
        df['rsi'] = self._calculate_rsi(df['close'], 14)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on breakouts."""
        signals = pd.Series(0, index=data.index)

        breakout_threshold = self.config.parameters['breakout_threshold']
        lookback = self.config.parameters['lookback_period']

        # Skip initial rows where indicators are NaN
        start_idx = lookback * 2

        for i in range(start_idx, len(data)):
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            rolling_high = data.iloc[i]['rolling_high']
            rolling_low = data.iloc[i]['rolling_low']

            # Breakout conditions
            bullish_breakout = current_high > (rolling_high * (1 + breakout_threshold))
            bearish_breakout = current_low < (rolling_low * (1 - breakout_threshold))

            # Confirmation conditions
            consolidating = data.iloc[i]['is_consolidating']
            volume_spike = data.iloc[i]['volume_ratio'] > 1.5  # 50% above average volume
            momentum_confirm = data.iloc[i]['momentum'] > 0.01  # Positive momentum
            rsi_confirm = data.iloc[i]['rsi'] > 50  # Not oversold

            # Generate signals
            if bullish_breakout and consolidating and volume_spike and momentum_confirm and rsi_confirm:
                signals.iloc[i] = 1  # Buy signal on breakout above resistance
            elif bearish_breakout and consolidating and volume_spike:
                signals.iloc[i] = -1  # Sell signal on breakdown below support

        return signals

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi