"""
Squeeze Momentum + ADX + TTM Strategy
Converted from Pine Script indicator to Python strategy for TradingIA platform

This strategy implements:
- Squeeze Momentum Oscillator
- Average Directional Index (ADX)
- Trade The Market Waves (A, B, C)
- Multi-timeframe analysis for enhanced decision making

Based on TradingLatino's strategy combined with John F. Carter's approach
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import talib
from strategies.base_strategy import BaseStrategy


class SqueezeMomentumADXTTMStrategy(BaseStrategy):
    """
    Advanced strategy combining multiple technical indicators:
    - Squeeze Momentum Oscillator for volatility compression
    - ADX for trend strength
    - TTM Waves for market structure
    - Multi-timeframe analysis for better entry timing
    """

    def __init__(self, name: str = "Squeeze_ADX_TTM_Strategy"):
        super().__init__(name)

        # Squeeze Momentum Parameters
        self.bb_length = 20
        self.bb_mult = 2.0
        self.kc_length = 20
        self.kc_mult = 1.5
        self.linear_momentum = 20
        self.use_true_range = True

        # ADX Parameters
        self.adx_length = 14
        self.di_length = 14
        self.key_level = 23

        # TTM Waves Parameters
        self.wave_a_length = 55
        self.wave_b_length = 144
        self.wave_c_length = 233
        self.fast_ma_period = 8

        # Multi-timeframe Parameters
        self.higher_tf_weight = 0.3  # Weight for higher timeframe confirmation
        self.lower_tf_weight = 0.2   # Weight for lower timeframe confirmation

        # Signal Parameters
        self.squeeze_threshold = 0.5  # Minimum squeeze compression for signals
        self.adx_threshold = 20       # Minimum ADX for trend signals
        self.momentum_threshold = 0.1 # Minimum momentum for signals

        # Risk Management con Ratio 2:1
        self.stop_loss_atr_mult = 1.0      # Stop loss = 1 ATR
        self.take_profit_atr_mult = 2.0    # Take profit = 2 ATR (ratio 2:1)
        self.trailing_stop = False         # Trailing stop loss
        self.trailing_activation = 0.5     # Activate trailing at 50% of target
        self.risk_per_trade = 0.02         # 2% risk per trade
        self.max_risk_per_trade = 0.05     # Max 5% risk per trade

        # Entry Filters
        self.min_adx_entry = 20            # Minimum ADX for entry
        self.max_adx_entry = 50            # Maximum ADX (avoid overbought)
        self.min_squeeze_momentum = 0.3    # Minimum squeeze momentum
        self.use_volume_filter = True      # Filter by volume
        self.volume_threshold = 1.2        # Volume must be 20% above average

        # POC Parameters (Point of Control)
        self.use_poc_filter = True
        self.poc_lookback = 500  # Bars to calculate POC
        self.poc_threshold = 0.0  # Price must be above/below POC by this percentage

        # ADX Advanced Parameters
        self.use_adx_slope = True
        self.use_adx_divergence = True
        self.adx_slope_threshold = 0.1  # Minimum ADX slope for signals
        self.adx_divergence_weight = 0.2  # Weight for divergence confirmation

        # Multi-timeframe control
        self.use_multitimeframe = True

        # Runtime variables for POC and ADX analysis
        self.current_poc = None
        self.current_adx_slope = None
        self.current_adx_divergence = None

    def get_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return {
            # Squeeze Parameters
            'bb_length': self.bb_length,
            'bb_mult': self.bb_mult,
            'kc_length': self.kc_length,
            'kc_mult': self.kc_mult,
            'linear_momentum': self.linear_momentum,
            'use_true_range': self.use_true_range,

            # ADX Parameters
            'adx_length': self.adx_length,
            'di_length': self.di_length,
            'key_level': self.key_level,

            # TTM Parameters
            'wave_a_length': self.wave_a_length,
            'wave_b_length': self.wave_b_length,
            'wave_c_length': self.wave_c_length,
            'fast_ma_period': self.fast_ma_period,

            # Multi-timeframe
            'higher_tf_weight': self.higher_tf_weight,
            'lower_tf_weight': self.lower_tf_weight,

            # Signal thresholds
            'squeeze_threshold': self.squeeze_threshold,
            'adx_threshold': self.adx_threshold,
            'momentum_threshold': self.momentum_threshold,

            # Risk management con Ratio 2:1
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'trailing_stop': self.trailing_stop,
            'trailing_activation': self.trailing_activation,
            'risk_per_trade': self.risk_per_trade,
            'max_risk_per_trade': self.max_risk_per_trade,

            # Entry Filters
            'min_adx_entry': self.min_adx_entry,
            'max_adx_entry': self.max_adx_entry,
            'min_squeeze_momentum': self.min_squeeze_momentum,
            'use_volume_filter': self.use_volume_filter,
            'volume_threshold': self.volume_threshold,

            # POC Parameters
            'use_poc_filter': self.use_poc_filter,
            'poc_lookback': self.poc_lookback,
            'poc_threshold': self.poc_threshold,

            # ADX Advanced Parameters
            'use_adx_slope': self.use_adx_slope,
            'use_adx_divergence': self.use_adx_divergence,
            'adx_slope_threshold': self.adx_slope_threshold,
            'adx_divergence_weight': self.adx_divergence_weight,

            # Multi-timeframe control
            'use_multitimeframe': self.use_multitimeframe
        }

    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _calculate_squeeze_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Squeeze Momentum Oscillator"""
        # Bollinger Bands
        df['bb_basis'] = df['close'].rolling(self.bb_length).mean()
        df['bb_std'] = df['close'].rolling(self.bb_length).std()
        df['bb_upper'] = df['bb_basis'] + (self.bb_mult * df['bb_std'])
        df['bb_lower'] = df['bb_basis'] - (self.bb_mult * df['bb_std'])

        # Keltner Channels
        df['kc_basis'] = df['close'].rolling(self.kc_length).mean()
        if self.use_true_range:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
        else:
            df['tr'] = df['high'] - df['low']

        df['tr_ma'] = df['tr'].rolling(self.bb_length).mean()
        df['kc_upper'] = df['kc_basis'] + (self.kc_mult * df['tr_ma'])
        df['kc_lower'] = df['kc_basis'] - (self.kc_mult * df['tr_ma'])

        # Squeeze conditions
        df['sqz_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        df['sqz_off'] = (df['bb_lower'] < df['kc_lower']) & (df['bb_upper'] > df['kc_upper'])
        df['no_sqz'] = ~(df['sqz_on'] | df['sqz_off'])

        # Linear Regression Momentum
        highest_high = df['high'].rolling(self.linear_momentum).max()
        lowest_low = df['low'].rolling(self.linear_momentum).min()
        sma_close = df['close'].rolling(self.linear_momentum).mean()

        df['avg_hl'] = (highest_high + lowest_low) / 2
        df['avg_hl_sma'] = (df['avg_hl'] + sma_close) / 2

        # Calculate linear regression
        def linreg(series, period):
            return talib.LINEARREG(np.asarray(series), timeperiod=period)

        df['linreg_value'] = linreg(df['close'] - df['avg_hl_sma'], self.linear_momentum)

        return df

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index"""
        # Calculate True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Calculate Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']

        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'], 0
        )
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'], 0
        )

        # Calculate Directional Indicators
        df['tr_rma'] = talib.EMA(np.asarray(df['tr']), timeperiod=self.di_length)
        df['plus_di'] = 100 * talib.EMA(np.asarray(df['plus_dm']), timeperiod=self.di_length) / df['tr_rma']
        df['minus_di'] = 100 * talib.EMA(np.asarray(df['minus_dm']), timeperiod=self.di_length) / df['tr_rma']

        # Calculate ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = talib.EMA(np.asarray(df['dx']), timeperiod=self.adx_length)

        return df

    def _calculate_ttm_waves(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Trade The Market Waves A, B, C"""
        # Wave A
        df['fast_ma_a'] = talib.EMA(np.asarray(df['close']), timeperiod=self.fast_ma_period)
        df['slow_ma_a'] = talib.EMA(np.asarray(df['close']), timeperiod=self.wave_a_length)
        df['macd_a'] = df['fast_ma_a'] - df['slow_ma_a']
        df['signal_a'] = talib.EMA(np.asarray(df['macd_a']), timeperiod=self.wave_a_length)
        df['hist_a'] = df['macd_a'] - df['signal_a']

        # Wave B
        df['fast_ma_b'] = talib.EMA(np.asarray(df['close']), timeperiod=self.fast_ma_period)
        df['slow_ma_b'] = talib.EMA(np.asarray(df['close']), timeperiod=self.wave_b_length)
        df['macd_b'] = df['fast_ma_b'] - df['slow_ma_b']
        df['signal_b'] = talib.EMA(np.asarray(df['macd_b']), timeperiod=self.wave_a_length)  # Note: uses wave_a_length for signal
        df['hist_b'] = df['macd_b'] - df['signal_b']

        # Wave C
        df['fast_ma_c'] = talib.EMA(np.asarray(df['close']), timeperiod=self.fast_ma_period)
        df['slow_ma_c'] = talib.EMA(np.asarray(df['close']), timeperiod=self.wave_c_length)
        df['macd_c'] = df['fast_ma_c'] - df['slow_ma_c']
        df['signal_c'] = talib.EMA(np.asarray(df['macd_c']), timeperiod=self.wave_c_length)
        df['hist_c'] = df['macd_c'] - df['signal_c']

        return df

    def _calculate_multi_tf_confirmation(self, df: pd.DataFrame, df_multi_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate multi-timeframe confirmation signals"""
        # df already contains the calculated indicators

        if not df.empty and len(df_multi_tf) > 1:
            # Get higher timeframe data (15Min, 1H)
            higher_tf_signals = {}

            for tf, tf_df in df_multi_tf.items():
                if tf in ['15Min', '1H'] and not tf_df.empty:
                    # Calculate trend direction for higher timeframes
                    tf_df['ema_20'] = talib.EMA(np.asarray(tf_df['close']), timeperiod=20)
                    tf_df['ema_50'] = talib.EMA(np.asarray(tf_df['close']), timeperiod=50)
                    tf_df['trend'] = np.where(tf_df['ema_20'] > tf_df['ema_50'], 1,
                                            np.where(tf_df['ema_20'] < tf_df['ema_50'], -1, 0))

                    # Resample to 5Min timeframe for alignment
                    trend_resampled = tf_df['trend'].resample('5Min').ffill()
                    higher_tf_signals[tf] = trend_resampled.reindex(df.index, method='ffill').fillna(0)

            # Combine higher timeframe signals
            if higher_tf_signals:
                df['higher_tf_trend'] = sum(higher_tf_signals.values()) / len(higher_tf_signals)
                df['higher_tf_confirmed'] = abs(df['higher_tf_trend']) >= 0.5

        return df

    def _generate_base_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate base signals from Squeeze + ADX + TTM"""
        signals = pd.Series(0.0, index=df.index, name='signal', dtype=float)

        # Squeeze Momentum signals
        squeeze_signal = np.where(
            (df['linreg_value'] > self.momentum_threshold) & df['sqz_off'],
            1,  # Buy signal
            np.where(
                (df['linreg_value'] < -self.momentum_threshold) & df['sqz_off'],
                -1,  # Sell signal
                0
            )
        )

        # ADX trend strength filter
        adx_strong = df['adx'] > self.adx_threshold
        adx_trend_up = (df['plus_di'] > df['minus_di']) & adx_strong
        adx_trend_down = (df['minus_di'] > df['plus_di']) & adx_strong

        # TTM Wave confirmation
        wave_a_bullish = df['hist_a'] > 0
        wave_a_bearish = df['hist_a'] < 0
        wave_b_bullish = df['hist_b'] > 0
        wave_b_bearish = df['hist_b'] < 0

        # Combine signals
        buy_conditions = (
            (squeeze_signal == 1) &
            (adx_trend_up | (df['adx'] < self.key_level)) &  # Allow trades in weak trends
            (wave_a_bullish | wave_b_bullish)
        )

        sell_conditions = (
            (squeeze_signal == -1) &
            (adx_trend_down | (df['adx'] < self.key_level)) &
            (wave_a_bearish | wave_b_bearish)
        )

        signals.loc[buy_conditions] = 1
        signals.loc[sell_conditions] = -1

        return signals

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals using Squeeze Momentum + ADX + TTM

        Args:
            df_multi_tf: Dictionary with timeframe keys ('5Min', '15Min', '1H') and OHLCV DataFrames

        Returns:
            Dict with 'entries', 'exits', 'signals' Series
        """
        if not df_multi_tf or '5Min' not in df_multi_tf:
            return {'entries': pd.Series(), 'exits': pd.Series(), 'signals': pd.Series()}

        df = df_multi_tf['5Min'].copy()

        # Calculate all indicators
        df = self._calculate_squeeze_momentum(df)
        df = self._calculate_adx(df)
        df = self._calculate_ttm_waves(df)
        df = self._calculate_multi_tf_confirmation(df, df_multi_tf)

        # Generate base signals
        signals = self._generate_base_signals(df)

        # Apply POC filter if enabled
        if self.use_poc_filter and self.current_poc is not None:
            signals = self._apply_poc_filter(signals, df)

        # Apply ADX slope/divergence filters if enabled
        if self.use_adx_slope and self.current_adx_slope is not None:
            signals = self._apply_adx_slope_filter(signals, df)

        if self.use_adx_divergence and self.current_adx_divergence is not None:
            signals = self._apply_adx_divergence_filter(signals, df)

        # Apply multi-timeframe confirmation if available
        if 'higher_tf_confirmed' in df.columns:
            # Reduce signal strength if higher timeframe doesn't confirm
            signals.loc[~df['higher_tf_confirmed']] = (signals * 0.5).loc[~df['higher_tf_confirmed']]

        # Create entries and exits
        entries = (signals == 1).astype(int)
        exits = (signals == -1).astype(int)

        return {
            'entries': entries,
            'exits': exits,
            'signals': signals,
            'trade_scores': signals.abs()  # Signal strength as trade score
        }

    def calculate_signal_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate signal strength based on multiple factors"""
        if 'signal' not in df.columns:
            df['signal_strength'] = 0
            return df

        strength = pd.Series(0, index=df.index)

        # Base strength from signal magnitude
        strength += abs(df['signal']) * 2

        # ADX contribution (trend strength)
        if 'adx' in df.columns:
            adx_score = np.clip(df['adx'] / 50, 0, 1)  # Normalize to 0-1
            strength += adx_score

        # Squeeze momentum contribution
        if 'linreg_value' in df.columns:
            momentum_score = np.clip(abs(df['linreg_value']) / 2, 0, 1)
            strength += momentum_score

        # TTM wave confirmation
        if 'hist_a' in df.columns and 'hist_b' in df.columns:
            wave_score = np.where(
                (df['hist_a'] > 0) | (df['hist_b'] > 0), 0.5, 0
            )
            strength += wave_score

        # Multi-timeframe confirmation
        if 'higher_tf_confirmed' in df.columns:
            mtf_score = df['higher_tf_confirmed'].astype(int) * 0.5
            strength += mtf_score

        # Scale to 0-5 range
        df['signal_strength'] = np.clip(strength, 0, 5)

        return df

    def _apply_poc_filter(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Apply Point of Control filter to signals"""
        if self.current_poc is None:
            return signals

        filtered_signals = signals.copy()

        for idx in signals.index:
            signal = signals.loc[idx]
            if signal == 0:
                continue

            current_price = df.loc[idx, 'close']
            poc = self.current_poc

            # For long signals, price must be above POC
            if signal == 1 and current_price <= poc * (1 + self.poc_threshold):
                filtered_signals.loc[idx] = 0
            # For short signals, price must be below POC
            elif signal == -1 and current_price >= poc * (1 - self.poc_threshold):
                filtered_signals.loc[idx] = 0

        return filtered_signals

    def _apply_adx_slope_filter(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Apply ADX slope filter to signals"""
        if self.current_adx_slope is None:
            return signals

        filtered_signals = signals.copy()

        for idx in signals.index:
            signal = signals.loc[idx]
            if signal == 0:
                continue

            adx_slope = self.current_adx_slope

            # For long signals, prefer positive ADX slope (strengthening trend)
            if signal == 1 and adx_slope < self.adx_slope_threshold:
                filtered_signals.loc[idx] *= 0.7  # Reduce signal strength
            # For short signals, prefer positive ADX slope
            elif signal == -1 and adx_slope < self.adx_slope_threshold:
                filtered_signals.loc[idx] *= 0.7

        return filtered_signals

    def _apply_adx_divergence_filter(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Apply ADX divergence filter to signals"""
        if self.current_adx_divergence is None:
            return signals

        filtered_signals = signals.copy()

        for idx in signals.index:
            signal = signals.loc[idx]
            if signal == 0:
                continue

            divergence = self.current_adx_divergence

            # Bullish divergence (1) strengthens long signals
            if signal == 1 and divergence == 1:
                filtered_signals.loc[idx] *= (1 + self.adx_divergence_weight)
            # Bearish divergence (-1) strengthens short signals
            elif signal == -1 and divergence == -1:
                filtered_signals.loc[idx] *= (1 + self.adx_divergence_weight)
            # Opposite divergence weakens signals
            elif signal == 1 and divergence == -1:
                filtered_signals.loc[idx] *= (1 - self.adx_divergence_weight)
            elif signal == -1 and divergence == 1:
                filtered_signals.loc[idx] *= (1 - self.adx_divergence_weight)

        return filtered_signals