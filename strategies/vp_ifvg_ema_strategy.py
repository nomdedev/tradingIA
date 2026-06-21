"""
Volume Profile + IFVG + EMAs Combined Strategy
Converted from Pine Script indicator to Python strategy for TradingIA platform

This strategy replicates the exact behavior of the Pine Script indicator:
- IFVG: Detects Implied Fair Value Gaps and their inversions
- Volume Profile: POC, VAH, VAL calculations
- EMAs: Trend filtering with exponential moving averages
- Signals: Triangle markers above/below bars like TradingView
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import talib
from strategies.base_strategy import BaseStrategy


class FVGData:
    """Data structure for Fair Value Gap tracking"""
    def __init__(self, left_idx: int, top_price: float, right_idx: int, bottom_price: float,
                 mid_price: float, direction: int, state: int = 0):
        self.left_idx = left_idx
        self.top_price = top_price
        self.right_idx = right_idx
        self.bottom_price = bottom_price
        self.mid_price = mid_price
        self.direction = direction  # 1 for bullish, -1 for bearish
        self.state = state  # 0: active, 1: inverted, 2: completed
        self.x_val = right_idx


class VPIFVGEmaStrategy(BaseStrategy):
    """
    Combined strategy implementing:
    - Volume Profile (VP) with POC, VAH, VAL
    - Implied Fair Value Gaps (IFVG) with inversion tracking
    - Exponential Moving Averages (EMAs) for trend filtering
    - Triangle signals above/below bars like TradingView
    """

    def __init__(self, name: str = "VP_IFVG_EMA_Strategy"):
        super().__init__(name)

        # IFVG Parameters (all configurable)
        self.disp_num = 5  # Show Last FVGs
        self.signal_pref = "Close"  # Signal Preference: "Close" or "Wick"
        self.atr_multi = 0.25  # ATR Multiplier

        # Volume Profile Parameters
        self.vp_show = True
        self.sp_show = True
        self.sd_show = True
        self.sd_threshold = 15
        self.pc_show = "Developing POC"
        self.vah_show = True
        self.val_show = True
        self.vp_va = 68  # Value Area %
        self.vp_polarity = "Bar Polarity"
        self.vp_lookback = "Fixed Range"
        self.vp_length = 360
        self.vp_rows = 100
        self.vp_width = 31
        self.vp_offset = 13
        self.vp_placement = "Right"

        # Volume Histogram Parameters
        self.vh_show = True
        self.vma_show = True
        self.vma_length = 21
        self.vh_placement = "Top"
        self.vh_height = 8
        self.vh_offset = 1

        # Volume Weighted Colored Bars
        self.vwcb_show = False
        self.vwcb_upper = 1.618
        self.vwcb_lower = 0.618

        # EMA Parameters
        self.ema1_length = 20
        self.ema2_length = 50
        self.ema3_length = 100
        self.ema4_length = 200

        # Signal filtering parameters
        self.use_volume_filter = True  # Use volume confirmation
        self.use_ema_filter = True     # Use EMA trend filter
        self.use_vp_levels = True      # Use VP levels for signals
        self.min_signal_strength = 1   # Minimum signal strength (1-5)

        # Internal state for FVG tracking
        self.bull_fvgs: List[FVGData] = []
        self.bear_fvgs: List[FVGData] = []
        self.max_fvgs = 100  # Maximum FVGs to track

    def get_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return {
            # IFVG Parameters
            'disp_num': self.disp_num,
            'signal_pref': self.signal_pref,
            'atr_multi': self.atr_multi,

            # Volume Profile Parameters
            'vp_show': self.vp_show,
            'sp_show': self.sp_show,
            'sd_show': self.sd_show,
            'sd_threshold': self.sd_threshold,
            'pc_show': self.pc_show,
            'vah_show': self.vah_show,
            'val_show': self.val_show,
            'vp_va': self.vp_va,
            'vp_polarity': self.vp_polarity,
            'vp_lookback': self.vp_lookback,
            'vp_length': self.vp_length,
            'vp_rows': self.vp_rows,
            'vp_width': self.vp_width,
            'vp_offset': self.vp_offset,
            'vp_placement': self.vp_placement,

            # Volume Histogram Parameters
            'vh_show': self.vh_show,
            'vma_show': self.vma_show,
            'vma_length': self.vma_length,
            'vh_placement': self.vh_placement,
            'vh_height': self.vh_height,
            'vh_offset': self.vh_offset,

            # Volume Weighted Parameters
            'vwcb_show': self.vwcb_show,
            'vwcb_upper': self.vwcb_upper,
            'vwcb_lower': self.vwcb_lower,

            # EMA Parameters
            'ema1_length': self.ema1_length,
            'ema2_length': self.ema2_length,
            'ema3_length': self.ema3_length,
            'ema4_length': self.ema4_length,

            # Signal filtering parameters
            'use_volume_filter': self.use_volume_filter,
            'use_ema_filter': self.use_ema_filter,
            'use_vp_levels': self.use_vp_levels,
            'min_signal_strength': self.min_signal_strength,
        }

    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 200) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr * self.atr_multi

    def _detect_ifvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect Implied Fair Value Gaps"""
        atr = self._calculate_atr(df)

        # Bullish FVG: low > high[2] and close[1] > high[2]
        bullish_condition = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
        bullish_size = abs(df['low'] - df['high'].shift(2))
        bullish_fvg = bullish_condition & (bullish_size > atr)

        # Bearish FVG: high < low[2] and close[1] < low[2]
        bearish_condition = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
        bearish_size = abs(df['low'].shift(2) - df['high'])
        bearish_fvg = bearish_condition & (bearish_size > atr)

        return bullish_fvg, bearish_fvg

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate Volume Profile components"""
        lookback = min(self.vp_length, len(df))

        # Get price range
        price_high = df['high'].tail(lookback).max()
        price_low = df['low'].tail(lookback).min()
        price_range = price_high - price_low

        if price_range == 0:
            return {
                'poc': df['close'].iloc[-1],
                'vah': df['close'].iloc[-1],
                'val': df['close'].iloc[-1],
                'profile_high': price_high,
                'profile_low': price_low
            }

        # Create price bins
        bins = np.linspace(price_low, price_high, self.vp_rows + 1)

        # Calculate volume per bin
        volume_profile = np.zeros(self.vp_rows)

        for i in range(lookback):
            idx = len(df) - lookback + i
            row_high = df['high'].iloc[idx]
            row_low = df['low'].iloc[idx]
            volume = df['volume'].iloc[idx]

            # Find bins that overlap with this candle
            for bin_idx in range(self.vp_rows):
                bin_low = bins[bin_idx]
                bin_high = bins[bin_idx + 1]

                # Calculate overlap
                overlap_low = max(bin_low, row_low)
                overlap_high = min(bin_high, row_high)

                if overlap_high > overlap_low:
                    overlap_ratio = (overlap_high - overlap_low) / (row_high - row_low)
                    volume_profile[bin_idx] += volume * overlap_ratio

        # Find POC (Point of Control)
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Calculate Value Area
        total_volume = volume_profile.sum()
        va_volume = total_volume * (self.vp_va / 100)

        # Find VAH and VAL
        cumsum = np.cumsum(volume_profile)
        vah_idx = np.where(cumsum >= va_volume)[0]
        vah_idx = vah_idx[0] if len(vah_idx) > 0 else len(volume_profile) - 1
        vah = (bins[vah_idx] + bins[vah_idx + 1]) / 2

        val_idx = np.where(cumsum >= (total_volume - va_volume))[0]
        val_idx = val_idx[0] if len(val_idx) > 0 else 0
        val = (bins[val_idx] + bins[val_idx + 1]) / 2

        return {
            'poc': poc,
            'vah': vah,
            'val': val,
            'profile_high': price_high,
            'profile_low': price_low,
            'volume_profile': volume_profile
        }

    def _calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        df = df.copy()

        df['ema1'] = talib.EMA(df['close'].values.astype(float), timeperiod=self.ema1_length)
        df['ema2'] = talib.EMA(df['close'].values.astype(float), timeperiod=self.ema2_length)
        df['ema3'] = talib.EMA(df['close'].values.astype(float), timeperiod=self.ema3_length)
        df['ema4'] = talib.EMA(df['close'].values.astype(float), timeperiod=self.ema4_length)

        return df

    def _calculate_volume_signals(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume-based signals"""
        if not self.vma_show:
            return pd.Series(0, index=df.index)

        volume_ma = talib.SMA(df['volume'].values.astype(float), timeperiod=self.vma_length)

        # High volume signal
        high_volume = (df['volume'] > volume_ma * self.vwcb_upper).astype(int)

        # Low volume signal (opposite)
        low_volume = (df['volume'] < volume_ma * self.vwcb_lower).astype(int)

        return high_volume - low_volume

    def _detect_fvgs(self, df: pd.DataFrame, idx: int) -> None:
        """Detect new Fair Value Gaps at the given index"""
        if idx < 2:  # Need at least 2 previous bars
            return

        # Calculate ATR for filtering
        atr_period = 200
        if idx >= atr_period:
            atr_values = []
            for i in range(max(0, idx - atr_period + 1), idx + 1):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                close_prev = df.iloc[i-1]['close'] if i > 0 else df.iloc[i]['open']
                tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
                atr_values.append(tr)
            atr = np.mean(atr_values) * self.atr_multi
        else:
            atr = 0

        current = df.iloc[idx]
        prev2 = df.iloc[idx-2]

        # Bullish FVG: low > high[2] and close[1] > high[2]
        if (current['low'] > prev2['high'] and
            df.iloc[idx-1]['close'] > prev2['high'] and
            abs(current['low'] - prev2['high']) > atr):

            fvg = FVGData(
                left_idx=idx-1,
                top_price=current['low'],
                right_idx=idx,
                bottom_price=prev2['high'],
                mid_price=(current['low'] + prev2['high']) / 2,
                direction=1  # Bullish
            )
            self.bull_fvgs.append(fvg)

        # Bearish FVG: high < low[2] and close[1] < low[2]
        elif (current['high'] < prev2['low'] and
              df.iloc[idx-1]['close'] < prev2['low'] and
              abs(prev2['low'] - current['high']) > atr):

            fvg = FVGData(
                left_idx=idx-1,
                top_price=prev2['low'],
                right_idx=idx,
                bottom_price=current['high'],
                mid_price=(prev2['low'] + current['high']) / 2,
                direction=-1  # Bearish
            )
            self.bear_fvgs.append(fvg)

        # Limit FVG storage
        if len(self.bull_fvgs) > self.max_fvgs:
            self.bull_fvgs.pop(0)
        if len(self.bear_fvgs) > self.max_fvgs:
            self.bear_fvgs.pop(0)

    def _check_fvg_inversions(self, df: pd.DataFrame, idx: int) -> int:
        """
        Check for FVG inversions and return signal strength

        Returns:
            3: Strong bullish signal (bearish FVG inverted)
            -3: Strong bearish signal (bullish FVG inverted)
            0: No signal
        """
        if idx < 1:
            return 0

        current = df.iloc[idx]
        wick_high = current['high'] if self.signal_pref == "Wick" else current['close']
        wick_low = current['low'] if self.signal_pref == "Wick" else current['close']

        signal = 0

        # Check bullish FVGs (bearish FVG getting inverted = bullish signal)
        for fvg in self.bear_fvgs[-self.disp_num:]:
            if fvg.state == 0:  # Active FVG
                # Check if price breaks above the FVG
                if wick_high > fvg.top_price:
                    fvg.state = 1  # Inverted
                    signal = max(signal, 3)  # Strong bullish signal

        # Check bearish FVGs (bullish FVG getting inverted = bearish signal)
        for fvg in self.bull_fvgs[-self.disp_num:]:
            if fvg.state == 0:  # Active FVG
                # Check if price breaks below the FVG
                if wick_low < fvg.bottom_price:
                    fvg.state = 1  # Inverted
                    signal = min(signal, -3)  # Strong bearish signal

        return signal

    def _check_vp_levels(self, price: float, vp_data: Dict) -> int:
        """Check Volume Profile levels for signals"""
        signal = 0

        # Buy near VAL (Value Area Low)
        if abs(price - vp_data['val']) / vp_data['val'] < 0.001:  # Within 0.1%
            signal += 1

        # Sell near VAH (Value Area High)
        elif abs(price - vp_data['vah']) / vp_data['vah'] < 0.001:  # Within 0.1%
            signal -= 1

        return signal

    def _check_ema_trend(self, df: pd.DataFrame, idx: int) -> int:
        """Check EMA trend for filtering"""
        if idx < 1 or 'ema1' not in df.columns or 'ema2' not in df.columns:
            return 0

        ema1_prev = df.iloc[idx-1]['ema1']
        ema2_prev = df.iloc[idx-1]['ema2']
        ema1_curr = df.iloc[idx]['ema1']
        ema2_curr = df.iloc[idx]['ema2']

        # Bullish trend: EMA1 above EMA2
        if ema1_curr > ema2_curr:
            return 1
        # Bearish trend: EMA1 below EMA2
        elif ema1_curr < ema2_curr:
            return -1

        return 0

    def _combine_signals(self, fvg_signal: int, vp_signal: int,
                        ema_trend: int, vol_confirm: int) -> int:
        """
        Combine all signals into final signal strength

        Signal hierarchy:
        1. FVG inversions (strongest, ±3)
        2. VP levels (±1)
        3. EMA trend (±1)
        4. Volume confirmation (±1)
        """
        final_signal = 0

        # FVG signals are strongest
        if fvg_signal != 0:
            final_signal += fvg_signal

        # VP levels add confirmation
        if vp_signal != 0:
            final_signal += vp_signal

        # EMA trend filtering
        if ema_trend != 0:
            # Only add trend signal if it aligns with existing signal
            if (final_signal > 0 and ema_trend > 0) or (final_signal < 0 and ema_trend < 0):
                final_signal += ema_trend
            elif final_signal == 0:
                final_signal += ema_trend * 0.5  # Weaker when no other signals

        # Volume confirmation
        if vol_confirm != 0:
            final_signal += vol_confirm

        return int(final_signal)

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals based on combined indicators

        This replicates the exact behavior of the Pine Script indicator:
        - IFVG signals: Triangles above/below bars when FVGs get inverted
        - Volume Profile levels: Support/resistance at VAH/VAL/POC
        - EMA crosses: Trend filtering
        - Volume confirmation: High/low volume signals

        Args:
            df_multi_tf: Dictionary with timeframe keys and OHLCV DataFrames

        Returns:
            Dict with 'entries', 'exits', 'signals' Series
        """
        # Extract 5min data for signal generation
        df = df_multi_tf['5min'].copy()
        
        if not self.validate_data(df):
            raise ValueError("DataFrame must contain OHLCV columns")

        df['signal'] = 0
        df['signal_strength'] = 0

        # Reset FVG tracking for new data
        self.bull_fvgs = []
        self.bear_fvgs = []

        # Calculate all indicators
        df = self._calculate_emas(df)
        volume_signals = self._calculate_volume_signals(df)
        vp_data = self._calculate_volume_profile(df)

        # Process each bar to detect FVGs and generate signals
        for i in range(len(df)):
            current_bar = df.iloc[i]

            # Detect new FVGs
            self._detect_fvgs(df, i)

            # Check for FVG inversions (main signal source)
            fvg_signal = self._check_fvg_inversions(df, i)

            # Volume Profile level signals
            vp_signal = 0
            if self.use_vp_levels and i == len(df) - 1:  # Only for current bar
                vp_signal = self._check_vp_levels(current_bar['close'], vp_data)

            # EMA trend filter
            ema_trend = 0
            if self.use_ema_filter and i > 0:
                ema_trend = self._check_ema_trend(df, i)

            # Volume confirmation
            vol_confirm = 0
            if self.use_volume_filter:
                vol_confirm = volume_signals.iloc[i]

            # Combine all signals
            final_signal = self._combine_signals(fvg_signal, vp_signal, ema_trend, vol_confirm)

            # Apply minimum signal strength filter
            if abs(final_signal) >= self.min_signal_strength:
                df.loc[df.index[i], 'signal'] = 1 if final_signal > 0 else -1 if final_signal < 0 else 0
                df.loc[df.index[i], 'signal_strength'] = min(5, abs(final_signal))

        # Convert signals to entries/exits format expected by backtester
        entries = (df['signal'] == 1).astype(bool)
        exits = (df['signal'] == -1).astype(bool)
        
        return {
            'entries': entries,
            'exits': exits,
            'signals': df['signal']
        }