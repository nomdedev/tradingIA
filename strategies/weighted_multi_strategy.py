#!/usr/bin/env python3
"""
Weighted Multi-Strategy: Squeeze Momentum + ADX + TTM + VP + IFVG + EMAs
Sistema de pesos para toma de decisiones con confirmación multi-timeframe

Esta estrategia combina múltiples indicadores técnicos con un sistema de pesos
para generar señales más robustas y confiables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from strategies.base_strategy import BaseStrategy


class WeightedMultiStrategy(BaseStrategy):
    """
    Estrategia multi-componente con sistema de pesos:

    Componentes principales:
    1. Squeeze Momentum + ADX + TTM (peso configurable)
    2. VP + IFVG + EMAs (peso configurable)
    3. Confirmación multi-timeframe (peso configurable)

    Sistema de scoring:
    - Cada componente aporta un score entre -1 y +1
    - Score total = suma ponderada de componentes
    - Señal cuando score total > threshold
    """

    def __init__(self, name: str = "Weighted_Multi_Strategy"):
        super().__init__(name)

        # === SISTEMA DE PESOS ===
        self.squeeze_weight = 0.35      # Peso para Squeeze + ADX + TTM
        self.vp_ifvg_weight = 0.35      # Peso para VP + IFVG + EMAs
        self.multitimeframe_weight = 0.20  # Peso para confirmación MT
        self.volume_weight = 0.10       # Peso para volumen

        # Threshold para generar señales
        self.signal_threshold = 0.3     # Score mínimo para señal (0-1) - REDUCIDO
        self.min_confirmation_score = 0.2  # Score mínimo para confirmación - REDUCIDO

        # === SQUEEZE MOMENTUM + ADX + TTM ===
        # Squeeze Parameters
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

        # === VP + IFVG + EMAs ===
        # IFVG Parameters
        self.disp_num = 5
        self.signal_pref = "Close"
        self.atr_multi = 0.25
        self.atr_period = 200

        # Volume Profile Parameters
        self.vp_length = 360
        self.vp_rows = 100
        self.vp_va = 68

        # EMA Parameters
        self.ema1_length = 20
        self.ema2_length = 50
        self.ema3_length = 100
        self.ema4_length = 200

        # === MULTI-TIMEFRAME CONFIRMATION ===
        self.higher_tf_weight = 0.3
        self.lower_tf_weight = 0.2
        self.use_multitimeframe = True

        # === RISK MANAGEMENT ===
        self.stop_loss_atr_mult = 1.0
        self.take_profit_atr_mult = 2.0
        self.risk_per_trade = 0.01  # 1% por trade
        self.max_risk_per_trade = 0.03

        # === FILTROS ADICIONALES ===
        self.use_volume_filter = True
        self.volume_threshold = 1.2
        self.min_adx_entry = 20
        self.max_adx_entry = 50
        self.min_squeeze_momentum = 0.3

        # Estado interno
        self.fvg_data = []
        self.current_poc = None
        self.current_vah = None
        self.current_val = None

    def get_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return {
            # Sistema de pesos
            'squeeze_weight': self.squeeze_weight,
            'vp_ifvg_weight': self.vp_ifvg_weight,
            'multitimeframe_weight': self.multitimeframe_weight,
            'volume_weight': self.volume_weight,
            'signal_threshold': self.signal_threshold,

            # Squeeze + ADX + TTM
            'bb_length': self.bb_length,
            'adx_length': self.adx_length,
            'wave_a_length': self.wave_a_length,

            # VP + IFVG + EMAs
            'vp_length': self.vp_length,
            'ema1_length': self.ema1_length,
            'atr_period': self.atr_period,

            # Risk management
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'risk_per_trade': self.risk_per_trade,
        }

    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _calculate_squeeze_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate squeeze momentum score (-1 to +1)"""
        # Bollinger Bands
        df['bb_basis'] = df['Close'].rolling(self.bb_length).mean()
        df['bb_std'] = df['Close'].rolling(self.bb_length).std()
        df['bb_upper'] = df['bb_basis'] + (self.bb_mult * df['bb_std'])
        df['bb_lower'] = df['bb_basis'] - (self.bb_mult * df['bb_std'])

        # Keltner Channels
        df['kc_basis'] = df['Close'].rolling(self.kc_length).mean()
        if self.use_true_range:
            df['tr'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
        else:
            df['tr'] = df['High'] - df['Low']

        df['tr_ma'] = df['tr'].rolling(self.bb_length).mean()
        df['kc_upper'] = df['kc_basis'] + (self.kc_mult * df['tr_ma'])
        df['kc_lower'] = df['kc_basis'] - (self.kc_mult * df['tr_ma'])

        # Squeeze conditions
        df['sqz_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        df['sqz_off'] = (df['bb_lower'] < df['kc_lower']) & (df['bb_upper'] > df['kc_upper'])

        # Linear Regression Momentum
        highest_high = df['High'].rolling(self.linear_momentum).max()
        lowest_low = df['Low'].rolling(self.linear_momentum).min()
        sma_close = df['Close'].rolling(self.linear_momentum).mean()

        df['avg_hl'] = (highest_high + lowest_low) / 2
        df['avg_hl_sma'] = (df['avg_hl'] + sma_close) / 2

        # Calculate linear regression (simplified)
        df['linreg_value'] = (df['Close'] - df['avg_hl_sma']) / df['avg_hl_sma']

        # Normalize momentum to -1 to +1 scale
        momentum_score = np.tanh(df['linreg_value'] * 10)  # Tanh for smooth scaling

        return momentum_score

    def _calculate_adx_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX trend strength score (0 to 1)"""
        df['adx'] = talib.ADX(np.asarray(df['High'], dtype=np.float64), np.asarray(df['Low'], dtype=np.float64), np.asarray(df['Close'], dtype=np.float64), timeperiod=self.adx_length)
        df['plus_di'] = talib.PLUS_DI(np.asarray(df['High'], dtype=np.float64), np.asarray(df['Low'], dtype=np.float64), np.asarray(df['Close'], dtype=np.float64), timeperiod=self.di_length)
        df['minus_di'] = talib.MINUS_DI(np.asarray(df['High'], dtype=np.float64), np.asarray(df['Low'], dtype=np.float64), np.asarray(df['Close'], dtype=np.float64), timeperiod=self.di_length)

        # ADX score: higher ADX = stronger trend
        adx_score = df['adx'] / 100.0  # Normalize to 0-1

        return adx_score

    def _calculate_ttm_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate TTM waves score (-1 to +1)"""
        # Wave A
        df['fast_ma_a'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.fast_ma_period)
        df['slow_ma_a'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.wave_a_length)
        df['macd_a'] = df['fast_ma_a'] - df['slow_ma_a']
        df['signal_a'] = talib.EMA(np.asarray(df['macd_a'], dtype=np.float64), timeperiod=self.wave_a_length)
        df['hist_a'] = df['macd_a'] - df['signal_a']

        # Wave B
        df['fast_ma_b'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.fast_ma_period)
        df['slow_ma_b'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.wave_b_length)
        df['macd_b'] = df['fast_ma_b'] - df['slow_ma_b']
        df['signal_b'] = talib.EMA(np.asarray(df['macd_b'], dtype=np.float64), timeperiod=self.wave_a_length)
        df['hist_b'] = df['macd_b'] - df['signal_b']

        # Combined TTM score
        ttm_score = np.sign(df['hist_a']) * 0.6 + np.sign(df['hist_b']) * 0.4
        ttm_score = ttm_score.fillna(0)

        return ttm_score

    def _calculate_vp_ifvg_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VP + IFVG + EMAs score (-1 to +1)"""
        # EMAs for trend
        df['ema20'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.ema1_length)
        df['ema50'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.ema2_length)
        df['ema100'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.ema3_length)
        df['ema200'] = talib.EMA(np.asarray(df['Close'], dtype=np.float64), timeperiod=self.ema4_length)

        # EMA trend score
        ema_trend = np.where(df['ema20'] > df['ema50'], 1, -1)
        ema_trend = np.where(df['ema50'] > df['ema100'], ema_trend * 1.2, ema_trend * 0.8)

        # Volume Profile (simplified POC calculation)
        lookback_period = min(self.vp_length, len(df))
        price_volume = {}

        for i in range(max(0, len(df) - lookback_period), len(df)):
            price = round(df.iloc[i]['Close'], 2)
            volume = df.iloc[i]['Volume']

            if price in price_volume:
                price_volume[price] += volume
            else:
                price_volume[price] = volume

        if price_volume:
            poc_price = max(price_volume, key=price_volume.get)
            self.current_poc = poc_price

            # Distance to POC score
            poc_distance = abs(df['Close'] - poc_price) / poc_price
            poc_score = 1.0 - (poc_distance * 10).clip(0, 1.0)  # Closer to POC = higher score
        else:
            poc_score = 0.5

        # IFVG calculation (simplified)
        ifvg_score = self._calculate_ifvg_signals(df)

        # Combined VP + IFVG + EMA score
        vp_ifvg_score = pd.Series(
            np.sign(ema_trend) * 0.4 +
            poc_score * 0.3 +
            ifvg_score * 0.3,
            index=df.index
        )

        return vp_ifvg_score

    def _calculate_ifvg_signals(self, df: pd.DataFrame) -> pd.Series:
        """Calculate IFVG signals (simplified version)"""
        ifvg_signals = pd.Series(0.0, index=df.index)

        # Simple gap detection (bearish gaps)
        for i in range(2, len(df)):
            # Bearish FVG: high[2] < low[0] (gap down)
            if df.iloc[i-2]['High'] < df.iloc[i]['Low']:
                gap_size = df.iloc[i]['Low'] - df.iloc[i-2]['High']
                if gap_size > df.iloc[i]['Close'] * 0.001:  # Minimum gap size
                    ifvg_signals.iloc[i] = -0.8  # Bearish signal

            # Bullish FVG: low[2] > high[0] (gap up)
            elif df.iloc[i-2]['Low'] > df.iloc[i]['High']:
                gap_size = df.iloc[i-2]['Low'] - df.iloc[i]['High']
                if gap_size > df.iloc[i]['Close'] * 0.001:
                    ifvg_signals.iloc[i] = 0.8  # Bullish signal

        return ifvg_signals

    def _calculate_multitimeframe_score(self, df: pd.DataFrame, df_multi_tf: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculate multi-timeframe confirmation score"""
        mtf_score = pd.Series(0.5, index=df.index)  # Neutral score by default

        if not self.use_multitimeframe or not df_multi_tf:
            return mtf_score

        # Check higher timeframes
        for tf, tf_df in df_multi_tf.items():
            if tf in ['15Min', '1H'] and not tf_df.empty:
                # Simple trend confirmation
                tf_df['ema20'] = talib.EMA(np.asarray(tf_df['Close'], dtype=np.float64), timeperiod=20)
                tf_df['ema50'] = talib.EMA(np.asarray(tf_df['Close'], dtype=np.float64), timeperiod=50)
                tf_df['trend'] = np.where(tf_df['ema20'] > tf_df['ema50'], 1, -1)

                # Resample to 5Min for alignment
                trend_resampled = tf_df['trend'].resample('5Min').ffill()
                aligned_trend = trend_resampled.reindex(df.index, method='ffill').fillna(0)

                # Add to score
                mtf_score += aligned_trend * 0.3

        return mtf_score.clip(-1, 1)

    def _calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume confirmation score"""
        if not self.use_volume_filter:
            return pd.Series(0.5, index=df.index)

        df['volume_ma'] = df['Volume'].rolling(20).mean()
        volume_ratio = df['Volume'] / df['volume_ma']

        # Volume score: higher volume = higher score
        volume_score = pd.Series(
            np.where(volume_ratio > self.volume_threshold, 0.8,
                    np.where(volume_ratio > 1.0, 0.5, 0.2)),
            index=df.index
        )

        return volume_score

    def _combine_scores(self, squeeze_score: pd.Series, adx_score: pd.Series,
                       ttm_score: pd.Series, vp_ifvg_score: pd.Series,
                       mtf_score: pd.Series, volume_score: pd.Series) -> pd.Series:
        """Combine all component scores into final signal score"""

        # Squeeze + ADX + TTM combined score
        squeeze_adx_ttm_score = (squeeze_score * 0.5 + adx_score * 0.3 + ttm_score * 0.2)

        # Final weighted score
        final_score = (
            squeeze_adx_ttm_score * self.squeeze_weight +
            vp_ifvg_score * self.vp_ifvg_weight +
            mtf_score * self.multitimeframe_weight +
            volume_score * self.volume_weight
        )

        return final_score

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate weighted multi-strategy signals

        Returns:
            Dict with 'entries', 'exits', 'signals', 'scores'
        """
        if not df_multi_tf or '5Min' not in df_multi_tf:
            return {'entries': pd.Series(), 'exits': pd.Series(), 'signals': pd.Series(), 'scores': pd.Series()}

        df = df_multi_tf['5Min'].copy()

        # Calculate all component scores
        squeeze_score = self._calculate_squeeze_score(df)
        adx_score = self._calculate_adx_score(df)
        ttm_score = self._calculate_ttm_score(df)
        vp_ifvg_score = self._calculate_vp_ifvg_score(df)
        mtf_score = self._calculate_multitimeframe_score(df, df_multi_tf)
        volume_score = self._calculate_volume_score(df)

        # Combine into final score
        final_score = self._combine_scores(
            squeeze_score, adx_score, ttm_score,
            vp_ifvg_score, mtf_score, volume_score
        )

        # Generate DISCRETE signals (not per candle)
        signals = pd.Series(0, index=df.index)

        # Find entry points: score crosses above threshold (from below)
        entry_condition = (
            (final_score > self.signal_threshold) &
            (final_score.shift(1) <= self.signal_threshold)
        )
        signals.loc[entry_condition] = 1

        # Find exit points: score crosses below negative threshold (from above)
        exit_condition = (
            (final_score < -self.signal_threshold) &
            (final_score.shift(1) >= -self.signal_threshold)
        )
        signals.loc[exit_condition] = -1

        # Create entries and exits series
        entries = (signals == 1).astype(int)
        exits = (signals == -1).astype(int)

        print(f"  📊 Total candles: {len(df)}")
        print(f"  📈 Entry signals: {entries.sum()}")
        print(f"  📉 Exit signals: {exits.sum()}")
        print(f"  🎯 Final score range: {final_score.min():.3f} to {final_score.max():.3f}")

        return {
            'entries': entries,
            'exits': exits,
            'signals': signals,
            'scores': final_score,
            'components': {
                'squeeze': squeeze_score,
                'adx': adx_score,
                'ttm': ttm_score,
                'vp_ifvg': vp_ifvg_score,
                'mtf': mtf_score,
                'volume': volume_score
            }
        }