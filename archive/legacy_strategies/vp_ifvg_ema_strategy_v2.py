"""
Volume Profile + IFVG + EMAs Strategy V2
VERSION MEJORADA con gestión de posiciones, stops/targets y risk management

Mejoras principales:
1. Gestión de posiciones (long/short/flat tracking)
2. Stop Loss / Take Profit dinámicos basados en ATR
3. Risk Management (Kelly, 2% rule)
4. Exit logic correcta
5. Trade scoring mejorado
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
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


class TradePosition:
    """Track current trade position"""
    def __init__(self, direction: int, entry_price: float, entry_idx: int, 
                 stop_loss: float, take_profit: float, size: float = 1.0):
        self.direction = direction  # 1=long, -1=short
        self.entry_price = entry_price
        self.entry_idx = entry_idx
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = size
        self.exit_price = None
        self.exit_idx = None
        self.exit_reason = None
        self.pnl = 0
        
    def update_exit(self, exit_price: float, exit_idx: int, reason: str):
        """Update exit information"""
        self.exit_price = exit_price
        self.exit_idx = exit_idx
        self.exit_reason = reason
        
        # Calculate PnL
        if self.direction == 1:  # Long
            self.pnl = (exit_price - self.entry_price) / self.entry_price
        else:  # Short
            self.pnl = (self.entry_price - exit_price) / self.entry_price


class VPIFVGEmaStrategyV2(BaseStrategy):
    """
    Version 2: Estrategia mejorada con gestión de posiciones completa
    
    Características:
    - Stops/Targets dinámicos basados en ATR
    - Risk management (2% capital por trade)
    - Position tracking correcto
    - Exit logic basada en múltiples condiciones
    - Scoring de trades mejorado
    """

    def __init__(self, name: str = "VP_IFVG_EMA_Strategy_V2"):
        super().__init__(name)

        # === RISK MANAGEMENT ===
        self.max_risk_per_trade = 0.02  # 2% del capital por trade
        self.max_daily_risk = 0.06  # 6% máximo pérdida diaria
        self.max_open_trades = 1  # Solo 1 posición a la vez
        
        # === STOP LOSS / TAKE PROFIT ===
        self.stop_loss_atr_mult = 2.0  # SL = 2x ATR
        self.take_profit_atr_mult = 4.0  # TP = 4x ATR (ratio 2:1)
        self.use_trailing_stop = True
        self.trailing_stop_activation = 1.5  # Activar trailing cuando profit > 1.5x ATR
        self.trailing_stop_distance = 1.0  # Trailing a 1x ATR
        
        # === IFVG Parameters ===
        self.disp_num = 5
        self.signal_pref = "Close"
        self.atr_multi = 0.25
        self.atr_period = 200

        # === Volume Profile Parameters ===
        self.vp_show = True
        self.vp_length = 360
        self.vp_rows = 100
        self.vp_va = 68
        
        # === EMA Parameters ===
        self.ema1_length = 20
        self.ema2_length = 50
        self.ema3_length = 100
        self.ema4_length = 200
        
        # === Signal filtering parameters ===
        self.use_volume_filter = True
        self.use_ema_filter = True
        self.use_vp_levels = True
        self.min_signal_strength = 2  # Mínimo 2 para entrar
        
        # === Internal state ===
        self.bull_fvgs: List[FVGData] = []
        self.bear_fvgs: List[FVGData] = []
        self.max_fvgs = 100
        
        # Position tracking
        self.current_position: Optional[TradePosition] = None
        self.closed_trades: List[TradePosition] = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
        # OPTIMIZACIÓN: Cache para FVG para evitar recálculos
        self.fvg_cache = {}
        
        # OPTIMIZACIÓN: Cache para VP pre-calculado
        self.current_vp_data = None
        
    def reset(self):
        """Reset strategy state for new backtest"""
        self.current_position = None
        self.capital = getattr(self, 'initial_capital', 10000)
        self.daily_pnl = 0
        self.trades_history = []
        self.open_trades = []
        self.closed_trades = []
        self.daily_risk_used = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.vp_va = 68

        # Reset internal state
        self.bull_fvgs = []
        self.bear_fvgs = []
        self.current_position = None
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
        # Limpiar caches para nueva ejecución
        self.fvg_cache = {}
        self.current_vp_data = None

    def get_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return {
            # Risk Management
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_daily_risk': self.max_daily_risk,
            'max_open_trades': self.max_open_trades,
            
            # Stop Loss / Take Profit
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'use_trailing_stop': self.use_trailing_stop,
            'trailing_stop_activation': self.trailing_stop_activation,
            'trailing_stop_distance': self.trailing_stop_distance,
            
            # IFVG Parameters
            'disp_num': self.disp_num,
            'signal_pref': self.signal_pref,
            'atr_multi': self.atr_multi,
            'atr_period': self.atr_period,
            
            # Volume Profile
            'vp_length': self.vp_length,
            'vp_rows': self.vp_rows,
            'vp_va': self.vp_va,
            
            # EMAs
            'ema1_length': self.ema1_length,
            'ema2_length': self.ema2_length,
            'ema3_length': self.ema3_length,
            'ema4_length': self.ema4_length,
            
            # Filters
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

    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range"""
        if period is None:
            period = self.atr_period
            
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        return pd.Series(atr, index=df.index)

    def _detect_ifvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect Implied Fair Value Gaps con cache para optimización"""
        # Crear clave de cache basada en el rango de datos
        cache_key = f"{len(df)}_{df.index[0]}_{df.index[-1]}"
        
        if cache_key in self.fvg_cache:
            return self.fvg_cache[cache_key]
        
        # Calcular ATR
        atr = self._calculate_atr(df) * self.atr_multi

        # Bullish FVG: low > high[2] and close[1] > high[2]
        bullish_condition = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
        bullish_size = abs(df['low'] - df['high'].shift(2))
        bullish_fvg = bullish_condition & (bullish_size > atr)

        # Bearish FVG: high < low[2] and close[1] < low[2]
        bearish_condition = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
        bearish_size = abs(df['low'].shift(2) - df['high'])
        bearish_fvg = bearish_condition & (bearish_size > atr)
        
        # Cachear resultado
        result = (bullish_fvg, bearish_fvg)
        self.fvg_cache[cache_key] = result
        
        return result

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
        price_step = price_range / self.vp_rows
        bins = np.arange(price_low, price_high + price_step, price_step)

        # Calculate volume at each price level
        volume_profile = np.zeros(len(bins) - 1)

        for i in range(len(df) - lookback, len(df)):
            bar_high = df['high'].iloc[i]
            bar_low = df['low'].iloc[i]
            bar_volume = df['volume'].iloc[i]

            # Distribute volume across price levels
            for j in range(len(bins) - 1):
                bin_low = bins[j]
                bin_high = bins[j + 1]

                # Check overlap
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)

                if overlap_high > overlap_low:
                    overlap_pct = (overlap_high - overlap_low) / (bar_high - bar_low + 1e-10)
                    volume_profile[j] += bar_volume * overlap_pct

        # Find POC (Point of Control)
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Calculate Value Area
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * (self.vp_va / 100)

        # Find VAH and VAL
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumsum = 0
        va_indices = []

        for idx in sorted_indices:
            cumsum += volume_profile[idx]
            va_indices.append(idx)
            if cumsum >= target_volume:
                break

        vah = bins[max(va_indices) + 1]
        val = bins[min(va_indices)]

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

        close_array = df['close'].values.astype(float)
        
        df['ema1'] = talib.EMA(close_array, timeperiod=self.ema1_length)
        df['ema2'] = talib.EMA(close_array, timeperiod=self.ema2_length)
        df['ema3'] = talib.EMA(close_array, timeperiod=self.ema3_length)
        df['ema4'] = talib.EMA(close_array, timeperiod=self.ema4_length)

        return df

    def _check_ema_trend(self, df: pd.DataFrame, idx: int) -> int:
        """
        Check EMA trend at given index
        Returns: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        if idx < 1:
            return 0

        ema1_curr = df.iloc[idx]['ema1']
        ema2_curr = df.iloc[idx]['ema2']

        if pd.isna(ema1_curr) or pd.isna(ema2_curr):
            return 0

        # Bullish trend: EMA1 above EMA2
        if ema1_curr > ema2_curr:
            return 1
        # Bearish trend: EMA1 below EMA2
        elif ema1_curr < ema2_curr:
            return -1

        return 0

    def _check_vp_levels(self, price: float, vp_data: Dict) -> int:
        """
        Check if price is near VP levels
        Returns: signal strength
        """
        poc = vp_data['poc']
        vah = vp_data['vah']
        val = vp_data['val']

        price_range = vah - val
        if price_range == 0:
            return 0

        threshold = price_range * 0.05  # 5% threshold (más razonable)

        # Near POC
        if abs(price - poc) < threshold:
            return 1

        # Near VAH or VAL
        if abs(price - vah) < threshold or abs(price - val) < threshold:
            return 1

        return 0

    def _check_volume_high(self, df: pd.DataFrame, idx: int, multiplier: float = 1.5) -> bool:
        """
        Check if current volume is significantly higher than average
        """
        if idx < 20:  # Need some history
            return False
        
        current_volume = df.iloc[idx]['volume']
        avg_volume = df.iloc[idx-20:idx]['volume'].mean()
        
        return current_volume > (avg_volume * multiplier)

    def _check_ema_cross(self, df: pd.DataFrame, idx: int, fast: int = 9, slow: int = 21, direction: str = 'bullish') -> bool:
        """
        Check for EMA cross in specified direction
        """
        if idx < 1:
            return False
        
        fast_col = f'ema{fast}'
        slow_col = f'ema{slow}'
        
        if fast_col not in df.columns or slow_col not in df.columns:
            return False
        
        fast_curr = df.iloc[idx][fast_col]
        slow_curr = df.iloc[idx][slow_col]
        fast_prev = df.iloc[idx-1][fast_col]
        slow_prev = df.iloc[idx-1][slow_col]
        
        if pd.isna(fast_curr) or pd.isna(slow_curr) or pd.isna(fast_prev) or pd.isna(slow_prev):
            return False
        
        # Bullish cross: fast crosses above slow
        if direction == 'bullish':
            return (fast_prev <= slow_prev) and (fast_curr > slow_curr)
        # Bearish cross: fast crosses below slow
        elif direction == 'bearish':
            return (fast_prev >= slow_prev) and (fast_curr < slow_curr)
        
        return False

    def _get_raw_signal(self, df: pd.DataFrame, vp_data: Dict, idx: int) -> Tuple[int, int]:
        """
        Get raw signal without position management
        
        Returns:
            (signal_direction, signal_strength)
            signal_direction: 1 (long), -1 (short), 0 (no signal)
            signal_strength: 1-5
        """
        if idx < 2:
            return 0, 0

        current_bar = df.iloc[idx]
        current_price = current_bar['close']

        # 1. Check IFVG inversions (strongest signal)
        fvg_signal = 0
        bullish_fvg, bearish_fvg = self._detect_ifvg(df.iloc[:idx+1])
        
        if bullish_fvg.iloc[idx]:
            fvg_signal = 3  # Strong bullish
        elif bearish_fvg.iloc[idx]:
            fvg_signal = -3  # Strong bearish

        # 2. Check for high volume + EMA cross pattern (from evaluation)
        pattern_bonus = 0
        if fvg_signal > 0:  # Bullish FVG
            has_high_vol = self._check_volume_high(df, idx, 1.5)
            has_ema_cross = self._check_ema_cross(df, idx, 9, 21, 'bullish')
            if has_high_vol and has_ema_cross:
                pattern_bonus = 2  # Boost signal for successful pattern

        # 3. Volume Profile level signals
        vp_signal = 0
        if self.use_vp_levels:
            vp_signal = self._check_vp_levels(current_price, vp_data)

        # 4. EMA trend filter
        ema_trend = 0
        if self.use_ema_filter:
            ema_trend = self._check_ema_trend(df, idx)

        # 5. Volume confirmation
        vol_confirm = 0
        if self.use_volume_filter and 'volume_signal' in df.columns:
            vol_confirm = df.iloc[idx]['volume_signal']

        # Combine signals
        final_signal_value = fvg_signal + pattern_bonus + (vp_signal * np.sign(fvg_signal)) + (ema_trend * 0.5)
        
        # Determine direction and strength
        if final_signal_value > 0:
            direction = 1  # Long
            strength = min(5, int(abs(final_signal_value)))
        elif final_signal_value < 0:
            direction = -1  # Short
            strength = min(5, int(abs(final_signal_value)))
        else:
            direction = 0
            strength = 0

        return direction, strength

    def _should_exit_position(self, df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Check if current position should be closed
        
        Returns:
            (should_exit, exit_reason)
        """
        if self.current_position is None:
            return False, ""

        current_price = df.iloc[idx]['close']
        position = self.current_position

        # 1. Check Stop Loss
        if position.direction == 1:  # Long
            if current_price <= position.stop_loss:
                return True, "stop_loss"
        else:  # Short
            if current_price >= position.stop_loss:
                return True, "stop_loss"

        # 2. Check Take Profit
        if position.direction == 1:  # Long
            if current_price >= position.take_profit:
                return True, "take_profit"
        else:  # Short
            if current_price <= position.take_profit:
                return True, "take_profit"

        # 3. Trailing Stop
        if self.use_trailing_stop:
            profit = abs(current_price - position.entry_price)
            atr = self._calculate_atr(df.iloc[:idx+1]).iloc[-1]
            
            if profit > atr * self.trailing_stop_activation:
                # Activate trailing stop
                if position.direction == 1:  # Long
                    trailing_sl = current_price - (atr * self.trailing_stop_distance)
                    if current_price <= trailing_sl:
                        return True, "trailing_stop"
                else:  # Short
                    trailing_sl = current_price + (atr * self.trailing_stop_distance)
                    if current_price >= trailing_sl:
                        return True, "trailing_stop"

        # 4. Opposite signal - usar VP pre-calculado si está disponible
        if self.current_vp_data is not None:
            vp_data = self.current_vp_data
        else:
            vp_data = self._calculate_volume_profile(df.iloc[:idx+1])
        
        signal_dir, signal_strength = self._get_raw_signal(df, vp_data, idx)
        
        if signal_dir != 0 and signal_dir != position.direction and signal_strength >= self.min_signal_strength:
            return True, "opposite_signal"

        # 5. Time-based exit (optional)
        bars_in_trade = idx - position.entry_idx
        if bars_in_trade > 100:  # Max 100 bars (adjust as needed)
            return True, "timeout"

        return False, ""

    def _can_open_position(self) -> bool:
        """Check if we can open a new position"""
        # Check if already in position
        if self.current_position is not None:
            return False

        # Check daily risk limit
        if abs(self.daily_pnl) >= self.max_daily_risk:
            return False

        # Check max trades per day
        if self.daily_trades >= 10:  # Max 10 trades per day
            return False

        return True

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals with proper position management
        
        OPTIMIZADO: Usa indicadores pre-calculados cuando estén disponibles
        """
        # Extract 5min data
        df = df_multi_tf['5min'].copy()
        
        if not self.validate_data(df):
            raise ValueError("DataFrame must contain OHLCV columns")

        # Reset internal state
        self.current_position = None
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.bull_fvgs = []
        self.bear_fvgs = []

        # OPTIMIZACIÓN: Usar indicadores pre-calculados si están disponibles
        if hasattr(df, 'ema5') and df['ema5'].notna().any():
            # Indicadores ya calculados, usarlos directamente
            pass  # Los indicadores ya están en el DataFrame
        else:
            # Calcular indicadores (fallback para compatibilidad)
            df = self._calculate_emas(df)
            atr = self._calculate_atr(df)
            df['atr'] = atr

        # Volume signals (simplified)
        if 'volume' in df.columns:
            vol_ma = talib.SMA(df['volume'].values.astype(float), timeperiod=21)
            df['volume_signal'] = ((df['volume'] > vol_ma * 1.5).astype(int) - 
                                  (df['volume'] < vol_ma * 0.5).astype(int))
        else:
            df['volume_signal'] = 0

        # Initialize output series
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        signals = pd.Series(0, index=df.index)
        trade_scores = pd.Series(0, index=df.index)

        # Process each bar
        for i in range(max(self.ema4_length, self.atr_period), len(df)):
            current_date = df.index[i].date() if hasattr(df.index[i], 'date') else None
            
            # Reset daily counters at start of new day
            if i > 0:
                prev_date = df.index[i-1].date() if hasattr(df.index[i-1], 'date') else None
                if current_date and prev_date and current_date != prev_date:
                    self.daily_pnl = 0.0
                    self.daily_trades = 0

            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['atr']

            # 1. Check exit conditions FIRST
            if self.current_position is not None:
                should_exit, exit_reason = self._should_exit_position(df, i)
                
                if should_exit:
                    # Close position
                    exits.iloc[i] = True
                    self.current_position.update_exit(current_price, i, exit_reason)
                    
                    # Update daily PnL
                    self.daily_pnl += self.current_position.pnl
                    
                    # Store closed trade
                    self.closed_trades.append(self.current_position)
                    
                    # Clear position
                    self.current_position = None
                    signals.iloc[i] = -1 if exit_reason != "opposite_signal" else 0
                    continue

            # 2. Check entry conditions
            if self._can_open_position():
                # Usar VP pre-calculado si está disponible, sino calcular
                if hasattr(self, 'current_vp_data') and self.current_vp_data is not None:
                    vp_data = self.current_vp_data
                else:
                    vp_data = self._calculate_volume_profile(df.iloc[:i+1])
                
                signal_dir, signal_strength = self._get_raw_signal(df, vp_data, i)

                if signal_strength >= self.min_signal_strength:
                    # Calculate stop loss and take profit
                    if signal_dir == 1:  # Long
                        stop_loss = current_price - (current_atr * self.stop_loss_atr_mult)
                        take_profit = current_price + (current_atr * self.take_profit_atr_mult)
                    else:  # Short
                        stop_loss = current_price + (current_atr * self.stop_loss_atr_mult)
                        take_profit = current_price - (current_atr * self.take_profit_atr_mult)

                    # Calculate position size based on risk
                    risk_amount = abs(current_price - stop_loss) / current_price
                    position_size = self.max_risk_per_trade / risk_amount if risk_amount > 0 else 1.0
                    position_size = min(position_size, 1.0)  # Cap at 100% capital

                    # Open position
                    self.current_position = TradePosition(
                        direction=signal_dir,
                        entry_price=current_price,
                        entry_idx=i,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        size=position_size
                    )

                    entries.iloc[i] = True
                    signals.iloc[i] = signal_dir
                    trade_scores.iloc[i] = signal_strength
                    self.daily_trades += 1

        return {
            'entries': entries,
            'exits': exits,
            'signals': signals,
            'trade_scores': trade_scores,
            'closed_trades': self.closed_trades
        }

    def get_strategy_info(self) -> Dict:
        """Get strategy information including closed trades"""
        return {
            'name': self.name,
            'total_trades': len(self.closed_trades),
            'winning_trades': sum(1 for t in self.closed_trades if t.pnl > 0),
            'losing_trades': sum(1 for t in self.closed_trades if t.pnl < 0),
            'total_pnl': sum(t.pnl for t in self.closed_trades),
            'avg_win': np.mean([t.pnl for t in self.closed_trades if t.pnl > 0]) if any(t.pnl > 0 for t in self.closed_trades) else 0,
            'avg_loss': np.mean([t.pnl for t in self.closed_trades if t.pnl < 0]) if any(t.pnl < 0 for t in self.closed_trades) else 0,
        }
