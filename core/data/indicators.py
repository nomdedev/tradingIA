"""
Multi-Timeframe Indicators for BTC IFVG Strategy

Implements:
- IFVG enhanced with ATR filter and mitigation tracking
- Volume Profile with 120 bins and POC/VAH/VAL calculation
- EMAs multi-TF with optimizable lengths
- Combined signal generation with confidence scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from enum import Enum
import talib

def calculate_ifvg_enhanced(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate enhanced IFVG (Inside/Outside Fair Value Gaps) with ATR filter and mitigation tracking.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters dictionary

    Returns:
        Tuple of (bull_signals, bear_signals, confidence_scores)
    """

    # Parameters
    atr_period = params.get('atr_period', 14)
    atr_multiplier = params.get('atr_multi', 0.2)
    mitigation_lookback = params.get('mitigation_lookback', 5)
    min_gap_size = params.get('min_gap_size', 0.001)  # 0.1%

    # Calculate ATR for filtering
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)

    # Find gaps
    gaps = []
    for i in range(2, len(df)):
        prev_high = df['high'].iloc[i-2]
        prev_low = df['low'].iloc[i-2]
        curr_high = df['high'].iloc[i-1]
        curr_low = df['low'].iloc[i-1]

        # Bullish IFVG: gap between prev high and curr low, filled by next high
        if curr_low > prev_high:
            gap_size = (curr_low - prev_high) / prev_high
            if gap_size >= min_gap_size:
                gaps.append({
                    'index': i-1,
                    'type': 'bullish',
                    'gap_start': prev_high,
                    'gap_end': curr_low,
                    'gap_size': gap_size,
                    'atr_filter': atr.iloc[i-1] * atr_multiplier
                })

        # Bearish IFVG: gap between curr high and prev low, filled by next low
        elif curr_high < prev_low:
            gap_size = (prev_low - curr_high) / prev_low
            if gap_size >= min_gap_size:
                gaps.append({
                    'index': i-1,
                    'type': 'bearish',
                    'gap_start': curr_high,
                    'gap_end': prev_low,
                    'gap_size': gap_size,
                    'atr_filter': atr.iloc[i-1] * atr_multiplier
                })

    # Convert to signals
    bull_signals = pd.Series(False, index=df.index)
    bear_signals = pd.Series(False, index=df.index)
    confidence = pd.Series(0.0, index=df.index)

    for gap in gaps:
        idx = df.index[gap['index']]

        # Check if gap is still valid (not mitigated)
        gap_filled = False
        for j in range(gap['index'] + 1, min(gap['index'] + mitigation_lookback + 1, len(df))):
            if gap['type'] == 'bullish':
                if df['high'].iloc[j] >= gap['gap_end']:
                    gap_filled = True
                    break
            else:  # bearish
                if df['low'].iloc[j] <= gap['gap_start']:
                    gap_filled = True
                    break

        if not gap_filled:
            # Gap strength based on size and ATR filter
            strength = min(gap['gap_size'] / (atr.iloc[gap['index']] * atr_multiplier), 1.0)
            confidence.loc[idx] = strength

            if gap['type'] == 'bullish':
                bull_signals.loc[idx] = True
            else:
                bear_signals.loc[idx] = True

    return bull_signals, bear_signals, confidence

def volume_profile_advanced(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate advanced Volume Profile with POC, VAH, VAL.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters dictionary

    Returns:
        Tuple of (poc_series, vah_series, val_series)
    """

    bins = params.get('vp_rows', 120)
    value_area_percent = params.get('va_percent', 0.7)

    # Calculate price bins
    price_min = df['low'].min()
    price_max = df['high'].max()
    bin_size = (price_max - price_min) / bins

    poc_series = pd.Series(np.nan, index=df.index)
    vah_series = pd.Series(np.nan, index=df.index)
    val_series = pd.Series(np.nan, index=df.index)

    # Rolling volume profile (last 50 bars for responsiveness)
    window = min(50, len(df))

    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i+1]

        # Create volume profile
        volume_profile = {}

        for j in range(len(window_df)):
            price_range = window_df['high'].iloc[j] - window_df['low'].iloc[j]
            volume = window_df['volume'].iloc[j]

            if price_range > 0:
                # Distribute volume across price bins
                bins_in_bar = max(1, int(price_range / bin_size))
                volume_per_bin = volume / bins_in_bar

                for k in range(bins_in_bar):
                    bin_price = window_df['low'].iloc[j] + (k * price_range / bins_in_bar)
                    bin_key = round(bin_price / bin_size) * bin_size

                    if bin_key not in volume_profile:
                        volume_profile[bin_key] = 0
                    volume_profile[bin_key] += volume_per_bin

        if volume_profile:
            # Find POC (Point of Control)
            poc_price = max(volume_profile, key=volume_profile.get)
            poc_series.iloc[i] = poc_price

            # Calculate Value Area (70% of total volume)
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * value_area_percent

            # Sort by volume descending
            sorted_prices = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)

            cumulative_volume = 0

            for price, volume in sorted_prices:
                cumulative_volume += volume
                if cumulative_volume >= target_volume:
                    break

            # VAH and VAL are the highest and lowest prices in value area
            value_area_prices = [p for p, v in sorted_prices[:len(sorted_prices)] if cumulative_volume <= target_volume]
            if value_area_prices:
                vah_series.iloc[i] = max(value_area_prices)
                val_series.iloc[i] = min(value_area_prices)

    return poc_series, vah_series, val_series

def emas_multi_tf(df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                 df_1h: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate EMAs across multiple timeframes with optimizable lengths.

    Args:
        df_5m, df_15m, df_1h: DataFrames for each timeframe
        params: Parameters dictionary

    Returns:
        DataFrame with EMA signals aligned to 5m timeframe
    """

    # EMA lengths (optimizable)
    ema_fast_5m = params.get('ema_fast_5m', 9)
    ema_slow_5m = params.get('ema_slow_5m', 21)
    ema_fast_15m = params.get('ema_fast_15m', 13)
    ema_slow_15m = params.get('ema_slow_15m', 34)
    ema_fast_1h = params.get('ema_fast_1h', 21)
    ema_slow_1h = params.get('ema_slow_1h', 55)

    # Calculate EMAs
    ema_fast_5m_series = talib.EMA(df_5m['close'].to_numpy(dtype=np.float64), timeperiod=ema_fast_5m)
    ema_slow_5m_series = talib.EMA(df_5m['close'].to_numpy(dtype=np.float64), timeperiod=ema_slow_5m)

    ema_fast_15m_series = talib.EMA(df_15m['close'].to_numpy(dtype=np.float64), timeperiod=ema_fast_15m)
    ema_slow_15m_series = talib.EMA(df_15m['close'].to_numpy(dtype=np.float64), timeperiod=ema_slow_15m)

    ema_fast_1h_series = talib.EMA(df_1h['close'].to_numpy(dtype=np.float64), timeperiod=ema_fast_1h)
    ema_slow_1h_series = talib.EMA(df_1h['close'].to_numpy(dtype=np.float64), timeperiod=ema_slow_1h)

    # Align to 5m timeframe
    result = pd.DataFrame(index=df_5m.index)

    # 5m EMAs
    result['ema_fast_5m'] = ema_fast_5m_series
    result['ema_slow_5m'] = ema_slow_5m_series
    result['ema_trend_5m'] = (ema_fast_5m_series > ema_slow_5m_series).astype(int)

    # 15m EMAs (forward fill to align with 5m)
    ema_trend_15m = pd.Series((ema_fast_15m_series > ema_slow_15m_series).astype(int), index=df_15m.index)
    result['ema_trend_15m'] = ema_trend_15m.reindex(df_5m.index, method='ffill')

    # 1h EMAs (forward fill to align with 5m)
    ema_trend_1h = pd.Series((ema_fast_1h_series > ema_slow_1h_series).astype(int), index=df_1h.index)
    result['ema_trend_1h'] = ema_trend_1h.reindex(df_5m.index, method='ffill')

    # Combined trend strength
    result['ema_strength'] = result[['ema_trend_5m', 'ema_trend_15m', 'ema_trend_1h']].sum(axis=1) / 3.0

    return result

def generate_filtered_signals(df_5m: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Generate combined signals from all indicators with multi-TF filtering.

    Args:
        df_5m: 5-minute DataFrame with OHLCV and cross-TF data
        params: Strategy parameters

    Returns:
        Tuple of (bull_signals, bear_signals, confidence_scores)
    """

    # Calculate individual indicators
    ifvg_bull, ifvg_bear, ifvg_conf = calculate_ifvg_enhanced(df_5m, params)
    vp_poc, vp_vah, vp_val = volume_profile_advanced(df_5m, params)

    # EMAs require multi-TF data - assume it's already in df_5m
    # (would need to be calculated separately and merged)

    # Volume cross signal
    vol_thresh = params.get('vol_thresh', 1.2)
    volume_ma = talib.SMA(df_5m['volume'].to_numpy(dtype=np.float64), timeperiod=20)
    vol_cross = df_5m['volume'] > volume_ma * vol_thresh

    # Momentum filter (15m)
    momentum_15m = df_5m.get('momentum_15m', pd.Series(True, index=df_5m.index))

    # Trend filter (1h) - HTF ALWAYS filters longs
    uptrend_1h = df_5m.get('uptrend_1h', pd.Series(True, index=df_5m.index))

    # Combine signals with confidence weighting
    bull_signals = pd.Series(False, index=df_5m.index)
    bear_signals = pd.Series(False, index=df_5m.index)
    confidence = pd.Series(0.0, index=df_5m.index)

    for i in range(len(df_5m)):
        # Base signals
        bull_base = ifvg_bull.iloc[i]
        bear_base = ifvg_bear.iloc[i]

        # Multi-TF filters
        vol_filter = vol_cross.iloc[i] if not pd.isna(vol_cross.iloc[i]) else True
        momentum_filter = momentum_15m.iloc[i] if not pd.isna(momentum_15m.iloc[i]) else True
        trend_filter = uptrend_1h.iloc[i] if not pd.isna(uptrend_1h.iloc[i]) else True

        # Volume profile filter (price near POC/VAH/VAL)
        current_price = df_5m['close'].iloc[i]
        vp_filter = False

        if not pd.isna(vp_poc.iloc[i]):
            poc_distance = abs(current_price - vp_poc.iloc[i]) / current_price
            vah_distance = abs(current_price - vp_vah.iloc[i]) / current_price if not pd.isna(vp_vah.iloc[i]) else 1.0
            val_distance = abs(current_price - vp_val.iloc[i]) / current_price if not pd.isna(vp_val.iloc[i]) else 1.0

            # Signal if price is near high volume areas
            vp_filter = min(poc_distance, vah_distance, val_distance) < 0.005  # Within 0.5%

        # EMA filter (if available)
        ema_trend = df_5m.get('ema_strength', pd.Series(0.5, index=df_5m.index)).iloc[i]
        ema_filter = ema_trend > 0.5  # Majority of EMAs trending up

        # Apply filters
        if bull_base:
            # Long signals need: volume + momentum + trend + VP + EMA
            filters_passed = vol_filter and momentum_filter and trend_filter and vp_filter and ema_filter
            if filters_passed:
                bull_signals.iloc[i] = True
                confidence.iloc[i] = ifvg_conf.iloc[i] * (1 + ema_trend) * 0.5  # Boost confidence

        elif bear_base:
            # Short signals need: volume + momentum + VP (trend can be down)
            filters_passed = vol_filter and momentum_filter and vp_filter
            if filters_passed:
                bear_signals.iloc[i] = True
                confidence.iloc[i] = ifvg_conf.iloc[i] * (2 - ema_trend) * 0.5  # Boost when EMAs down

    # Normalize confidence to 0-1 range
    if confidence.max() > 0:
        confidence = confidence / confidence.max()

    return bull_signals, bear_signals, confidence

def validate_params(params: Dict[str, Any]) -> bool:
    """
    Validate strategy parameters for correlations and constraints.

    Args:
        params: Parameters dictionary

    Returns:
        True if parameters are valid
    """

    # ATR multiplier should be reasonable
    if not (0.1 <= params.get('atr_multi', 0.2) <= 0.5):
        return False

    # VA percent should be between 60-80%
    if not (0.6 <= params.get('va_percent', 0.7) <= 0.8):
        return False

    # VP rows should be reasonable
    if not (80 <= params.get('vp_rows', 120) <= 200):
        return False

    # Volume threshold reasonable
    if not (0.8 <= params.get('vol_thresh', 1.2) <= 2.0):
        return False

    # TP RR should be reasonable
    if not (1.5 <= params.get('tp_rr', 2.2) <= 3.0):
        return False

    # Confidence threshold reasonable
    if not (0.4 <= params.get('min_confidence', 0.6) <= 0.8):
        return False

    # EMA lengths should be in reasonable ranges and fast < slow
    ema_lengths = [
        ('ema_fast_5m', 'ema_slow_5m', 5, 50),
        ('ema_fast_15m', 'ema_slow_15m', 8, 80),
        ('ema_fast_1h', 'ema_slow_1h', 13, 130)
    ]

    for fast_key, slow_key, min_len, max_len in ema_lengths:
        fast_len = params.get(fast_key, 9)
        slow_len = params.get(slow_key, 21)

        if not (min_len <= fast_len < slow_len <= max_len):
            return False

    return True


# ===== ALIASES FOR BACKWARD COMPATIBILITY =====
# Wrappers para compatibilidad con tests que no pasan parámetros

def calculate_ifvg(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Wrapper para calculate_ifvg_enhanced con parámetros por defecto.
    Devuelve un DataFrame con todas las columnas calculadas.
    """
    # Crear copia y normalizar nombres de columnas
    df_work = df.copy()
    
    # Normalizar a minúsculas para compatibilidad con funciones internas
    column_mapping = {}
    for col in df_work.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close', 'volume']:
            column_mapping[col] = col_lower
    
    if column_mapping:
        df_work = df_work.rename(columns=column_mapping)
    
    # Parámetros por defecto
    default_params = {
        'atr_period': 14,
        'atr_multi': kwargs.get('atr_multi', 0.2),
        'mitigation_lookback': 5,
        'min_gap_size': 0.001,
        'wick_based': kwargs.get('wick_based', False)
    }
    
    # Actualizar con kwargs adicionales
    default_params.update(kwargs)
    
    # Calcular IFVG
    bull_signals, bear_signals, confidence = calculate_ifvg_enhanced(df_work, default_params)
    
    # Crear DataFrame resultado con todas las columnas esperadas
    result = df.copy()  # Usar DataFrame original para mantener nombres de columnas
    result['bull_signal'] = bull_signals
    result['bear_signal'] = bear_signals
    result['confidence'] = confidence
    
    # Calcular columnas auxiliares para compatibilidad con tests
    # Usar columnas normalizadas para cálculos
    result['ATR'] = talib.ATR(df_work['high'], df_work['low'], df_work['close'], 
                              timeperiod=default_params['atr_period'])
    
    # Calcular gaps - usar nombres originales de columnas
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    
    result['gap_up'] = False
    result['gap_down'] = False
    result['fvg_size'] = 0.0
    result['valid_fvg'] = False
    
    for i in range(2, len(result)):
        prev_high = df[high_col].iloc[i-2]
        prev_low = df[low_col].iloc[i-2]
        curr_high = df[high_col].iloc[i-1]
        curr_low = df[low_col].iloc[i-1]
        
        # Bullish gap
        if curr_low > prev_high:
            result.iloc[i, result.columns.get_loc('gap_up')] = True
            result.iloc[i, result.columns.get_loc('fvg_size')] = (curr_low - prev_high) / prev_high
            result.iloc[i, result.columns.get_loc('valid_fvg')] = \
                result.iloc[i, result.columns.get_loc('fvg_size')] >= default_params['min_gap_size']
        
        # Bearish gap
        elif curr_high < prev_low:
            result.iloc[i, result.columns.get_loc('gap_down')] = True
            result.iloc[i, result.columns.get_loc('fvg_size')] = (prev_low - curr_high) / prev_low
            result.iloc[i, result.columns.get_loc('valid_fvg')] = \
                result.iloc[i, result.columns.get_loc('fvg_size')] >= default_params['min_gap_size']
    
    return result


def volume_profile(df: pd.DataFrame, rows: int = 120, va_percent: float = 0.7) -> pd.DataFrame:
    """
    Wrapper para volume_profile_advanced con parámetros por defecto.
    Devuelve un DataFrame con las columnas calculadas.
    """
    # Normalizar nombres de columnas
    df_work = df.copy()
    column_mapping = {}
    for col in df_work.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close', 'volume']:
            column_mapping[col] = col_lower
    
    if column_mapping:
        df_work = df_work.rename(columns=column_mapping)
    
    params = {
        'vp_rows': rows,
        'va_percent': va_percent,
        'vp_lookback': 200
    }
    
    poc_series, vah_series, val_series = volume_profile_advanced(df_work, params)
    
    result = df.copy()
    result['vp_poc'] = poc_series
    result['vp_vah'] = vah_series
    result['vp_val'] = val_series
    
    # Aliases para compatibilidad con tests (nombres sin prefijo)
    result['POC'] = result['vp_poc']
    result['VAH'] = result['vp_vah']
    result['VAL'] = result['vp_val']
    result['SD_Thresh'] = result['vp_vah'] - result['vp_val']  # Simple threshold
    
    return result


def emas_simple(df: pd.DataFrame, timeframes: list = None, **kwargs) -> dict:
    """
    Wrapper simplificado para calcular EMAs con un solo DataFrame.
    Para compatibilidad con tests que no usan multi-timeframe.
    
    Args:
        df: DataFrame con datos OHLCV
        timeframes: Lista de timeframes (solo usa el primero por ahora)
        **kwargs: Parámetros adicionales
        
    Returns:
        Dict con timeframes como keys y DataFrames con EMAs como values
    """
    # Parámetros por defecto
    default_params = {
        'ema_fast_5m': 20,
        'ema_slow_5m': 50,
        'ema_fast_15m': 13,
        'ema_slow_15m': 34,
        'ema_fast_1h': 9,
        'ema_slow_1h': 21
    }
    default_params.update(kwargs)
    
    result_df = df.copy()
    
    # Determinar nombre de columna Close
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    # Calcular EMAs básicas
    result_df['EMA_20'] = talib.EMA(df[close_col], timeperiod=default_params['ema_fast_5m'])
    result_df['EMA_50'] = talib.EMA(df[close_col], timeperiod=default_params['ema_slow_5m'])
    result_df['EMA_100'] = talib.EMA(df[close_col], timeperiod=100)
    result_df['EMA_200'] = talib.EMA(df[close_col], timeperiod=200)
    
    # Retornar dict con timeframe base
    return {'5Min': result_df}


def generate_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Wrapper para generate_filtered_signals con parámetros por defecto.
    Genera señales long/short con niveles de confianza.
    
    Args:
        df: DataFrame con datos OHLCV e indicadores
        **kwargs: Parámetros adicionales (use_ema_filter, use_vp_filter, etc)
        
    Returns:
        DataFrame con columnas de señales añadidas
    """
    # Normalizar nombres de columnas
    df_work = df.copy()
    column_mapping = {}
    for col in df_work.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close', 'volume']:
            column_mapping[col] = col_lower
    
    if column_mapping:
        df_work = df_work.rename(columns=column_mapping)
    
    # Parámetros por defecto
    default_params = {
        'atr_period': 14,
        'atr_multi': 0.2,
        'vp_rows': 120,
        'va_percent': 0.7,
        'vol_thresh': 1.2,
        'min_confidence': 0.6,
        'ema_fast_5m': 20,
        'ema_slow_5m': 50
    }
    
    # Aplicar filtros opcionales desde kwargs
    if 'use_ema_filter' in kwargs:
        default_params['use_ema_filter'] = kwargs['use_ema_filter']
    if 'use_vp_filter' in kwargs:
        default_params['use_vp_filter'] = kwargs['use_vp_filter']
    
    default_params.update({k: v for k, v in kwargs.items() 
                          if k not in ['use_ema_filter', 'use_vp_filter']})
    
    # Generar señales usando el DataFrame normalizado
    bull_signals, bear_signals, confidence = generate_filtered_signals(df_work, default_params)
    
    result = df.copy()
    result['signal'] = 0
    result.loc[bull_signals, 'signal'] = 1
    result.loc[bear_signals, 'signal'] = -1
    result['confidence'] = confidence
    result['bull_signal'] = bull_signals
    result['bear_signal'] = bear_signals
    
    return result


def calculate_all_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Calcula todos los indicadores principales en un DataFrame.
    Función helper para testing y uso rápido.
    
    Args:
        df: DataFrame con datos OHLCV
        **kwargs: Parámetros opcionales para los indicadores
        
    Returns:
        DataFrame con todos los indicadores calculados
    """
    result = df.copy()
    
    # Asegurar que las columnas tengan nombres estándar
    if 'open' in result.columns:
        result.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
    
    # 1. Calcular IFVG
    ifvg_df = calculate_ifvg(result, **kwargs)
    for col in ['gap_up', 'gap_down', 'fvg_size', 'ATR', 'valid_fvg', 'bull_signal', 'bear_signal']:
        if col in ifvg_df.columns:
            result[col] = ifvg_df[col]
    
    # 2. Calcular Volume Profile
    vp_df = volume_profile(result,
                          rows=kwargs.get('vp_rows', 120),
                          va_percent=kwargs.get('va_percent', 0.7))
    for col in ['vp_poc', 'vp_vah', 'vp_val']:
        if col in vp_df.columns:
            result[col] = vp_df[col]
    
    # Aliases para compatibilidad
    if 'vp_poc' in result.columns:
        result['POC'] = result['vp_poc']
        result['VAH'] = result['vp_vah']
        result['VAL'] = result['vp_val']
    
    # 3. Calcular EMAs simples en el timeframe actual
    # Note: emas_multi_tf requiere múltiples timeframes, aquí solo calculamos EMAs básicas
    ema_periods = kwargs.get('ema_periods', [20, 50, 100, 200])
    for period in ema_periods:
        result[f'EMA_{period}'] = result['Close'].ewm(span=period, adjust=False).mean()
    
    # 4. Indicadores técnicos adicionales
    result['RSI'] = talib.RSI(result['Close'], timeperiod=14)
    result['MACD'], result['MACD_signal'], result['MACD_hist'] = talib.MACD(
        result['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # 5. Bollinger Bands
    result['BB_upper'], result['BB_middle'], result['BB_lower'] = talib.BBANDS(
        result['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    
    # 6. Generar señales consolidadas basadas en todos los indicadores
    result['signal'] = 0  # 0: neutral, 1: buy, -1: sell
    result['confidence'] = 0.0
    
    # Señal alcista si: bull_signal Y close > EMA_20 Y RSI < 70
    bullish = (result.get('bull_signal', False) == True) & \
              (result['Close'] > result['EMA_20']) & \
              (result['RSI'] < 70)
    result.loc[bullish, 'signal'] = 1
    result.loc[bullish, 'confidence'] = 0.7
    
    # Señal bajista si: bear_signal Y close < EMA_20 Y RSI > 30
    bearish = (result.get('bear_signal', False) == True) & \
              (result['Close'] < result['EMA_20']) & \
              (result['RSI'] > 30)
    result.loc[bearish, 'signal'] = -1
    result.loc[bearish, 'confidence'] = 0.7
    
    return result


# Signal enum para compatibilidad

class Signal(Enum):
    """Signal enum for trading signals"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    NEUTRAL = 0  # Alias de HOLD
    SELL = -1
    STRONG_SELL = -2
    
    # Aliases adicionales
    LONG = 1
    SHORT = -1
    
    @classmethod
    def from_value(cls, value):
        """Convert numeric value to signal type"""
        if value >= 2:
            return cls.STRONG_BUY
        elif value >= 1:
            return cls.BUY
        elif value <= -2:
            return cls.STRONG_SELL
        elif value <= -1:
            return cls.SELL
        else:
            return cls.HOLD
