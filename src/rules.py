#!/usr/bin/env python3
"""
Trading Rules Engine
===================
Implementa las reglas concretas de entrada/salida para la estrategia
IFVG + Volume Profile + EMAs Multi-TF basado en strategy_definition.md

Funciones principales:
- check_long_conditions: Valida condiciones de entrada long
- check_short_conditions: Valida condiciones de entrada short
- calculate_confluence_score: Calcula score 0-5 de confluencia
- calculate_position_params: Calcula entry, SL, TP para un trade
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def check_htf_bias(df_5m: pd.DataFrame, direction: str = 'long') -> pd.Series:
    """
    Verifica bias de Higher Timeframe (1h) - CONDICIÓN OBLIGATORIA

    Args:
        df_5m: DataFrame con datos de 5m incluyendo EMA_210_1h resampleada
        direction: 'long' o 'short'

    Returns:
        Serie booleana indicando si cumple bias HTF
    """
    if 'ema_210_1h' not in df_5m.columns:
        logger.warning("EMA_210_1h no encontrada en df_5m. Retornando False.")
        return pd.Series(False, index=df_5m.index)

    if direction == 'long':
        # Long: precio debe estar por encima de EMA 210 de 1h
        bias = df_5m['close'] > df_5m['ema_210_1h']
    else:
        # Short: precio debe estar por debajo de EMA 210 de 1h
        bias = df_5m['close'] < df_5m['ema_210_1h']

    return bias


def check_momentum_15m(df_5m: pd.DataFrame, direction: str = 'long') -> pd.Series:
    """
    Verifica momentum de 15m - CONDICIÓN OBLIGATORIA

    Args:
        df_5m: DataFrame con EMAs calculadas (EMA_18_5m, EMA_48_15m)
        direction: 'long' o 'short'

    Returns:
        Serie booleana indicando si cumple momentum
    """
    if 'ema_18_5m' not in df_5m.columns or 'ema_48_15m' not in df_5m.columns:
        logger.warning("EMAs de momentum no encontradas. Retornando False.")
        return pd.Series(False, index=df_5m.index)

    if direction == 'long':
        # Long: EMA rápida 5m debe estar por encima de EMA 48 de 15m
        momentum = df_5m['ema_18_5m'] > df_5m['ema_48_15m']
    else:
        # Short: EMA rápida 5m debe estar por debajo de EMA 48 de 15m
        momentum = df_5m['ema_18_5m'] < df_5m['ema_48_15m']

    return momentum


def check_ifvg_condition(df_5m: pd.DataFrame, direction: str = 'long') -> pd.Series:
    """
    Verifica condición IFVG (Inverse Fair Value Gap) - ADICIONAL

    Args:
        df_5m: DataFrame con señales IFVG precalculadas
        direction: 'long' o 'short'

    Returns:
        Serie booleana indicando si cumple IFVG
    """
    if direction == 'long':
        # Long: señal bullish de IFVG (mitigación de gap bajista)
        if 'ifvg_bull' in df_5m.columns:
            return df_5m['ifvg_bull']
        else:
            logger.warning("ifvg_bull no encontrada. Retornando False.")
            return pd.Series(False, index=df_5m.index)
    else:
        # Short: señal bearish de IFVG
        if 'ifvg_bear' in df_5m.columns:
            return df_5m['ifvg_bear']
        else:
            logger.warning("ifvg_bear no encontrada. Retornando False.")
            return pd.Series(False, index=df_5m.index)


def check_volume_cross(df_5m: pd.DataFrame) -> pd.Series:
    """
    Verifica condición de Volume Cross Multi-TF - ADICIONAL

    Condiciones:
    1. Volume 5m > 1.2 * SMA(volume_5m, 21)
    2. SMA(volume_5m, 21) > rolling_mean(SMA(volume_1h, 21), 10)

    Args:
        df_5m: DataFrame con volumen y SMAs calculadas

    Returns:
        Serie booleana indicando si cumple volume cross
    """
    if 'volume' not in df_5m.columns:
        logger.warning("Volume no encontrada. Retornando False.")
        return pd.Series(False, index=df_5m.index)

    # SMA de volumen 5m
    vol_sma_5m = talib.SMA(df_5m['volume'].to_numpy(dtype=np.float64), timeperiod=21)

    # Condición 1: Vol actual > 1.2 * SMA
    vol_thresh = 1.2
    cond1 = df_5m['volume'] > (vol_sma_5m * vol_thresh)

    # Condición 2: SMA 5m > rolling mean de SMA 1h (si está disponible)
    if 'vol_sma_1h' in df_5m.columns:
        vol_sma_1h_rolling = df_5m['vol_sma_1h'].rolling(window=10).mean()
        cond2 = pd.Series(vol_sma_5m, index=df_5m.index) > vol_sma_1h_rolling
    else:
        # Si no hay SMA 1h, solo validar condición 1
        logger.warning("vol_sma_1h no disponible. Solo validando condición 1.")
        cond2 = pd.Series(True, index=df_5m.index)

    return cond1 & cond2


def check_volume_profile(df_5m: pd.DataFrame, direction: str = 'long') -> pd.Series:
    """
    Verifica condición de Volume Profile - ADICIONAL

    Long:
    - Close > VAL_5m (Value Area Low)
    - |Close - POC_1h| < 0.5 * ATR_1h

    Short:
    - Close < VAH_5m (Value Area High)
    - |Close - POC_1h| < 0.5 * ATR_1h

    Args:
        df_5m: DataFrame con niveles VP calculados
        direction: 'long' o 'short'

    Returns:
        Serie booleana indicando si cumple VP
    """
    required_cols = ['vp_val', 'vp_vah', 'vp_poc_1h', 'atr_1h']

    for col in required_cols:
        if col not in df_5m.columns:
            logger.warning(f"{col} no encontrada. Retornando False.")
            return pd.Series(False, index=df_5m.index)

    close = df_5m['close']

    if direction == 'long':
        # Long: precio por encima de VAL
        price_filter = close > df_5m['vp_val']
    else:
        # Short: precio por debajo de VAH
        price_filter = close < df_5m['vp_vah']

    # Proximidad a POC de 1h
    poc_distance = (close - df_5m['vp_poc_1h']).abs()
    poc_filter = poc_distance < (0.5 * df_5m['atr_1h'])

    return price_filter & poc_filter


def calculate_confluence_score(df_5m: pd.DataFrame, direction: str = 'long') -> pd.Series:
    """
    Calcula score de confluencia (0-5) para cada barra

    Score Components:
    1. HTF Bias 1h (OBLIGATORIO)
    2. Momentum 15m (OBLIGATORIO)
    3. IFVG Signal (ADICIONAL)
    4. Volume Cross (ADICIONAL)
    5. Volume Profile (ADICIONAL)

    Entry requiere: Score >= 4

    Args:
        df_5m: DataFrame con todos los indicadores calculados
        direction: 'long' o 'short'

    Returns:
        Serie con scores 0-5
    """
    score = pd.Series(0, index=df_5m.index, dtype=int)

    # Componente 1: HTF Bias (OBLIGATORIO)
    htf_bias = check_htf_bias(df_5m, direction)
    score += htf_bias.astype(int)

    # Componente 2: Momentum 15m (OBLIGATORIO)
    momentum = check_momentum_15m(df_5m, direction)
    score += momentum.astype(int)

    # Componente 3: IFVG (ADICIONAL)
    ifvg = check_ifvg_condition(df_5m, direction)
    score += ifvg.astype(int)

    # Componente 4: Volume Cross (ADICIONAL)
    vol_cross = check_volume_cross(df_5m)
    score += vol_cross.astype(int)

    # Componente 5: Volume Profile (ADICIONAL)
    vp = check_volume_profile(df_5m, direction)
    score += vp.astype(int)

    return score


def calculate_position_params(
    df_5m: pd.DataFrame,
    idx: int,
    direction: str = 'long',
    capital: float = 10000,
    risk_pct: float = 0.01,
    max_exposure_pct: float = 0.05
) -> Dict[str, float]:
    """
    Calcula parámetros de posición para un trade

    Args:
        df_5m: DataFrame con datos de mercado
        idx: Índice de la barra de entrada
        direction: 'long' o 'short'
        capital: Capital disponible
        risk_pct: Porcentaje de riesgo (0.01 = 1%)
        max_exposure_pct: Máxima exposición por posición (0.05 = 5%)

    Returns:
        Dict con entry_price, stop_loss, take_profit, position_size, risk_amount
    """
    if idx >= len(df_5m):
        raise ValueError(f"Índice {idx} fuera de rango")

    entry_price = df_5m['close'].iloc[idx]

    # ATR para cálculo de SL
    if 'ATR' in df_5m.columns:
        atr = df_5m['ATR'].iloc[idx]
    else:
        # Fallback: usar 2% del precio como ATR estimado
        atr = entry_price * 0.02
        logger.warning(f"ATR no disponible en idx {idx}. Usando 2% del precio.")

    # Risk = 1.5 * ATR
    risk = 1.5 * atr

    # Stop Loss
    if direction == 'long':
        stop_loss = entry_price - risk
    else:
        stop_loss = entry_price + risk

    # Take Profit (RR = 2.2)
    tp_rr = 2.2
    if direction == 'long':
        take_profit = entry_price + (risk * tp_rr)
    else:
        take_profit = entry_price - (risk * tp_rr)

    # Position Sizing
    risk_amount = capital * risk_pct

    # Método 1: Basado en riesgo
    position_size_risk = risk_amount / risk

    # Método 2: Basado en exposición máxima
    max_position_value = capital * max_exposure_pct
    position_size_exposure = max_position_value / entry_price

    # Tomar el mínimo para respetar ambas restricciones
    position_size = min(position_size_risk, position_size_exposure)

    return {
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'position_size': position_size,
        'risk_amount': risk_amount,
        'risk': risk,
        'atr': atr
    }


def check_long_conditions(
    df_5m: pd.DataFrame,
    idx: int,
    min_score: int = 4
) -> Tuple[bool, int, Optional[Dict[str, float]]]:
    """
    Verifica todas las condiciones de entrada LONG

    Args:
        df_5m: DataFrame con indicadores calculados
        idx: Índice de la barra a evaluar
        min_score: Score mínimo requerido para entrada (default 4)

    Returns:
        Tuple (puede_entrar, score, params_dict)
        - puede_entrar: True si cumple condiciones
        - score: Score de confluencia 0-5
        - params_dict: Parámetros de posición si puede_entrar=True, None otherwise
    """
    if idx >= len(df_5m):
        return False, 0, None

    # Calcular score de confluencia
    scores = calculate_confluence_score(df_5m, direction='long')
    score = int(scores.iloc[idx])

    # Verificar si cumple score mínimo
    if score >= min_score:
        # Calcular parámetros de posición
        params = calculate_position_params(df_5m, idx, direction='long')
        return True, score, params
    else:
        return False, score, None


def check_short_conditions(
    df_5m: pd.DataFrame,
    idx: int,
    min_score: int = 4
) -> Tuple[bool, int, Optional[Dict[str, float]]]:
    """
    Verifica todas las condiciones de entrada SHORT

    Args:
        df_5m: DataFrame con indicadores calculados
        idx: Índice de la barra a evaluar
        min_score: Score mínimo requerido para entrada (default 4)

    Returns:
        Tuple (puede_entrar, score, params_dict)
    """
    if idx >= len(df_5m):
        return False, 0, None

    # Calcular score de confluencia
    scores = calculate_confluence_score(df_5m, direction='short')
    score = int(scores.iloc[idx])

    # Verificar si cumple score mínimo
    if score >= min_score:
        # Calcular parámetros de posición
        params = calculate_position_params(df_5m, idx, direction='short')
        return True, score, params
    else:
        return False, score, None


def check_global_filters(df_5m: pd.DataFrame, idx: int) -> Tuple[bool, str]:
    """
    Verifica filtros globales que pueden pausar el trading

    Filtros:
    1. Volatilidad extrema: ATR_1h > 2.0 * SMA(ATR_1h, 20)
    2. Baja volatilidad: Volume_1h < 0.5 * SMA(Volume_1h, 20)

    Args:
        df_5m: DataFrame con datos
        idx: Índice de la barra

    Returns:
        Tuple (can_trade, reason)
        - can_trade: True si puede operar
        - reason: Razón si no puede operar
    """
    if idx >= len(df_5m):
        return False, "INVALID_INDEX"

    # Filtro 1: Volatilidad extrema
    if 'atr_1h' in df_5m.columns:
        atr_1h = df_5m['atr_1h'].iloc[idx]
        atr_1h_sma = df_5m['atr_1h'].rolling(20).mean().iloc[idx]

        if not pd.isna(atr_1h_sma) and atr_1h > 2.0 * atr_1h_sma:
            return False, "EXTREME_VOLATILITY"

    # Filtro 2: Baja liquidez
    if 'volume_1h' in df_5m.columns:
        vol_1h = df_5m['volume_1h'].iloc[idx]
        vol_1h_sma = df_5m['volume_1h'].rolling(20).mean().iloc[idx]

        if not pd.isna(vol_1h_sma) and vol_1h < 0.5 * vol_1h_sma:
            return False, "LOW_VOLUME_PERIOD"

    return True, "OK"


def should_exit_position(
    df_5m: pd.DataFrame,
    idx: int,
    position: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Determina si una posición debe cerrarse

    Exit Conditions:
    1. TP/SL alcanzado
    2. HTF Bias flip
    3. Time limit (12h = 144 bars @ 5m)
    4. RSI extremo (>70 long, <30 short)
    5. Volume spike (>4.5 * SMA)

    Args:
        df_5m: DataFrame con datos
        idx: Índice actual
        position: Dict con info de posición (direction, entry_idx, entry_price, sl, tp)

    Returns:
        Tuple (should_exit, exit_reason)
    """
    if idx >= len(df_5m):
        return False, "INVALID_INDEX"

    current_price = df_5m['close'].iloc[idx]
    direction = position['direction']
    entry_price = position['entry_price']
    stop_loss = position['stop_loss']
    take_profit = position['take_profit']
    entry_idx = position.get('entry_idx', idx)

    # 1. TP/SL alcanzado
    if direction == 'long':
        if current_price >= take_profit:
            return True, "TAKE_PROFIT"
        if current_price <= stop_loss:
            return True, "STOP_LOSS"
    else:  # short
        if current_price <= take_profit:
            return True, "TAKE_PROFIT"
        if current_price >= stop_loss:
            return True, "STOP_LOSS"

    # 2. HTF Bias flip
    if 'ema_210_1h' in df_5m.columns:
        ema_210 = df_5m['ema_210_1h'].iloc[idx]
        if direction == 'long' and current_price < ema_210:
            return True, "HTF_BIAS_FLIP"
        if direction == 'short' and current_price > ema_210:
            return True, "HTF_BIAS_FLIP"

    # 3. Time limit (12h = 144 bars)
    bars_open = idx - entry_idx
    if bars_open > 144:
        return True, "EOD_TIME_LIMIT"

    # 4. RSI extremo
    if 'rsi_15m' in df_5m.columns:
        rsi = df_5m['rsi_15m'].iloc[idx]
        if direction == 'long' and rsi > 70:
            return True, "EXHAUSTION_RSI"
        if direction == 'short' and rsi < 30:
            return True, "EXHAUSTION_RSI"

    # 5. Volume spike
    if 'volume' in df_5m.columns:
        vol = df_5m['volume'].iloc[idx]
        vol_sma = talib.SMA(df_5m['volume'].to_numpy(dtype=np.float64), timeperiod=21)

        if idx < len(vol_sma) and not pd.isna(vol_sma[idx]):
            if vol > 4.5 * vol_sma[idx]:
                return True, "EXHAUSTION_VOLUME_SPIKE"

    return False, "HOLDING"


def update_trailing_stop(
    position: Dict[str, Any],
    current_price: float,
    atr: float
) -> Dict[str, Any]:
    """
    Actualiza trailing stop si el trade está en profit

    Regla:
    - Activación: Profit > 1R (risk)
    - Breakeven: Entry + 0.5 * ATR
    - Trail: +1 ATR por cada +0.5R adicional

    Args:
        position: Dict con info de posición
        current_price: Precio actual
        atr: ATR actual

    Returns:
        Position dict actualizado con nuevo stop_loss si aplica
    """
    direction = position['direction']
    entry_price = position['entry_price']
    risk = position['risk']
    current_sl = position['stop_loss']

    # Calcular profit actual
    if direction == 'long':
        profit = current_price - entry_price
    else:
        profit = entry_price - current_price

    # Activar trailing si profit > 1R
    if profit >= risk:
        # Calcular nuevo SL
        if direction == 'long':
            new_sl = entry_price + (0.5 * atr)

            # Trail adicional por cada +0.5R
            if profit >= 1.5 * risk:
                new_sl = entry_price + (1.0 * atr)
            if profit >= 2.0 * risk:
                new_sl = entry_price + (1.5 * atr)

            # Solo actualizar si el nuevo SL es mayor (protege más)
            if new_sl > current_sl:
                position['stop_loss'] = new_sl
                position['trailing_active'] = True

        else:  # short
            new_sl = entry_price - (0.5 * atr)

            if profit >= 1.5 * risk:
                new_sl = entry_price - (1.0 * atr)
            if profit >= 2.0 * risk:
                new_sl = entry_price - (1.5 * atr)

            # Solo actualizar si el nuevo SL es menor (protege más en short)
            if new_sl < current_sl:
                position['stop_loss'] = new_sl
                position['trailing_active'] = True

    return position


# Ejemplo de uso (para testing)
if __name__ == "__main__":
    # Crear datos de ejemplo
    dates = pd.date_range('2025-01-01', periods=500, freq='5min')

    df_test = pd.DataFrame({
        'timestamp': dates,
        'close': 45000 + np.random.randn(500) * 100,
        'high': 45000 + np.random.randn(500) * 100 + 50,
        'low': 45000 + np.random.randn(500) * 100 - 50,
        'volume': 500 + np.random.randn(500) * 50,
        'ATR': 150 + np.random.randn(500) * 10,
        'ema_210_1h': 44500 + np.random.randn(500) * 50,
        'ema_18_5m': 45000 + np.random.randn(500) * 80,
        'ema_48_15m': 44900 + np.random.randn(500) * 70,
        'ifvg_bull': np.random.choice([True, False], 500, p=[0.1, 0.9]),
        'ifvg_bear': np.random.choice([True, False], 500, p=[0.1, 0.9]),
        'vp_val': 44800 + np.random.randn(500) * 50,
        'vp_vah': 45200 + np.random.randn(500) * 50,
        'vp_poc_1h': 45000 + np.random.randn(500) * 30,
        'atr_1h': 300 + np.random.randn(500) * 20,
    })

    df_test.set_index('timestamp', inplace=True)

    # Test check_long_conditions
    can_enter, score, params = check_long_conditions(df_test, 250)

    print("✅ Prueba de reglas completada")
    print(f"   Puede entrar: {can_enter}")
    print(f"   Score: {score}/5")
    if params:
        print(f"   Entry: ${params['entry_price']:.2f}")
        print(f"   SL: ${params['stop_loss']:.2f}")
        print(f"   TP: ${params['take_profit']:.2f}")
        print(f"   Position Size: {params['position_size']:.4f} BTC")
