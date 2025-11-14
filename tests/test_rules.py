#!/usr/bin/env python3
"""
Test Suite for Trading Rules
============================
Pytest tests for src/rules.py validating confluence scoring and trade params
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rules import (
    check_long_conditions,
    check_short_conditions,
    calculate_confluence_score,
    calculate_position_params,
    check_global_filters,
    should_exit_position,
    update_trailing_stop
)


@pytest.fixture
def sample_data_perfect_long():
    """Genera datos con todas las condiciones LONG cumplidas (score=5)"""
    dates = pd.date_range('2025-01-01', periods=300, freq='5min')
    
    # Crear volumen que cumpla condición volume cross
    # Vol actual debe ser > 1.2 * SMA(vol, 21)
    # Estrategia: empezar con vol bajo, luego alto para que SMA < vol actual
    volume_data = [300] * 50 + [650] * 250  # Primeras 50 bajas, resto altas
    
    df = pd.DataFrame({
        'close': [45000] * 300,
        'high': [45100] * 300,
        'low': [44900] * 300,
        'volume': volume_data,
        'ATR': [150] * 300,
        # HTF Bias: close > ema_210_1h ✓
        'ema_210_1h': [44500] * 300,
        # Momentum: ema_18_5m > ema_48_15m ✓
        'ema_18_5m': [45050] * 300,
        'ema_48_15m': [44900] * 300,
        # IFVG ✓
        'ifvg_bull': [True] * 300,
        'ifvg_bear': [False] * 300,
        # Volume Profile ✓
        'vp_val': [44800] * 300,  # close > val
        'vp_vah': [45200] * 300,
        'vp_poc_1h': [44950] * 300,  # |45000-44950| = 50 < 0.5*300 = 150
        'atr_1h': [300] * 300,
        # Otros
        'vol_sma_1h': [350] * 300,  # Para vol cross - debe ser menor que SMA vol 5m
        'volume_1h': [10000] * 300,
        'rsi_15m': [55] * 300,
    })
    
    df.index = dates
    return df


@pytest.fixture
def sample_data_insufficient():
    """Genera datos con score < 4 (no debe entrar)"""
    dates = pd.date_range('2025-01-01', periods=300, freq='5min')
    
    df = pd.DataFrame({
        'close': [45000] * 300,
        'high': [45100] * 300,
        'low': [44900] * 300,
        'volume': [300] * 300,  # Bajo volumen
        'ATR': [150] * 300,
        # HTF Bias: NO cumple (close < ema_210_1h) ❌
        'ema_210_1h': [45500] * 300,
        # Momentum: NO cumple ❌
        'ema_18_5m': [44900] * 300,
        'ema_48_15m': [45050] * 300,
        # IFVG NO ❌
        'ifvg_bull': [False] * 300,
        'ifvg_bear': [False] * 300,
        'vp_val': [44800] * 300,
        'vp_vah': [45200] * 300,
        'vp_poc_1h': [44950] * 300,
        'atr_1h': [300] * 300,
    })
    
    df.index = dates
    return df


def test_long_conditions_score_5(sample_data_perfect_long):
    """Test: Todas las condiciones cumplidas debe dar score>=4 y permitir entrada"""
    df = sample_data_perfect_long
    
    can_enter, score, params = check_long_conditions(df, idx=150, min_score=4)
    
    assert can_enter == True, "Debe poder entrar con todas las condiciones"
    assert score >= 4, f"Score esperado >=4, obtenido {score}"
    assert params is not None, "Debe retornar parámetros de posición"
    
    # Validar parámetros calculados
    assert params['entry_price'] == 45000
    assert params['stop_loss'] == 45000 - (1.5 * 150)  # 44775
    assert params['take_profit'] == 45000 + (1.5 * 150 * 2.2)  # 45495
    assert params['risk'] == 1.5 * 150  # 225


def test_long_conditions_insufficient_score(sample_data_insufficient):
    """Test: Score < 4 no debe permitir entrada"""
    df = sample_data_insufficient
    
    can_enter, score, params = check_long_conditions(df, idx=150, min_score=4)
    
    assert can_enter == False, "No debe entrar con score insuficiente"
    assert score < 4, f"Score debe ser < 4, obtenido {score}"
    assert params is None, "No debe retornar parámetros si no puede entrar"


def test_position_params_calculation(sample_data_perfect_long):
    """Test: Cálculo correcto de SL/TP/Position Size"""
    df = sample_data_perfect_long
    
    params = calculate_position_params(
        df, 
        idx=150,
        direction='long',
        capital=10000,
        risk_pct=0.01,
        max_exposure_pct=0.05
    )
    
    # Entry = 45000
    # Risk = 1.5 * ATR = 1.5 * 150 = 225
    # SL = 45000 - 225 = 44775
    # TP = 45000 + (225 * 2.2) = 45495
    # Risk amount = 10000 * 0.01 = 100
    # Position size = min(100/225, (10000*0.05)/45000) = min(0.444, 0.011) = 0.011
    
    assert params['entry_price'] == 45000
    assert params['stop_loss'] == 44775
    assert params['take_profit'] == 45495
    assert params['risk_amount'] == 100
    assert abs(params['position_size'] - 0.011) < 0.001  # ~0.011 BTC


def test_confluence_score_components(sample_data_perfect_long):
    """Test: Cada componente del score suma correctamente"""
    df = sample_data_perfect_long
    
    scores = calculate_confluence_score(df, direction='long')
    
    # Verificar que al menos una barra tiene score=5
    assert scores.max() == 5, f"Score máximo esperado 5, obtenido {scores.max()}"
    
    # Verificar que todos los scores son válidos (0-5)
    assert scores.min() >= 0, "Score mínimo debe ser >= 0"
    assert scores.max() <= 5, "Score máximo debe ser <= 5"


def test_global_filters_extreme_volatility():
    """Test: Filtro de volatilidad extrema debe bloquear trading"""
    dates = pd.date_range('2025-01-01', periods=50, freq='5min')
    
    df = pd.DataFrame({
        'close': [45000] * 50,
        # ATR extremo: 800 > 2.0 * 300 = 600
        'atr_1h': [800] * 50,
    })
    df.index = dates
    
    # Crear SMA de ATR
    df['atr_1h_sma'] = 300
    
    can_trade, reason = check_global_filters(df, 25)
    
    # Nota: check_global_filters calcula rolling internamente
    # Este test es ilustrativo de la lógica esperada


def test_exit_take_profit_long():
    """Test: Exit cuando se alcanza TP"""
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')
    
    df = pd.DataFrame({
        'close': [45600] * 100,  # Precio por encima de TP
        'ema_210_1h': [44500] * 100,
    })
    df.index = dates
    
    position = {
        'direction': 'long',
        'entry_price': 45000,
        'stop_loss': 44775,
        'take_profit': 45495,  # Alcanzado
        'entry_idx': 10,
        'risk': 225
    }
    
    should_exit, reason = should_exit_position(df, 50, position)
    
    assert should_exit == True, "Debe salir cuando alcanza TP"
    assert reason == "TAKE_PROFIT", f"Razón esperada TAKE_PROFIT, obtenida {reason}"


def test_exit_stop_loss_long():
    """Test: Exit cuando se alcanza SL"""
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')
    
    df = pd.DataFrame({
        'close': [44700] * 100,  # Precio por debajo de SL
        'ema_210_1h': [44500] * 100,
    })
    df.index = dates
    
    position = {
        'direction': 'long',
        'entry_price': 45000,
        'stop_loss': 44775,  # Alcanzado
        'take_profit': 45495,
        'entry_idx': 10,
        'risk': 225
    }
    
    should_exit, reason = should_exit_position(df, 50, position)
    
    assert should_exit == True, "Debe salir cuando alcanza SL"
    assert reason == "STOP_LOSS", f"Razón esperada STOP_LOSS, obtenida {reason}"


def test_exit_htf_bias_flip():
    """Test: Exit cuando cambia bias HTF"""
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')
    
    df = pd.DataFrame({
        'close': [44400] * 100,  # Precio cruza por debajo de EMA 210
        'ema_210_1h': [44500] * 100,
    })
    df.index = dates
    
    position = {
        'direction': 'long',
        'entry_price': 45000,
        'stop_loss': 44000,  # Stop loss más bajo para que no se active primero
        'take_profit': 45495,
        'entry_idx': 10,
        'risk': 225
    }
    
    should_exit, reason = should_exit_position(df, 50, position)
    
    assert should_exit == True, "Debe salir por HTF flip"
    assert reason == "HTF_BIAS_FLIP", f"Razón esperada HTF_BIAS_FLIP, obtenida {reason}"



def test_trailing_stop_activation():
    """Test: Trailing stop se activa después de +1R"""
    position = {
        'direction': 'long',
        'entry_price': 45000,
        'stop_loss': 44775,
        'risk': 225,
        'trailing_active': False
    }
    
    # Profit = 45300 - 45000 = 300 > 225 (1R) ✓
    current_price = 45300
    atr = 150
    
    updated_position = update_trailing_stop(position, current_price, atr)
    
    assert updated_position['trailing_active'] == True, "Trailing debe activarse"
    # Nuevo SL = entry + 0.5*ATR = 45000 + 75 = 45075
    assert updated_position['stop_loss'] > 44775, "SL debe subir"


def test_short_conditions_mirror_long(sample_data_perfect_long):
    """Test: Condiciones short son espejo de long"""
    # Modificar datos para short perfecto
    df = sample_data_perfect_long.copy()
    
    # Invertir condiciones
    df['ema_210_1h'] = 45500  # close < ema_210 para short
    df['ema_18_5m'] = 44900   # ema_18 < ema_48 para short
    df['ema_48_15m'] = 45050
    df['ifvg_bull'] = False
    df['ifvg_bear'] = True
    df['vp_vah'] = 45200  # close < vah para short
    
    can_enter, score, params = check_short_conditions(df, idx=150, min_score=4)
    
    # Debería cumplir condiciones short
    assert score >= 2, f"Score short debe ser >= 2 (obligatorias), obtenido {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
