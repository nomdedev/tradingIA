"""
Estrategias Simples para Validación del Sistema.
Implementaciones de referencia de MA Crossover, RSI Mean Reversion y Volatility Breakout.
"""
from typing import Dict, Any
import pandas as pd
import pandas_ta as ta
from strategies.base_strategy import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(name="MA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        # Usamos el timeframe principal (asumimos '5m' o el primero disponible)
        tf = list(df_multi_tf.keys())[0]
        df = df_multi_tf[tf].copy()

        # Calcular indicadores
        df['fast_ma'] = ta.sma(df['close'], length=self.fast_period)
        df['slow_ma'] = ta.sma(df['close'], length=self.slow_period)

        # Generar señales
        signals = pd.Series(0, index=df.index)
        
        # Crossover alcista
        bull_cond = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        signals[bull_cond] = 1
        
        # Crossover bajista
        bear_cond = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
        signals[bear_cond] = -1

        return {
            'signals': signals,
            'entries': signals != 0,
            'exits': pd.Series(False, index=df.index) # Salidas gestionadas por TP/SL o señal opuesta
        }

    def get_parameters(self) -> Dict:
        return {'fast_period': self.fast_period, 'slow_period': self.slow_period}

    def set_parameters(self, params: Dict) -> None:
        self.fast_period = params.get('fast_period', self.fast_period)
        self.slow_period = params.get('slow_period', self.slow_period)


class RSIStrategy(BaseStrategy):
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__(name="RSI Mean Reversion")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        tf = list(df_multi_tf.keys())[0]
        df = df_multi_tf[tf].copy()

        df['rsi'] = ta.rsi(df['close'], length=self.period)

        signals = pd.Series(0, index=df.index)
        
        # Compra en sobreventa (cruce hacia arriba de 30)
        bull_cond = (df['rsi'] < self.oversold)
        signals[bull_cond] = 1
        
        # Venta en sobrecompra (cruce hacia abajo de 70)
        bear_cond = (df['rsi'] > self.overbought)
        signals[bear_cond] = -1

        return {
            'signals': signals,
            'entries': signals != 0,
            'exits': pd.Series(False, index=df.index)
        }

    def get_parameters(self) -> Dict:
        return {'period': self.period, 'overbought': self.overbought, 'oversold': self.oversold}

    def set_parameters(self, params: Dict) -> None:
        self.period = params.get('period', self.period)
        self.overbought = params.get('overbought', self.overbought)
        self.oversold = params.get('oversold', self.oversold)


class BreakoutStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, factor: float = 1.5):
        super().__init__(name="Volatility Breakout")
        self.lookback = lookback
        self.factor = factor

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        tf = list(df_multi_tf.keys())[0]
        df = df_multi_tf[tf].copy()

        # Bandas de Bollinger o Canal de Donchian
        df['high_max'] = df['high'].rolling(self.lookback).max()
        df['low_min'] = df['low'].rolling(self.lookback).min()
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        signals = pd.Series(0, index=df.index)
        
        # Breakout alcista: Cierre > Max High previo + factor * ATR
        # Simplificado: Cierre > Max High previo
        bull_cond = df['close'] > df['high_max'].shift(1)
        signals[bull_cond] = 1
        
        # Breakout bajista
        bear_cond = df['close'] < df['low_min'].shift(1)
        signals[bear_cond] = -1

        return {
            'signals': signals,
            'entries': signals != 0,
            'exits': pd.Series(False, index=df.index)
        }

    def get_parameters(self) -> Dict:
        return {'lookback': self.lookback, 'factor': self.factor}

    def set_parameters(self, params: Dict) -> None:
        self.lookback = params.get('lookback', self.lookback)
        self.factor = params.get('factor', self.factor)
