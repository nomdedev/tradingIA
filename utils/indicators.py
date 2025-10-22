import pandas as pd
import numpy as np
import talib
from .advanced_indicators import add_all_advanced_indicators

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade todos los indicadores técnicos al DataFrame
    Incluye indicadores básicos + indicadores avanzados institucionales
    """
    df = df.copy()

    # Indicadores básicos de TA-Lib
    # RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    df['MACD_hist'] = macdhist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower

    # Keltner Channels
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=20)
    df['KC_upper'] = typical_price + (atr * 1.5)
    df['KC_middle'] = typical_price
    df['KC_lower'] = typical_price - (atr * 1.5)

    # Volume indicators
    df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Momentum
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)

    # Stochastic
    slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd

    # Williams %R
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Añadir indicadores avanzados (Squeeze Momentum, IFVG, EMAs institucionales)
    df = add_all_advanced_indicators(df)

    # Limpiar NaN values
    df = df.dropna()

    return df