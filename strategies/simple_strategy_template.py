"""Plantilla simple de estrategia: momentum basada en SMA.

Implementa `SimpleMomentumStrategy.generate_signals(price_series)` -> pd.Series de {-1,0,1}
"""
from typing import Optional
import pandas as pd


class SimpleMomentumStrategy:
    def __init__(self, lookback: int = 20):
        self.lookback = int(lookback)

    def generate_signals(self, price_series: pd.Series) -> pd.Series:
        sma = price_series.rolling(self.lookback, min_periods=1).mean()
        sig = pd.Series(0, index=price_series.index)
        sig[price_series > sma] = 1
        sig[price_series < sma] = -1
        return sig.astype(int)

    def get_parameters(self) -> dict:
        return {"lookback": self.lookback}

    def set_parameters(self, **params) -> None:
        if "lookback" in params:
            self.lookback = int(params["lookback"]) 
