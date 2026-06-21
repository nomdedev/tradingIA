"""Demo rápido que muestra uso del Council y una estrategia simple.

Ejecutar desde la raíz del repo: `python scripts/council_demo.py`
"""

import pandas as pd
import numpy as np
from core.council import Council
from strategies.simple_strategy_template import SimpleMomentumStrategy


def simple_momentum_rule(context):
    # Espera context['signals'] como pd.Series
    sigs = context.get("signals")
    if sigs is None:
        return {"error": "no signals in context"}
    last = int(sigs.iloc[-1])
    return {"signal": last, "score": float(last), "weight": 1.0}


def run_demo(data_path: str = "data/btc_15Min.csv"):
    df = pd.read_csv(data_path)
    # intentar detectar columna de cierre
    if "close" in df.columns:
        price = df["close"]
    else:
        price = df.iloc[:, -1]

    strat = SimpleMomentumStrategy(lookback=50)
    signals = strat.generate_signals(price)

    council = Council()
    council.add_rule("momentum_simple", simple_momentum_rule, "Vota según señal momentum (SMA)")

    decision = council.decide({"signals": signals})
    print("Decision del Council:", decision["decision"], "aggregate_score:", decision["aggregate_score"])

    # Backtest simple (posición retiene hasta señal contraria, sin costos)
    positions = signals.shift(1).fillna(0)
    returns = price.pct_change().fillna(0)
    strat_returns = positions * returns
    equity = (1 + strat_returns).cumprod()

    print("Total return (demo):", equity.iloc[-1])
    print("Mean return:", strat_returns.mean(), "Std return:", strat_returns.std())


if __name__ == "__main__":
    run_demo()
