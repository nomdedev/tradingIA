import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.data.sql_data_handler import SQLDataHandler
from core.data.indicators import generate_filtered_signals


def main():
    data_handler = SQLDataHandler()
    dfs = data_handler.get_multi_tf_data(symbol="BTC", start_date="2024-01-01", end_date="2024-02-01")

    if not dfs:
        print("No data loaded")
        return

    df_5m = dfs["entry"]

    params = {
        "atr_multi": 0.418,
        "va_percent": 0.668,
        "vp_rows": 139,
        "vol_thresh": 1.21,
        "tp_rr": 2.11,
        "min_confidence": 0.53,
    }

    print("Generating signals...")
    bull, bear, conf = generate_filtered_signals(df_5m, params)

    print(f"Bull signals: {bull.sum()}")
    print(f"Bear signals: {bear.sum()}")
    print(f"Avg Confidence: {conf.mean()}")

    if bull.sum() == 0 and bear.sum() == 0:
        print("No signals generated! Checking intermediate indicators...")
        from core.data.indicators import calculate_ifvg_enhanced, volume_profile_advanced

        ifvg_bull, ifvg_bear, ifvg_conf = calculate_ifvg_enhanced(df_5m, params)
        print(f"IFVG Bull: {ifvg_bull.sum()}")
        print(f"IFVG Bear: {ifvg_bear.sum()}")

        vp_poc, vp_vah, vp_val = volume_profile_advanced(df_5m, params)
        print(f"VP POC NaNs: {vp_poc.isna().sum()}")

    # Check Council
    print("\nChecking Council decisions for first 5 signals...")
    from core.council import Council
    from core.risk.risk_manager import RiskManager

    council = Council(rules_dir=str(project_root / "core" / "rules"))
    risk_manager = RiskManager(config={"max_daily_drawdown": 0.05})

    # Find indices of signals
    bull_indices = bull[bull].index
    bear_indices = bear[bear].index

    signals_to_check = []
    for idx in bull_indices[:5]:
        signals_to_check.append((idx, 1))
    for idx in bear_indices[:5]:
        signals_to_check.append((idx, -1))

    for idx, signal_type in signals_to_check:
        # Construct context for Council
        context = {
            "price": df_5m.loc[idx, "close"],
            "atr": df_5m.loc[idx, "ATR"] if "ATR" in df_5m.columns else 0.0,
            "confidence": 0.8,
            "capital": 10000.0,
            "equity": 10000.0,
            "current_drawdown_pct": 0.0,
            "uptrend_1h": True,
            "momentum_15m": True,
        }

        decision = council.decide(context)

        # Check Risk Manager
        risk_check = risk_manager.check_order(
            {"type": "long" if signal_type == 1 else "short", "price": context["price"]}
        )

        print(f"Time: {idx}, Type: {'Bull' if signal_type==1 else 'Bear'}")
        print(f"  Council Decision: {decision}")
        print(f"  Risk Check: {risk_check}")


if __name__ == "__main__":
    main()
