"""
SQL Data Handler
================
Replacement for MultiTFDataHandler using SQLite database.
"""

import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.mtf_config import MTF_CONFIG, TRADING_CONFIG
from core.data.db_manager import DBManager

logger = logging.getLogger(__name__)


class SQLDataHandler:
    """
    Fetches multi-timeframe data from SQLite database.
    """

    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_manager = DBManager(db_path)
        logger.info("✅ SQLDataHandler initialized")

    def get_multi_tf_data(
        self, symbol: str = "BTCUSD", start_date: str = None, end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes from DB.

        Args:
            symbol: Trading pair (default: BTCUSD)
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD

        Returns:
            Dict: {'entry': df_5m, 'momentum': df_15m, 'trend': df_1h}
        """
        start = start_date or TRADING_CONFIG["start_date"]
        end = end_date or TRADING_CONFIG["end_date"]

        logger.info(f"📊 Fetching multi-TF data from DB for {symbol}: {start} to {end}")

        timeframes = MTF_CONFIG["timeframes"]  # e.g. {'entry': '5Min', ...}
        dfs = {}

        # Map config timeframes to DB timeframes
        tf_map_db = {"5Min": "5m", "15Min": "15m", "1H": "1h", "1D": "1d"}

        for tf_name, tf_value in timeframes.items():
            db_tf = tf_map_db.get(tf_value, tf_value)
            logger.info(f"  Loading {tf_name} ({db_tf})...")

            df = self.db_manager.load_data(symbol, db_tf, start, end)

            if not df.empty:
                # Ensure columns are lowercase to match indicators.py expectations
                df.columns = [c.lower() for c in df.columns]
                dfs[tf_name] = df
                logger.info(f"  ✅ {tf_name}: {len(df)} bars loaded")
            else:
                logger.warning(f"  ⚠️ No data for {tf_name} ({db_tf}) in DB")

        return dfs

    def add_multi_tf_filters(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Añade filtros cross-TF: HTF trend, MTF momentum, vol interconectado.
        Copied logic from MultiTFDataHandler to ensure consistency.
        """
        logger.info("🔧 Adding multi-TF filters (SQL)...")

        if "entry" not in dfs or "momentum" not in dfs or "trend" not in dfs:
            logger.error("Missing required timeframes for filters")
            return dfs

        df_5m = dfs.get("entry").copy()
        df_15m = dfs.get("momentum").copy()
        df_1h = dfs.get("trend").copy()

        # ========================================================================
        # HTF TREND FILTER (1H) - ALWAYS BIAS
        # ========================================================================
        # Calculate EMA200 on 1h
        df_1h["EMA200"] = df_1h["close"].ewm(span=200, adjust=False).mean()

        # Resample 1h to 5min (forward fill)
        df_1h_resampled = df_1h[["EMA200"]].resample("5Min").ffill()
        df_1h_resampled.columns = ["EMA200_1h"]

        # Merge to 5min
        df_5m = df_5m.join(df_1h_resampled, how="left")
        df_5m["EMA200_1h"] = df_5m["EMA200_1h"].ffill()

        # Define uptrend (CRITICAL FILTER)
        df_5m["uptrend_1h"] = df_5m["close"] > df_5m["EMA200_1h"]

        # ========================================================================
        # MTF MOMENTUM FILTER (15MIN EMA50)
        # ========================================================================
        # Calculate EMA20 on 5min
        df_5m["EMA20"] = df_5m["close"].ewm(span=20, adjust=False).mean()

        # Calculate EMA50 on 15min
        df_15m["EMA50"] = df_15m["close"].ewm(span=50, adjust=False).mean()

        # Resample 15min to 5min
        df_15m_resampled = df_15m[["EMA50"]].resample("5Min").ffill()
        df_15m_resampled.columns = ["EMA50_15m"]

        # Merge to 5min
        df_5m = df_5m.join(df_15m_resampled, how="left")
        df_5m["EMA50_15m"] = df_5m["EMA50_15m"].ffill()

        # Define momentum
        df_5m["momentum_15m"] = df_5m["EMA20"] > df_5m["EMA50_15m"]

        # ========================================================================
        # VOL CROSS-TF FILTER
        # ========================================================================
        # SMA21 on 5min volume
        df_5m["SMA_vol_21"] = df_5m["volume"].rolling(21).mean()

        # SMA on 1h volume
        df_1h["SMA_vol"] = df_1h["volume"].rolling(10).mean()

        # Resample 1h vol to 5min
        df_1h_vol_resampled = df_1h[["SMA_vol"]].resample("5Min").ffill()
        df_1h_vol_resampled.columns = ["SMA_vol_1h"]

        # Merge to 5min
        df_5m = df_5m.join(df_1h_vol_resampled, how="left")
        df_5m["SMA_vol_1h"] = df_5m["SMA_vol_1h"].ffill()

        # Vol cross filter
        vol_thresh = 1.2
        df_5m["high_vol_5m"] = df_5m["volume"] > (vol_thresh * df_5m["SMA_vol_21"])
        df_5m["high_vol_cross"] = df_5m["volume"] > df_5m["SMA_vol_1h"]
        df_5m["vol_filter"] = df_5m["high_vol_5m"] & df_5m["high_vol_cross"]

        # ========================================================================
        # COMBINED FILTERS
        # ========================================================================
        # Bull filter: uptrend AND momentum AND vol
        df_5m["bull_filter"] = df_5m["uptrend_1h"] & df_5m["momentum_15m"] & df_5m["vol_filter"]

        # Update dfs
        dfs["entry"] = df_5m
        dfs["momentum"] = df_15m
        dfs["trend"] = df_1h

        return dfs

    def add_noise(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Placeholder for noise addition if needed.
        """
        return dfs
