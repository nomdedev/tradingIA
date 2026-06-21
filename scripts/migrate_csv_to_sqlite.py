"""
Script to migrate existing CSV data to SQLite database.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.data.db_manager import DBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_csvs():
    data_dir = project_root / "data"
    db_manager = DBManager(str(data_dir / "trading_data.db"))

    # Pattern: {symbol}_{timeframe}.csv
    # Example: btc_15Min.csv

    csv_files = list(data_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to migrate.")

    for csv_file in csv_files:
        try:
            filename = csv_file.stem  # e.g., btc_15Min
            parts = filename.split("_")

            if len(parts) < 2:
                logger.warning(f"Skipping {csv_file.name}: filename format not recognized.")
                continue

            symbol = parts[0].upper()  # BTC
            timeframe = parts[1]  # 15Min

            # Normalize timeframe format if needed (e.g., 15Min -> 15m)
            # Assuming current format is what we want to keep or mapping it.
            # Let's map to standard: 5Min->5m, 15Min->15m, 1H->1h
            tf_map = {"5Min": "5m", "15Min": "15m", "1H": "1h", "1D": "1d"}
            timeframe = tf_map.get(timeframe, timeframe)

            logger.info(f"Migrating {symbol} {timeframe} from {csv_file.name}...")

            df = pd.read_csv(csv_file)

            # Ensure index is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "Date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["Date"])
                df.set_index("timestamp", inplace=True)
            else:
                # Try to infer or use first column
                df.index = pd.to_datetime(df.iloc[:, 0])
                df.index.name = "timestamp"

            # Standardize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            db_manager.save_data(df, symbol, timeframe)
            logger.info(f"Successfully migrated {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Failed to migrate {csv_file.name}: {e}")


if __name__ == "__main__":
    migrate_csvs()
