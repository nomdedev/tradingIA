"""
Database Manager for Trading Data.
Handles connection and schema for SQLite database.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DBManager:
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Table for market data (OHLCV)
        # We use a composite primary key (symbol, timeframe, timestamp)
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
        """
        )

        # Index for faster queries
        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_market_data_lookup 
        ON market_data (symbol, timeframe, timestamp)
        """
        )

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Save DataFrame to database.
        Expects DataFrame index to be datetime.
        """
        if df.empty:
            return

        conn = self._get_connection()

        # Prepare data for insertion
        data_to_insert = df.copy()
        data_to_insert["symbol"] = symbol
        data_to_insert["timeframe"] = timeframe
        data_to_insert["timestamp"] = data_to_insert.index

        # Ensure columns exist and are in order
        cols = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]
        # Handle missing columns if any (e.g. if df only has close)
        for col in cols:
            if col not in data_to_insert.columns:
                data_to_insert[col] = None

        data_to_insert = data_to_insert[cols]

        # Use pandas to_sql with 'append'
        try:
            data_to_insert.to_sql("market_data", conn, if_exists="append", index=False, method="multi", chunksize=1000)
            logger.info(f"Saved {len(df)} rows for {symbol} {timeframe}")
        except sqlite3.IntegrityError:
            # If duplicates exist, we might want to update or ignore.
            # For now, let's try to insert row by row or use INSERT OR REPLACE via raw SQL if needed.
            # But pandas to_sql doesn't support INSERT OR REPLACE easily without custom method.
            # Let's stick to append and catch error, or maybe clean up first?
            # For this implementation, let's assume we are appending new data or filling history.
            # If we need upsert, we can implement a custom method.
            logger.warning(f"IntegrityError saving data for {symbol} {timeframe}. Some records might already exist.")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
        finally:
            conn.close()

    def load_data(
        self, symbol: str, timeframe: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from database."""
        conn = self._get_connection()

        query = "SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = ? AND timeframe = ?"
        params = [symbol, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        try:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])
            if not df.empty:
                df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
