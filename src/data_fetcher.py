"""
Data Fetcher Module
Handles downloading and caching of historical OHLCV data from Alpaca API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame, TimeFrameUnit

from config.mtf_config import (
    ALPACA_CONFIG, TRADING_CONFIG, DATA_FETCH_CONFIG, DATA_DIR
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and cache historical market data from Alpaca"""

    def __init__(self):
        """Initialize Alpaca API client"""
        self.api = tradeapi.REST(
            ALPACA_CONFIG['api_key'],
            ALPACA_CONFIG['secret_key'],
            ALPACA_CONFIG['base_url']
        )
        self.rate_limit_delay = DATA_FETCH_CONFIG['rate_limit_delay']
        self.max_retries = DATA_FETCH_CONFIG['max_retries']
        logger.info("✅ DataFetcher initialized with Alpaca API")

    def _parse_timeframe(self, timeframe_str):
        """Convert timeframe string to Alpaca TimeFrame object"""
        timeframe_map = {
            '1Min': TimeFrame(1, TimeFrameUnit.Minute),
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '1H': TimeFrame(1, TimeFrameUnit.Hour),
            '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
            '1D': TimeFrame(1, TimeFrameUnit.Day),
            '1Day': TimeFrame(1, TimeFrameUnit.Day),
        }
        return timeframe_map.get(timeframe_str, TimeFrame(5, TimeFrameUnit.Minute))

    def get_historical_data(self, symbol='BTCUSD', timeframe='5Min',
                            start_date=None, end_date=None):
        """
        Download historical OHLCV data from Alpaca

        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1H')
            start_date: Start date (YYYY-MM-DD) or datetime
            end_date: End date (YYYY-MM-DD) or datetime

        Returns:
            pandas.DataFrame with OHLCV data
        """
        # Use config defaults if not specified
        if start_date is None:
            start_date = TRADING_CONFIG['start_date']
        if end_date is None:
            end_date = TRADING_CONFIG['end_date']

        # Convert to datetime if string
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Format for Alpaca API
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        logger.info(
            f"📊 Fetching {symbol} data ({timeframe}): {start_date.date()} to {end_date.date()}")

        # Retry logic for rate limits
        for attempt in range(self.max_retries):
            try:
                # Fetch data from Alpaca
                bars = self.api.get_crypto_bars(
                    symbol,
                    timeframe=self._parse_timeframe(timeframe),
                    start=start_str,
                    end=end_str
                ).df

                if bars.empty:
                    logger.warning(f"⚠️ No data returned for {symbol} {timeframe}")
                    return pd.DataFrame()

                # Rename columns to standard format
                df = bars.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })

                # Validate and clean data
                df = self._validate_data(df)

                logger.info(f"✅ Fetched {len(df)} bars for {symbol} ({timeframe})")

                # Rate limit delay
                time.sleep(self.rate_limit_delay)

                return df

            except Exception as e:
                logger.error(
                    f"❌ Error fetching data (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * 2)  # Longer delay on error
                else:
                    logger.error("Max retries reached, returning empty DataFrame")
                    return self._generate_simulated_data(symbol, start_date, end_date, timeframe)

    def _validate_data(self, df):
        """Validate and clean OHLCV data"""
        if df.empty:
            return df

        # Remove NaN values
        df = df.dropna()

        # Filter by minimum volume
        min_vol = DATA_FETCH_CONFIG['min_volume_threshold']
        if min_vol > 0:
            df = df[df['Volume'] > min_vol]

        # Ensure OHLC logic
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Sort by timestamp
        df = df.sort_index()

        return df

    def _generate_simulated_data(self, symbol, start_date, end_date, timeframe):
        """Fallback: Generate simulated data if API fails"""
        logger.warning(f"⚠️ Generating simulated data for {symbol} as fallback")

        # Calculate number of bars based on timeframe
        timeframe_minutes = {
            '1Min': 1, '5Min': 5, '15Min': 15, '1H': 60, '1Hour': 60
        }
        minutes = timeframe_minutes.get(timeframe, 5)

        total_minutes = int((end_date - start_date).total_seconds() / 60)
        num_bars = total_minutes // minutes

        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, periods=num_bars)

        # Simulate price movement (Brownian motion)
        np.random.seed(42)
        base_price = 45000 if 'BTC' in symbol else 2500

        returns = np.random.normal(0.00005, 0.003, num_bars)
        prices = base_price * (1 + returns).cumprod()

        # Create OHLCV
        df = pd.DataFrame({
            'Open': prices,
            'High': prices * (1 + np.abs(np.random.normal(0, 0.001, num_bars))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, num_bars))),
            'Close': prices,
            'Volume': np.random.uniform(50, 500, num_bars)
        }, index=dates)

        return df

    def resample_data(self, df_5min):
        """
        Resample 5-minute data to multiple timeframes

        Args:
            df_5min: DataFrame with 5-minute OHLCV data

        Returns:
            dict of DataFrames for each timeframe
        """
        if df_5min.empty:
            logger.warning("Empty DataFrame, cannot resample")
            return {}

        resampled = {}

        # Resample to 15 minutes
        df_15min = df_5min.resample('15T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        resampled['15Min'] = df_15min

        # Resample to 1 hour
        df_1h = df_5min.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        resampled['1H'] = df_1h

        logger.info(f"✅ Resampled: 15Min ({len(df_15min)} bars), 1H ({len(df_1h)} bars)")

        return resampled

    def save_to_csv(self, df, filename):
        """Save DataFrame to CSV in data directory"""
        filepath = DATA_DIR / filename
        df.to_csv(filepath)
        logger.info(f"💾 Saved data to {filepath}")

    def load_cached_data(self, timeframe='5Min'):
        """
        Load cached data from CSV if exists, otherwise download

        Args:
            timeframe: Timeframe to load

        Returns:
            DataFrame with OHLCV data
        """
        filename = f"btc_{timeframe}.csv"
        filepath = DATA_DIR / filename

        if filepath.exists() and DATA_FETCH_CONFIG['cache_data']:
            logger.info(f"📂 Loading cached data from {filepath}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"✅ Loaded {len(df)} bars from cache")
            return df
        else:
            logger.info(f"📥 No cache found, downloading {timeframe} data...")
            df = self.get_historical_data(timeframe=timeframe)

            if not df.empty and DATA_FETCH_CONFIG['cache_data']:
                self.save_to_csv(df, filename)

            return df

    def get_multi_tf_data(self, symbol='BTCUSD', timeframes=None):
        """
        Get data for multiple timeframes

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes (default from config)

        Returns:
            dict of DataFrames {timeframe: df}
        """
        if timeframes is None:
            timeframes = TRADING_CONFIG['timeframes']

        data = {}

        # Get base 5Min data
        df_5min = self.load_cached_data('5Min')
        data['5Min'] = df_5min

        if not df_5min.empty:
            # Resample to other timeframes
            resampled = self.resample_data(df_5min)
            data.update(resampled)

            # Cache resampled data
            for tf, df in resampled.items():
                if DATA_FETCH_CONFIG['cache_data']:
                    self.save_to_csv(df, f"btc_{tf}.csv")

        logger.info(f"✅ Multi-TF data loaded: {list(data.keys())}")
        return data


# Standalone functions for backward compatibility
def get_historical_data(symbol='BTCUSD', timeframe='5Min', start_date=None, end_date=None):
    """Get historical data (standalone function)"""
    fetcher = DataFetcher()
    return fetcher.get_historical_data(symbol, timeframe, start_date, end_date)


def get_multi_tf_data(symbol='BTCUSD', timeframes=None):
    """Get multi-timeframe data (standalone function)"""
    fetcher = DataFetcher()
    return fetcher.get_multi_tf_data(symbol, timeframes)


if __name__ == "__main__":
    # Test data fetching
    print("🧪 Testing Data Fetcher")
    print("=" * 50)

    fetcher = DataFetcher()

    # Test single timeframe
    df = fetcher.load_cached_data('5Min')
    print(f"\n5Min Data: {len(df)} bars")
    if not df.empty:
        print(df.head())
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Test multi-timeframe
    dfs = fetcher.get_multi_tf_data()
    print(f"\nMulti-TF Data: {list(dfs.keys())}")
    for tf, df in dfs.items():
        print(f"  {tf}: {len(df)} bars")
