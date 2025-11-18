"""
Multi-Timeframe Data Handler
============================
Maneja descarga, resampleo y filtros cross-TF para BTC con Alpaca API.

Funcionalidades:
- get_multi_tf_data(): Descarga 5Min/15Min/1H, resample, caching
- add_multi_tf_filters(): Interconexiones HTF/MTF (uptrend_1h, momentum_15m, vol_cross)
- add_noise(): Stress testing augmentation (+50% vol)
"""

from config.mtf_config import (
    ALPACA_CONFIG, MTF_CONFIG, TRADING_CONFIG,
    DATA_DIR
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class MultiTFDataHandler:
    """
    Descarga y procesa datos multi-timeframe desde Alpaca

    Critical: HTF (1h) SIEMPRE marca bias para longs/shorts
    """

    def __init__(self):
        """Initialize Alpaca API client"""
        self.api = tradeapi.REST(
            ALPACA_CONFIG['api_key'],
            ALPACA_CONFIG['secret_key'],
            ALPACA_CONFIG['base_url']
        )
        self.data_dir = DATA_DIR
        self.rate_delay = MTF_CONFIG['rate_delay']
        logger.info("✅ MultiTFDataHandler initialized")

    def get_multi_tf_data(
        self,
        symbol: str = 'BTCUSD',
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos para múltiples timeframes desde Alpaca

        Args:
            symbol: Trading pair (default: BTCUSD)
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD

        Returns:
            Dict: {'5min': df_5m, '15min': df_15m, '1h': df_1h}
        """
        start = start_date or TRADING_CONFIG['start_date']
        end = end_date or TRADING_CONFIG['end_date']

        logger.info(f"📊 Fetching multi-TF data for {symbol}: {start} to {end}")

        timeframes = MTF_CONFIG['timeframes']
        dfs = {}

        for tf_name, tf_value in timeframes.items():
            logger.info(f"  Downloading {tf_name} ({tf_value})...")

            # Check cache first
            cache_file = self.data_dir / f"{symbol}_{tf_value}_{start}_{end}.csv"

            if cache_file.exists():
                logger.info(f"  ✅ Loading from cache: {cache_file.name}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            else:
                # Download from Alpaca
                df = self._download_timeframe(symbol, tf_value, start, end)

                if df is not None and len(df) > 0:
                    # Save to cache
                    df.to_csv(cache_file)
                    logger.info(f"  💾 Saved to cache: {cache_file.name}")
                else:
                    logger.warning(f"  ⚠️ No data for {tf_value}")
                    continue

            # Validate data
            if self._validate_data(df, tf_value):
                dfs[tf_name] = df
                logger.info(f"  ✅ {tf_name}: {len(df)} bars loaded")

            # Rate limiting
            time.sleep(self.rate_delay)

        logger.info(f"✅ Multi-TF data loaded: {list(dfs.keys())}")
        return dfs

    def _download_timeframe(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Download data for single timeframe with retry logic"""
        # Map to Alpaca TimeFrame
        tf_map = MTF_CONFIG['alpaca_map']
        alpaca_tf = tf_map.get(timeframe, timeframe)

        # Parse timeframe
        if alpaca_tf == '5Min':
            tf_obj = TimeFrame(5, TimeFrameUnit.Minute)
        elif alpaca_tf == '15Min':
            tf_obj = TimeFrame(15, TimeFrameUnit.Minute)
        elif alpaca_tf == '1Hour':
            tf_obj = TimeFrame(1, TimeFrameUnit.Hour)
        else:
            logger.error(f"Unknown timeframe: {alpaca_tf}")
            return None

        # Convert dates
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        for attempt in range(max_retries):
            try:
                # Fetch from Alpaca
                bars = self.api.get_crypto_bars(
                    symbol,
                    timeframe=tf_obj,
                    start=start_dt.isoformat(),
                    end=end_dt.isoformat()
                ).df

                if bars.empty:
                    logger.warning(f"Empty data returned for {symbol} {timeframe}")
                    return None

                # Rename columns
                bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'TradeCount', 'VWAP']

                # Remove timezone for consistency
                if bars.index.tz is not None:
                    bars.index = bars.index.tz_localize(None)

                return bars

            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {symbol} {timeframe}")
                    return None

    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Validate downloaded data quality"""
        if df is None or len(df) == 0:
            return False

        # Check required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            logger.error(f"Missing required columns in {timeframe}")
            return False

        # Check for nulls
        if df[required].isnull().any().any():
            logger.warning(f"Null values found in {timeframe}, forward filling...")
            df = df.ffill()

        # Validate OHLC
        invalid = (df['High'] < df['Low']) | (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        if invalid.any():
            logger.warning(f"Invalid OHLC in {timeframe}: {invalid.sum()} bars, dropping...")
            df = df[~invalid]

        # Validate volume > 0
        if (df['Volume'] <= 0).any():
            logger.warning(f"Zero/negative volume in {timeframe}, filtering...")
            df = df[df['Volume'] > 0]

        return len(df) > 0

    def add_multi_tf_filters(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Añade filtros cross-TF: HTF trend, MTF momentum, vol interconectado

        Critical interconnections:
        - uptrend_1h: close_5m > EMA200_1h (resampled) - ALWAYS filters
        - momentum_15m: EMA20_5m > EMA50_15m (resampled)
        - vol_cross: vol_5m > SMA21_5m AND vol_5m > SMA_vol_1h
        - vp_proximity: price near POC_1h

        Args:
            dfs: Dict from get_multi_tf_data()

        Returns:
            Dict with added filter columns
        """
        logger.info("🔧 Adding multi-TF filters...")

        df_5m = dfs.get('entry').copy()
        df_15m = dfs.get('momentum').copy()
        df_1h = dfs.get('trend').copy()

        # ========================================================================
        # HTF TREND FILTER (1H) - ALWAYS BIAS
        # ========================================================================
        logger.info("  📈 HTF Trend Filter (1H EMA200)...")

        # Calculate EMA200 on 1h
        df_1h['EMA200'] = df_1h['Close'].ewm(span=200, adjust=False).mean()

        # Resample 1h to 5min (forward fill)
        df_1h_resampled = df_1h[['EMA200']].resample('5Min').ffill()
        df_1h_resampled.columns = ['EMA200_1h']

        # Merge to 5min
        df_5m = df_5m.join(df_1h_resampled, how='left')
        df_5m['EMA200_1h'] = df_5m['EMA200_1h'].ffill()

        # Define uptrend (CRITICAL FILTER)
        df_5m['uptrend_1h'] = df_5m['Close'] > df_5m['EMA200_1h']

        uptrend_pct = df_5m['uptrend_1h'].mean() * 100
        logger.info(f"    ✅ Uptrend 1h: {uptrend_pct:.1f}% of time")

        # ========================================================================
        # MTF MOMENTUM FILTER (15MIN EMA50)
        # ========================================================================
        logger.info("  ⚡ MTF Momentum Filter (15Min EMA50)...")

        # Calculate EMA20 on 5min
        df_5m['EMA20'] = df_5m['Close'].ewm(span=20, adjust=False).mean()

        # Calculate EMA50 on 15min
        df_15m['EMA50'] = df_15m['Close'].ewm(span=50, adjust=False).mean()

        # Resample 15min to 5min
        df_15m_resampled = df_15m[['EMA50']].resample('5Min').ffill()
        df_15m_resampled.columns = ['EMA50_15m']

        # Merge to 5min
        df_5m = df_5m.join(df_15m_resampled, how='left')
        df_5m['EMA50_15m'] = df_5m['EMA50_15m'].ffill()

        # Define momentum
        df_5m['momentum_15m'] = df_5m['EMA20'] > df_5m['EMA50_15m']

        momentum_pct = df_5m['momentum_15m'].mean() * 100
        logger.info(f"    ✅ Momentum 15m: {momentum_pct:.1f}% of time")

        # ========================================================================
        # VOL CROSS-TF FILTER
        # ========================================================================
        logger.info("  📊 Volume Cross-TF Filter...")

        # SMA21 on 5min volume
        df_5m['SMA_vol_21'] = df_5m['Volume'].rolling(21).mean()

        # SMA on 1h volume
        df_1h['SMA_vol'] = df_1h['Volume'].rolling(10).mean()

        # Resample 1h vol to 5min
        df_1h_vol_resampled = df_1h[['SMA_vol']].resample('5Min').ffill()
        df_1h_vol_resampled.columns = ['SMA_vol_1h']

        # Merge to 5min
        df_5m = df_5m.join(df_1h_vol_resampled, how='left')
        df_5m['SMA_vol_1h'] = df_5m['SMA_vol_1h'].ffill()

        # Vol cross filter
        vol_thresh = 1.2
        df_5m['high_vol_5m'] = df_5m['Volume'] > (vol_thresh * df_5m['SMA_vol_21'])
        df_5m['high_vol_cross'] = df_5m['Volume'] > df_5m['SMA_vol_1h']
        df_5m['vol_filter'] = df_5m['high_vol_5m'] & df_5m['high_vol_cross']

        vol_filter_pct = df_5m['vol_filter'].mean() * 100
        logger.info(f"    ✅ Vol filter: {vol_filter_pct:.1f}% of time")

        # ========================================================================
        # COMBINED FILTERS
        # ========================================================================
        # Bull filter: uptrend AND momentum AND vol
        df_5m['bull_filter'] = (
            df_5m['uptrend_1h'] &
            df_5m['momentum_15m'] &
            df_5m['vol_filter']
        )

        # Bear filter: NOT uptrend AND vol (no momentum required for shorts)
        df_5m['bear_filter'] = (
            ~df_5m['uptrend_1h'] &
            df_5m['vol_filter']
        )

        bull_pct = df_5m['bull_filter'].mean() * 100
        bear_pct = df_5m['bear_filter'].mean() * 100

        logger.info(f"✅ Combined filters: Bull {bull_pct:.1f}%, Bear {bear_pct:.1f}%")

        # Update dfs
        dfs['entry'] = df_5m
        dfs['momentum'] = df_15m
        dfs['trend'] = df_1h

        return dfs

    def add_noise(
        self,
        df: pd.DataFrame,
        vol_pct: float = 50,
        seed: int = None
    ) -> pd.DataFrame:
        """
        Add noise for stress testing (+50% volatility)

        Args:
            df: DataFrame to augment
            vol_pct: Volatility increase percentage (default: 50%)
            seed: Random seed

        Returns:
            DataFrame with augmented data
        """
        if seed:
            np.random.seed(seed)

        df_stressed = df.copy()

        # Calculate returns
        returns = df['Close'].pct_change()
        std_returns = returns.std()

        # Generate noise
        noise_factor = vol_pct / 100
        noise = np.random.normal(0, std_returns * noise_factor, len(df))

        # Apply to OHLC
        multiplier = 1 + noise
        df_stressed['Open'] = df['Open'] * multiplier
        df_stressed['High'] = df['High'] * multiplier
        df_stressed['Low'] = df['Low'] * multiplier
        df_stressed['Close'] = df['Close'] * multiplier

        # Validate OHLC
        df_stressed['High'] = df_stressed[['High', 'Close']].max(axis=1)
        df_stressed['Low'] = df_stressed[['Low', 'Close']].min(axis=1)

        # Volume noise (±30%)
        vol_noise = np.random.uniform(0.7, 1.3, len(df))
        df_stressed['Volume'] = df['Volume'] * vol_noise

        logger.info(f"✅ Added {vol_pct}% noise to data")

        return df_stressed


def example_usage():
    """Example of how to use MultiTFDataHandler"""
    print("=" * 80)
    print("Multi-TF Data Handler Example")
    print("=" * 80)

    # Initialize
    handler = MultiTFDataHandler()

    # Get multi-TF data
    dfs = handler.get_multi_tf_data(
        symbol='BTCUSD',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    if not dfs:
        print("❌ No data loaded")
        return

    print(f"\n📊 Data loaded:")
    for tf_name, df in dfs.items():
        print(f"  {tf_name}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Add multi-TF filters
    dfs_filtered = handler.add_multi_tf_filters(dfs)

    # Check results
    df_5m = dfs_filtered['entry']
    print(f"\n✅ Filters added to 5min data:")
    print(f"  uptrend_1h: {df_5m['uptrend_1h'].sum()} bars ({df_5m['uptrend_1h'].mean()*100:.1f}%)")
    print(
        f"  momentum_15m: {df_5m['momentum_15m'].sum()} bars ({df_5m['momentum_15m'].mean()*100:.1f}%)")
    print(
        f"  bull_filter: {df_5m['bull_filter'].sum()} bars ({df_5m['bull_filter'].mean()*100:.1f}%)")
    print(
        f"  bear_filter: {df_5m['bear_filter'].sum()} bars ({df_5m['bear_filter'].mean()*100:.1f}%)")

    # Stress test
    print(f"\n🔬 Creating stressed data (+50% vol)...")
    df_stressed = handler.add_noise(df_5m, vol_pct=50, seed=42)

    orig_std = df_5m['Close'].pct_change().std()
    stressed_std = df_stressed['Close'].pct_change().std()
    print(f"  Original volatility: {orig_std*100:.3f}%")
    print(f"  Stressed volatility: {stressed_std*100:.3f}%")
    print(f"  Increase: {(stressed_std/orig_std - 1)*100:.1f}%")

    print("\n✅ Example complete!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()
