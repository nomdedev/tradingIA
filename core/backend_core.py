import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import logging
import json
import datetime
import traceback
import pytz
from pathlib import Path
from .data.data_validator import DataValidator, validate_dataframe
from config.mtf_config import ALPACA_CONFIG


class DataManager:
    def __init__(self, api_key=None, secret_key=None, cache_dir='data/cache'):
        # Use provided credentials or fall back to config
        self.api_key = api_key or ALPACA_CONFIG['api_key']
        self.secret_key = secret_key or ALPACA_CONFIG['secret_key']
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, filename='../logs/platform.log',
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize Alpaca API if credentials are available
        if self.api_key and self.secret_key:
            try:
                self.api = tradeapi.REST(
                    self.api_key,
                    self.secret_key,
                    ALPACA_CONFIG['base_url'],
                    api_version='v2')
                self.logger.info("Alpaca API connection initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Alpaca API: {e}")
                self.api = None
        else:
            self.logger.warning("No Alpaca API credentials provided - data loading will be limited")

    def load_alpaca_data(
            self,
            symbol='BTC/USD',
            start_date='2020-01-01',
            end_date=None,
            timeframe='5Min'):
        try:
            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # Map timeframe to Alpaca format
            timeframe_map = {
                '5Min': '5Min',
                '15Min': '15Min',
                '1H': '1Hour',
                '1Hour': '1Hour',
                '4H': '4Hour',
                '1D': '1Day'
            }
            alpaca_timeframe = timeframe_map.get(timeframe, '5Min')

            # Create safe filename from symbol
            safe_symbol = symbol.replace('/', '_')
            cache_file = self.cache_dir / f"{safe_symbol}_{timeframe}.csv"

            # Try to load from cache first
            if cache_file.exists():
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    self.logger.info(f"Loaded {len(df)} bars from cache for {symbol}")
                    # Validate cache data
                    if self._validate_data(df):
                        return df
                    else:
                        self.logger.warning("Cached data invalid, fetching fresh data")
                except Exception as e:
                    self.logger.warning(f"Failed to load cache: {e}")

            # Fetch fresh data from Alpaca
            if self.api is None:
                raise Exception("Alpaca API not initialized")

            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)

            # Alpaca API call - use date only format
            bars = self.api.get_crypto_bars(
                symbol,
                alpaca_timeframe,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d')).df

            if bars.empty:
                raise Exception(f"No data returned for {symbol}")

            # Rename columns to standard format
            df = bars[['open', 'high', 'low', 'close', 'volume']].copy()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Calculate ATR and SMA Volume
            df['ATR'] = self._calculate_atr(df, 14)
            df['SMA_Vol'] = df['Volume'].rolling(21).mean()

            # Validate data
            if not self._validate_data(df):
                raise Exception("Fetched data validation failed")

            # Save to cache
            df.to_csv(cache_file)
            self.logger.info(f"Fetched and cached {len(df)} bars for {symbol}")

            return df

        except Exception as e:
            error_msg = f"Error loading data for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def resample_multi_tf(self, df_5m):
        try:
            # Resample to 15min
            df_15m = df_5m.resample('15Min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            # Resample to 1h
            df_1h = df_5m.resample('1h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            return {
                '5min': df_5m,
                '15min': df_15m,
                '1h': df_1h
            }

        except Exception as e:
            error_msg = f"Error resampling data: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def get_data_info(self):
        try:
            # Find all cache files
            cache_files = list(self.cache_dir.glob("*.csv"))
            if not cache_files:
                return {'error': 'No cache files found', 'status': 'EMPTY'}

            # Get info from the first file (assuming BTCUSD_5Min.csv exists)
            btc_file = self.cache_dir / "BTCUSD_5Min.csv"
            if btc_file.exists():
                df = pd.read_csv(btc_file, index_col=0, parse_dates=True)
                return {
                    'symbol': 'BTCUSD',
                    'n_bars': len(df),
                    'start_date': df.index.min().strftime('%Y-%m-%d') if not df.empty else 'N/A',
                    'end_date': df.index.max().strftime('%Y-%m-%d') if not df.empty else 'N/A',
                    'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'status': 'OK'
                }
            else:
                return {'error': 'BTCUSD_5Min.csv not found', 'status': 'NOT_FOUND'}
        except Exception as e:
            return {'error': str(e), 'traceback': traceback.format_exc()}

    def save_cache(self, df, symbol, timeframe):
        try:
            cache_file = self.cache_dir / f"{symbol}_{timeframe}.csv"
            df.to_csv(cache_file)
            self.logger.info(f"Saved cache for {symbol} {timeframe}")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def _calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _validate_data(self, df):
        """Comprehensive data validation using DataValidator"""
        if df is None or df.empty:
            self.logger.error("Data validation failed: DataFrame is None or empty")
            return False

        try:
            # Use the new DataValidator
            validator = DataValidator(strict_mode=False)

            # Map column names to expected format
            df_mapped = df.copy()
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }

            # Rename columns if needed
            df_mapped = df_mapped.rename(columns=column_mapping)

            # Add timestamp column if not present
            if 'timestamp' not in df_mapped.columns:
                if isinstance(df_mapped.index, pd.DatetimeIndex):
                    df_mapped['timestamp'] = df_mapped.index
                else:
                    # Assume data is in chronological order
                    df_mapped['timestamp'] = pd.date_range(
                        start='2020-01-01',
                        periods=len(df_mapped),
                        freq='5min',
                        tz='UTC'
                    )

            # Run validations
            results = validator.run_all_validations(df_mapped)

            # Check for critical errors
            critical_errors = [r for r in results if r.severity.value == 'critical']
            errors = [r for r in results if r.severity.value == 'error']

            if critical_errors:
                self.logger.error(f"Critical validation errors: {len(critical_errors)}")
                for error in critical_errors:
                    self.logger.error(f"  - {error.message}")
                return False

            if errors:
                self.logger.warning(f"Validation errors found: {len(errors)}")
                for error in errors:
                    self.logger.warning(f"  - {error.message}")
                # Don't fail on warnings/errors in non-strict mode
                return True

            # Log warnings
            warnings = [r for r in results if r.severity.value == 'warning']
            if warnings:
                self.logger.info(f"Validation warnings: {len(warnings)}")
                for warning in warnings:
                    self.logger.info(f"  - {warning.message}")

            self.logger.info("Data validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Data validation failed with exception: {e}")
            return False

    def validate_ohlc(self, df, auto_correct=False):
        """Validate OHLC relationships: High >= max(Open, Close), Low <= min(Open, Close)"""
        if df.empty:
            raise ValueError("Cannot validate empty DataFrame")

        # Check High >= Low
        invalid_hl = df[df['High'] < df['Low']]
        if not invalid_hl.empty:
            if auto_correct:
                self.logger.warning(f"Auto-correcting {len(invalid_hl)} High < Low violations")
                # Swap High and Low where High < Low
                df.loc[df['High'] < df['Low'], ['High', 'Low']
                       ] = df.loc[df['High'] < df['Low'], ['Low', 'High']].values
            else:
                raise ValueError(f"High must be >= Low. Found {len(invalid_hl)} invalid bars")

        # Check Close within [Low, High]
        invalid_close_high = df[df['Close'] > df['High']]
        if not invalid_close_high.empty:
            if auto_correct:
                self.logger.warning(
                    f"Auto-correcting {len(invalid_close_high)} Close > High violations")
                df.loc[df['Close'] > df['High'], 'High'] = df.loc[df['Close'] > df['High'], 'Close']
            else:
                raise ValueError(
                    f"Close must be <= High. Found {len(invalid_close_high)} invalid bars")

        invalid_close_low = df[df['Close'] < df['Low']]
        if not invalid_close_low.empty:
            if auto_correct:
                self.logger.warning(
                    f"Auto-correcting {len(invalid_close_low)} Close < Low violations")
                df.loc[df['Close'] < df['Low'], 'Low'] = df.loc[df['Close'] < df['Low'], 'Close']
            else:
                raise ValueError(
                    f"Close must be >= Low. Found {len(invalid_close_low)} invalid bars")

        # Check Open within [Low, High]
        invalid_open_high = df[df['Open'] > df['High']]
        if not invalid_open_high.empty:
            if auto_correct:
                self.logger.warning(
                    f"Auto-correcting {len(invalid_open_high)} Open > High violations")
                df.loc[df['Open'] > df['High'], 'High'] = df.loc[df['Open'] > df['High'], 'Open']
            else:
                self.logger.warning(
                    f"Open > High in {len(invalid_open_high)} bars, auto-correcting")
                df.loc[df['Open'] > df['High'], 'High'] = df.loc[df['Open'] > df['High'], 'Open']

        invalid_open_low = df[df['Open'] < df['Low']]
        if not invalid_open_low.empty:
            if auto_correct:
                self.logger.warning(
                    f"Auto-correcting {len(invalid_open_low)} Open < Low violations")
                df.loc[df['Open'] < df['Low'], 'Low'] = df.loc[df['Open'] < df['Low'], 'Open']
            else:
                self.logger.warning(f"Open < Low in {len(invalid_open_low)} bars, auto-correcting")
                df.loc[df['Open'] < df['Low'], 'Low'] = df.loc[df['Open'] < df['Low'], 'Open']

        return df

    def validate_volume(self, df):
        """Validate volume is positive"""
        if df.empty:
            raise ValueError("Cannot validate empty DataFrame")

        # Check for negative volume
        negative_vol = df[df['Volume'] < 0]
        if not negative_vol.empty:
            raise ValueError(f"Negative volume detected in {len(negative_vol)} bars")

        # Warn about zero volume
        zero_vol = df[df['Volume'] == 0]
        if not zero_vol.empty:
            self.logger.warning(
                f"Zero volume detected in {len(zero_vol)} bars ({len(zero_vol)/len(df)*100:.1f}%)")

        return True

    def validate_no_future_data(self, df, tolerance_minutes=5):
        """Detect future data that would cause look-ahead bias"""
        if df.empty:
            return True

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")

        now = pd.Timestamp.now(tz=pytz.UTC)

        # Convert index to UTC if not already
        if df.index.tz is None:
            df_index_utc = df.index.tz_localize('UTC')
        else:
            df_index_utc = df.index.tz_convert('UTC')

        # Allow tolerance for API delays
        tolerance = pd.Timedelta(minutes=tolerance_minutes)
        future_data = df_index_utc[df_index_utc > (now + tolerance)]

        if not future_data.empty:
            raise ValueError(
                f"Future data detected: {len(future_data)} bars beyond {now + tolerance}")

        return True

    def normalize_timezone(self, df, target_tz='UTC', assume_utc=False):
        """Normalize all timestamps to UTC"""
        if df.empty:
            return df

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                # Keep the Date column and set it as index temporarily for processing
                date_col = df['Date'].copy()
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")

        # Check if index has timezone
        if df.index.tz is None:
            if assume_utc:
                self.logger.info("Assuming naive datetime is UTC")
                df.index = df.index.tz_localize('UTC')
            else:
                import warnings
                warnings.warn("Naive datetime detected, assuming UTC", UserWarning)
                self.logger.warning("Naive datetime detected, assuming UTC")
                df.index = df.index.tz_localize('UTC')
        elif df.index.tz != pytz.UTC:
            self.logger.info(f"Converting from {df.index.tz} to UTC")
            df.index = df.index.tz_convert('UTC')

        # If we had a Date column originally, restore it
        if 'date_col' in locals():
            df['Date'] = df.index
            df = df.reset_index(drop=True)

        return df

    def detect_data_gaps(self, df, expected_freq='5min', ignore_weekends=False):
        """Detect gaps in time series data"""
        if df.empty or len(df) < 2:
            return []

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")

        gaps = []
        expected_delta = pd.Timedelta(expected_freq)

        for i in range(1, len(df)):
            actual_delta = df.index[i] - df.index[i - 1]

            # Skip weekend gaps if ignore_weekends is True
            if ignore_weekends:
                # Check if gap spans weekend (Friday evening to Monday morning)
                day1 = df.index[i - 1].weekday()  # 0=Monday, 4=Friday
                day2 = df.index[i].weekday()
                if day1 == 4 and day2 == 0:  # Friday to Monday
                    continue

            if actual_delta > expected_delta * 1.5:  # 50% tolerance
                gaps.append({
                    'start': df.index[i - 1],
                    'end': df.index[i],
                    'duration_minutes': actual_delta.total_seconds() / 60
                })

        if gaps:
            self.logger.warning(f"Detected {len(gaps)} data gaps")

        return gaps

    def handle_gaps(self, df, method='raise', max_gap_hours=24, expected_freq='5min'):
        """Handle gaps in data"""
        gaps = self.detect_data_gaps(df, expected_freq)

        if not gaps:
            return df

        # Check for large gaps
        large_gaps = [g for g in gaps if g['duration_minutes'] > max_gap_hours * 60]
        if large_gaps and method == 'raise':
            gap_duration = large_gaps[0]['duration_minutes'] / 60
            raise ValueError(f"Data gap exceeds maximum {max_gap_hours}h: {gap_duration:.1f}h gap")

        if method == 'ffill':
            # Forward fill to create complete time series
            full_index = pd.date_range(df.index[0], df.index[-1], freq=expected_freq)
            df = df.reindex(full_index, method='ffill')
            self.logger.info(f"Forward filled {len(gaps)} gaps")

        return df

    def detect_duplicate_timestamps(self, df):
        """Detect duplicate timestamps"""
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")

        duplicates = df.index[df.index.duplicated()].unique()
        if len(duplicates) > 0:
            self.logger.warning(f"Found {len(duplicates)} duplicate timestamps")
        return duplicates

    def remove_duplicate_timestamps(self, df, keep='first'):
        """Remove duplicate timestamps"""
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")

        if df.index.duplicated().any():
            original_len = len(df)
            if keep == 'average':
                # Group by index and take mean
                df = df.groupby(df.index).mean()
            else:
                df = df[~df.index.duplicated(keep=keep)]

            removed = original_len - len(df)
            self.logger.info(f"Removed {removed} duplicate timestamps (keep={keep})")

        return df

    def detect_zero_volume(self, df):
        """Detect bars with zero volume"""
        if df.empty:
            return []

        zero_volume_mask = df['Volume'] == 0
        zero_volume_indices = df.index[zero_volume_mask].tolist()

        if zero_volume_indices:
            self.logger.warning(f"Found {len(zero_volume_indices)} bars with zero volume")

        return zero_volume_indices

    def handle_zero_volume(self, df, method='interpolate'):
        """Handle zero volume bars"""
        if df.empty:
            return df

        zero_volume_mask = df['Volume'] == 0

        if not zero_volume_mask.any():
            return df

        if method == 'interpolate':
            # Interpolate volume using linear interpolation
            df['Volume'] = df['Volume'].replace(0, np.nan).interpolate(method='linear')
            # Fill any remaining NaN with forward fill
            df['Volume'] = df['Volume'].fillna(method='ffill')
            self.logger.info(f"Interpolated {zero_volume_mask.sum()} zero volume bars")

        return df

    def validate_timezone_consistency(self, df):
        """Validate that all timestamps are in the same timezone"""
        if df.empty:
            return True

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")

        # Check if all timestamps have the same timezone
        timezones = df.index.map(lambda x: x.tz if hasattr(x, 'tz') else None).unique()

        if len(timezones) > 1:
            raise ValueError(f"Mixed timezones detected: {timezones}")

        return True

    def calculate_sma_chunked(self, df, period=200, chunk_size=50000):
        """Calculate SMA on large datasets in chunks to avoid memory issues"""
        if df.empty:
            return df

        # For small datasets, calculate normally
        if len(df) <= chunk_size:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            return df

        # For large datasets, process in chunks
        result_chunks = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()

            # Calculate SMA for this chunk
            chunk[f'SMA_{period}'] = chunk['Close'].rolling(window=period).mean()

            # For overlapping periods, we need to handle the edges
            if i > 0 and period > 1:
                # Get previous chunk's last values for proper calculation
                prev_chunk_end = i
                prev_chunk_start = max(0, prev_chunk_end - period + 1)
                prev_values = df.iloc[prev_chunk_start:prev_chunk_end]['Close']

                # Recalculate SMA for the first few bars of this chunk
                for j in range(min(period - 1, len(chunk))):
                    window_data = pd.concat(
                        [prev_values.iloc[-(period - j - 1):], chunk.iloc[:j + 1]['Close']])
                    if len(window_data) >= period:
                        chunk.iloc[j, chunk.columns.get_loc(f'SMA_{period}')] = window_data.mean()

            result_chunks.append(chunk)

        # Combine chunks
        result_df = pd.concat(result_chunks)
        self.logger.info(
            f"Calculated SMA_{period} on {len(df)} bars in {len(result_chunks)} chunks")

        return result_df

    def validate_complete(self, df):
        """Run complete validation pipeline"""
        if df is None or df.empty:
            raise ValueError("Cannot validate empty DataFrame")

        # 1. Basic structure validation
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")

        # 2. OHLC validation
        self.validate_ohlc(df)

        # 3. Volume validation
        self.validate_volume(df)

        # 4. Future data validation
        self.validate_no_future_data(df)

        # 5. Timezone normalization
        df = self.normalize_timezone(df)

        # 6. Duplicate timestamps
        duplicates = self.detect_duplicate_timestamps(df)
        if len(duplicates) > 0:
            df = self.remove_duplicate_timestamps(df)

        # 7. Data gaps
        gaps = self.detect_data_gaps(df)
        if gaps:
            df = self.handle_gaps(df, method='ffill')

        # 8. Zero volume handling
        zero_vol_indices = self.detect_zero_volume(df)
        if zero_vol_indices:
            df = self.handle_zero_volume(df)

        return df

    def process_large_dataset(self, data, chunk_size=100000):
        """Process large datasets in chunks to avoid memory issues"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        if len(data) <= chunk_size:
            # Small enough, return as-is
            return data

        # Process in chunks
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)

        # Recombine chunks
        result = pd.concat(chunks, ignore_index=False)
        self.logger.info(f"Processed {len(data)} rows in {len(chunks)} chunks of size {chunk_size}")

        return result


class StrategyEngine:
    def __init__(self, strategies_config_file='config/strategies_registry.json'):
        self.strategies_config_file = strategies_config_file
        self.strategies_config = {}
        self.logger = logging.getLogger(__name__)

        try:
            with open(strategies_config_file, 'r') as f:
                config_data = json.load(f)
                
                # Si el config es una lista, convertirla a diccionario
                if isinstance(config_data, list):
                    self.strategies_config = {item['name']: item for item in config_data}
                else:
                    self.strategies_config = config_data
                    
            self.logger.info("Strategies config loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load strategies config: {e}")
            # Create default config
            self._create_default_config()

    def list_available_strategies(self):
        return list(self.strategies_config.keys())

    def get_strategy_params(self, strategy_name):
        if strategy_name not in self.strategies_config:
            return {'error': f'Strategy {strategy_name} not found'}

        params = self.strategies_config[strategy_name]['params']
        return params

    def validate_params(self, strategy_name, params_dict):
        """Enhanced parameter validation with dependency checking"""
        if strategy_name not in self.strategies_config:
            return False, f'Strategy {strategy_name} not found'

        config_params = self.strategies_config[strategy_name]['params']

        # 1. Check all required parameters are present
        for param_name, param_config in config_params.items():
            if param_name not in params_dict:
                return False, f'Missing parameter: {param_name}'

            value = params_dict[param_name]
            min_val = param_config.get('min', float('-inf'))
            max_val = param_config.get('max', float('inf'))

            # 2. Validate parameter bounds
            if not min_val <= value <= max_val:
                range_msg = (f'Parameter {param_name} out of range: {value} '
                           f'not in [{min_val}, {max_val}]')
                return False, range_msg

        # 3. Validate parameter dependencies
        dependency_check = self._validate_param_dependencies(strategy_name, params_dict)
        if not dependency_check[0]:
            return dependency_check

        # 4. Validate indicator data sufficiency
        sufficiency_check = self._validate_indicator_sufficiency(strategy_name, params_dict)
        if not sufficiency_check[0]:
            return sufficiency_check

        return True, 'Parameters valid'

    def _validate_param_dependencies(self, strategy_name, params_dict):
        """Validate parameter dependencies (e.g., fast < slow for MA crossover)"""

        # MACD strategy: fast must be < slow
        if strategy_name == 'MACD_ADX':
            if 'macd_fast' in params_dict and 'macd_slow' in params_dict:
                fast = params_dict['macd_fast']
                slow = params_dict['macd_slow']
                if fast >= slow:
                    return False, f"MACD fast period ({fast}) must be < slow period ({slow})"

        # Pairs trading: lookback must be >= entry_threshold
        if strategy_name == 'Pairs':
            if 'lookback' in params_dict and 'entry_threshold' in params_dict:
                lookback = params_dict['lookback']
                if lookback < 20:
                    return False, (f"Pairs lookback ({lookback}) must be >= 20 "
                                  "for stable spread calculation")

        # LSTM: sequence_length must be < prediction_horizon
        if strategy_name == 'LSTM_ML':
            if 'sequence_length' in params_dict and 'prediction_horizon' in params_dict:
                pred_horizon = params_dict['prediction_horizon']
                seq_length = params_dict['sequence_length']
                if pred_horizon > seq_length:
                    return False, (f"LSTM prediction_horizon ({pred_horizon}) must be <= "
                                  f"sequence_length ({seq_length})")

        return True, 'Dependencies valid'

    def _validate_indicator_sufficiency(self, strategy_name, params_dict):
        """Validate that strategy has enough data for indicators"""

        # Calculate minimum bars needed for each strategy
        min_bars_needed = 0

        if strategy_name == 'IBS_BB':
            bb_length = params_dict.get('bb_length', 20)
            min_bars_needed = bb_length + 10  # BB needs length + buffer

        elif strategy_name == 'MACD_ADX':
            macd_slow = params_dict.get('macd_slow', 26)
            adx_period = params_dict.get('adx_period', 14)
            macd_signal = 9  # Standard MACD signal period
            min_bars_needed = max(macd_slow + macd_signal, adx_period * 2) + 10

        elif strategy_name == 'Pairs':
            lookback = params_dict.get('lookback', 100)
            min_bars_needed = lookback + 20

        elif strategy_name == 'HFT_VMA':
            vma_period = params_dict.get('vma_period', 20)
            min_bars_needed = vma_period + 10

        elif strategy_name == 'LSTM_ML':
            sequence_length = params_dict.get('sequence_length', 60)
            min_bars_needed = sequence_length + 20

        # Store minimum requirement in params_dict for later validation
        params_dict['_min_bars_required'] = min_bars_needed

        return True, f'Minimum {min_bars_needed} bars required'

    def check_preset_collision(self, strategy_name, params_dict, existing_presets):
        """Check if parameters match an existing preset"""
        for preset_name, preset_params in existing_presets.items():
            if preset_params == params_dict:
                return True, f"Parameters match existing preset: {preset_name}"

        return False, "No preset collision"

    def load_strategy_module(self, strategy_name):
        if strategy_name not in self.strategies_config:
            return {'error': f'Strategy {strategy_name} not found'}

        try:
            module_path = self.strategies_config[strategy_name]['module']
            class_name = self.strategies_config[strategy_name]['class']

            # Dynamic import
            import importlib
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)

            return strategy_class

        except Exception as e:
            error_msg = f"Failed to load strategy {strategy_name}: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def _create_default_config(self):
        self.strategies_config = {
            "IBS_BB": {
                "module": "mean_reversion_ibs_bb_crypto",
                "class": "IBSBBStrategy",
                "params": {
                    "bb_length": {"default": 20, "min": 10, "max": 50, "step": 1, "description": "Bollinger Bands length"},
                    "vol_mult": {"default": 1.2, "min": 0.8, "max": 2.0, "step": 0.1, "description": "Volume multiplier for IBS filter"}
                }
            },
            "MACD_ADX": {
                "module": "momentum_macd_adx_stocks",
                "class": "MACDADXStrategy",
                "params": {
                    "macd_fast": {"default": 12, "min": 8, "max": 20, "step": 1, "description": "MACD fast period"},
                    "macd_slow": {"default": 26, "min": 20, "max": 40, "step": 1, "description": "MACD slow period"},
                    "adx_period": {"default": 14, "min": 10, "max": 25, "step": 1, "description": "ADX period"}
                }
            },
            "Pairs": {
                "module": "pairs_trading_crypto",
                "class": "PairsTradingStrategy",
                "params": {
                    "lookback": {"default": 100, "min": 50, "max": 200, "step": 10, "description": "Lookback period for spread calculation"},
                    "entry_threshold": {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1, "description": "Entry threshold in standard deviations"}
                }
            },
            "HFT_VMA": {
                "module": "hft_momentum_vma_forex",
                "class": "HFTVMAForexStrategy",
                "params": {
                    "vma_period": {"default": 20, "min": 10, "max": 50, "step": 1, "description": "Volume Moving Average period"},
                    "momentum_period": {"default": 5, "min": 3, "max": 15, "step": 1, "description": "Momentum calculation period"}
                }
            },
            "LSTM_ML": {
                "module": "lstm_ml_reversion_commodities",
                "class": "LSTMMLReversionStrategy",
                "params": {
                    "sequence_length": {"default": 60, "min": 30, "max": 120, "step": 10, "description": "LSTM sequence length"},
                    "prediction_horizon": {"default": 5, "min": 1, "max": 10, "step": 1, "description": "Prediction horizon in bars"}
                }
            }
        }

        # Save default config
        try:
            with open(self.strategies_config_file, 'w') as f:
                json.dump(self.strategies_config, f, indent=2)
            self.logger.info("Default strategies config created")
        except Exception as e:
            self.logger.error(f"Failed to save default config: {e}")
