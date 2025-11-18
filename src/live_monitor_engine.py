import threading
import time
import logging
import alpaca_trade_api as tradeapi
from PySide6.QtCore import QObject, Signal


class LiveMonitorEngine(QObject):
    # Signals for GUI updates
    # {'timestamp': datetime, 'signal_type': str, 'price': float, 'score': int}
    signal_detected = Signal(dict)
    pnl_updated = Signal(float)  # current PnL value
    metrics_updated = Signal(dict)  # live metrics dict
    status_changed = Signal(str)  # status updates for GUI

    def __init__(self, alpaca_api_key, alpaca_secret_key):
        super().__init__()
        self.api_key = alpaca_api_key
        self.secret_key = alpaca_secret_key

        # Initialize API only if keys are provided
        if alpaca_api_key and alpaca_secret_key:
            self.api = self._initialize_api_with_retry(alpaca_api_key, alpaca_secret_key)
        else:
            self.api = None  # Demo mode without API

        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        self._api_call_timestamps = []  # For rate limiting
        self._max_calls_per_minute = 200
        logging.basicConfig(filename='../logs/live_monitor.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def _initialize_api_with_retry(self, api_key, secret_key, max_retries=3):
        """Initialize Alpaca API with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                api = tradeapi.REST(
                    api_key,
                    secret_key,
                    'https://paper-api.alpaca.markets',
                    api_version='v2')
                # Test connection
                api.get_account()
                self.logger.info(f"Alpaca API initialized successfully on attempt {attempt + 1}")
                return api
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                self.logger.warning(
                    f"API initialization failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("API initialization failed after all retries")
                    return None

    def _reconnect_api(self):
        """Reconnect to Alpaca API if connection is lost"""
        if not self.api_key or not self.secret_key:
            return False

        self.logger.info("Attempting to reconnect to Alpaca API...")
        self.api = self._initialize_api_with_retry(self.api_key, self.secret_key)
        return self.api is not None

    def _rate_limit_check(self):
        """Check if we're within rate limits (200 calls/minute)"""
        now = time.time()

        # Remove timestamps older than 1 minute
        self._api_call_timestamps = [ts for ts in self._api_call_timestamps if now - ts < 60]

        if len(self._api_call_timestamps) >= self._max_calls_per_minute:
            # Calculate wait time until oldest call expires
            oldest_call = self._api_call_timestamps[0]
            wait_time = 60 - (now - oldest_call) + 0.1  # Add 100ms buffer
            self.logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            # Clean up again after waiting
            self._api_call_timestamps = [
                ts for ts in self._api_call_timestamps if time.time() - ts < 60]

        # Record this call
        self._api_call_timestamps.append(time.time())

    def _api_call_with_retry(self, api_func, *args, max_retries=3, **kwargs):
        """Execute API call with rate limiting and retry logic"""
        for attempt in range(max_retries):
            try:
                # Check rate limit before call
                self._rate_limit_check()

                # Make the API call
                result = api_func(*args, **kwargs)
                return result

            except Exception as e:
                error_str = str(e).lower()

                # Check if connection error
                if any(keyword in error_str for keyword in ['connection', 'timeout', 'network']):
                    self.logger.warning(
                        f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")

                    # Try to reconnect
                    if self._reconnect_api():
                        wait_time = 2 ** attempt
                        self.logger.info(f"Reconnected, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.error("Reconnection failed")
                        raise

                # Check if rate limit error
                elif 'rate limit' in error_str or '429' in error_str:
                    wait_time = 60  # Wait 1 minute for rate limit
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    self._api_call_timestamps.clear()  # Reset rate limit tracker
                    continue

                # Other errors - retry with backoff
                else:
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        raise

        raise Exception(f"API call failed after {max_retries} attempts")

    def monitor_signals(self, strategy_engine, data_manager, interval_sec=300):
        """Start monitoring for signals in a separate thread"""
        if self.monitoring:
            self.logger.warning("Monitoring already running")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop,
                                               args=(strategy_engine, data_manager, interval_sec))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info(f"Started signal monitoring with {interval_sec}s interval")

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            if self.monitor_thread.is_alive():
                self.logger.warning("Monitor thread did not stop cleanly")
        self.logger.info("Stopped signal monitoring")

    def _monitor_loop(self, strategy_engine, data_manager, interval_sec):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Fetch latest data
                df_latest = data_manager.load_alpaca_data(symbol='BTCUSD', timeframe='5Min',
                                                          start_date='2024-01-01')  # Last few days

                if df_latest is not None and not df_latest.empty:
                    # Get latest bar
                    # latest_bar = df_latest.iloc[-1]  # Removed unused variable

                    # Generate signals using strategy engine
                    signals = strategy_engine.generate_signals(df_latest)

                    if signals and len(signals) > 0:
                        latest_signal = signals[-1]  # Most recent signal
                        self.signal_detected.emit(latest_signal)
                        self.logger.info(f"Signal detected: {latest_signal}")

                    # Update PnL
                    pnl = self.get_live_pnl()
                    self.pnl_updated.emit(pnl)

                    # Update metrics (simplified)
                    metrics = self._calculate_live_metrics(df_latest)
                    self.metrics_updated.emit(metrics)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

            time.sleep(interval_sec)

    def manual_entry(self, symbol, qty, side, stop_loss=None, take_profit=None):
        """Place a manual market order"""
        if not self.api:
            self.logger.error("API not initialized - cannot place order")
            return None

        try:
            order = self._api_call_with_retry(
                self.api.submit_order,
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )

            # Set stop loss and take profit if provided
            if stop_loss:
                self._api_call_with_retry(
                    self.api.submit_order,
                    symbol=symbol,
                    qty=qty,
                    side='sell' if side == 'buy' else 'buy',
                    type='stop',
                    stop_price=stop_loss,
                    time_in_force='gtc'
                )

            if take_profit:
                self._api_call_with_retry(
                    self.api.submit_order,
                    symbol=symbol,
                    qty=qty,
                    side='sell' if side == 'buy' else 'buy',
                    type='limit',
                    limit_price=take_profit,
                    time_in_force='gtc'
                )

            if order:
                self.logger.info(f"Manual order placed: {order.id}")
                return order.id
            else:
                self.logger.error("Failed to place manual order")
                return None

        except Exception as e:
            self.logger.error(f"Failed to place manual order: {e}")
            return None

    def close_position(self, position_id):
        """Close a position"""
        if not self.api:
            self.logger.error("API not initialized - cannot close position")
            return None

        try:
            # Close position with retry logic
            order = self._api_call_with_retry(self.api.close_position, position_id)
            self.logger.info(f"Position closed: {position_id}")
            return order.id

        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            return None

    def get_live_pnl(self):
        """Get current PnL from Alpaca"""
        if not self.api:
            return 0.0

        try:
            account = self._api_call_with_retry(self.api.get_account)
            return float(account.equity) - float(account.last_equity)
        except Exception as e:
            self.logger.error(f"Failed to get PnL: {e}")
            return 0.0

    def _calculate_live_metrics(self, df):
        """Calculate basic live metrics"""
        try:
            returns = df['close'].pct_change().dropna()
            
            # Sharpe con risk-free rate (5-min data)
            rf_per_period = 0.04 / (252 * 24 * 12)  # Risk-free rate por período 5-min
            excess_returns = returns - rf_per_period
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252 * 24 * 12) if excess_returns.std() > 0 else 0.0
            
            win_rate = (returns > 0).mean()
            max_dd = (df['close'] / df['close'].expanding().max() - 1).min()

            return {
                'sharpe': sharpe,
                'win_rate': win_rate,
                'max_dd': max_dd,
                'trades_today': 0  # Placeholder
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {e}")
            return {}
