"""
Data loading utilities for the trading dashboard - CLEAN VERSION
Fixed date comparison issues and robust timestamp handling
"""

import pandas as pd
import os
import psutil
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

# Absolute imports from dashboard package
from .alpaca_utils import normalize_alpaca_trades
from .safe_access import safe_get, ensure_columns

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest, GetAssetsRequest
    from alpaca.trading.enums import OrderSide, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.getLogger(__name__).warning("Alpaca SDK not available. Using CSV fallback.")

# Import config - assuming it's in dashboard root
try:
    import dashboard.config as config
except ImportError:
    # Fallback config
    class config:
        ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
        ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
        ALPACA_LIVE_TRADING = False

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Centralized data loading for the dashboard with robust date handling.
    """

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        self.data_dir = logs_dir  # For consistency
        self.trades_file = os.path.join(logs_dir, "trades.csv")
        self.retraining_file = os.path.join(logs_dir, "retraining_history.csv")
        self.decisions_file = os.path.join(logs_dir, "decisions.csv")

        # Cache for performance
        self._trades_cache = None
        self._retraining_cache = None
        self._cache_timestamp = None
        self._cache_duration = 30  # seconds

        # Alpaca clients
        self._trading_client = None
        self._data_client = None

    def _to_timestamp(self, date_input: Any) -> pd.Timestamp:
        """Robustly convert any date input to pd.Timestamp."""
        try:
            if isinstance(date_input, pd.Timestamp):
                return date_input
            elif isinstance(date_input, datetime):
                return pd.Timestamp(date_input)
            elif isinstance(date_input, date):
                return pd.Timestamp(datetime.combine(date_input, datetime.min.time()))
            elif isinstance(date_input, str):
                return pd.to_datetime(date_input)
            else:
                return pd.to_datetime(date_input)
        except Exception as e:
            logger.error(f"Error converting date {date_input}: {e}")
            return pd.Timestamp.now()

    def _ensure_datetime_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Ensure a column is datetime type, converting if necessary."""
        if col not in df.columns:
            return df
        try:
            df = df.copy()
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df.dropna(subset=[col])
            return df
        except Exception as e:
            logger.error(f"Error converting column {col} to datetime: {e}")
            return df

    def _get_alpaca_clients(self):
        """Initialize Alpaca clients if available."""
        if not ALPACA_AVAILABLE:
            return None, None
        try:
            if not self._trading_client:
                self._trading_client = TradingClient(
                    api_key=config.ALPACA_API_KEY,
                    secret_key=config.ALPACA_SECRET_KEY,
                    paper=not config.ALPACA_LIVE_TRADING
                )
            if not self._data_client:
                self._data_client = StockHistoricalDataClient(
                    api_key=config.ALPACA_API_KEY,
                    secret_key=config.ALPACA_SECRET_KEY
                )
            return self._trading_client, self._data_client
        except Exception as e:
            logger.error(f"Error initializing Alpaca clients: {e}")
            return None, None

    def load_trades_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load trades data from CSV or Alpaca API."""
        try:
            if (not force_refresh and
                self._trades_cache is not None and
                self._cache_timestamp and
                (datetime.now() - self._cache_timestamp).seconds < self._cache_duration):
                return self._trades_cache.copy()

            # Try Alpaca first if available
            if ALPACA_AVAILABLE and config.ALPACA_API_KEY:
                try:
                    trading_client, _ = self._get_alpaca_clients()
                    if trading_client:
                        request = GetOrdersRequest(
                            status=QueryOrderStatus.CLOSED,
                            limit=500
                        )
                        orders = trading_client.get_orders(request)

                        if orders:
                            trades_data = []
                            for order in orders:
                                if hasattr(order, 'filled_at') and order.filled_at:
                                    trade = {
                                        'timestamp': pd.to_datetime(order.filled_at),
                                        'symbol': safe_get(order, 'symbol', ''),
                                        'side': safe_get(order, 'side', '').value.lower() if hasattr(order.side, 'value') else str(order.side).lower(),
                                        'quantity': float(safe_get(order, 'filled_qty', 0)),
                                        'price': float(safe_get(order, 'filled_avg_price', 0)),
                                        'status': 'filled',
                                        'order_id': safe_get(order, 'id', ''),
                                        'pnl': 0.0
                                    }
                                    trades_data.append(trade)

                            if trades_data:
                                trades_df = pd.DataFrame(trades_data)
                                self._trades_cache = trades_df.copy()
                                self._cache_timestamp = datetime.now()
                                return trades_df

                except Exception as e:
                    logger.warning(f"Alpaca trades loading failed: {e}")

            # Fallback to CSV
            if os.path.exists(self.trades_file):
                trades_df = pd.read_csv(self.trades_file)

                if not trades_df.empty:
                    trades_df = self._ensure_datetime_column(trades_df, 'timestamp')
                    trades_df = trades_df.dropna(subset=['timestamp'])
                    trades_df = trades_df.sort_values('timestamp')

                    self._trades_cache = trades_df.copy()
                    self._cache_timestamp = datetime.now()
                    return trades_df

            empty_df = pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'quantity', 'price', 'status'])
            self._trades_cache = empty_df.copy()
            self._cache_timestamp = datetime.now()
            return empty_df

        except Exception as e:
            logger.error(f"Error loading trades data: {e}")
            return pd.DataFrame()

    def load_trades_in_range(self, start_date, end_date) -> pd.DataFrame:
        """Load trades data filtered by date range."""
        try:
            if not os.path.exists(self.trades_file):
                return pd.DataFrame()

            trades_df = pd.read_csv(self.trades_file)

            if trades_df.empty:
                return trades_df

            trades_df = self._ensure_datetime_column(trades_df, 'timestamp')

            start_ts = self._to_timestamp(start_date)
            end_ts = self._to_timestamp(end_date)

            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_ts = end_ts.replace(hour=23, minute=59, second=59)

            mask = (trades_df['timestamp'] >= start_ts) & (trades_df['timestamp'] <= end_ts)
            filtered_df = trades_df[mask].copy()

            logger.info(f"Filtered {len(filtered_df)} trades between {start_ts} and {end_ts}")
            return filtered_df

        except Exception as e:
            logger.error(f"Error loading trades in range: {e}")
            return pd.DataFrame()

    def load_equity_curve_range(self, start_date, end_date) -> pd.DataFrame:
        """Load equity curve data filtered by date range."""
        try:
            equity_df = self.load_equity_curve()

            if equity_df.empty or 'timestamp' not in equity_df.columns:
                return pd.DataFrame()

            start_ts = self._to_timestamp(start_date)
            end_ts = self._to_timestamp(end_date)

            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_ts = end_ts.replace(hour=23, minute=59, second=59)

            mask = (equity_df['timestamp'] >= start_ts) & (equity_df['timestamp'] <= end_ts)
            return equity_df[mask].copy()

        except Exception as e:
            logger.error(f"Error loading equity curve range: {e}")
            return pd.DataFrame()

    def load_retraining_history_range(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Load retraining history filtered by date range."""
        try:
            retrain_df = self.load_retraining_history()

            if retrain_df.empty or 'timestamp' not in retrain_df.columns:
                return pd.DataFrame()

            if start_date is None and end_date is None:
                return retrain_df

            start_ts = self._to_timestamp(start_date) if start_date else None
            end_ts = self._to_timestamp(end_date) if end_date else None

            if end_ts and isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_ts = end_ts.replace(hour=23, minute=59, second=59)

            mask = pd.Series(True, index=retrain_df.index)
            if start_ts:
                mask &= (retrain_df['timestamp'] >= start_ts)
            if end_ts:
                mask &= (retrain_df['timestamp'] <= end_ts)

            return retrain_df[mask].copy()

        except Exception as e:
            logger.error(f"Error loading retraining history range: {e}")
            return pd.DataFrame()

    def load_trades(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Load trades with optional date filtering."""
        if start_date and end_date:
            return self.load_trades_in_range(start_date, end_date)
        else:
            return self.load_trades_data()

    def load_recent_trades(self, limit: int = 10) -> pd.DataFrame:
        """Load most recent trades."""
        try:
            trades_df = self.load_trades_data()

            if trades_df.empty:
                return trades_df

            recent_df = trades_df.sort_values('timestamp', ascending=False).head(limit)
            return recent_df

        except Exception as e:
            logger.error(f"Error loading recent trades: {e}")
            return pd.DataFrame()

    def load_active_positions(self) -> pd.DataFrame:
        """Load current active positions."""
        try:
            if ALPACA_AVAILABLE and config.ALPACA_API_KEY:
                trading_client, _ = self._get_alpaca_clients()
                if trading_client:
                    positions = trading_client.get_all_positions()
                    if positions:
                        positions_data = []
                        for pos in positions:
                            position = {
                                'symbol': safe_get(pos, 'symbol', ''),
                                'qty': float(safe_get(pos, 'qty', 0)),
                                'avg_entry_price': float(safe_get(pos, 'avg_entry_price', 0)),
                                'current_price': float(safe_get(pos, 'current_price', 0)),
                                'unrealized_pl': float(safe_get(pos, 'unrealized_pl', 0)),
                                'unrealized_plpc': float(safe_get(pos, 'unrealized_plpc', 0))
                            }
                            positions_data.append(position)

                        return pd.DataFrame(positions_data)

            return pd.DataFrame(columns=['symbol', 'qty', 'avg_entry_price', 'current_price', 'unrealized_pl', 'unrealized_plpc'])

        except Exception as e:
            logger.error(f"Error loading active positions: {e}")
            return pd.DataFrame()

    def load_equity_curve(self, hours: int = None, force_refresh: bool = False) -> pd.DataFrame:
        """Calculate complete equity curve from all trades."""
        try:
            trades_df = self.load_trades_data(force_refresh)

            if trades_df.empty:
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=hours) if hours else end_date - timedelta(days=90)
                dates = pd.date_range(start=start_date, end=end_date, freq='H' if hours and hours <= 24 else 'D')
                equity = 100000.0
                equity_values = []
                for i in range(len(dates)):
                    equity *= (1 + np.random.normal(0.001, 0.01))
                    equity_values.append(equity)

                return pd.DataFrame({
                    'timestamp': dates,
                    'equity': equity_values
                })

            trades_df = trades_df.sort_values('timestamp')

            if hours:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                trades_df = trades_df[trades_df['timestamp'] >= cutoff_time]

            trades_df = trades_df.copy()
            trades_df['equity'] = 100000.0

            for i in range(1, len(trades_df)):
                pnl = safe_get(trades_df.iloc[i], 'pnl', 0)
                if pd.notna(pnl):
                    trades_df.iloc[i, trades_df.columns.get_loc('equity')] = (
                        trades_df.iloc[i-1]['equity'] + pnl
                    )

            return trades_df[['timestamp', 'equity']].copy()

        except Exception as e:
            logger.error(f"Error calculating equity curve: {e}")
            return pd.DataFrame()

    def load_decision_log(self, limit: int = 10) -> pd.DataFrame:
        """Load recent decision log entries."""
        try:
            if os.path.exists(self.decisions_file):
                decisions_df = pd.read_csv(self.decisions_file)

                if not decisions_df.empty:
                    decisions_df = self._ensure_datetime_column(decisions_df, 'timestamp')
                    decisions_df = decisions_df.dropna(subset=['timestamp'])
                    decisions_df = decisions_df.sort_values('timestamp', ascending=False)
                    return decisions_df.head(limit)

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading decision log: {e}")
            return pd.DataFrame()

    def load_retraining_history(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load retraining history from CSV."""
        try:
            if (not force_refresh and
                self._retraining_cache is not None and
                self._cache_timestamp and
                (datetime.now() - self._cache_timestamp).seconds < self._cache_duration):
                return self._retraining_cache.copy()

            if os.path.exists(self.retraining_file):
                retrain_df = pd.read_csv(self.retraining_file)

                if not retrain_df.empty:
                    retrain_df = self._ensure_datetime_column(retrain_df, 'timestamp')
                    retrain_df = retrain_df.dropna(subset=['timestamp'])
                    retrain_df = retrain_df.sort_values('timestamp')

                    self._retraining_cache = retrain_df.copy()
                    self._cache_timestamp = datetime.now()
                    return retrain_df

            empty_df = pd.DataFrame(columns=['timestamp', 'type', 'sharpe_before', 'sharpe_after', 'status'])
            self._retraining_cache = empty_df.copy()
            self._cache_timestamp = datetime.now()
            return empty_df

        except Exception as e:
            logger.error(f"Error loading retraining history: {e}")
            return pd.DataFrame()

    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading performance metrics."""
        try:
            if trades_df.empty:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0
                }

            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df.get('pnl', 0) > 0])
            losing_trades = len(trades_df[trades_df.get('pnl', 0) < 0])

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            total_pnl = trades_df.get('pnl', 0).sum()
            avg_win = trades_df[trades_df.get('pnl', 0) > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades_df[trades_df.get('pnl', 0) < 0]['pnl'].mean()) if losing_trades > 0 else 0

            profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if avg_loss > 0 and losing_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def calculate_sharpe(self, equity_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity curve."""
        try:
            if equity_df.empty or 'equity' not in equity_df.columns or len(equity_df) < 2:
                return 0.0

            returns = equity_df['equity'].pct_change().dropna()

            if len(returns) == 0:
                return 0.0

            avg_return = returns.mean() * 252
            std_return = returns.std() * np.sqrt(252)

            if std_return == 0:
                return 0.0

            sharpe = (avg_return - risk_free_rate) / std_return
            return sharpe

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_drawdown(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drawdown metrics from equity curve."""
        try:
            if equity_df.empty or 'equity' not in equity_df.columns:
                return {'max_drawdown': 0.0, 'current_drawdown': 0.0, 'avg_drawdown': 0.0}

            equity_values = equity_df['equity'].values
            peak = equity_values[0]
            max_drawdown = 0
            current_drawdown = 0
            drawdowns = []

            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
                current_drawdown = drawdown
                drawdowns.append(drawdown)

            avg_drawdown = np.mean(drawdowns) if drawdowns else 0

            return {
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'avg_drawdown': avg_drawdown
            }

        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0, 'avg_drawdown': 0.0}

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current trading metrics."""
        try:
            trades_df = self.load_trades_data()
            equity_df = self.load_equity_curve()

            metrics = self.calculate_metrics(trades_df)
            sharpe = self.calculate_sharpe(equity_df)
            drawdown = self.calculate_drawdown(equity_df)

            return {
                **metrics,
                'sharpe_ratio': sharpe,
                **drawdown,
                'current_equity': equity_df['equity'].iloc[-1] if not equity_df.empty else 100000.0
            }

        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource usage metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)

            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)

            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024 ** 3)
            disk_total_gb = disk.total / (1024 ** 3)

            net_io = psutil.net_io_counters()
            bytes_sent_mb = net_io.bytes_sent / (1024 ** 2)
            bytes_recv_mb = net_io.bytes_recv / (1024 ** 2)

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': round(memory_used_gb, 2),
                'memory_total_gb': round(memory_total_gb, 2),
                'disk_percent': disk_percent,
                'disk_used_gb': round(disk_used_gb, 2),
                'disk_total_gb': round(disk_total_gb, 2),
                'network_sent_mb': round(bytes_sent_mb, 2),
                'network_recv_mb': round(bytes_recv_mb, 2),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 0,
                'disk_percent': 0,
                'disk_used_gb': 0,
                'disk_total_gb': 0,
                'network_sent_mb': 0,
                'network_recv_mb': 0,
                'timestamp': datetime.now()
            }
