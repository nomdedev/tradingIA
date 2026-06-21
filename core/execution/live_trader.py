"""
LiveTrader - Live Trading Engine for TradingIA

This module provides a production-ready live trading implementation
with robust error handling, reconnection logic, and rate limiting.

Features:
- Common interface with BacktesterCore
- Automatic API reconnection with exponential backoff
- Order retry logic with configurable attempts
- Rate limiting (200 req/min for Alpaca)
- Kill switch integration
- Real-time position tracking

Author: TradingIA Team
Date: 13 de Enero 2026
"""

import threading
import time
import logging
from datetime import datetime
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import queue

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    APIError = Exception

from core.constants import (
    LOGS_DIR,
    DEFAULT_MAX_DAILY_DRAWDOWN,
    DEFAULT_MAX_TOTAL_DRAWDOWN,
)

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    error_message: Optional[str] = None


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float
    side: str  # 'long' or 'short'


class LiveTrader:
    """
    Production-ready live trading engine with Alpaca integration.
    
    Provides common interface with BacktesterCore for easy transition
    from backtesting to live trading.
    """
    
    # Rate limiting constants
    MAX_CALLS_PER_MINUTE = 200
    RATE_LIMIT_BUFFER_MS = 100
    
    # Retry constants
    DEFAULT_MAX_RETRIES = 3
    BASE_RETRY_DELAY_S = 1.0
    MAX_RETRY_DELAY_S = 30.0
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        initial_capital: float = 100000.0,
        max_daily_drawdown: float = DEFAULT_MAX_DAILY_DRAWDOWN,
        max_total_drawdown: float = DEFAULT_MAX_TOTAL_DRAWDOWN,
        risk_manager=None,
        council=None,
    ):
        """
        Initialize LiveTrader.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: API endpoint (paper or live)
            initial_capital: Starting capital for risk calculations
            max_daily_drawdown: Maximum allowed daily drawdown (0.05 = 5%)
            max_total_drawdown: Maximum allowed total drawdown
            risk_manager: Optional RiskManager instance
            council: Optional Council instance for signal validation
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.initial_capital = initial_capital
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.risk_manager = risk_manager
        self.council = council
        
        # API connection
        self.api: Optional[tradeapi.REST] = None
        self._connected = False
        
        # Rate limiting
        self._api_call_timestamps: List[float] = []
        self._rate_limit_lock = threading.Lock()
        
        # State tracking
        self.current_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.peak_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Kill switch
        self._kill_switch_active = False
        self._kill_switch_lock = threading.Lock()
        self._kill_switch_reason: Optional[str] = None
        
        # Threading
        self._running = False
        self._stop_event = threading.Event()
        self._order_queue: queue.Queue = queue.Queue()
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_fill_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []
        self._on_position_update_callbacks: List[Callable] = []
        
        # Logger setup
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize connection
        if api_key and secret_key and ALPACA_AVAILABLE:
            self._connect()
    
    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================
    
    def _connect(self) -> bool:
        """Establish initial API connection"""
        return self.reconnect_api()
    
    def reconnect_api(self, max_retries: int = None) -> bool:
        """
        Reconnect to Alpaca API with exponential backoff.
        
        Args:
            max_retries: Maximum retry attempts (default: DEFAULT_MAX_RETRIES)
            
        Returns:
            True if reconnection successful, False otherwise
        """
        if not ALPACA_AVAILABLE:
            self.logger.error("Alpaca API not available - install alpaca-trade-api")
            return False
            
        if not self.api_key or not self.secret_key:
            self.logger.error("API credentials not provided")
            return False
        
        max_retries = max_retries or self.DEFAULT_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                self.api = tradeapi.REST(
                    self.api_key,
                    self.secret_key,
                    self.base_url,
                    api_version="v2"
                )
                
                # Test connection
                account = self.api.get_account()
                self.current_capital = float(account.equity)
                self._connected = True
                
                self.logger.info(
                    f"✅ API connected successfully (attempt {attempt + 1}). "
                    f"Equity: ${self.current_capital:,.2f}"
                )
                return True
                
            except Exception as e:
                delay = min(
                    self.BASE_RETRY_DELAY_S * (2 ** attempt),
                    self.MAX_RETRY_DELAY_S
                )
                self.logger.warning(
                    f"⚠️ Connection failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error("❌ All connection attempts failed")
                    self._connected = False
                    return False
        
        return False
    
    def is_connected(self) -> bool:
        """Check if API is connected and responsive"""
        if not self._connected or not self.api:
            return False
        
        try:
            self._rate_limit_check()
            self.api.get_account()
            return True
        except Exception:
            self._connected = False
            return False
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    def _rate_limit_check(self):
        """
        Check and enforce rate limits (200 calls/minute for Alpaca).
        Thread-safe implementation.
        """
        with self._rate_limit_lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute
            self._api_call_timestamps = [
                ts for ts in self._api_call_timestamps 
                if now - ts < 60
            ]
            
            if len(self._api_call_timestamps) >= self.MAX_CALLS_PER_MINUTE:
                # Calculate wait time
                oldest_call = self._api_call_timestamps[0]
                wait_time = 60 - (now - oldest_call) + (self.RATE_LIMIT_BUFFER_MS / 1000)
                
                self.logger.warning(f"⏳ Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                
                # Clean up after waiting
                now = time.time()
                self._api_call_timestamps = [
                    ts for ts in self._api_call_timestamps 
                    if now - ts < 60
                ]
            
            # Record this call
            self._api_call_timestamps.append(time.time())
    
    # =========================================================================
    # ORDER EXECUTION WITH RETRY
    # =========================================================================
    
    def submit_order_with_retry(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float = None,
        stop_price: float = None,
        max_retries: int = None,
        time_in_force: str = "day",
    ) -> Optional[Order]:
        """
        Submit order with automatic retry logic.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            qty: Order quantity
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: Limit price (for limit/stop_limit orders)
            stop_price: Stop price (for stop/stop_limit orders)
            max_retries: Maximum retry attempts
            time_in_force: Order duration ('day', 'gtc', 'ioc', 'fok')
            
        Returns:
            Order object if successful, None if failed
        """
        # Check kill switch
        if self._is_kill_switch_active():
            self.logger.error(f"❌ Order rejected: Kill switch active - {self._kill_switch_reason}")
            return None
        
        # Validate inputs
        if qty <= 0:
            self.logger.error(f"❌ Invalid quantity: {qty}")
            return None
        
        if side not in ('buy', 'sell'):
            self.logger.error(f"❌ Invalid side: {side}")
            return None
        
        # Check with Council if available
        if self.council:
            try:
                council_decision = self.council.evaluate_signal({
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'price': limit_price or self._get_current_price(symbol),
                })
                if not council_decision.get('approved', True):
                    self.logger.warning(f"⚠️ Order rejected by Council: {council_decision.get('reason')}")
                    return None
            except Exception as e:
                self.logger.warning(f"Council evaluation failed: {e}")
        
        # Check with Risk Manager if available
        if self.risk_manager:
            try:
                risk_check = self.risk_manager.check_order(symbol, qty, side)
                if not risk_check.get('approved', True):
                    self.logger.warning(f"⚠️ Order rejected by Risk Manager: {risk_check.get('reason')}")
                    return None
            except Exception as e:
                self.logger.warning(f"Risk check failed: {e}")
        
        max_retries = max_retries or self.DEFAULT_MAX_RETRIES
        
        # Create order object
        order = Order(
            id=f"order_{int(time.time() * 1000)}",
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        
        for attempt in range(max_retries):
            try:
                order.attempts = attempt + 1
                self._rate_limit_check()
                
                # Build order parameters
                order_params = {
                    'symbol': symbol.replace('/', ''),  # Alpaca format: BTCUSD
                    'qty': qty,
                    'side': side,
                    'type': order_type,
                    'time_in_force': time_in_force,
                }
                
                if limit_price and order_type in ('limit', 'stop_limit'):
                    order_params['limit_price'] = limit_price
                if stop_price and order_type in ('stop', 'stop_limit'):
                    order_params['stop_price'] = stop_price
                
                # Submit to Alpaca
                alpaca_order = self.api.submit_order(**order_params)
                
                # Update order with response
                order.id = alpaca_order.id
                order.status = OrderStatus.SUBMITTED
                order.updated_at = datetime.now()
                
                self.pending_orders[order.id] = order
                
                self.logger.info(
                    f"✅ Order submitted: {side.upper()} {qty} {symbol} "
                    f"(ID: {order.id}, attempt {attempt + 1})"
                )
                
                return order
                
            except APIError as e:
                delay = min(
                    self.BASE_RETRY_DELAY_S * (2 ** attempt),
                    self.MAX_RETRY_DELAY_S
                )
                order.error_message = str(e)
                
                self.logger.warning(
                    f"⚠️ Order failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                # Check if error is retryable
                if self._is_retryable_error(e):
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        order.status = OrderStatus.FAILED
                        self.logger.error(f"❌ Order failed after {max_retries} attempts")
                else:
                    order.status = OrderStatus.REJECTED
                    self.logger.error(f"❌ Order rejected (non-retryable): {e}")
                    break
                    
            except Exception as e:
                order.status = OrderStatus.FAILED
                order.error_message = str(e)
                self.logger.error(f"❌ Unexpected error submitting order: {e}")
                
                # Try to reconnect if connection lost
                if not self.is_connected():
                    self.logger.info("Attempting to reconnect...")
                    if self.reconnect_api():
                        continue
                break
        
        self.order_history.append(order)
        return order if order.status == OrderStatus.SUBMITTED else None
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        retryable_codes = {
            'rate_limit_exceeded',
            'service_unavailable',
            'internal_server_error',
            'timeout',
        }
        
        error_str = str(error).lower()
        return any(code in error_str for code in retryable_codes)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            self._rate_limit_check()
            quote = self.api.get_latest_crypto_quote(symbol.replace('/', ''))
            return float(quote.ap)  # Ask price
        except Exception as e:
            self.logger.warning(f"Failed to get price for {symbol}: {e}")
            return None
    
    # =========================================================================
    # KILL SWITCH
    # =========================================================================
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        Activate kill switch to halt all trading.
        
        Args:
            reason: Reason for activation
        """
        with self._kill_switch_lock:
            self._kill_switch_active = True
            self._kill_switch_reason = reason
            
        self.logger.critical(f"🛑 KILL SWITCH ACTIVATED: {reason}")
        
        # Cancel all pending orders
        self._cancel_all_orders()
        
        # Notify callbacks
        for callback in self._on_error_callbacks:
            try:
                callback({'type': 'kill_switch', 'reason': reason})
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch to resume trading"""
        with self._kill_switch_lock:
            self._kill_switch_active = False
            self._kill_switch_reason = None
            
        self.logger.info("✅ Kill switch deactivated")
    
    def _is_kill_switch_active(self) -> bool:
        """Thread-safe kill switch check"""
        with self._kill_switch_lock:
            return self._kill_switch_active
    
    def _check_drawdown_kill_switch(self):
        """Check if drawdown limits are exceeded"""
        if self.current_capital <= 0:
            return
            
        # Daily drawdown
        daily_drawdown = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        if daily_drawdown >= self.max_daily_drawdown:
            self.activate_kill_switch(
                f"Daily drawdown limit exceeded: {daily_drawdown:.2%} >= {self.max_daily_drawdown:.2%}"
            )
            return
        
        # Total drawdown
        total_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if total_drawdown >= self.max_total_drawdown:
            self.activate_kill_switch(
                f"Total drawdown limit exceeded: {total_drawdown:.2%} >= {self.max_total_drawdown:.2%}"
            )
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions from API"""
        if not self.is_connected():
            return self.positions
        
        try:
            self._rate_limit_check()
            alpaca_positions = self.api.list_positions()
            
            self.positions = {}
            for pos in alpaca_positions:
                self.positions[pos.symbol] = Position(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                    market_value=float(pos.market_value),
                    side='long' if float(pos.qty) > 0 else 'short',
                )
            
            return self.positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return self.positions
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close a position completely"""
        positions = self.get_positions()
        
        if symbol not in positions:
            self.logger.warning(f"No position found for {symbol}")
            return None
        
        pos = positions[symbol]
        side = 'sell' if pos.side == 'long' else 'buy'
        
        return self.submit_order_with_retry(
            symbol=symbol,
            qty=abs(pos.qty),
            side=side,
        )
    
    def close_all_positions(self):
        """Close all open positions"""
        positions = self.get_positions()
        
        for symbol in positions:
            self.close_position(symbol)
    
    def _cancel_all_orders(self):
        """Cancel all pending orders"""
        if not self.is_connected():
            return
        
        try:
            self._rate_limit_check()
            self.api.cancel_all_orders()
            self.pending_orders.clear()
            self.logger.info("All pending orders cancelled")
        except Exception as e:
            self.logger.error(f"Failed to cancel orders: {e}")
    
    # =========================================================================
    # ACCOUNT INFO
    # =========================================================================
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.is_connected():
            return {
                'equity': self.current_capital,
                'buying_power': 0,
                'connected': False,
            }
        
        try:
            self._rate_limit_check()
            account = self.api.get_account()
            
            info = {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'connected': True,
            }
            
            # Update internal state
            self.current_capital = info['equity']
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            
            # Check drawdown limits
            self._check_drawdown_kill_switch()
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return {'connected': False, 'error': str(e)}
    
    # =========================================================================
    # MONITORING THREAD
    # =========================================================================
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """
        Start background monitoring thread.
        
        Args:
            interval_seconds: Polling interval
        """
        if self._running:
            self.logger.warning("Monitoring already running")
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=False,  # Not daemon - ensure cleanup
            name="LiveTrader-Monitor"
        )
        self._monitor_thread.start()
        self.logger.info(f"📡 Monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self, timeout: float = 5.0):
        """
        Stop monitoring thread with proper cleanup.
        
        Args:
            timeout: Maximum wait time for thread to stop
        """
        if not self._running:
            return
        
        self.logger.info("Stopping monitoring...")
        self._running = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
            
            if self._monitor_thread.is_alive():
                self.logger.warning(f"⚠️ Monitor thread did not stop within {timeout}s")
            else:
                self.logger.info("✅ Monitoring stopped cleanly")
        
        self._monitor_thread = None
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        self.logger.info("Monitor thread started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Update account info
                account = self.get_account_info()
                self.logger.debug(f"Account status: {account}")
                
                # Update positions
                positions = self.get_positions()
                
                # Notify position callbacks
                for callback in self._on_position_update_callbacks:
                    try:
                        callback(positions)
                    except Exception as e:
                        self.logger.error(f"Error in position callback: {e}")
                
                # Check for filled orders
                self._check_order_fills()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
                # Try to reconnect
                if not self.is_connected():
                    self.reconnect_api()
            
            # Wait for next iteration or stop signal
            self._stop_event.wait(timeout=interval)
        
        self.logger.info("Monitor thread exiting")
    
    def _check_order_fills(self):
        """Check pending orders for fills"""
        if not self.pending_orders:
            return
        
        try:
            self._rate_limit_check()
            orders = self.api.list_orders(status='all', limit=100)
            
            order_map = {o.id: o for o in orders}
            
            filled_ids = []
            for order_id, order in self.pending_orders.items():
                if order_id in order_map:
                    alpaca_order = order_map[order_id]
                    
                    if alpaca_order.status == 'filled':
                        order.status = OrderStatus.FILLED
                        order.filled_qty = float(alpaca_order.filled_qty)
                        order.filled_avg_price = float(alpaca_order.filled_avg_price)
                        order.updated_at = datetime.now()
                        filled_ids.append(order_id)
                        
                        self.logger.info(
                            f"✅ Order filled: {order.side.upper()} {order.filled_qty} "
                            f"@ ${order.filled_avg_price:.2f}"
                        )
                        
                        # Notify callbacks
                        for callback in self._on_fill_callbacks:
                            try:
                                callback(order)
                            except Exception as e:
                                self.logger.error(f"Error in fill callback: {e}")
                    
                    elif alpaca_order.status in ('cancelled', 'expired', 'rejected'):
                        order.status = OrderStatus.CANCELLED
                        filled_ids.append(order_id)
            
            # Move filled orders to history
            for order_id in filled_ids:
                order = self.pending_orders.pop(order_id)
                self.order_history.append(order)
                
        except Exception as e:
            self.logger.error(f"Error checking order fills: {e}")
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def on_fill(self, callback: Callable[[Order], None]):
        """Register callback for order fills"""
        self._on_fill_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Dict], None]):
        """Register callback for errors"""
        self._on_error_callbacks.append(callback)
    
    def on_position_update(self, callback: Callable[[Dict[str, Position]], None]):
        """Register callback for position updates"""
        self._on_position_update_callbacks.append(callback)
    
    # =========================================================================
    # DAILY RESET
    # =========================================================================
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at market open)"""
        self.daily_start_capital = self.current_capital
        self.logger.info(f"📊 Daily stats reset. Starting capital: ${self.current_capital:,.2f}")
    
    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup"""
        self.stop_monitoring(timeout=5.0)
        return False
