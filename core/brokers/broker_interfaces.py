"""
Broker interfaces and base classes for live trading integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import pandas as pd


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    fees: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: PositionSide
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    market_value: float = 0.0

    @property
    def total_value(self) -> float:
        """Total value of the position."""
        return self.quantity * self.current_price


@dataclass
class Account:
    """Trading account information."""
    id: str
    balance: float
    buying_power: float
    cash: float
    currency: str = "USD"
    margin_used: float = 0.0
    margin_available: float = 0.0
    day_trades: int = 0
    day_trade_limit: int = 0


@dataclass
class MarketData:
    """Market data for a symbol."""
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    timestamp: datetime = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BrokerError(Exception):
    """Base exception for broker-related errors."""
    pass


class ConnectionError(BrokerError):
    """Connection-related errors."""
    pass


class AuthenticationError(BrokerError):
    """Authentication-related errors."""
    pass


class OrderError(BrokerError):
    """Order-related errors."""
    pass


class InsufficientFundsError(BrokerError):
    """Insufficient funds errors."""
    pass


class BrokerInterface(ABC):
    """Abstract base class for broker implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        pass

    @abstractmethod
    def connect(self, credentials: Dict[str, str]) -> bool:
        """Connect to the broker."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    def get_account(self) -> Account:
        """Get account information."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status."""
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: OrderSide, quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   limit_price: Optional[float] = None) -> Order:
        """Place a new order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol."""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime,
                           end_date: datetime, timeframe: str = "1D") -> pd.DataFrame:
        """Get historical market data."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        pass

    @abstractmethod
    def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, List]:
        """Get order book for a symbol."""
        pass


@dataclass
class BrokerConfig:
    """Configuration for a broker."""
    name: str
    broker_type: str
    credentials: Dict[str, str]
    enabled: bool = True
    max_orders_per_minute: int = 60
    risk_limits: Dict[str, float] = None
    supported_assets: List[str] = None

    def __post_init__(self):
        if self.risk_limits is None:
            self.risk_limits = {
                'max_position_size': 10000.0,
                'max_daily_loss': 1000.0,
                'max_drawdown': 0.1
            }
        if self.supported_assets is None:
            self.supported_assets = ['stocks', 'crypto', 'forex']


@dataclass
class TradeExecution:
    """Represents a trade execution."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    execution_id: Optional[str] = None

    @property
    def total_cost(self) -> float:
        """Total cost including fees."""
        return (self.quantity * self.price) + self.fees