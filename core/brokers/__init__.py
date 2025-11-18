"""
Broker integration module for live trading.
"""

from .broker_interfaces import (
    BrokerInterface, BrokerConfig, Order, OrderType, OrderSide, OrderStatus,
    Position, PositionSide, Account, MarketData, BrokerError,
    ConnectionError, AuthenticationError, OrderError, InsufficientFundsError
)
from .broker_manager import BrokerManager
from .alpaca_broker import AlpacaBroker

try:
    from .brokers_panel import BrokersPanel, BrokerConfigDialog, OrderDialog
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    BrokersPanel = None
    BrokerConfigDialog = None
    OrderDialog = None

__all__ = [
    'BrokerInterface', 'BrokerConfig', 'Order', 'OrderType', 'OrderSide', 'OrderStatus',
    'Position', 'PositionSide', 'Account', 'MarketData', 'BrokerError',
    'ConnectionError', 'AuthenticationError', 'OrderError', 'InsufficientFundsError',
    'BrokerManager', 'AlpacaBroker'
]

if PYQT6_AVAILABLE:
    __all__.extend(['BrokersPanel', 'BrokerConfigDialog', 'OrderDialog'])