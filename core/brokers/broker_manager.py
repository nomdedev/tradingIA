"""
Broker manager for handling multiple broker connections and live trading.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
import json
import os

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from .broker_interfaces import (
    BrokerInterface, BrokerConfig, Order, Position, Account,
    MarketData, BrokerError, OrderStatus
)
from .alpaca_broker import AlpacaBroker


class BrokerManager:
    """Manages multiple broker connections and live trading operations."""

    def __init__(self, config_file: str = "config/brokers_config.json"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

        # Broker registry
        self.broker_classes: Dict[str, Type[BrokerInterface]] = {
            'alpaca': AlpacaBroker,
        }

        # Active brokers
        self.brokers: Dict[str, BrokerInterface] = {}
        self.configs: Dict[str, BrokerConfig] = {}

        # Monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_update = datetime.now()

        # Load configuration
        self._load_config()

    def register_broker_class(self, name: str, broker_class: Type[BrokerInterface]):
        """Register a new broker implementation."""
        self.broker_classes[name] = broker_class
        self.logger.info(f"Registered broker class: {name}")

    def add_broker(self, config: BrokerConfig) -> bool:
        """Add and connect to a broker."""
        try:
            if config.name in self.brokers:
                self.logger.warning(f"Broker {config.name} already exists")
                return False

            # Get broker class
            broker_class = self.broker_classes.get(config.broker_type)
            if not broker_class:
                raise BrokerError(f"Unknown broker type: {config.broker_type}")

            # Create broker instance
            broker = broker_class(**config.credentials)

            # Connect
            if broker.connect(config.credentials):
                self.brokers[config.name] = broker
                self.configs[config.name] = config
                self._save_config()
                self.logger.info(f"Added broker: {config.name}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Failed to add broker {config.name}: {e}")
            return False

    def remove_broker(self, name: str) -> bool:
        """Remove and disconnect from a broker."""
        try:
            if name not in self.brokers:
                return False

            broker = self.brokers[name]
            broker.disconnect()
            del self.brokers[name]
            del self.configs[name]
            self._save_config()
            self.logger.info(f"Removed broker: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to remove broker {name}: {e}")
            return False

    def get_broker(self, name: str) -> Optional[BrokerInterface]:
        """Get a broker instance by name."""
        return self.brokers.get(name)

    def get_available_brokers(self) -> List[str]:
        """Get list of available broker names."""
        return list(self.brokers.keys())

    def get_broker_config(self, name: str) -> Optional[BrokerConfig]:
        """Get broker configuration."""
        return self.configs.get(name)

    def start_monitoring(self):
        """Start monitoring all brokers."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Broker monitoring started")

    def stop_monitoring(self):
        """Stop monitoring all brokers."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Broker monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_connections()
                self._update_positions()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in broker monitoring: {e}")
                time.sleep(60)  # Wait longer on error

    def _check_connections(self):
        """Check broker connections and reconnect if needed."""
        for name, broker in self.brokers.items():
            if not broker.is_connected:
                self.logger.warning(f"Broker {name} disconnected, attempting reconnect")
                config = self.configs[name]
                try:
                    broker.connect(config.credentials)
                    self.logger.info(f"Reconnected to broker {name}")
                except Exception as e:
                    self.logger.error(f"Failed to reconnect to broker {name}: {e}")

    def _update_positions(self):
        """Update position information for all brokers."""
        self.last_update = datetime.now()

    def get_all_positions(self) -> Dict[str, List[Position]]:
        """Get positions from all brokers."""
        result = {}
        for name, broker in self.brokers.items():
            try:
                if broker.is_connected:
                    result[name] = broker.get_positions()
            except Exception as e:
                self.logger.error(f"Failed to get positions for broker {name}: {e}")
                result[name] = []
        return result

    def get_all_accounts(self) -> Dict[str, Account]:
        """Get account information from all brokers."""
        result = {}
        for name, broker in self.brokers.items():
            try:
                if broker.is_connected:
                    result[name] = broker.get_account()
            except Exception as e:
                self.logger.error(f"Failed to get account for broker {name}: {e}")
        return result

    def get_all_orders(self, status: Optional[OrderStatus] = None) -> Dict[str, List[Order]]:
        """Get orders from all brokers."""
        result = {}
        for name, broker in self.brokers.items():
            try:
                if broker.is_connected:
                    result[name] = broker.get_orders(status)
            except Exception as e:
                self.logger.error(f"Failed to get orders for broker {name}: {e}")
                result[name] = []
        return result

    def place_order(self, broker_name: str, symbol: str, side: str, quantity: float,
                   order_type: str = "market", **kwargs) -> Optional[Order]:
        """Place an order through a specific broker."""
        broker = self.get_broker(broker_name)
        if not broker or not broker.is_connected:
            self.logger.error(f"Broker {broker_name} not available")
            return None

        try:
            from .broker_interfaces import OrderSide, OrderType
            order_side = OrderSide(side.lower())
            order_type_enum = OrderType(order_type.lower())

            order = broker.place_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                order_type=order_type_enum,
                **kwargs
            )
            self.logger.info(f"Order placed via {broker_name}: {order.id}")
            return order

        except Exception as e:
            self.logger.error(f"Failed to place order via {broker_name}: {e}")
            return None

    def cancel_order(self, broker_name: str, order_id: str) -> bool:
        """Cancel an order through a specific broker."""
        broker = self.get_broker(broker_name)
        if not broker or not broker.is_connected:
            self.logger.error(f"Broker {broker_name} not available")
            return False

        try:
            return broker.cancel_order(order_id)
        except Exception as e:
            self.logger.error(f"Failed to cancel order via {broker_name}: {e}")
            return False

    def get_market_data(self, broker_name: str, symbol: str) -> Optional[MarketData]:
        """Get market data from a specific broker."""
        broker = self.get_broker(broker_name)
        if not broker or not broker.is_connected:
            return None

        try:
            return broker.get_market_data(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get market data from {broker_name}: {e}")
            return None

    def get_historical_data(self, broker_name: str, symbol: str, start_date: datetime,
                           end_date: datetime, timeframe: str = "1D") -> Optional[Any]:
        """Get historical data from a specific broker."""
        broker = self.get_broker(broker_name)
        if not broker or not broker.is_connected:
            return None

        try:
            return broker.get_historical_data(symbol, start_date, end_date, timeframe)
        except Exception as e:
            self.logger.error(f"Failed to get historical data from {broker_name}: {e}")
            return None

    def _load_config(self):
        """Load broker configurations from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)

                for broker_data in data.get('brokers', []):
                    config = BrokerConfig(**broker_data)
                    if config.enabled:
                        self.add_broker(config)

                self.logger.info(f"Loaded broker configurations from {self.config_file}")
            else:
                self.logger.info("No broker configuration file found, starting with empty config")

        except Exception as e:
            self.logger.error(f"Failed to load broker config: {e}")

    def _save_config(self):
        """Save broker configurations to file."""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            data = {
                'brokers': [config.__dict__ for config in self.configs.values()]
            }

            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.info(f"Saved broker configurations to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Failed to save broker config: {e}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of broker statuses."""
        summary = {
            'total_brokers': len(self.brokers),
            'connected_brokers': 0,
            'total_positions': 0,
            'total_orders': 0,
            'last_update': self.last_update.isoformat(),
            'brokers': {}
        }

        for name, broker in self.brokers.items():
            broker_info = {
                'connected': broker.is_connected,
                'type': type(broker).__name__,
                'positions': 0,
                'orders': 0
            }

            if broker.is_connected:
                summary['connected_brokers'] += 1
                try:
                    positions = broker.get_positions()
                    orders = broker.get_orders()
                    broker_info['positions'] = len(positions)
                    broker_info['orders'] = len(orders)
                    summary['total_positions'] += len(positions)
                    summary['total_orders'] += len(orders)
                except Exception as e:
                    self.logger.error(f"Failed to get status for broker {name}: {e}")

            summary['brokers'][name] = broker_info

        return summary