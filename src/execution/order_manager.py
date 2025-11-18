"""
Order Management System with Realistic Order Types

Implements:
- Market orders (immediate execution)
- Limit orders (price-conditional)
- Stop orders (trigger-based)
- Trailing stop orders (dynamic)
- Order timeout/expiration
- Partial fills simulation
- Order rejection handling
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging


class OrderType(Enum):
    """Order types supported"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order with all parameters.
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    trailing_offset: Optional[float] = None  # For trailing stops (percentage)
    time_in_force: str = "GTC"  # GTC, IOC, FOK, DAY
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    
    # Rejection/cancellation
    rejection_reason: Optional[str] = None
    
    # Trailing stop tracking
    highest_price: Optional[float] = None  # For trailing stops
    lowest_price: Optional[float] = None

    def __post_init__(self):
        """Validate order parameters"""
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive: {self.quantity}")
        
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit order requires price")
        
        if self.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            if self.stop_price is None:
                raise ValueError(f"{self.order_type} requires stop_price")
        
        if self.order_type == OrderType.TRAILING_STOP:
            if self.trailing_offset is None:
                raise ValueError("Trailing stop requires trailing_offset")

    @property
    def remaining_quantity(self) -> float:
        """Quantity still to be filled"""
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]

    def add_fill(self, quantity: float, price: float, timestamp: datetime):
        """Record a partial or full fill"""
        if quantity <= 0:
            raise ValueError(f"Fill quantity must be positive: {quantity}")
        
        if self.filled_quantity + quantity > self.quantity:
            raise ValueError(f"Fill would exceed order quantity")
        
        # Record fill
        self.fills.append({
            'timestamp': timestamp,
            'quantity': quantity,
            'price': price
        })
        
        # Update totals
        total_value = self.avg_fill_price * self.filled_quantity + price * quantity
        self.filled_quantity += quantity
        self.avg_fill_price = total_value / self.filled_quantity
        
        # Update status
        if self.is_complete:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL


class OrderManager:
    """
    Manages order lifecycle and execution simulation.
    """

    def __init__(self, account_balance: float = 10000, enable_partial_fills: bool = True):
        """
        Initialize order manager.

        Args:
            account_balance: Available capital
            enable_partial_fills: Whether to simulate partial fills
        """
        self.account_balance = account_balance
        self.enable_partial_fills = enable_partial_fills
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.logger = logging.getLogger(__name__)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trailing_offset: Optional[float] = None,
        timeout_bars: Optional[int] = None
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            trailing_offset: Trailing offset percentage (for trailing stops)
            timeout_bars: Auto-cancel after N bars

        Returns:
            Order object
        """
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"

        # Calculate expiration
        expires_at = None
        if timeout_bars is not None:
            # This will be set properly when processing
            expires_at = datetime.now() + timedelta(seconds=timeout_bars * 60)  # Assume 1min bars

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trailing_offset=trailing_offset,
            expires_at=expires_at
        )

        self.orders[order_id] = order
        self.logger.info(f"Created order: {order_id} {side.value} {quantity} {symbol} @ {order_type.value}")
        
        return order

    def process_orders(
        self,
        current_price: float,
        current_timestamp: datetime,
        high: float,
        low: float,
        volume: float,
        avg_volume: float
    ) -> List[Order]:
        """
        Process all active orders against current market data.

        Args:
            current_price: Current market price
            current_timestamp: Current timestamp
            high: Bar high
            low: Bar low
            volume: Current bar volume
            avg_volume: Average volume

        Returns:
            List of orders that were filled (fully or partially)
        """
        filled_orders = []

        for order in list(self.orders.values()):
            if not order.is_active:
                continue

            # Check expiration
            if order.expires_at and current_timestamp >= order.expires_at:
                order.status = OrderStatus.EXPIRED
                self.logger.info(f"Order {order.order_id} expired")
                continue

            # Process based on order type
            if order.order_type == OrderType.MARKET:
                filled = self._process_market_order(order, current_price, current_timestamp, volume, avg_volume)
            
            elif order.order_type == OrderType.LIMIT:
                filled = self._process_limit_order(order, current_price, current_timestamp, high, low, volume, avg_volume)
            
            elif order.order_type == OrderType.STOP_MARKET:
                filled = self._process_stop_market_order(order, current_price, current_timestamp, high, low, volume, avg_volume)
            
            elif order.order_type == OrderType.TRAILING_STOP:
                filled = self._process_trailing_stop_order(order, current_price, current_timestamp, high, low, volume, avg_volume)
            
            else:
                self.logger.warning(f"Unsupported order type: {order.order_type}")
                filled = False

            if filled:
                filled_orders.append(order)

        return filled_orders

    def _process_market_order(
        self,
        order: Order,
        price: float,
        timestamp: datetime,
        volume: float,
        avg_volume: float
    ) -> bool:
        """
        Process market order (immediate execution).

        Market orders execute immediately but may have partial fills
        if order size is large relative to volume.
        """
        remaining = order.remaining_quantity

        # Check if we can fill completely
        if self.enable_partial_fills:
            # Partial fill logic: can only fill up to X% of current volume
            max_fill_pct = 0.1  # Can fill max 10% of bar volume
            max_fillable = (volume * max_fill_pct) if volume > 0 else remaining

            fill_quantity = min(remaining, max_fillable)
        else:
            fill_quantity = remaining

        if fill_quantity > 0:
            order.add_fill(fill_quantity, price, timestamp)
            self.logger.info(
                f"Market order {order.order_id} filled: {fill_quantity:.4f} @ {price:.2f} "
                f"({order.filled_quantity/order.quantity:.1%} complete)"
            )
            return True

        return False

    def _process_limit_order(
        self,
        order: Order,
        price: float,
        timestamp: datetime,
        high: float,
        low: float,
        volume: float,
        avg_volume: float
    ) -> bool:
        """
        Process limit order (price-conditional execution).

        Buy limit: Executes if price <= limit_price
        Sell limit: Executes if price >= limit_price
        """
        limit_price = order.price

        # Check if limit price was touched during this bar
        if order.side == OrderSide.BUY:
            # Buy limit: execute if low <= limit_price
            if low <= limit_price:
                fill_price = min(limit_price, price)  # Best available price
                fill_quantity = order.remaining_quantity
                
                if self.enable_partial_fills and volume > 0:
                    max_fill_pct = 0.15  # Limit orders can take more volume
                    max_fillable = volume * max_fill_pct
                    fill_quantity = min(fill_quantity, max_fillable)
                
                if fill_quantity > 0:
                    order.add_fill(fill_quantity, fill_price, timestamp)
                    self.logger.info(f"Buy limit {order.order_id} filled: {fill_quantity:.4f} @ {fill_price:.2f}")
                    return True

        elif order.side == OrderSide.SELL:
            # Sell limit: execute if high >= limit_price
            if high >= limit_price:
                fill_price = max(limit_price, price)
                fill_quantity = order.remaining_quantity
                
                if self.enable_partial_fills and volume > 0:
                    max_fill_pct = 0.15
                    max_fillable = volume * max_fill_pct
                    fill_quantity = min(fill_quantity, max_fillable)
                
                if fill_quantity > 0:
                    order.add_fill(fill_quantity, fill_price, timestamp)
                    self.logger.info(f"Sell limit {order.order_id} filled: {fill_quantity:.4f} @ {fill_price:.2f}")
                    return True

        return False

    def _process_stop_market_order(
        self,
        order: Order,
        price: float,
        timestamp: datetime,
        high: float,
        low: float,
        volume: float,
        avg_volume: float
    ) -> bool:
        """
        Process stop market order (trigger then market).

        Buy stop: Triggers if price >= stop_price
        Sell stop: Triggers if price <= stop_price
        """
        stop_price = order.stop_price

        triggered = False

        if order.side == OrderSide.BUY:
            # Buy stop: trigger if high >= stop_price
            if high >= stop_price:
                triggered = True
                fill_price = max(stop_price, price)  # Slippage: execute at worse price
        
        elif order.side == OrderSide.SELL:
            # Sell stop: trigger if low <= stop_price
            if low <= stop_price:
                triggered = True
                fill_price = min(stop_price, price)  # Slippage: execute at worse price

        if triggered:
            fill_quantity = order.remaining_quantity
            
            if self.enable_partial_fills and volume > 0:
                max_fill_pct = 0.1
                max_fillable = volume * max_fill_pct
                fill_quantity = min(fill_quantity, max_fillable)
            
            if fill_quantity > 0:
                order.add_fill(fill_quantity, fill_price, timestamp)
                self.logger.info(f"Stop market {order.order_id} triggered and filled: {fill_quantity:.4f} @ {fill_price:.2f}")
                return True

        return False

    def _process_trailing_stop_order(
        self,
        order: Order,
        price: float,
        timestamp: datetime,
        high: float,
        low: float,
        volume: float,
        avg_volume: float
    ) -> bool:
        """
        Process trailing stop order (dynamic stop that follows price).

        For long positions:
        - Trailing stop rises with price but never falls
        - Triggers if price drops by trailing_offset from highest

        For short positions:
        - Trailing stop falls with price but never rises
        - Triggers if price rises by trailing_offset from lowest
        """
        trailing_offset = order.trailing_offset

        if order.side == OrderSide.SELL:  # Trailing stop for long position
            # Update highest price seen
            if order.highest_price is None or high > order.highest_price:
                order.highest_price = high
            
            # Calculate current stop price
            stop_price = order.highest_price * (1 - trailing_offset)
            
            # Check if stop was hit
            if low <= stop_price:
                fill_price = min(stop_price, price)
                fill_quantity = order.remaining_quantity
                
                order.add_fill(fill_quantity, fill_price, timestamp)
                self.logger.info(
                    f"Trailing stop {order.order_id} triggered: {fill_quantity:.4f} @ {fill_price:.2f} "
                    f"(high: {order.highest_price:.2f}, trail: {trailing_offset:.2%})"
                )
                return True

        elif order.side == OrderSide.BUY:  # Trailing stop for short position
            # Update lowest price seen
            if order.lowest_price is None or low < order.lowest_price:
                order.lowest_price = low
            
            # Calculate current stop price
            stop_price = order.lowest_price * (1 + trailing_offset)
            
            # Check if stop was hit
            if high >= stop_price:
                fill_price = max(stop_price, price)
                fill_quantity = order.remaining_quantity
                
                order.add_fill(fill_quantity, fill_price, timestamp)
                self.logger.info(
                    f"Trailing stop {order.order_id} triggered: {fill_quantity:.4f} @ {fill_price:.2f} "
                    f"(low: {order.lowest_price:.2f}, trail: {trailing_offset:.2%})"
                )
                return True

        return False

    def cancel_order(self, order_id: str, reason: str = "Manual cancellation"):
        """Cancel an active order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                order.rejection_reason = reason
                self.logger.info(f"Order {order_id} cancelled: {reason}")

    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return [order for order in self.orders.values() if order.is_active]

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("Order Management System - Example Usage")
    print("=" * 70)

    # Create order manager
    manager = OrderManager(account_balance=10000, enable_partial_fills=True)

    # Create sample orders
    print("\n1. Creating orders...")
    
    # Market order
    market_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )
    
    # Limit order
    limit_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.2,
        price=49000
    )
    
    # Stop market order
    stop_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.15,
        stop_price=48000
    )
    
    # Trailing stop
    trailing_order = manager.create_order(
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.TRAILING_STOP,
        quantity=0.1,
        trailing_offset=0.02  # 2% trailing
    )

    print(f"\nCreated {len(manager.orders)} orders")

    # Simulate market data
    print("\n2. Processing orders with market data...")
    
    market_data = [
        {'timestamp': datetime(2024, 1, 1, 9, 0), 'open': 50000, 'high': 50200, 'low': 49800, 'close': 50100, 'volume': 100},
        {'timestamp': datetime(2024, 1, 1, 9, 5), 'open': 50100, 'high': 50300, 'low': 49900, 'close': 49950, 'volume': 120},
        {'timestamp': datetime(2024, 1, 1, 9, 10), 'open': 49950, 'high': 50000, 'low': 48900, 'close': 49000, 'volume': 150},
        {'timestamp': datetime(2024, 1, 1, 9, 15), 'open': 49000, 'high': 49200, 'low': 47800, 'close': 48000, 'volume': 200},
    ]

    for bar in market_data:
        print(f"\n--- Bar: {bar['timestamp']} | Price: {bar['close']} | Volume: {bar['volume']} ---")
        
        filled = manager.process_orders(
            current_price=bar['close'],
            current_timestamp=bar['timestamp'],
            high=bar['high'],
            low=bar['low'],
            volume=bar['volume'],
            avg_volume=100
        )
        
        if filled:
            print(f"  ✓ {len(filled)} order(s) filled")
        
        active = manager.get_active_orders()
        print(f"  Active orders: {len(active)}")

    # Print final order status
    print("\n3. Final Order Status:")
    print("=" * 70)
    
    for order_id, order in manager.orders.items():
        print(f"\n{order_id}:")
        print(f"  Type: {order.order_type.value}")
        print(f"  Side: {order.side.value}")
        print(f"  Quantity: {order.quantity}")
        print(f"  Status: {order.status.value}")
        print(f"  Filled: {order.filled_quantity:.4f} ({order.filled_quantity/order.quantity:.1%})")
        if order.status == OrderStatus.FILLED:
            print(f"  Avg Fill Price: ${order.avg_fill_price:,.2f}")
            print(f"  Fills: {len(order.fills)}")

    print("\n" + "=" * 70)
