"""
Alpaca broker implementation for live trading.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    REST = None
    TimeFrame = None

from .broker_interfaces import (
    BrokerInterface, Order, OrderType, OrderSide, OrderStatus,
    Position, PositionSide, Account, MarketData, BrokerError,
    ConnectionError, AuthenticationError, OrderError, InsufficientFundsError
)


class AlpacaBroker(BrokerInterface):
    """Alpaca broker implementation."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.alpaca.markets"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.api: Optional[REST] = None
        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "Alpaca"

    @property
    def is_connected(self) -> bool:
        return self.api is not None

    def connect(self, credentials: Dict[str, str]) -> bool:
        """Connect to Alpaca API."""
        if not ALPACA_AVAILABLE:
            raise BrokerError("Alpaca SDK not installed. Install with: pip install alpaca-trade-api")

        try:
            api_key = credentials.get('api_key', self.api_key)
            api_secret = credentials.get('api_secret', self.api_secret)
            base_url = credentials.get('base_url', self.base_url)

            self.api = REST(api_key, api_secret, base_url)
            # Test connection
            self.api.get_account()
            self.logger.info("Successfully connected to Alpaca")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.api = None
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")

    def disconnect(self) -> bool:
        """Disconnect from Alpaca API."""
        self.api = None
        self.logger.info("Disconnected from Alpaca")
        return True

    def get_account(self) -> Account:
        """Get account information."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            account = self.api.get_account()

            return Account(
                id=account.id,
                balance=float(account.portfolio_value),
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                currency=account.currency,
                margin_used=float(getattr(account, 'margin_used', 0)),
                margin_available=float(getattr(account, 'margin_available', 0)),
                day_trades=int(getattr(account, 'daytrade_count', 0))
            )

        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise BrokerError(f"Failed to get account info: {e}")

    def get_positions(self) -> List[Position]:
        """Get current positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            positions = self.api.list_positions()
            result = []

            for pos in positions:
                # Get current price
                try:
                    current_price = float(self.api.get_latest_quote(pos.symbol).askprice)
                except:
                    current_price = float(pos.current_price)

                side = PositionSide.LONG if int(pos.qty) > 0 else PositionSide.SHORT

                position = Position(
                    symbol=pos.symbol,
                    side=side,
                    quantity=abs(float(pos.qty)),
                    avg_price=float(pos.avg_entry_price),
                    current_price=current_price,
                    unrealized_pnl=float(pos.unrealized_pl),
                    realized_pnl=float(pos.realized_pl),
                    market_value=float(pos.market_value)
                )
                result.append(position)

            return result

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Map our status to Alpaca status
            alpaca_status = None
            if status:
                status_mapping = {
                    OrderStatus.PENDING: 'new',
                    OrderStatus.FILLED: 'filled',
                    OrderStatus.CANCELLED: 'canceled',
                    OrderStatus.REJECTED: 'rejected'
                }
                alpaca_status = status_mapping.get(status.value)

            orders = self.api.list_orders(status=alpaca_status)
            result = []

            for order in orders:
                # Map Alpaca order to our Order
                side = OrderSide.BUY if order.side == 'buy' else OrderSide.SELL

                order_type = OrderType.MARKET
                if order.type == 'limit':
                    order_type = OrderType.LIMIT
                elif order.type == 'stop':
                    order_type = OrderType.STOP
                elif order.type == 'stop_limit':
                    order_type = OrderType.STOP_LIMIT

                status_mapping = {
                    'new': OrderStatus.PENDING,
                    'filled': OrderStatus.FILLED,
                    'partially_filled': OrderStatus.PARTIALLY_FILLED,
                    'canceled': OrderStatus.CANCELLED,
                    'rejected': OrderStatus.REJECTED,
                    'expired': OrderStatus.EXPIRED
                }
                order_status = status_mapping.get(order.status, OrderStatus.PENDING)

                our_order = Order(
                    id=order.id,
                    symbol=order.symbol,
                    side=side,
                    type=order_type,
                    quantity=float(order.qty),
                    price=float(order.filled_avg_price) if order.filled_avg_price else None,
                    stop_price=float(order.stop_price) if order.stop_price else None,
                    limit_price=float(order.limit_price) if order.limit_price else None,
                    status=order_status,
                    filled_quantity=float(order.filled_qty),
                    remaining_quantity=float(order.qty) - float(order.filled_qty),
                    created_at=order.created_at,
                    updated_at=order.updated_at,
                    fees=float(getattr(order, 'fees', 0))
                )
                result.append(our_order)

            return result

        except Exception as e:
            self.logger.error(f"Failed to get orders: {e}")
            raise BrokerError(f"Failed to get orders: {e}")

    def place_order(self, symbol: str, side: OrderSide, quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   limit_price: Optional[float] = None) -> Order:
        """Place a new order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Map our types to Alpaca
            alpaca_side = side.value
            alpaca_type = order_type.value

            # Prepare order data
            order_data = {
                'symbol': symbol,
                'qty': str(quantity),
                'side': alpaca_side,
                'type': alpaca_type,
                'time_in_force': 'gtc'  # Good till cancelled
            }

            if order_type == OrderType.LIMIT and limit_price:
                order_data['limit_price'] = str(limit_price)
            elif order_type == OrderType.STOP and stop_price:
                order_data['stop_price'] = str(stop_price)
            elif order_type == OrderType.STOP_LIMIT and stop_price and limit_price:
                order_data['stop_price'] = str(stop_price)
                order_data['limit_price'] = str(limit_price)

            # Place order
            alpaca_order = self.api.submit_order(**order_data)

            # Convert back to our Order
            our_order = Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                side=OrderSide.BUY if alpaca_order.side == 'buy' else OrderSide.SELL,
                type=OrderType(alpaca_order.type),
                quantity=float(alpaca_order.qty),
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                status=OrderStatus.PENDING,
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at
            )

            self.logger.info(f"Order placed: {our_order.id} for {quantity} {symbol}")
            return our_order

        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            else:
                raise OrderError(f"Failed to place order: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise OrderError(f"Failed to cancel order: {e}")

    def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)
            # Get latest trade
            trade = self.api.get_latest_trade(symbol)

            return MarketData(
                symbol=symbol,
                price=float(trade.price),
                bid=float(quote.bidprice),
                ask=float(quote.askprice),
                volume=float(getattr(trade, 'size', 0)),
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            raise BrokerError(f"Failed to get market data: {e}")

    def get_historical_data(self, symbol: str, start_date: datetime,
                           end_date: datetime, timeframe: str = "1D") -> pd.DataFrame:
        """Get historical market data."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Map timeframe
            timeframe_mapping = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame.Minute,
                '15Min': TimeFrame.Minute,
                '1H': TimeFrame.Hour,
                '1D': TimeFrame.Day
            }

            tf = timeframe_mapping.get(timeframe, TimeFrame.Day)

            # Alpaca expects different adjustment for timeframe
            adjustment = 'raw'
            if timeframe == '5Min':
                adjustment = '5Min'
            elif timeframe == '15Min':
                adjustment = '15Min'

            barset = self.api.get_bars(
                symbol, tf, start_date, end_date,
                adjustment=adjustment
            )

            # Convert to DataFrame
            data = []
            for bar in barset:
                data.append({
                    'timestamp': bar.t,
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': float(bar.v)
                })

            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise BrokerError(f"Failed to get historical data: {e}")

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Get active assets
            assets = self.api.list_assets(status='active')
            return [asset.symbol for asset in assets if asset.tradable]

        except Exception as e:
            self.logger.error(f"Failed to get supported symbols: {e}")
            raise BrokerError(f"Failed to get supported symbols: {e}")

    def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, List]:
        """Get order book for a symbol."""
        # Alpaca doesn't provide order book data through their API
        # This would need to be implemented differently or return empty
        return {'bids': [], 'asks': []}