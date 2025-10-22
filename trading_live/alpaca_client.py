"""
Alpaca Paper Trading Client
Cliente completo para paper trading con Alpaca Markets API
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import pytz

from alpaca_trade_api import REST, TimeFrame
from alpaca_trade_api.entity import Account, Position, Order

logger = logging.getLogger(__name__)

class AlpacaClient:
    """
    Cliente completo para interactuar con Alpaca Markets API
    Maneja paper trading con todas las funcionalidades necesarias
    """

    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None):
        """
        Inicializar cliente Alpaca

        Args:
            api_key: API key de Alpaca (opcional, usa env var si no se proporciona)
            secret_key: Secret key de Alpaca (opcional, usa env var si no se proporciona)
            base_url: Base URL (opcional, usa env var si no se proporciona)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not all([self.api_key, self.secret_key]):
            raise ValueError("API_KEY y SECRET_KEY son requeridos. Configurar variables de entorno o pasar como parámetros.")

        try:
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url
            )
            logger.info("✅ Alpaca client inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando Alpaca client: {e}")
            raise

    def get_account_info(self) -> Account:
        """
        Obtener información completa de la cuenta

        Returns:
            Account: Información de la cuenta Alpaca
        """
        try:
            account = self.api.get_account()
            logger.info(f"📊 Cuenta: ${account.cash} cash, ${account.portfolio_value} portfolio")
            return account
        except Exception as e:
            logger.error(f"❌ Error obteniendo info de cuenta: {e}")
            raise

    def get_positions(self) -> List[Position]:
        """
        Obtener todas las posiciones abiertas

        Returns:
            List[Position]: Lista de posiciones
        """
        try:
            positions = self.api.list_positions()
            logger.info(f"📈 Posiciones abiertas: {len(positions)}")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.qty} @ ${pos.avg_entry_price}")
            return positions
        except Exception as e:
            logger.error(f"❌ Error obteniendo posiciones: {e}")
            raise

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Obtener posición específica

        Args:
            symbol: Símbolo del activo

        Returns:
            Position or None: Posición si existe
        """
        try:
            position = self.api.get_position(symbol)
            return position
        except Exception as e:
            logger.debug(f"No position found for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """
        Obtener precio actual de un símbolo

        Args:
            symbol: Símbolo del activo

        Returns:
            float: Precio actual
        """
        try:
            quote = self.api.get_latest_quote(symbol)
            # Alpaca API v2 usa 'ap' y 'bp' para ask price y bid price
            ask_price = getattr(quote, 'ap', getattr(quote, 'askprice', None))
            bid_price = getattr(quote, 'bp', getattr(quote, 'bidprice', None))

            if ask_price is None or bid_price is None:
                logger.warning(f"⚠️ No se pudieron obtener precios ask/bid para {symbol}, intentando último trade")
                # Fallback: usar último trade price
                last_trade = self.api.get_latest_trade(symbol)
                price = last_trade.price
            else:
                price = (ask_price + bid_price) / 2  # Mid price

            logger.debug(f"💰 {symbol} precio actual: ${price:.2f}")
            return price
        except Exception as e:
            logger.error(f"❌ Error obteniendo precio de {symbol}: {e}")
            raise

    def get_historical_data(self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day) -> pd.DataFrame:
        """
        Obtener datos históricos

        Args:
            symbol: Símbolo del activo
            days: Número de días de historial
            timeframe: Marco temporal

        Returns:
            pd.DataFrame: Datos históricos
        """
        try:
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days)

            bars = self.api.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                limit=1000
            )

            # Convertir a DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })

            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            logger.info(f"📊 Datos históricos obtenidos para {symbol}: {len(df)} barras")
            return df

        except Exception as e:
            logger.error(f"❌ Error obteniendo datos históricos de {symbol}: {e}")
            raise

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market',
                    time_in_force: str = 'day', limit_price: float = None, stop_price: float = None) -> Order:
        """
        Enviar orden de compra/venta

        Args:
            symbol: Símbolo del activo
            qty: Cantidad
            side: 'buy' o 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', etc.
            limit_price: Precio límite (para limit orders)
            stop_price: Precio stop (para stop orders)

        Returns:
            Order: Orden enviada
        """
        try:
            # Validar parámetros
            if side not in ['buy', 'sell']:
                raise ValueError("Side debe ser 'buy' o 'sell'")
            if order_type not in ['market', 'limit', 'stop', 'stop_limit']:
                raise ValueError("Order type inválido")

            # Preparar parámetros de orden
            order_params = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }

            if limit_price:
                order_params['limit_price'] = limit_price
            if stop_price:
                order_params['stop_price'] = stop_price

            # Enviar orden
            order = self.api.submit_order(**order_params)

            logger.info(f"📝 Orden enviada: {side.upper()} {qty} {symbol} @ {order_type.upper()}")
            logger.info(f"   Order ID: {order.id}, Status: {order.status}")

            return order

        except Exception as e:
            logger.error(f"❌ Error enviando orden {side} {qty} {symbol}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancelar orden

        Args:
            order_id: ID de la orden

        Returns:
            bool: True si cancelada exitosamente
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"❌ Orden cancelada: {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cancelando orden {order_id}: {e}")
            return False

    def get_orders(self, status: str = None, limit: int = 100) -> List[Order]:
        """
        Obtener órdenes

        Args:
            status: Filtrar por status ('open', 'closed', 'all')
            limit: Número máximo de órdenes

        Returns:
            List[Order]: Lista de órdenes
        """
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            logger.info(f"📋 Órdenes encontradas: {len(orders)}")
            return orders
        except Exception as e:
            logger.error(f"❌ Error obteniendo órdenes: {e}")
            raise

    def get_open_orders(self) -> List[Order]:
        """
        Obtener órdenes abiertas

        Returns:
            List[Order]: Órdenes abiertas
        """
        return self.get_orders(status='open')

    def is_market_open(self) -> bool:
        """
        Verificar si el mercado está abierto

        Returns:
            bool: True si mercado abierto
        """
        try:
            clock = self.api.get_clock()
            is_open = clock.is_open
            next_open = clock.next_open if not is_open else None
            next_close = clock.next_close if is_open else None

            logger.debug(f"🕐 Mercado {'abierto' if is_open else 'cerrado'}")
            if next_open:
                logger.debug(f"   Próxima apertura: {next_open}")
            if next_close:
                logger.debug(f"   Próximo cierre: {next_close}")

            return is_open

        except Exception as e:
            logger.error(f"❌ Error verificando estado del mercado: {e}")
            return False

    def get_market_hours(self) -> Dict[str, Any]:
        """
        Obtener información de horarios del mercado

        Returns:
            Dict: Información de horarios
        """
        try:
            clock = self.api.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'timestamp': clock.timestamp
            }
        except Exception as e:
            logger.error(f"❌ Error obteniendo horarios del mercado: {e}")
            raise

    def calculate_position_size(self, symbol: str, risk_percentage: float = 0.02,
                              stop_loss_percentage: float = 0.02) -> float:
        """
        Calcular tamaño de posición basado en riesgo

        Args:
            symbol: Símbolo del activo
            risk_percentage: Porcentaje de riesgo (ej: 0.02 = 2%)
            stop_loss_percentage: Porcentaje de stop loss (ej: 0.02 = 2%)

        Returns:
            float: Cantidad de acciones a comprar/vender
        """
        try:
            account = self.get_account_info()
            current_price = self.get_current_price(symbol)

            # Capital disponible para riesgo
            risk_amount = float(account.portfolio_value) * risk_percentage

            # Pérdida máxima por acción
            max_loss_per_share = current_price * stop_loss_percentage

            # Cantidad máxima de acciones
            max_qty = risk_amount / max_loss_per_share

            logger.info(f"📊 Position sizing for {symbol}:")
            logger.info(f"   Risk amount: ${risk_amount:.2f}")
            logger.info(f"   Max loss per share: ${max_loss_per_share:.2f}")
            logger.info(f"   Max quantity: {max_qty:.0f}")

            return max_qty

        except Exception as e:
            logger.error(f"❌ Error calculando position size para {symbol}: {e}")
            raise

    def get_portfolio_value(self) -> float:
        """
        Obtener valor total del portfolio

        Returns:
            float: Valor del portfolio
        """
        try:
            account = self.get_account_info()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error(f"❌ Error obteniendo valor del portfolio: {e}")
            raise

    def get_buying_power(self) -> float:
        """
        Obtener buying power disponible

        Returns:
            float: Buying power
        """
        try:
            account = self.get_account_info()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"❌ Error obteniendo buying power: {e}")
            raise

    def get_cash(self) -> float:
        """
        Obtener efectivo disponible

        Returns:
            float: Efectivo disponible
        """
        try:
            account = self.get_account_info()
            return float(account.cash)
        except Exception as e:
            logger.error(f"❌ Error obteniendo efectivo: {e}")
            raise

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen completo de la cuenta

        Returns:
            Dict: Resumen de cuenta
        """
        try:
            account = self.get_account_info()
            positions = self.get_positions()
            open_orders = self.get_open_orders()

            summary = {
                'account_id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'positions_count': len(positions),
                'open_orders_count': len(open_orders),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc)
                    } for pos in positions
                ],
                'open_orders': [
                    {
                        'id': order.id,
                        'symbol': order.symbol,
                        'qty': order.qty,
                        'side': order.side,
                        'type': order.type,
                        'status': order.status,
                        'submitted_at': order.submitted_at
                    } for order in open_orders
                ]
            }

            logger.info("📊 Resumen de cuenta generado")
            return summary

        except Exception as e:
            logger.error(f"❌ Error generando resumen de cuenta: {e}")
            raise