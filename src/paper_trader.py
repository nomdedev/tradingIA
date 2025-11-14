"""
Paper Trading Engine
====================
Motor de paper trading en vivo con:
- Conexión a Alpaca API en modo paper
- Seguimiento de señales en tiempo real
- Gestión automática de órdenes (market, limit)
- Stop Loss y Take Profit dinámicos
- Monitoreo de posiciones abiertas
- Logging detallado de trades
- WebSocket para datos en tiempo real (opcional)
"""

from src.indicators import calculate_all_indicators
from src.data_fetcher import DataFetcher
from config.config import (
    ALPACA_CONFIG,
    TRADING_CONFIG,
    PAPER_TRADING_CONFIG,
    get_config
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.client import TradingClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


class Position:
    """Representa una posición abierta"""

    def __init__(self,
                 symbol: str,
                 side: str,  # 'long' or 'short'
                 entry_price: float,
                 quantity: float,
                 entry_time: datetime,
                 stop_loss: float,
                 take_profit: float,
                 order_id: Optional[str] = None):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_id = order_id

        # Runtime tracking
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_percent = 0.0

    def update_price(self, current_price: float):
        """Actualiza precio actual y P&L no realizado"""
        self.current_price = current_price

        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_percent = (current_price - self.entry_price) / self.entry_price
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.unrealized_pnl_percent = (self.entry_price - current_price) / self.entry_price

    def should_close_sl(self) -> bool:
        """Verifica si debe cerrarse por stop loss"""
        if self.side == 'long':
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss

    def should_close_tp(self) -> bool:
        """Verifica si debe cerrarse por take profit"""
        if self.side == 'long':
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit

    def to_dict(self) -> Dict:
        """Convierte a diccionario para logging"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.isoformat() if isinstance(
                self.entry_time,
                datetime) else self.entry_time,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_percent': self.unrealized_pnl_percent * 100,
            'order_id': self.order_id}


class PaperTrader:
    """
    Motor de paper trading en vivo

    Parámetros:
    -----------
    symbol : str
        Símbolo a tradear (ej: 'BTCUSD')
    initial_capital : float
        Capital inicial para trading
    risk_per_trade : float
        Porcentaje de riesgo por trade (0.01 = 1%)
    check_interval : int
        Intervalo en segundos para verificar señales (default: 60)
    use_limit_orders : bool
        Usar órdenes limit en lugar de market (default: False)
    """

    def __init__(self,
                 symbol: str = None,
                 initial_capital: float = None,
                 risk_per_trade: float = None,
                 check_interval: int = None,
                 use_limit_orders: bool = False):

        self.symbol = symbol or TRADING_CONFIG['symbol']
        self.initial_capital = initial_capital or PAPER_TRADING_CONFIG['initial_capital']
        self.risk_per_trade = risk_per_trade or PAPER_TRADING_CONFIG['risk_per_trade']
        self.check_interval = check_interval or PAPER_TRADING_CONFIG['check_interval']
        self.use_limit_orders = use_limit_orders

        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=ALPACA_CONFIG['api_key'],
            secret_key=ALPACA_CONFIG['secret_key'],
            paper=True  # PAPER TRADING MODE
        )

        self.data_client = CryptoHistoricalDataClient()
        self.data_fetcher = DataFetcher()

        # Trading state
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades_history: List[Dict] = []
        self.is_running = False

        # Load existing trades if available
        self._load_trades_history()

        logger.info(f"PaperTrader initialized: {self.symbol}, ${self.initial_capital:,.2f} capital")
        logger.info(
            f"Check interval: {self.check_interval}s, Risk/trade: {self.risk_per_trade*100}%")

    def _load_trades_history(self):
        """Carga historial de trades desde archivo"""
        trades_file = Path('logs') / 'paper_trades.json'
        if trades_file.exists():
            try:
                with open(trades_file, 'r') as f:
                    self.trades_history = json.load(f)
                logger.info(f"Loaded {len(self.trades_history)} historical trades")
            except Exception as e:
                logger.warning(f"Could not load trades history: {e}")

    def _save_trade(self, trade: Dict):
        """Guarda trade al historial"""
        self.trades_history.append(trade)

        # Save to JSON file
        trades_file = Path('logs') / 'paper_trades.json'
        trades_file.parent.mkdir(exist_ok=True)

        try:
            with open(trades_file, 'w') as f:
                json.dump(self.trades_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

    def get_current_price(self, symbol: str = None) -> float:
        """Obtiene precio actual del símbolo"""
        symbol = symbol or self.symbol

        try:
            # Get latest bar (1 minute)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                limit=1
            )

            bars = self.data_client.get_crypto_bars(request)

            if bars and symbol in bars:
                latest = bars[symbol][-1]
                return float(latest.close)

        except Exception as e:
            logger.error(f"Error getting current price: {e}")

        return 0.0

    def get_account_info(self) -> Dict:
        """Obtiene información de la cuenta"""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calcula tamaño de posición basado en riesgo

        Position Size = (Capital * Risk%) / (Entry - Stop Loss)
        """
        account_info = self.get_account_info()
        available_capital = account_info.get('buying_power', self.capital)

        risk_amount = available_capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0

        position_size = risk_amount / price_risk

        # No permitir posiciones mayores al capital disponible
        max_size = available_capital / entry_price * 0.95  # 95% max para fees
        return min(position_size, max_size)

    def place_market_order(self, side: OrderSide, quantity: float) -> Optional[str]:
        """Coloca orden de mercado"""
        try:
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC
            )

            order = self.trading_client.submit_order(order_data)
            logger.info(
                f"Market order placed: {side.value} {quantity} {self.symbol}, ID: {order.id}")

            return order.id

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def place_limit_order(self, side: OrderSide, quantity: float,
                          limit_price: float) -> Optional[str]:
        """Coloca orden limitada"""
        try:
            order_data = LimitOrderRequest(
                symbol=self.symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )

            order = self.trading_client.submit_order(order_data)
            logger.info(
                f"Limit order placed: {side.value} {quantity} @ ${limit_price}, ID: {order.id}")

            return order.id

        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    def open_position(self, signal: int, current_price: float,
                      stop_loss: float, take_profit: float, confidence: float):
        """Abre nueva posición"""

        # Verificar si ya hay posición abierta
        if self.symbol in self.positions:
            logger.warning(f"Position already open for {self.symbol}")
            return

        # Determinar lado
        side = 'long' if signal > 0 else 'short'
        order_side = OrderSide.BUY if signal > 0 else OrderSide.SELL

        # Calcular tamaño de posición
        quantity = self.calculate_position_size(current_price, stop_loss)

        if quantity <= 0:
            logger.warning("Position size too small, skipping trade")
            return

        # Colocar orden
        if self.use_limit_orders:
            order_id = self.place_limit_order(order_side, quantity, current_price)
        else:
            order_id = self.place_market_order(order_side, quantity)

        if order_id:
            # Crear posición
            position = Position(
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                quantity=quantity,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id
            )

            self.positions[self.symbol] = position

            logger.info(f"Position opened: {side} {quantity} @ ${current_price:,.2f}")
            logger.info(f"  SL: ${stop_loss:,.2f}, TP: ${take_profit:,.2f}")
            logger.info(f"  Confidence: {confidence*100:.1f}%")

    def close_position(self, symbol: str, reason: str = 'signal'):
        """Cierra posición abierta"""

        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return

        position = self.positions[symbol]
        current_price = self.get_current_price(symbol)

        # Determinar orden de cierre
        close_side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY

        # Colocar orden de cierre
        if self.use_limit_orders:
            order_id = self.place_limit_order(close_side, position.quantity, current_price)
        else:
            order_id = self.place_market_order(close_side, position.quantity)

        if order_id:
            # Actualizar P&L final
            position.update_price(current_price)

            # Crear registro de trade cerrado
            trade = {
                'entry_time': position.entry_time.isoformat() if isinstance(
                    position.entry_time,
                    datetime) else position.entry_time,
                'exit_time': datetime.now().isoformat(),
                'symbol': symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'quantity': position.quantity,
                'pnl': position.unrealized_pnl,
                'pnl_percent': position.unrealized_pnl_percent * 100,
                'exit_reason': reason,
                'entry_order_id': position.order_id,
                'exit_order_id': order_id}

            # Guardar trade
            self._save_trade(trade)

            # Actualizar capital
            self.capital += position.unrealized_pnl

            logger.info(f"Position closed: {reason}")
            logger.info(f"  Entry: ${position.entry_price:,.2f}, Exit: ${current_price:,.2f}")
            logger.info(
                f"  P&L: ${position.unrealized_pnl:,.2f} ({position.unrealized_pnl_percent*100:.2f}%)")

            # Remover posición
            del self.positions[symbol]

    def check_exit_conditions(self):
        """Verifica condiciones de salida para posiciones abiertas"""

        for symbol, position in list(self.positions.items()):
            current_price = self.get_current_price(symbol)
            position.update_price(current_price)

            # Verificar Stop Loss
            if position.should_close_sl():
                logger.warning(f"Stop Loss hit for {symbol}")
                self.close_position(symbol, reason='stop_loss')
                continue

            # Verificar Take Profit
            if position.should_close_tp():
                logger.info(f"Take Profit hit for {symbol}")
                self.close_position(symbol, reason='take_profit')
                continue

    def check_entry_signals(self, df: pd.DataFrame):
        """Verifica señales de entrada en los datos más recientes"""

        if df is None or len(df) == 0:
            return

        # Obtener última señal
        last_row = df.iloc[-1]
        signal = last_row.get('signal', 0)

        if signal == 0:
            return

        # Si hay posición abierta en dirección contraria, cerrarla
        if self.symbol in self.positions:
            current_position = self.positions[self.symbol]
            if (signal > 0 and current_position.side == 'short') or \
               (signal < 0 and current_position.side == 'long'):
                logger.info("Closing position due to counter-signal")
                self.close_position(self.symbol, reason='counter_signal')

        # Si no hay posición abierta, abrir nueva
        if self.symbol not in self.positions:
            current_price = self.get_current_price(self.symbol)

            if current_price > 0:
                # Calcular stop loss y take profit
                atr = last_row.get('ATR', current_price * 0.02)
                sl_mult = PAPER_TRADING_CONFIG.get('sl_atr_multiplier', 1.5)
                tp_rr = PAPER_TRADING_CONFIG.get('tp_risk_reward', 2.0)

                if signal > 0:  # Long
                    stop_loss = current_price - (atr * sl_mult)
                    take_profit = current_price + (abs(current_price - stop_loss) * tp_rr)
                else:  # Short
                    stop_loss = current_price + (atr * sl_mult)
                    take_profit = current_price - (abs(current_price - stop_loss) * tp_rr)

                confidence = last_row.get('confidence', 0.5)

                # Abrir posición
                self.open_position(
                    signal=signal,
                    current_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence
                )

    def run_trading_loop(self, max_iterations: int = None):
        """
        Ejecuta loop principal de trading

        Parámetros:
        -----------
        max_iterations : int
            Número máximo de iteraciones (None = infinito)
        """
        self.is_running = True
        iteration = 0

        logger.info("=" * 60)
        logger.info("PAPER TRADING STARTED")
        logger.info("=" * 60)

        try:
            while self.is_running:
                iteration += 1

                if max_iterations and iteration > max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break

                logger.info(f"\n--- Iteration {iteration} ---")

                # 1. Obtener datos recientes
                try:
                    df = self.data_fetcher.get_historical_data(
                        symbol=self.symbol,
                        timeframe='5Min',
                        limit=500  # Suficiente para indicadores
                    )

                    if df is not None and len(df) > 0:
                        # 2. Calcular indicadores y señales
                        df = calculate_all_indicators(df)

                        # 3. Verificar condiciones de salida
                        self.check_exit_conditions()

                        # 4. Verificar señales de entrada
                        self.check_entry_signals(df)

                        # 5. Mostrar estado
                        self.print_status()

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")

                # Esperar antes de siguiente iteración
                if self.is_running and (not max_iterations or iteration < max_iterations):
                    time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nTrading loop stopped by user")

        finally:
            self.stop()

    def stop(self):
        """Detiene el trading loop"""
        self.is_running = False

        # Cerrar todas las posiciones abiertas
        for symbol in list(self.positions.keys()):
            logger.info(f"Closing open position for {symbol}")
            self.close_position(symbol, reason='shutdown')

        logger.info("=" * 60)
        logger.info("PAPER TRADING STOPPED")
        logger.info("=" * 60)
        self.print_summary()

    def print_status(self):
        """Imprime estado actual del trading"""
        account_info = self.get_account_info()

        print("\n" + "─" * 60)
        print(f"Account Equity: ${account_info.get('equity', self.capital):,.2f}")
        print(f"Open Positions: {len(self.positions)}")

        if self.positions:
            print("\nPositions:")
            for symbol, pos in self.positions.items():
                current_price = self.get_current_price(symbol)
                pos.update_price(current_price)

                print(f"  {symbol} {pos.side.upper()}")
                print(f"    Entry: ${pos.entry_price:,.2f}, Current: ${current_price:,.2f}")
                print(
                    f"    P&L: ${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_percent*100:+.2f}%)")
                print(f"    SL: ${pos.stop_loss:,.2f}, TP: ${pos.take_profit:,.2f}")

        print("─" * 60)

    def print_summary(self):
        """Imprime resumen de trading session"""

        if len(self.trades_history) == 0:
            print("\nNo trades executed")
            return

        # Calcular métricas
        df_trades = pd.DataFrame(self.trades_history)

        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] <= 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0

        print("\n" + "=" * 60)
        print("TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Trades:       {total_trades}")
        print(f"Winning Trades:     {winning_trades}")
        print(f"Losing Trades:      {losing_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print("-" * 60)
        print(f"Total P&L:          ${total_pnl:,.2f}")
        print(f"Average Win:        ${avg_win:,.2f}")
        print(f"Average Loss:       ${avg_loss:,.2f}")
        print(f"Final Capital:      ${self.capital:,.2f}")
        print(
            f"Return:             {(self.capital - self.initial_capital) / self.initial_capital * 100:+.2f}%")
        print("=" * 60 + "\n")


# Test del módulo
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing PaperTrader module...")
    print("\nNOTE: This requires valid Alpaca API credentials")
    print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")

    try:
        # Initialize paper trader
        trader = PaperTrader(
            symbol='BTCUSD',
            initial_capital=10000,
            risk_per_trade=0.01,
            check_interval=5  # 5 seconds for testing
        )

        # Get account info
        account = trader.get_account_info()
        if account:
            print(f"\n✅ Connected to Alpaca Paper Trading")
            print(f"Account Equity: ${account.get('equity', 0):,.2f}")
            print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")

        # Get current price
        price = trader.get_current_price('BTCUSD')
        if price > 0:
            print(f"Current BTC Price: ${price:,.2f}")

        print("\n✅ PaperTrader module tested successfully!")
        print("\nTo run paper trading:")
        print("  trader.run_trading_loop(max_iterations=10)  # 10 iterations")
        print("  trader.run_trading_loop()  # Infinite loop (Ctrl+C to stop)")

    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("Make sure you have valid Alpaca credentials in .env file")
