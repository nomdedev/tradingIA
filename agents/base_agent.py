import logging
import sys
import os
from typing import Dict, List, Any
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

# Añadir el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BaseAgent(ABC):
    """
    Clase base para todos los agentes de trading.
    Inspirada en Moon Dev AI Agents pero simplificada para stocks.

    Todos los agentes (RL, GA, Risk, Sentiment) heredan de aquí.
    """

    def __init__(self, name: str, initial_balance: float = 10000):
        self.name = name
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares = 0
        self.positions: List[Dict] = []  # Historial de posiciones
        self.trades_history: List[Dict] = []  # Historial completo de trades
        self.logger = self._setup_logger()
        self.metrics: Dict[str, Any] = {}

        # Estado del portfolio
        self.portfolio_value = initial_balance
        self.peak_value = initial_balance
        self.current_drawdown = 0.0

        self.logger.info(f"🤖 {self.name} inicializado con balance: ${initial_balance:,.2f}")

    def _setup_logger(self) -> logging.Logger:
        """Configurar logger con nombre del agente"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Evitar duplicar handlers
        if not logger.handlers:
            # Handler para consola
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Formato
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        return logger

    @abstractmethod
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método abstracto - cada agente lo implementa.

        Args:
            market_data: Diccionario con datos de mercado actuales
                {
                    'Close': float,
                    'Open': float,
                    'High': float,
                    'Low': float,
                    'Volume': float,
                    'RSI': float,
                    'MACD': float,
                    ... otros indicadores
                }

        Returns:
            Dict con decisión del agente:
            {
                'action': int (0=HOLD, 1=BUY, 2=SELL),
                'confidence': float (0-1),
                'reasoning': str (explicación de la decisión)
            }
        """
        pass

    def execute_trade(self, action: int, price: float, size: float = None, stop_loss: float = None) -> bool:
        """
        Lógica común de ejecución de trades.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            price: Precio de ejecución
            size: Número de shares (opcional, usa todo disponible para SELL)
            stop_loss: Precio de stop loss (opcional)

        Returns:
            bool: True si trade ejecutado exitosamente
        """
        if action == 0:  # HOLD
            return True

        if action == 1:  # BUY
            if size is None or size <= 0:
                self.logger.warning("❌ BUY sin tamaño especificado")
                return False

            cost = size * price
            if cost > self.balance:
                self.logger.warning(f"❌ Fondos insuficientes: ${self.balance:,.2f} < ${cost:,.2f}")
                return False

            # Ejecutar compra
            self.balance -= cost
            self.shares += size

            trade = {
                'timestamp': datetime.now(),
                'action': 'BUY',
                'price': price,
                'size': size,
                'cost': cost,
                'stop_loss': stop_loss,
                'portfolio_value': self.get_portfolio_value(price)
            }

            self.positions.append(trade)
            self.trades_history.append(trade)

            self.logger.info(f"✅ BUY: {size:.2f} shares @ ${price:.2f} = ${cost:,.2f}")

        elif action == 2:  # SELL
            if self.shares <= 0:
                self.logger.warning("❌ SELL sin posición abierta")
                return False

            # Si no se especifica size, vender todo
            sell_size = size if size is not None else self.shares
            sell_size = min(sell_size, self.shares)  # No vender más de lo que tenemos

            revenue = sell_size * price
            self.balance += revenue
            self.shares -= sell_size

            # Calcular P&L
            # Buscar posición correspondiente (simplificado: FIFO)
            pnl = 0
            remaining_to_sell = sell_size

            for position in reversed(self.positions):
                if position['action'] == 'BUY' and remaining_to_sell > 0:
                    sell_from_position = min(remaining_to_sell, position['size'])
                    position_pnl = sell_from_position * (price - position['price'])
                    pnl += position_pnl
                    remaining_to_sell -= sell_from_position

                    if remaining_to_sell <= 0:
                        break

            trade = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'price': price,
                'size': sell_size,
                'revenue': revenue,
                'pnl': pnl,
                'portfolio_value': self.get_portfolio_value(price)
            }

            self.trades_history.append(trade)

            self.logger.info(f"✅ SELL: {sell_size:.2f} shares @ ${price:.2f} = ${revenue:,.2f} | P&L: ${pnl:,.2f}")

        return True

    def update_portfolio(self, current_price: float):
        """Actualizar valor del portfolio y métricas de riesgo"""
        self.portfolio_value = self.get_portfolio_value(current_price)

        # Actualizar peak y drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

    def get_portfolio_value(self, current_price: float = None) -> float:
        """Calcular valor total del portfolio"""
        if current_price is None and self.positions:
            # Usar precio de la última posición
            current_price = self.positions[-1]['price']

        if current_price is None:
            return self.balance

        return self.balance + (self.shares * current_price)

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Obtener estado completo del portfolio"""
        return {
            'balance': self.balance,
            'shares': self.shares,
            'net_worth': self.portfolio_value,
            'peak_value': self.peak_value,
            'current_drawdown': self.current_drawdown,
            'total_trades': len(self.trades_history),
            'open_positions': len([p for p in self.positions if p['action'] == 'BUY'])
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calcular métricas de performance completas"""
        if not self.trades_history:
            return {
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0
            }

        # Calcular retornos
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

        # Trades con P&L
        pnl_trades = [t for t in self.trades_history if 'pnl' in t]
        winning_trades = [t for t in pnl_trades if t['pnl'] > 0]
        losing_trades = [t for t in pnl_trades if t['pnl'] < 0]

        win_rate = len(winning_trades) / len(pnl_trades) if pnl_trades else 0

        # Profit factor
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Sharpe ratio (simplificado)
        if len(pnl_trades) > 1:
            returns = [t['pnl'] / self.initial_balance for t in pnl_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Average trade
        avg_trade = np.mean([t['pnl'] for t in pnl_trades]) if pnl_trades else 0

        return {
            'total_return': total_return,
            'total_trades': len(pnl_trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.current_drawdown,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'total_wins': len(winning_trades),
            'total_losses': len(losing_trades)
        }

    def log_decision(self, market_data: Dict[str, Any], decision: Dict[str, Any]):
        """Log estandarizado de decisiones"""
        self.logger.info(f"📊 {self.name} | Price: ${market_data.get('Close', 0):.2f} | "
                        f"Action: {decision.get('action', 0)} | "
                        f"Confidence: {decision.get('confidence', 0):.2f} | "
                        f"Reasoning: {decision.get('reasoning', 'No reasoning')}")

    def reset(self):
        """Resetear agente a estado inicial"""
        self.balance = self.initial_balance
        self.shares = 0
        self.positions = []
        self.trades_history = []
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.current_drawdown = 0.0
        self.logger.info(f"🔄 {self.name} reseteado")