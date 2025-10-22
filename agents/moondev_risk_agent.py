from base_agent import BaseAgent
import numpy as np
from typing import Dict, Any

class MoonDevRiskAgent(BaseAgent):
    """
    Risk Management Agent inspirado en Moon Dev AI Agents.
    Adaptado para trading de stocks (SPY) en lugar de crypto.

    Funciones principales:
    - Validar trades propuestos por otros agentes
    - Calcular position sizing óptimo (Kelly Criterion simplificado)
    - Monitorear portfolio heat (riesgo total)
    - Stop loss dinámico basado en volatilidad
    - Max drawdown protection
    - Risk-adjusted position sizing
    """

    def __init__(self,
                 max_risk_per_trade: float = 0.02,      # 2% por trade
                 max_portfolio_heat: float = 0.10,      # 10% portfolio en riesgo
                 max_drawdown: float = 0.15,            # 15% drawdown máximo
                 volatility_lookback: int = 20,         # Períodos para calcular volatilidad
                 initial_balance: float = 10000.0):     # Balance inicial

        super().__init__(name="MoonDev_Risk_Agent", initial_balance=initial_balance)

        # Límites de riesgo
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        self.max_drawdown = max_drawdown
        self.volatility_lookback = volatility_lookback

        # Estado de riesgo
        self.portfolio_peak = self.initial_balance
        self.current_drawdown = 0.0
        self.price_history = []  # Para calcular volatilidad
        self.risk_history = []   # Historial de evaluaciones de riesgo

        # Estadísticas
        self.trades_blocked = 0
        self.trades_adjusted = 0
        self.total_evaluations = 0

        self.logger.info(f"🛡️ Risk Agent configurado: "
                        f"Max risk/trade: {max_risk_per_trade:.1%}, "
                        f"Max portfolio heat: {max_portfolio_heat:.1%}, "
                        f"Max drawdown: {max_drawdown:.1%}")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Risk Agent no toma decisiones directas de trading.
        Solo valida y ajusta trades propuestos por otros agentes.
        """
        return {
            'action': 0,  # HOLD
            'confidence': 0.0,
            'reasoning': 'Risk agent monitors and validates trades only'
        }

    def evaluate_trade_risk(self,
                           proposed_trade: Dict[str, Any],
                           portfolio_state: Dict[str, Any],
                           market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluar si un trade propuesto es seguro y calcular ajustes necesarios.

        Args:
            proposed_trade: {
                'action': int (0=HOLD, 1=BUY, 2=SELL),
                'price': float (precio de ejecución),
                'size': float (shares propuestas),
                'confidence': float (confianza del agente que propone)
            }
            portfolio_state: {
                'balance': float,
                'shares': float,
                'net_worth': float,
                'peak_value': float,
                'current_drawdown': float
            }
            market_data: Dict con datos de mercado para análisis de volatilidad

        Returns: {
            'approved': bool,
            'adjusted_size': float,
            'stop_loss': float,
            'take_profit': float (opcional),
            'risk_score': float (0-1, 1=max risk),
            'reason': str
        }
        """

        self.total_evaluations += 1

        # Actualizar price history para volatilidad
        if market_data and 'Close' in market_data:
            self.price_history.append(market_data['Close'])
            if len(self.price_history) > self.volatility_lookback:
                self.price_history.pop(0)

        # Si es HOLD, siempre aprobar
        if proposed_trade['action'] == 0:
            return {
                'approved': True,
                'adjusted_size': 0,
                'stop_loss': None,
                'take_profit': None,
                'risk_score': 0.0,
                'reason': 'HOLD approved (no risk)'
            }

        # Si es SELL, siempre aprobar (reduce riesgo)
        if proposed_trade['action'] == 2:
            current_shares = portfolio_state.get('shares', 0)
            sell_size = proposed_trade.get('size', current_shares)

            return {
                'approved': True,
                'adjusted_size': min(sell_size, current_shares),
                'stop_loss': None,
                'take_profit': None,
                'risk_score': 0.0,
                'reason': f'SELL approved (reduces risk from {current_shares:.2f} to {current_shares-sell_size:.2f} shares)'
            }

        # Validaciones para BUY
        price = proposed_trade['price']
        proposed_size = proposed_trade.get('size', 0)
        confidence = proposed_trade.get('confidence', 0.5)

        # 1. Check drawdown limit
        current_dd = portfolio_state.get('current_drawdown', 0)
        if current_dd >= self.max_drawdown:
            self.trades_blocked += 1
            return {
                'approved': False,
                'adjusted_size': 0,
                'stop_loss': None,
                'take_profit': None,
                'risk_score': 1.0,
                'reason': f'❌ BLOCKED: Drawdown {current_dd:.2%} exceeds limit {self.max_drawdown:.2%}'
            }

        # 2. Check portfolio heat
        portfolio_heat = self._calculate_portfolio_heat(portfolio_state, price)
        if portfolio_heat > self.max_portfolio_heat:
            self.trades_blocked += 1
            return {
                'approved': False,
                'adjusted_size': 0,
                'stop_loss': None,
                'take_profit': None,
                'risk_score': 0.9,
                'reason': f'❌ BLOCKED: Portfolio heat {portfolio_heat:.2%} exceeds limit {self.max_portfolio_heat:.2%}'
            }

        # 3. Calculate safe position size
        safe_size = self._calculate_safe_position_size(
            price, portfolio_state, market_data, confidence
        )

        # 4. Calculate stop loss dinámico
        stop_loss_price = self._calculate_dynamic_stop_loss(price, market_data)

        # 5. Calculate take profit (opcional)
        take_profit_price = self._calculate_take_profit(price, stop_loss_price)

        # 6. Ajustar size si es necesario
        adjusted_size = min(safe_size, proposed_size) if proposed_size > 0 else safe_size

        # 7. Calcular risk score final
        risk_score = self._calculate_risk_score(
            adjusted_size, price, portfolio_state, stop_loss_price
        )

        # Decidir si aprobar basado en risk score
        approved = risk_score < 0.7  # Threshold conservador

        if not approved:
            self.trades_blocked += 1
            reason = f'❌ BLOCKED: Risk score {risk_score:.2f} too high'
        else:
            if adjusted_size != proposed_size:
                self.trades_adjusted += 1
                reason = f'✅ APPROVED: Size adjusted from {proposed_size:.2f} to {adjusted_size:.2f} shares'
            else:
                reason = '✅ APPROVED: Trade within risk limits'

            reason += f' | SL: ${stop_loss_price:.2f} | Risk: {risk_score:.2f}'

        return {
            'approved': approved,
            'adjusted_size': adjusted_size if approved else 0,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'risk_score': risk_score,
            'reason': reason
        }

    def _calculate_portfolio_heat(self, portfolio_state: Dict[str, Any], current_price: float) -> float:
        """
        Calcular % del portfolio en riesgo.
        Portfolio heat = (valor posiciones * riesgo por posición) / net_worth
        """
        shares = portfolio_state.get('shares', 0)
        net_worth = portfolio_state.get('net_worth', portfolio_state.get('balance', 0))

        if shares <= 0 or net_worth <= 0:
            return 0.0

        # Estimar riesgo por posición basado en volatilidad
        position_value = shares * current_price
        risk_per_position = self._estimate_position_risk(current_price)

        total_risk = position_value * risk_per_position
        portfolio_heat = total_risk / net_worth

        return portfolio_heat

    def _estimate_position_risk(self, current_price: float) -> float:
        """Estimar riesgo por posición basado en volatilidad histórica"""
        if len(self.price_history) < 5:
            return 0.03  # 3% default para posiciones nuevas

        # Calcular volatilidad (std dev de retornos)
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.02

        # Risk = 2 * volatilidad (aproximación conservadora)
        return min(volatility * 2, 0.10)  # Máximo 10%

    def _calculate_safe_position_size(self,
                                    price: float,
                                    portfolio_state: Dict[str, Any],
                                    market_data: Dict[str, Any] = None,
                                    confidence: float = 0.5) -> float:
        """
        Calcular position size seguro usando múltiples métodos.

        Methods:
        1. % Risk method (max_risk_per_trade del portfolio)
        2. Volatility-adjusted sizing
        3. Kelly Criterion simplificado
        4. Confidence-adjusted sizing
        """

        # Validar precio para evitar división por cero
        if price <= 0:
            return 0

        balance = portfolio_state.get('balance', 0)
        net_worth = portfolio_state.get('net_worth', balance)

        if balance <= 0 or net_worth <= 0:
            return 0

        # Método 1: Risk-based sizing
        max_capital_at_risk = net_worth * self.max_risk_per_trade

        # Estimar stop loss distance
        stop_distance = self._estimate_stop_distance(price, market_data)
        risk_per_share = stop_distance

        risk_based_size = max_capital_at_risk / risk_per_share if risk_per_share > 0 else 0

        # Método 2: Balance-based sizing (máximo 70% del balance)
        balance_based_size = (balance * 0.70) / price

        # Método 3: Confidence adjustment
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5-1.0 basado en confidence

        # Método 4: Volatility adjustment
        volatility = self._calculate_current_volatility()
        volatility_multiplier = 1.0 / (1.0 + volatility * 10)  # Reduce size en alta volatilidad

        # Combinar métodos (tomar el mínimo más conservador)
        safe_size = min(
            risk_based_size,
            balance_based_size
        ) * confidence_multiplier * volatility_multiplier

        return max(0, safe_size)

    def _estimate_stop_distance(self, price: float, market_data: Dict[str, Any] = None) -> float:
        """Estimar distancia al stop loss"""
        if price <= 0:
            return 0.01  # Valor mínimo para evitar división por cero

        if market_data and 'ATR' in market_data:
            # Usar ATR si está disponible
            return market_data['ATR'] * 1.5
        elif len(self.price_history) >= 5:
            # Calcular basado en volatilidad reciente
            prices = np.array(self.price_history[-5:])
            volatility = np.std(np.diff(prices) / prices[:-1])
            return price * volatility * 2  # 2 std dev
        else:
            # Default conservador: 3% del precio
            return price * 0.03

    def _calculate_dynamic_stop_loss(self, entry_price: float, market_data: Dict[str, Any] = None) -> float:
        """Calcular stop loss dinámico"""
        stop_distance = self._estimate_stop_distance(entry_price, market_data)
        return entry_price - stop_distance

    def _calculate_take_profit(self, entry_price: float, stop_loss: float) -> float:
        """Calcular take profit (risk-reward ratio 1:2)"""
        risk = entry_price - stop_loss
        return entry_price + (risk * 2)

    def _calculate_current_volatility(self) -> float:
        """Calcular volatilidad actual"""
        if len(self.price_history) < 5:
            return 0.02  # 2% default

        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    def _calculate_risk_score(self,
                            size: float,
                            price: float,
                            portfolio_state: Dict[str, Any],
                            stop_loss: float) -> float:
        """
        Calcular score de riesgo del trade (0-1, donde 1 es máximo riesgo)

        Factores:
        - Size relative to portfolio
        - Distance to stop loss
        - Current drawdown
        - Portfolio concentration
        """

        if size <= 0 or price <= 0:
            return 0.0

        net_worth = portfolio_state.get('net_worth', portfolio_state.get('balance', 0))
        if net_worth <= 0:
            return 1.0

        # Factor 1: Position size relative to portfolio
        position_value = size * price
        size_factor = min(position_value / net_worth, 1.0)

        # Factor 2: Risk per share (stop loss distance)
        risk_per_share = price - stop_loss if stop_loss < price else price * 0.03
        risk_factor = min(risk_per_share / price, 0.1) / 0.1  # Normalize to 0-1

        # Factor 3: Current drawdown
        dd_factor = portfolio_state.get('current_drawdown', 0) / self.max_drawdown
        dd_factor = min(dd_factor, 1.0)

        # Factor 4: Portfolio concentration
        current_shares = portfolio_state.get('shares', 0)
        concentration_factor = (current_shares + size) * price / net_worth
        concentration_factor = min(concentration_factor, 1.0)

        # Risk score = weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # Size, Risk, DD, Concentration
        risk_score = (
            size_factor * weights[0] +
            risk_factor * weights[1] +
            dd_factor * weights[2] +
            concentration_factor * weights[3]
        )

        return min(risk_score, 1.0)

    def get_risk_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del risk management"""
        return {
            'total_evaluations': self.total_evaluations,
            'trades_blocked': self.trades_blocked,
            'trades_adjusted': self.trades_adjusted,
            'block_rate': self.trades_blocked / self.total_evaluations if self.total_evaluations > 0 else 0,
            'adjustment_rate': self.trades_adjusted / self.total_evaluations if self.total_evaluations > 0 else 0,
            'current_drawdown': self.current_drawdown,
            'portfolio_peak': self.portfolio_peak
        }

    def reset_risk_state(self):
        """Resetear estado de riesgo"""
        self.portfolio_peak = self.initial_balance
        self.current_drawdown = 0.0
        self.price_history = []
        self.risk_history = []
        self.trades_blocked = 0
        self.trades_adjusted = 0
        self.total_evaluations = 0
        self.logger.info("🔄 Risk state reseteado")